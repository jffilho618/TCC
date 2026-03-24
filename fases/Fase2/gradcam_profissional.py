import os
import cv2
import numpy as np
import torch
import glob
import timm
from PIL import Image, ImageDraw, ImageFont
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

# --- UI IMPORTS ---
from rich.console import Console
from rich.progress import track
from rich.panel import Panel

console = Console()

# ================= CONFIGURAÇÕES =================
# Caminhos relativos ao diretório do script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASE2_DIR = SCRIPT_DIR  # Este script já está em Fase2

# Verificar onde está classificacao_crops (pode estar em Fase1.5 ou Fase2)
PASTA_BASE_FASE2 = os.path.join(SCRIPT_DIR, 'classificacao_crops')
PASTA_BASE_FASE15 = os.path.join(os.path.dirname(SCRIPT_DIR), 'Fase1.5', 'classificacao_crops')

if os.path.exists(PASTA_BASE_FASE2):
    PASTA_BASE = PASTA_BASE_FASE2
elif os.path.exists(PASTA_BASE_FASE15):
    PASTA_BASE = PASTA_BASE_FASE15
else:
    PASTA_BASE = 'classificacao_crops'  # Fallback

# Pastas (Ajustadas para sua estrutura Fase 2)
PASTA_MODELOS = os.path.join(FASE2_DIR, 'modelos_fase2_final')
PASTA_TESTE_MALIGNO = os.path.join(PASTA_BASE, 'teste_final', 'maligno')
OUTPUT_FOLDER = os.path.join(FASE2_DIR, 'HEATMAPS_FASE2_HD')

# Quantas imagens gerar para testar?
QTD_IMAGENS = 10 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =================================================

class ApplyCLAHE(object):
    """Mesmo filtro usado no treino para garantir fidelidade."""
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_np):
        # Espera numpy array (H, W, C)
        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            final = clahe.apply(img_np)
            final = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
        return final

# --- FUNÇÕES DE SUPORTE PARA TRANSFORMERS ---
def reshape_transform_vit(tensor):
    height = int(np.sqrt(tensor.shape[1] - 1))
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, height, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_swin(tensor):
    # Swin outputa (B, H, W, C), GradCAM precisa de (B, C, H, W)
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    return tensor

def get_target_layer(model_name, model):
    """Identifica automaticamente a última camada convolucional/atenção."""
    if 'efficientnet' in model_name:
        return [model.conv_head]
    elif 'resnet' in model_name:
        return [model.layer4[-1]]
    elif 'densenet' in model_name:
        return [model.features[-1]]
    elif 'swin' in model_name:
        # Tenta pegar a norma final do último bloco
        return [model.layers[-1].blocks[-1].norm1]
    elif 'beit' in model_name or 'vit' in model_name:
        return [model.blocks[-1].norm1]
    else:
        # Fallback genérico
        return [list(model.children())[-2]]

def preprocess_image_custom(img_path, img_size):
    """Lê, redimensiona e aplica CLAHE igual ao treino."""
    # 1. Ler Original
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Aplicar CLAHE (Importante para o GradCAM ver a textura)
    clahe_filter = ApplyCLAHE()
    img_clahe = clahe_filter(img)
    
    # 3. Preparar para o Modelo (Normalização ImageNet)
    img_resized = cv2.resize(img_clahe, (img_size, img_size))
    img_float = np.float32(img_resized) / 255.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(Image.fromarray(img_resized)).unsqueeze(0)
    
    return img_clahe, img_float, input_tensor

def create_activation_matrix(heatmap, grid_size=9):
    """Cria visualização da matriz de ativação com números."""
    # 1. Redimensiona o heatmap para grid_size x grid_size
    heatmap_small = cv2.resize(heatmap, (grid_size, grid_size), interpolation=cv2.INTER_AREA)

    # 2. Define tamanho de cada célula (maior para ver bem os números)
    cell_size = 60  # pixels por célula
    matrix_h = grid_size * cell_size
    matrix_w = grid_size * cell_size

    # 3. Cria imagem da matriz com fundo branco
    matrix_img = np.ones((matrix_h, matrix_w, 3), dtype=np.uint8) * 255

    # 4. Preenche cada célula com cor e valor
    for i in range(grid_size):
        for j in range(grid_size):
            value = heatmap_small[i, j]

            # Coordenadas da célula
            y1 = i * cell_size
            y2 = (i + 1) * cell_size
            x1 = j * cell_size
            x2 = (j + 1) * cell_size

            # Cor de fundo baseada na ativação (JET colormap)
            color_val = int(value * 255)
            color_bgr = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            color_rgb = tuple(map(int, [color_bgr[2], color_bgr[1], color_bgr[0]]))

            # Desenha retângulo colorido
            cv2.rectangle(matrix_img, (x1, y1), (x2, y2), color_rgb, -1)

            # Borda da célula
            cv2.rectangle(matrix_img, (x1, y1), (x2, y2), (200, 200, 200), 1)

            # Texto com o valor (em preto ou branco dependendo do fundo)
            text = f"{value:.2f}"
            text_color = (255, 255, 255) if value > 0.5 else (0, 0, 0)

            # Centraliza o texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x1 + (cell_size - text_w) // 2
            text_y = y1 + (cell_size + text_h) // 2

            cv2.putText(matrix_img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return matrix_img

def create_output_image(original_hd, heatmap, prediction_score, model_name):
    """Monta a imagem final em Alta Resolução com matriz de ativação."""
    h_orig, w_orig = original_hd.shape[:2]

    # 1. Upscale do Heatmap para o tamanho original HD
    heatmap_resized = cv2.resize(heatmap, (w_orig, h_orig))

    # 2. Criar Overlay
    # Heatmap colorido
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Mistura: 60% Original + 40% Calor
    overlay = cv2.addWeighted(original_hd, 0.6, heatmap_color, 0.4, 0)

    # 3. Criar matriz de ativação 9x9
    activation_matrix = create_activation_matrix(heatmap, grid_size=9)

    # 4. Ajustar altura da matriz para coincidir com as imagens
    matrix_h_target = h_orig
    matrix_w_current = activation_matrix.shape[1]
    matrix_h_current = activation_matrix.shape[0]

    # Redimensiona mantendo proporção
    scale = matrix_h_target / matrix_h_current
    matrix_w_new = int(matrix_w_current * scale)
    activation_matrix_resized = cv2.resize(activation_matrix, (matrix_w_new, matrix_h_target), interpolation=cv2.INTER_LINEAR)

    # 5. Montagem: Original | Overlay | Matriz
    combined = np.hstack((original_hd, overlay, activation_matrix_resized))

    # 6. Adicionar Rodapé com Info
    footer_h = 50
    footer = np.ones((footer_h, combined.shape[1], 3), dtype=np.uint8) * 255 # Branco

    # Texto
    cv2.putText(footer,
                f"Modelo: {model_name} | Predicao Maligno: {prediction_score:.2f}% | Matriz de Ativacao 9x9",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    final_img = np.vstack((combined, footer))
    return final_img

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    console.rule("[bold blue]GERADOR DE GRAD-CAM FASE 2 (HD)[/]")

    # Debug: Mostrar caminhos configurados
    console.print(f"[dim]PASTA_BASE: {PASTA_BASE}[/]")
    console.print(f"[dim]PASTA_MODELOS: {PASTA_MODELOS}[/]")
    console.print(f"[dim]PASTA_TESTE_MALIGNO: {PASTA_TESTE_MALIGNO}[/]")
    console.print(f"[dim]OUTPUT_FOLDER: {OUTPUT_FOLDER}[/]")
    console.print()

    # 1. Buscar Modelos Treinados
    model_files = glob.glob(os.path.join(PASTA_MODELOS, "*_fase2.pth"))
    if not model_files:
        console.print(f"[bold red]Nenhum modelo encontrado em {PASTA_MODELOS}![/]")
        console.print(f"[yellow]Pasta existe? {os.path.exists(PASTA_MODELOS)}[/]")
        if os.path.exists(PASTA_MODELOS):
            console.print(f"[yellow]Conteúdo da pasta:[/]")
            for f in os.listdir(PASTA_MODELOS):
                console.print(f"  - {f}")
        return

    # 2. Buscar Imagens de Teste
    img_paths = glob.glob(os.path.join(PASTA_TESTE_MALIGNO, "*.png")) + \
                glob.glob(os.path.join(PASTA_TESTE_MALIGNO, "*.jpg")) + \
                glob.glob(os.path.join(PASTA_TESTE_MALIGNO, "*.jpeg"))

    if not img_paths:
        console.print(f"[bold red]Nenhuma imagem encontrada em {PASTA_TESTE_MALIGNO}![/]")
        return

    # Pega as primeiras X imagens (ou aleatórias se preferir)
    img_paths = img_paths[:QTD_IMAGENS]

    console.print(f"[cyan]Processando {len(model_files)} modelos em {len(img_paths)} imagens...[/]")

    for model_path in model_files:
        filename = os.path.basename(model_path)
        # Extrai nome do modelo (ex: resnet50_fase2.pth -> resnet50)
        clean_name = filename.replace("_fase2.pth", "")
        
        console.print(f"\n[bold yellow]➤ Carregando: {clean_name}[/]")
        
        try:
            # Detecta tamanho ideal baseado no modelo
            if 'beit' in clean_name:
                img_size = 224
            elif 'swin' in clean_name and '256' in clean_name:
                img_size = 256
            else:
                img_size = 384
            
            # Carrega Modelo
            model = timm.create_model(clean_name, pretrained=False, num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            
            # Configura GradCAM
            target_layers = get_target_layer(clean_name, model)
            reshape_func = None
            
            if 'swin' in clean_name: reshape_func = reshape_transform_swin
            if 'beit' in clean_name or 'vit' in clean_name: reshape_func = reshape_transform_vit
            
            # Usa GradCAM++ (Geralmente melhor para localização fina)
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_func)
            
            for img_path in track(img_paths, description=f"Gerando Heatmaps ({clean_name})..."):
                img_name = os.path.basename(img_path)
                
                # Preprocessamento (Original com CLAHE, Tensor normalizado)
                orig_hd, _, input_tensor = preprocess_image_custom(img_path, img_size)
                input_tensor = input_tensor.to(DEVICE)
                
                # Inferência (Para pegar a porcentagem de certeza)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    score_maligno = probs[0][1].item() * 100 # %
                
                # Gera Heatmap para a classe 1 (Maligno)
                targets = [ClassifierOutputTarget(1)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                
                # Monta Imagem Final HD
                final_image = create_output_image(orig_hd, grayscale_cam, score_maligno, clean_name)
                
                # Salva: HEATMAPS/resnet50_IMG001_Prob99.jpg
                # Remove extensão do nome original (pode ser .png, .jpg ou .jpeg)
                img_name_noext = os.path.splitext(img_name)[0]
                save_name = f"{clean_name}_{img_name_noext}_Prob{int(score_maligno)}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, save_name), cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
                
        except Exception as e:
            console.print(f"[bold red]Erro no modelo {clean_name}: {e}[/]")
            # import traceback
            # traceback.print_exc()

    console.print(f"\n[bold green]Concluído! Verifique a pasta '{OUTPUT_FOLDER}'[/]")

if __name__ == "__main__":
    main()