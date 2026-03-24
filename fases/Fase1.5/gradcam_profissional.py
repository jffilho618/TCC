import os
import cv2
import numpy as np
import torch
import glob
import timm
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI para evitar erros
import matplotlib.pyplot as plt
from matplotlib import cm

# ================= CONFIGURAÇÕES =================
# Obtém o diretório onde este script está localizado
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PASTA_IMAGENS_ORIGINAIS = os.path.join(SCRIPT_DIR, '..', 'BTXRD', 'annotated_images')  # Imagens segmentadas
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'GRADCAM_PROFISSIONAL')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
QTD_IMAGENS = 5  # Quantas imagens gerar por modelo
TAMANHO_MATRIZ = 8  # Tamanho da matriz de ativação (8x8)

# Dicionário de tamanhos de imagem por modelo
MODEL_IMG_SIZES = {
    'swinv2_tiny_window16_256': 256,
    'maxvit_rmlp_tiny_rw_256': 256,
}
# =================================================

# --- FUNÇÕES AUXILIARES PARA TRANSFORMERS ---
def reshape_transform_vit(tensor):
    # Para ViT: tensor shape é [batch, num_tokens, channels]
    # Precisamos descobrir quantos tokens espaciais temos
    total_tokens = tensor.shape[1]

    # Tentar diferentes números de tokens especiais (1 ou 2)
    # e ver qual resulta em um quadrado perfeito
    for num_special in [1, 2]:
        num_spatial = total_tokens - num_special
        height = int(np.sqrt(num_spatial))
        if height * height == num_spatial:
            # Encontramos um quadrado perfeito!
            tensor_without_special = tensor[:, num_special:, :]
            result = tensor_without_special.reshape(tensor.size(0), height, height, tensor.size(2))
            result = result.permute(0, 3, 1, 2)
            return result

    # Fallback: assume 1 token especial
    tensor_without_cls = tensor[:, 1:, :]
    num_spatial = tensor_without_cls.shape[1]
    height = int(np.sqrt(num_spatial))
    result = tensor_without_cls.reshape(tensor.size(0), height, height, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

def reshape_transform_swin(tensor):
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    return tensor

def reshape_transform_maxvit(tensor):
    # MaxViT pode retornar tensores em diferentes formatos
    # Formato CNN: [B, C, H, W] - já está correto
    if len(tensor.shape) == 4 and tensor.shape[1] > tensor.shape[3]:
        return tensor  # Já está em formato [B, C, H, W]

    # Formato atenção: [B, H, W, C] - precisa permutar
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    return tensor

# --- MAPEAMENTO DE CAMADAS ALVO ---
def get_target_layer(model_name, model):
    """Define onde o GradCAM deve 'olhar' dependendo da arquitetura"""

    # IMPORTANTE: Verificar substrings mais específicas primeiro!
    if 'maxvit' in model_name:
        # MaxViT: model.stages[-1].blocks[-1]
        return model.stages[-1].blocks[-1]
    elif 'swin' in model_name:
        return model.layers[-1].blocks[-1].norm1
    elif 'coatnet' in model_name:
        # CoAtNet tem stages
        return model.stages[-1]
    elif 'efficientnet' in model_name:
        return model.conv_head
    elif 'densenet' in model_name:
        return model.features[-1]
    elif 'resnet' in model_name:
        return model.layer4[-1]
    elif 'deit' in model_name:
        # DeiT distilled tem estrutura diferente
        return model.blocks[-1].norm1
    elif 'vit' in model_name or 'beit' in model_name:
        return model.blocks[-1].norm1
    else:
        print(f"Aviso: Arquitetura {model_name} desconhecida, tentando layer genérico.")
        return list(model.children())[-2]

def criar_matriz_ativacao(grayscale_cam, tamanho=8):
    """
    Cria uma visualização da matriz de ativação 8x8 com valores numéricos
    Similar ao artigo de pólipos
    """
    # Redimensionar o mapa de calor para 8x8
    cam_resized = cv2.resize(grayscale_cam, (tamanho, tamanho), interpolation=cv2.INTER_LINEAR)

    # Configurar matplotlib para gerar a matriz
    # Ajustar tamanho da figura baseado no tamanho da matriz
    fig_size = max(8, tamanho * 0.6)  # Escala dinâmica
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Criar heatmap com colormap personalizado
    im = ax.imshow(cam_resized, cmap='viridis', aspect='auto', vmin=0, vmax=1)

    # Ajustar tamanho da fonte baseado no tamanho da matriz
    font_size = max(6, 14 - tamanho // 2)  # Fonte menor para matrizes maiores

    # Adicionar valores numéricos em cada célula
    for i in range(tamanho):
        for j in range(tamanho):
            valor = cam_resized[i, j]
            # Cor do texto: branco para valores escuros, preto para claros
            cor_texto = 'white' if valor < 0.5 else 'black'
            ax.text(j, i, f'{valor:.1f}',
                   ha="center", va="center",
                   color=cor_texto, fontsize=font_size, fontweight='bold')

    # Remover ticks e labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Adicionar título
    ax.set_title(f'Ativação da Rede ({tamanho}x{tamanho})',
                fontsize=14, fontweight='bold', pad=15)

    # Converter figura matplotlib para imagem numpy (método compatível)
    fig.canvas.draw()

    # Usar buffer_rgba() ao invés de tostring_rgb()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    matriz_img = buf.reshape(h, w, 4)

    # Converter RGBA para RGB
    matriz_img = matriz_img[:, :, :3]

    plt.close(fig)

    return matriz_img

def adicionar_titulo(img, titulo, posicao='top'):
    """
    Adiciona título elegante à imagem
    """
    h, w = img.shape[:2]

    # Criar barra para o título
    barra_altura = 50
    barra = np.zeros((barra_altura, w, 3), dtype=np.uint8)

    # Adicionar texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Calcular posição centralizada do texto
    (text_width, text_height), _ = cv2.getTextSize(titulo, font, font_scale, thickness)
    x = (w - text_width) // 2
    y = (barra_altura + text_height) // 2

    cv2.putText(barra, titulo, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Combinar barra com imagem
    if posicao == 'top':
        img_com_titulo = np.vstack([barra, img])
    else:
        img_com_titulo = np.vstack([img, barra])

    return img_com_titulo

def criar_visualizacao_completa(img_original, grayscale_cam, nome_arquivo, nome_modelo, img_size):
    """
    Cria a visualização profissional estilo artigo:
    [Original Segmentada] | [Foco da IA (Heatmap)] | [Matriz 7x7]
    """
    # 1. Imagem Original (já segmentada do annotated_images)
    img_esq = cv2.resize(img_original, (img_size, img_size))

    # 2. Mapa de Calor Sobreposto (Centro)
    rgb_img_float = np.float32(img_original) / 255
    rgb_img_float = cv2.resize(rgb_img_float, (img_size, img_size))
    heatmap_overlay = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    # 3. Matriz de Ativação 7x7 (Direita)
    matriz_img = criar_matriz_ativacao(grayscale_cam, TAMANHO_MATRIZ)
    # Redimensionar matriz para mesma altura que outras imagens
    altura_alvo = img_esq.shape[0]
    fator_escala = altura_alvo / matriz_img.shape[0]
    nova_largura = int(matriz_img.shape[1] * fator_escala)
    matriz_img_resized = cv2.resize(matriz_img, (nova_largura, altura_alvo))

    # Combinar as três imagens lado a lado
    # Garantir que todas tenham a mesma altura
    altura_maxima = max(img_esq.shape[0], heatmap_overlay.shape[0], matriz_img_resized.shape[0])

    def centralizar_vertical(img, altura_alvo):
        if img.shape[0] < altura_alvo:
            pad_top = (altura_alvo - img.shape[0]) // 2
            pad_bottom = altura_alvo - img.shape[0] - pad_top
            img = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)
        return img

    img_esq = centralizar_vertical(img_esq, altura_maxima)
    heatmap_overlay = centralizar_vertical(heatmap_overlay, altura_maxima)
    matriz_img_resized = centralizar_vertical(matriz_img_resized, altura_maxima)

    # Adicionar pequeno espaçamento entre as imagens
    espacamento = np.zeros((altura_maxima, 20, 3), dtype=np.uint8)

    # Montar visualização final
    resultado = np.hstack([
        img_esq,
        espacamento,
        heatmap_overlay,
        espacamento,
        matriz_img_resized
    ])

    # Adicionar barra superior com informações do modelo e imagem
    barra_info_altura = 60
    largura_total = resultado.shape[1]
    barra_info = np.zeros((barra_info_altura, largura_total, 3), dtype=np.uint8)
    barra_info[:] = (40, 40, 40)  # Cor de fundo cinza escuro

    # Adicionar texto na barra
    font = cv2.FONT_HERSHEY_SIMPLEX
    texto_modelo = f'Modelo: {nome_modelo}'
    texto_imagem = f'Imagem: {nome_arquivo}'

    cv2.putText(barra_info, texto_modelo, (20, 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(barra_info, texto_imagem, (20, 50), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # Combinar barra com resultado
    visualizacao_final = np.vstack([barra_info, resultado])

    return visualizacao_final

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Achar os modelos treinados (.pth)
    model_files = glob.glob(os.path.join(SCRIPT_DIR, "modelos_salvos_crops", "*_best.pth"))
    if not model_files:
        print("Nenhum arquivo .pth encontrado em modelos_salvos_crops/!")
        return

    # 2. Selecionar imagens de teste (das anotadas)
    img_paths = glob.glob(os.path.join(PASTA_IMAGENS_ORIGINAIS, "*.jpeg"))[:QTD_IMAGENS]
    if not img_paths:
        print(f"Nenhuma imagem encontrada em {PASTA_IMAGENS_ORIGINAIS}")
        return

    print(f"Encontrados {len(model_files)} modelos e {len(img_paths)} imagens.")
    print(f"Gerando visualizações profissionais estilo artigo científico...")

    for model_path in model_files:
        # Extrair nome do modelo
        nome_arquivo = os.path.basename(model_path)
        nome_limpo = nome_arquivo.replace("_best.pth", "").replace(".pth", "")

        print(f"\n--- Processando Modelo: {nome_limpo} ---")

        try:
            # Determinar tamanho de imagem para este modelo
            IMG_SIZE = MODEL_IMG_SIZES.get(nome_limpo, 224)
            print(f"    Usando tamanho de imagem: {IMG_SIZE}x{IMG_SIZE}")

            # Carregar Modelo
            model = timm.create_model(nome_limpo, pretrained=False, num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()

            # Configurar GradCAM
            target_layers = [get_target_layer(nome_limpo, model)]
            reshape_func = None

            if 'maxvit' in nome_limpo:
                reshape_func = reshape_transform_maxvit
            elif any(x in nome_limpo for x in ['vit', 'deit', 'beit']):
                reshape_func = reshape_transform_vit
            elif 'swin' in nome_limpo:
                reshape_func = reshape_transform_swin

            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_func)

            # Loop nas imagens
            for img_path in img_paths:
                # Ler imagem original (BGR -> RGB)
                img_original = cv2.imread(img_path, 1)
                img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

                # Preparar para o modelo
                rgb_img_resized = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))
                rgb_img_float = np.float32(rgb_img_resized) / 255

                input_tensor = preprocess_image(rgb_img_resized,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).to(DEVICE)

                # Gerar Grad-CAM (alvo = classe maligno)
                targets = [ClassifierOutputTarget(1)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]

                # Criar visualização profissional completa
                nome_img = os.path.basename(img_path)
                visualizacao = criar_visualizacao_completa(
                    img_original,
                    grayscale_cam,
                    nome_img,
                    nome_limpo,
                    IMG_SIZE
                )

                # Salvar
                nome_img_sem_ext = os.path.splitext(nome_img)[0]
                save_name = f"{nome_limpo}_{nome_img_sem_ext}_gradcam_profissional.png"
                save_path = os.path.join(OUTPUT_FOLDER, save_name)

                # Converter RGB -> BGR para salvar com OpenCV
                cv2.imwrite(save_path, cv2.cvtColor(visualizacao, cv2.COLOR_RGB2BGR))
                print(f"✓ Salvo: {save_name}")

        except Exception as e:
            print(f"✗ Erro ao processar {nome_limpo}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Concluído! Visualizações salvas em: {OUTPUT_FOLDER}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
