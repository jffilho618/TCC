import os
import cv2
import numpy as np
import torch
import glob
import timm
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# ================= CONFIGURAÇÕES =================
PASTA_IMAGENS = os.path.join('classificacao', 'teste_final', 'maligno')
OUTPUT_FOLDER = 'HEATMAPS_FINAIS'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
QTD_IMAGENS = 3 # Quantas imagens gerar por modelo
# =================================================

# --- FUNÇÕES AUXILIARES PARA TRANSFORMERS (VIT/SWIN) ---
def reshape_transform_vit(tensor):
    # ViTs geralmente saem como [Batch, Tokens, Channels]
    # Precisamos ignorar o CLS token (índice 0) e transformar em 2D
    height = int(np.sqrt(tensor.shape[1] - 1))
    target_size = (height, height)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, height, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_swin(tensor):
    # Swin Transformers já costumam sair espaciais ou precisam de permuta
    # Tentativa genérica para Swin V2 do Timm
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    return tensor

# --- MAPEAMENTO DE CAMADAS ALVO (O SEGREDO DO SUCESSO) ---
def get_target_layer(model_name, model):
    """Define onde o GradCAM deve 'olhar' dependendo da arquitetura"""
    
    # 1. CNNs
    if 'efficientnet' in model_name:
        return model.conv_head # Última convolução
    elif 'densenet' in model_name:
        return model.features[-1] # Último bloco denso
    elif 'resnet' in model_name:
        return model.layer4[-1]
    
    # 2. Vision Transformers (ViT, DeiT, BEiT)
    elif 'vit' in model_name or 'deit' in model_name or 'beit' in model_name:
        # Geralmente pegamos a Normalização do último bloco antes do head
        return model.blocks[-1].norm1
        
    # 3. Híbridos (Swin, MaxViT, CoAtNet)
    elif 'swin' in model_name:
        return model.layers[-1].blocks[-1].norm1
    elif 'coatnet' in model_name:
        # CoAtNet termina com blocos de atenção ou conv
        # Tentativa de pegar o último estágio
        return model.stages[-1].blocks[-1]
    elif 'maxvit' in model_name:
        return model.stages[-1].blocks[-1]
        
    else:
        # Fallback genérico (Tenta achar a última camada conv)
        print(f"Aviso: Arquitetura {model_name} desconhecida, tentando layer genérico.")
        return list(model.children())[-2]

def adicionar_texto(img, texto):
    """Escreve o nome do modelo na imagem"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Achar os modelos treinados (.pth) na pasta modelos_salvos
    model_files = glob.glob(os.path.join("modelos_salvos", "*_best.pth"))
    if not model_files:
        print("Nenhum arquivo .pth encontrado em modelos_salvos/!")
        return

    # 2. Selecionar imagens de teste
    img_paths = glob.glob(os.path.join(PASTA_IMAGENS, "*"))[:QTD_IMAGENS]
    if not img_paths:
        print(f"Nenhuma imagem encontrada em {PASTA_IMAGENS}")
        return

    print(f"Encontrados {len(model_files)} modelos. Gerando Grad-CAMs...")

    for model_path in model_files:
        # Extrair nome do modelo do arquivo (ex: modelos_salvos/densenet201_best.pth -> densenet201)
        nome_arquivo = os.path.basename(model_path)
        nome_limpo = nome_arquivo.replace("_best.pth", "").replace(".pth", "")

        print(f"\n--- Processando: {nome_limpo} ---")
        
        try:
            # Carregar Modelo
            model = timm.create_model(nome_limpo, pretrained=False, num_classes=2)
            model.load_state_dict(torch.load(model_path))
            model.to(DEVICE)
            model.eval()
            
            # Configurar GradCAM (Reshape vs Standard)
            target_layers = [get_target_layer(nome_limpo, model)]
            reshape_func = None
            
            if any(x in nome_limpo for x in ['vit', 'deit', 'beit']):
                reshape_func = reshape_transform_vit
            elif 'swin' in nome_limpo:
                reshape_func = reshape_transform_swin
            
            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_func)

            # Loop nas imagens
            for img_path in img_paths:
                # Ler e preparar imagem
                rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
                rgb_img = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
                rgb_img_float = np.float32(rgb_img) / 255
                
                input_tensor = preprocess_image(rgb_img, 
                                                mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225]).to(DEVICE)

                # Gerar o Mapa de Calor (Alvo = Classe 1 Maligno)
                targets = [ClassifierOutputTarget(1)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                
                # Sobreposição
                visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
                
                # Montagem Profissional: [Original] [Heatmap] [Overlay]
                heatmap_color = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                # Converter para RGB para bater com a visualização
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                
                # Juntar lado a lado
                combined = np.hstack((rgb_img, heatmap_color, visualization))
                
                # Salvar com nome da imagem incluído
                nome_img = os.path.basename(img_path)
                nome_img_sem_ext = os.path.splitext(nome_img)[0]
                save_name = f"{nome_limpo}_{nome_img_sem_ext}_gradcam.png"
                save_path = os.path.join(OUTPUT_FOLDER, save_name)

                # Converter de volta para BGR para o OpenCV salvar certo
                cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                print(f"Salvo: {save_path} (Imagem: {nome_img})")

        except Exception as e:
            print(f"Erro ao gerar GradCAM para {nome_limpo}: {e}")
            # Em transformers complexos, o target layer as vezes precisa de ajuste manual
            print("Dica: Se for erro de 'layer', verifique a função get_target_layer.")

    print(f"\nConcluído! Verifique a pasta '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    main()