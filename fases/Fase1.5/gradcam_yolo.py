import os
import cv2
import numpy as np
import torch
import glob
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= CONFIGURAÇÕES =================
# Obtém o diretório onde este script está localizado
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PASTA_IMAGENS_ORIGINAIS = os.path.join(SCRIPT_DIR, '..', 'BTXRD', 'annotated_images')
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'GRADCAM_PROFISSIONAL')

# Tentar diferentes localizações do modelo
MODELO_YOLO_OPTIONS = [
    os.path.join(SCRIPT_DIR, 'YOLO_Resultados', 'experimento_yolo_balanceado', 'weights', 'best.pt'),
    os.path.join(SCRIPT_DIR, 'YOLO_Resultados', 'experimento_yolo_balanceado', 'weights', 'last.pt'),
    'yolov8s-cls.pt',  # Modelo de CLASSIFICAÇÃO pré-treinado
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
QTD_IMAGENS = 5
IMG_SIZE = 224
TAMANHO_MATRIZ = 8
# =================================================

def criar_matriz_ativacao(grayscale_cam, tamanho=8):
    """Cria visualização da matriz de ativação com valores numéricos"""
    cam_resized = cv2.resize(grayscale_cam, (tamanho, tamanho), interpolation=cv2.INTER_LINEAR)

    fig_size = max(8, tamanho * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cam_resized, cmap='viridis', aspect='auto', vmin=0, vmax=1)

    font_size = max(6, 14 - tamanho // 2)

    for i in range(tamanho):
        for j in range(tamanho):
            valor = cam_resized[i, j]
            cor_texto = 'white' if valor < 0.5 else 'black'
            ax.text(j, i, f'{valor:.1f}',
                   ha="center", va="center",
                   color=cor_texto, fontsize=font_size, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Ativação da Rede ({tamanho}x{tamanho})',
                fontsize=14, fontweight='bold', pad=15)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    matriz_img = buf.reshape(h, w, 4)[:, :, :3]

    plt.close(fig)
    return matriz_img

def gerar_gradcam_manual(model, img_tensor, target_class=1):
    """
    Gera GradCAM manual para YOLO classificação usando hooks
    """
    # Armazenar ativações e gradientes
    activations = []
    gradients = []

    # Hook para capturar ativações
    def forward_hook(module, input, output):
        # YOLO pode retornar tuplas, pegar apenas tensor
        if isinstance(output, tuple):
            output = output[0]
        activations.append(output)

    # Hook para capturar gradientes
    def backward_hook(module, grad_input, grad_output):
        if isinstance(grad_output[0], torch.Tensor):
            gradients.append(grad_output[0])

    # Encontrar última camada convolucional (não a camada Classify)
    target_layer = None
    backbone = model.model.model if hasattr(model.model, 'model') else model.model

    for i in range(len(backbone) - 1, -1, -1):
        layer = backbone[i]

        # Pular camadas de classificação (Classify, Linear, etc)
        if 'Classify' in type(layer).__name__ or isinstance(layer, torch.nn.Linear):
            continue

        # Procurar camadas com convoluções
        if hasattr(layer, 'conv'):
            target_layer = layer
            print(f"Target layer encontrada na posição [{i}]: {type(layer).__name__}")
            break
        elif isinstance(layer, torch.nn.Conv2d):
            target_layer = layer
            print(f"Target layer Conv2d na posição [{i}]")
            break

    if target_layer is None:
        # Fallback: pegar camada -3 (antes de pooling e classify)
        target_layer = backbone[-3] if len(backbone) > 3 else backbone[-2]
        print(f"Usando fallback: {type(target_layer).__name__}")

    # Registrar hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        model.model.eval()
        img_tensor.requires_grad = True

        # Passar pela rede
        output = model.model(img_tensor)

        # Se output é tupla, pegar primeiro elemento
        if isinstance(output, tuple):
            output = output[0]

        # Zerar gradientes
        model.model.zero_grad()

        # Backward na classe alvo
        class_score = output[0, target_class]
        class_score.backward()

        # Calcular GradCAM
        if len(activations) > 0 and len(gradients) > 0:
            activation = activations[0]
            gradient = gradients[0]

            # Verificar dimensões
            print(f"Activation shape: {activation.shape}, Gradient shape: {gradient.shape}")

            # Se gradiente é 4D [B, C, H, W]
            if len(gradient.shape) == 4:
                # Global average pooling dos gradientes
                weights = torch.mean(gradient, dim=(2, 3), keepdim=True)

                # Weighted combination
                cam = torch.sum(weights * activation, dim=1, keepdim=True)

            # Se gradiente é 2D [B, C] (camada fully connected)
            elif len(gradient.shape) == 2:
                # Usar apenas a ativação (não há dimensão espacial nos gradientes)
                # Fazer média global na ativação
                cam = torch.mean(activation, dim=1, keepdim=True)

            else:
                print(f"Formato inesperado de gradiente: {gradient.shape}")
                return np.zeros((IMG_SIZE, IMG_SIZE))

            # ReLU
            cam = torch.nn.functional.relu(cam)

            # Normalizar
            cam = cam.squeeze().cpu().detach().numpy()

            # Se cam é 1D ou escalar, criar mapa de ativação uniforme
            if cam.ndim == 0 or (cam.ndim == 1 and cam.size == 1):
                print("Aviso: CAM resultou em escalar, criando mapa uniforme")
                return np.ones((IMG_SIZE, IMG_SIZE)) * 0.5

            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            # Redimensionar para tamanho da imagem
            cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

            return cam
        else:
            print("Aviso: Ativações ou gradientes não foram capturados")
            return np.zeros((IMG_SIZE, IMG_SIZE))

    finally:
        forward_handle.remove()
        backward_handle.remove()

def criar_visualizacao_completa(img_original, grayscale_cam, nome_arquivo, modelo_nome):
    """Cria visualização profissional: [Original] | [Heatmap] | [Matriz]"""

    # 1. Imagem Original
    img_esq = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))

    # 2. Mapa de Calor Sobreposto
    rgb_img_float = np.float32(img_original) / 255
    rgb_img_float = cv2.resize(rgb_img_float, (IMG_SIZE, IMG_SIZE))

    # Criar heatmap manualmente
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # Sobrepor
    cam_weight = 0.5
    heatmap_overlay = (1 - cam_weight) * rgb_img_float + cam_weight * heatmap
    heatmap_overlay = np.uint8(255 * heatmap_overlay)

    # 3. Matriz de Ativação
    matriz_img = criar_matriz_ativacao(grayscale_cam, TAMANHO_MATRIZ)
    altura_alvo = img_esq.shape[0]
    fator_escala = altura_alvo / matriz_img.shape[0]
    nova_largura = int(matriz_img.shape[1] * fator_escala)
    matriz_img_resized = cv2.resize(matriz_img, (nova_largura, altura_alvo))

    def centralizar_vertical(img, altura_alvo):
        if img.shape[0] < altura_alvo:
            pad_top = (altura_alvo - img.shape[0]) // 2
            pad_bottom = altura_alvo - img.shape[0] - pad_top
            img = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)
        return img

    altura_maxima = max(img_esq.shape[0], heatmap_overlay.shape[0], matriz_img_resized.shape[0])
    img_esq = centralizar_vertical(img_esq, altura_maxima)
    heatmap_overlay = centralizar_vertical(heatmap_overlay, altura_maxima)
    matriz_img_resized = centralizar_vertical(matriz_img_resized, altura_maxima)

    espacamento = np.zeros((altura_maxima, 20, 3), dtype=np.uint8)
    resultado = np.hstack([img_esq, espacamento, heatmap_overlay, espacamento, matriz_img_resized])

    # Barra superior
    barra_info_altura = 60
    largura_total = resultado.shape[1]
    barra_info = np.zeros((barra_info_altura, largura_total, 3), dtype=np.uint8)
    barra_info[:] = (40, 40, 40)

    font = cv2.FONT_HERSHEY_SIMPLEX
    texto_modelo = f'Modelo: {modelo_nome}'
    texto_imagem = f'Imagem: {nome_arquivo}'

    cv2.putText(barra_info, texto_modelo, (20, 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(barra_info, texto_imagem, (20, 50), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    visualizacao_final = np.vstack([barra_info, resultado])
    return visualizacao_final

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Encontrar modelo YOLO disponível
    MODELO_YOLO = None
    for modelo_path in MODELO_YOLO_OPTIONS:
        if os.path.exists(modelo_path):
            MODELO_YOLO = modelo_path
            print(f"✓ Modelo encontrado: {MODELO_YOLO}")
            break

    if MODELO_YOLO is None:
        print("ERRO: Nenhum modelo YOLO encontrado!")
        print("Locais verificados:")
        for path in MODELO_YOLO_OPTIONS:
            print(f"  - {path}")
        print("\nExecute o treinamento YOLO primeiro com: python yolo_fase1.py")
        return

    # Carregar imagens
    img_paths = glob.glob(os.path.join(PASTA_IMAGENS_ORIGINAIS, "*.jpeg"))[:QTD_IMAGENS]
    if not img_paths:
        print(f"Nenhuma imagem encontrada em {PASTA_IMAGENS_ORIGINAIS}")
        return

    print(f"Carregando modelo YOLO: {MODELO_YOLO}")
    yolo_model = YOLO(MODELO_YOLO)
    yolo_model.to(DEVICE)

    print(f"\nGerando GradCAM para {len(img_paths)} imagens...\n")

    for img_path in img_paths:
        try:
            # Carregar imagem
            img_original = cv2.imread(img_path, 1)
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

            # Preprocessar
            img_resized = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))

            # Converter para tensor PyTorch
            input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            input_tensor = input_tensor.to(DEVICE)

            # Gerar GradCAM manual
            grayscale_cam = gerar_gradcam_manual(yolo_model, input_tensor, target_class=1)

            # Criar visualização
            nome_img = os.path.basename(img_path)
            modelo_nome = os.path.basename(MODELO_YOLO).replace('.pt', '')

            visualizacao = criar_visualizacao_completa(
                img_original,
                grayscale_cam,
                nome_img,
                modelo_nome
            )

            # Salvar
            nome_img_sem_ext = os.path.splitext(nome_img)[0]
            save_name = f"yolo_{nome_img_sem_ext}_gradcam.png"
            save_path = os.path.join(OUTPUT_FOLDER, save_name)

            cv2.imwrite(save_path, cv2.cvtColor(visualizacao, cv2.COLOR_RGB2BGR))
            print(f"✓ Salvo: {save_name}")

        except Exception as e:
            print(f"✗ Erro ao processar {os.path.basename(img_path)}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Concluído! Visualizações salvas em: {OUTPUT_FOLDER}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
