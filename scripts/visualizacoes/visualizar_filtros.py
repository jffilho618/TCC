import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ================= CONFIGURAÇÃO =================
# Caminhos relativos ao diretório do script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_IMAGENS = os.path.join(SCRIPT_DIR, 'Fase2', 'classificacao_crops', 'maligno')
# ================================================

def aplicar_filtros_demo():
    # Pega a primeira imagem que encontrar
    print(f"Procurando imagens em: {PASTA_IMAGENS}")
    print(f"Pasta existe? {os.path.exists(PASTA_IMAGENS)}")
    img_files = glob.glob(os.path.join(PASTA_IMAGENS, "*.png")) + glob.glob(os.path.join(PASTA_IMAGENS, "*.jpg")) + glob.glob(os.path.join(PASTA_IMAGENS, "*.jpeg"))

    if not img_files:
        print("Nenhuma imagem encontrada para teste.")
        return

    img_path = img_files[5] # Pega a primeira
    print(f"Gerando demo para: {img_path}")

    # 1. Carregar Original (Grayscale)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Aplicar CLAHE
    # ClipLimit alto (4.0) para você ver bem o efeito
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    
    # 3. Aplicar Sharpening (Unsharp Masking via Kernel)
    # Esse kernel realça as bordas subtraindo a vizinhança do centro
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img, -1, kernel)

    # 4. Plotar Comparativo
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original (Lavada)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_clahe, cmap='gray')
    plt.title("CLAHE (Textura Óssea Explícita)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_sharp, cmap='gray')
    plt.title("Sharpening (Bordas Duras)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    print("Janela de visualização aberta!")

if __name__ == "__main__":
    aplicar_filtros_demo()