import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ================= CONFIGURAÇÃO =================
# Tenta achar uma imagem maligna na sua estrutura
PASTAS_BUSCA = [
    os.path.join('Fase2.5', 'classificacao_crops', 'teste_final', 'maligno'),
    os.path.join('Fase2', 'classificacao_crops', 'teste_final', 'maligno'),
    os.path.join('Fase1.5', 'classificacao_crops', 'teste_final', 'maligno'),
    os.path.join('classificacao', 'teste_final', 'maligno'),
]
ARQUIVO_SAIDA = 'teste_bilateral_vs_clahe.png'
# ================================================

def aplicar_clahe(img):
    """O filtro que usamos na Fase 2.5 (Realça Textura)"""
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    return clahe.apply(img)

def aplicar_bilateral(img):
    """O filtro do Artigo 1 (Remove Ruído + Mantém Bordas)"""
    # d=15: Diâmetro maior do pixel vizinho para suavização mais forte
    # sigmaColor=30: Valor menor = mais agressivo na suavização de áreas similares
    # sigmaSpace=30: Valor menor = suavização mais local e pronunciada
    # Parâmetros ajustados para tornar o efeito visível em raio-X
    return cv2.bilateralFilter(img, 15, 30, 30)

def main():
    # 1. Achar imagem
    img_path = None
    for pasta in PASTAS_BUSCA:
        arquivos = glob.glob(os.path.join(pasta, "*.png")) + glob.glob(os.path.join(pasta, "*.jpg")) + glob.glob(os.path.join(pasta, "*.jpeg"))
        if arquivos:
            img_path = arquivos[1] # Pega a primeira que achar
            break
    
    if not img_path:
        print("ERRO: Nenhuma imagem encontrada nas pastas padrão.")
        return

    print(f"Testando filtros na imagem: {os.path.basename(img_path)}")

    # 2. Carregar e Processar
    img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        print("Erro ao abrir imagem.")
        return

    # A: Apenas CLAHE (Sua abordagem atual)
    img_clahe = aplicar_clahe(img_original)

    # B: Apenas Bilateral (Abordagem do Artigo 1)
    img_bilateral = aplicar_bilateral(img_original)

    # C: Híbrido (Bilateral para limpar ruído -> CLAHE para realçar osso)
    # *Minha aposta para o pipeline ideal*
    img_hibrida = aplicar_clahe(aplicar_bilateral(img_original))

    # 3. Montar Painel Comparativo
    plt.figure(figsize=(20, 10))

    # Original
    plt.subplot(1, 4, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title("Original (Com Ruído)", fontsize=14)
    plt.axis('off')

    # Bilateral (Artigo 1)
    plt.subplot(1, 4, 2)
    plt.imshow(img_bilateral, cmap='gray')
    plt.title("Só Bilateral (Artigo 1)\n(Limpa Ruído, Mantém Borda)", fontsize=14)
    plt.axis('off')

    # CLAHE (Seu TCC)
    plt.subplot(1, 4, 3)
    plt.imshow(img_clahe, cmap='gray')
    plt.title("Só CLAHE (Fase 2)\n(Realça Textura e Ruído)", fontsize=14)
    plt.axis('off')

    # Híbrido
    plt.subplot(1, 4, 4)
    plt.imshow(img_hibrida, cmap='gray')
    plt.title("Híbrido (Bilateral + CLAHE)\n(Melhor dos dois mundos?)", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(ARQUIVO_SAIDA)
    print(f"\n✅ Comparação salva em: {ARQUIVO_SAIDA}")
    print("Abra a imagem e veja qual revela melhor os detalhes do tumor!")

if __name__ == "__main__":
    main()