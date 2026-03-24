import os
import json
import cv2
import numpy as np
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from colorama import init, Fore, Style
import albumentations as A  # Biblioteca profissional de Augmentation (mais segura)

# Inicializa colorama
init(autoreset=True)

# ================= CONFIGURAÇÕES FASE 3 =================
BASE_ORIGEM = 'classificacao'
BASE_DESTINO_CNN = os.path.join('Fase3', 'classificacao_crops_cnn')
BASE_DESTINO_VIT = os.path.join('Fase3', 'classificacao_crops_vit') # Vai ter CSV próprio
BASE_DESTINO_TESTE = os.path.join('Fase3', 'classificacao_crops_teste') # Mudado nome para clareza

DIR_ANOTACOES = os.path.join('BTXRD', 'Annotations')
ARQUIVO_CSV_ORIGINAL = os.path.join('classificacao', 'dataset_kfold_controle.csv')

PADDING = 0.40
N_AUGMENTATIONS = 10 # 10x mais dados de treino para ViT
# ========================================================

def carregar_mapa_folds():
    """Lê o CSV original para saber qual imagem pertence a qual Fold."""
    if not os.path.exists(ARQUIVO_CSV_ORIGINAL):
        print(f"{Fore.RED}Erro: CSV original não encontrado em {ARQUIVO_CSV_ORIGINAL}")
        return None
    df = pd.read_csv(ARQUIVO_CSV_ORIGINAL)
    # Cria um dicionário: {'benigno/img1.png': 0, 'maligno/img2.png': 3}
    # Chave é 'classe/nome_arquivo' para ser único
    mapa = {}
    for _, row in df.iterrows():
        chave = f"{row['class_name']}/{row['image_id']}"
        mapa[chave] = row['fold']
    return df, mapa

def calcular_bbox_geral(shapes):
    all_points = []
    for shape in shapes:
        points = shape.get('points', [])
        for p in points: all_points.append(p)
    if not all_points: return None
    all_points = np.array(all_points)
    return int(np.min(all_points[:,0])), int(np.min(all_points[:,1])), int(np.max(all_points[:,0])), int(np.max(all_points[:,1]))

def aplicar_clahe(img):
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) na imagem."""
    if len(img.shape) == 3:
        # Imagem colorida: converte para LAB e aplica no canal L
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        # Imagem em escala de cinza
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        final = clahe.apply(img)
    return final

def criar_augmentations():
    """Define pipeline de augmentation leve/médio para osso."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), # Simula ruído de raio-x
        A.Affine(scale=(0.85, 1.15), translate_percent=(0.1, 0.1), p=0.5) # Zoom e Shift
    ])

def processar_fase3():
    print(f"{Fore.CYAN}--- INICIANDO GERAÇÃO FASE 3 (CORRIGIDA) ---")
    
    # 1. Carregar Mapa de Folds
    df_original, mapa_folds = carregar_mapa_folds()
    if mapa_folds is None: return

    # Listas para o NOVO CSV do ViT
    novas_linhas_vit = []
    
    # Augmenter
    augmenter = criar_augmentations()

    # Copiar CSV original para CNN (pois CNN usa dados normais)
    os.makedirs(BASE_DESTINO_CNN, exist_ok=True)
    shutil.copy(ARQUIVO_CSV_ORIGINAL, os.path.join(BASE_DESTINO_CNN, 'dataset_kfold_controle.csv'))
    print(f"CSV CNN copiado.")

    # Processar Pastas
    for pasta_split in ['treino_kfold', 'teste_final']:
        caminho_origem_split = os.path.join(BASE_ORIGEM, pasta_split)
        arquivos = list(Path(caminho_origem_split).rglob("*.*"))
        arquivos = [f for f in arquivos if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        print(f"\nProcessando {pasta_split} ({len(arquivos)} imagens)...")
        
        for caminho_img in tqdm(arquivos):
            try:
                nome_arq = caminho_img.name
                classe = caminho_img.parent.name # benigno/maligno
                nome_sem_ext = caminho_img.stem
                
                # Identificar Fold
                chave_mapa = f"{classe}/{nome_arq}"
                fold_atual = mapa_folds.get(chave_mapa, -1) # -1 se for teste ou não achar
                
                # Ler Imagem e JSON
                caminho_json = os.path.join(DIR_ANOTACOES, f"{nome_sem_ext}.json")
                if not os.path.exists(caminho_json): continue
                
                with open(caminho_json, 'r') as f: data = json.load(f)
                img = cv2.imread(str(caminho_img))
                if img is None: continue
                
                # Crop
                bbox = calcular_bbox_geral(data.get('shapes', []))
                if bbox is None: continue
                x1, y1, x2, y2 = bbox
                h, w = img.shape[:2]
                pad_x, pad_y = int((x2-x1)*PADDING), int((y2-y1)*PADDING)
                nx1, ny1 = max(0, x1-pad_x), max(0, y1-pad_y)
                nx2, ny2 = min(w, x2+pad_x), min(h, y2+pad_y)
                crop = img[ny1:ny2, nx1:nx2]

                # Aplica CLAHE no crop (100% das imagens agora são pré-processadas)
                crop = aplicar_clahe(crop)

                # --- LÓGICA 1: DESTINO CNN e TESTE (Sempre Crop Limpo com CLAHE) ---
                if pasta_split == 'teste_final':
                    # Salva no destino de teste (CROPADO, corrigindo o erro anterior)
                    dest_path = os.path.join(BASE_DESTINO_TESTE, classe)
                    os.makedirs(dest_path, exist_ok=True)
                    cv2.imwrite(os.path.join(dest_path, nome_arq), crop)
                
                elif pasta_split == 'treino_kfold':
                    # Salva no destino CNN (Crop Limpo)
                    dest_cnn = os.path.join(BASE_DESTINO_CNN, pasta_split, classe)
                    os.makedirs(dest_cnn, exist_ok=True)
                    cv2.imwrite(os.path.join(dest_cnn, nome_arq), crop)

                # --- LÓGICA 2: DESTINO VIT (Augmentation Inteligente) ---
                # Se for Teste ou Validação (Fold 0), salva SÓ O ORIGINAL
                # Se for Treino (Fold 1,2,3,4), salva ORIGINAL + 10 AUGMENTATIONS
                
                dest_vit = os.path.join(BASE_DESTINO_VIT, pasta_split, classe)
                if pasta_split == 'teste_final': # Vit Teste também precisa existir
                    dest_vit = os.path.join(BASE_DESTINO_VIT, 'teste_final', classe)
                
                os.makedirs(dest_vit, exist_ok=True)

                # 2.1 Salvar Crop Original no ViT (Sempre necessário)
                cv2.imwrite(os.path.join(dest_vit, nome_arq), crop)
                
                # Adiciona ao CSV do ViT (Mantendo dados originais)
                # Recupera a linha original do dataframe pra não perder metadados
                linha_original = df_original[(df_original['image_id'] == nome_arq) & (df_original['class_name'] == classe)]
                if not linha_original.empty:
                    dado_base = linha_original.iloc[0].to_dict()
                    novas_linhas_vit.append(dado_base)

                # 2.2 Gerar Augmentations (Apenas Treino e Fold != 0)
                if pasta_split == 'treino_kfold' and fold_atual != 0:
                    for i in range(N_AUGMENTATIONS):
                        # Aplica Augmentation
                        aug = augmenter(image=crop)['image']
                        
                        # Nome novo: img1_aug0.png
                        novo_nome = f"{nome_sem_ext}_aug{i}.png"
                        cv2.imwrite(os.path.join(dest_vit, novo_nome), aug)
                        
                        # Adiciona linha no CSV do ViT
                        nova_linha = dado_base.copy()
                        nova_linha['image_id'] = novo_nome # Atualiza ID
                        # As outras colunas (fold, class_name) ficam iguais
                        novas_linhas_vit.append(nova_linha)

            except Exception as e:
                pass # Ignora erros pontuais de arquivo

    # --- SALVAR NOVO CSV PARA VIT ---
    print(f"\nGerando CSV específico para ViT com {len(novas_linhas_vit)} linhas...")
    df_vit = pd.DataFrame(novas_linhas_vit)
    caminho_csv_vit = os.path.join(BASE_DESTINO_VIT, 'dataset_kfold_controle_vit.csv')
    df_vit.to_csv(caminho_csv_vit, index=False)
    
    print(f"{Fore.GREEN}Concluído!")
    print(f"CSV CNN (Original): {os.path.join(BASE_DESTINO_CNN, 'dataset_kfold_controle.csv')}")
    print(f"CSV ViT (Expandido): {caminho_csv_vit}")

if __name__ == "__main__":
    # Instalar albumentations se não tiver: pip install albumentations
    try:
        import albumentations
        processar_fase3()
    except ImportError:
        print("Por favor instale: pip install albumentations")