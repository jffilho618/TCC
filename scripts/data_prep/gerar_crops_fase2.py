import os
import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURAÇÕES FASE 2 (CORRIGIDO) =================
BASE_ORIGEM = 'classificacao' # Pasta raiz da Fase 1
BASE_DESTINO = os.path.join('Fase2', 'classificacao_crops')
DIR_ANOTACOES = os.path.join('BTXRD', 'Annotations')
ARQUIVO_CSV_CONTROLE = os.path.join('classificacao', 'dataset_kfold_controle.csv')

# Padding de 40% (Contexto Ósseo)
PADDING = 0.40
# ====================================================================

def calcular_bbox_geral(shapes):
    """Calcula caixa envolvente de todos os shapes."""
    all_points = []
    for shape in shapes:
        points = shape.get('points', [])
        for p in points:
            all_points.append(p)
    if not all_points: return None
    all_points = np.array(all_points)
    return int(np.min(all_points[:,0])), int(np.min(all_points[:,1])), int(np.max(all_points[:,0])), int(np.max(all_points[:,1]))

def processar_crops_completo():
    print(f"--- FASE 2: GERAÇÃO DE CROPS (TREINO + TESTE) ---")
    
    # 1. Copiar CSV de controle
    if os.path.exists(ARQUIVO_CSV_CONTROLE):
        os.makedirs(BASE_DESTINO, exist_ok=True)
        shutil.copy(ARQUIVO_CSV_CONTROLE, os.path.join(BASE_DESTINO, 'dataset_kfold_controle.csv'))
    else:
        print("ERRO: CSV de controle não encontrado.")
        return

    # 2. Listar as duas pastas principais (treino_kfold e teste_final)
    pastas_para_processar = ['treino_kfold', 'teste_final']
    
    total_sucesso = 0
    
    for pasta_split in pastas_para_processar:
        caminho_origem_split = os.path.join(BASE_ORIGEM, pasta_split)
        
        if not os.path.exists(caminho_origem_split):
            print(f"Aviso: Pasta {pasta_split} não encontrada em {BASE_ORIGEM}. Pulando.")
            continue
            
        print(f"\nProcessando grupo: {pasta_split} ...")
        
        # Busca recursiva de imagens dentro desse split (ex: treino_kfold/benigno/img.png)
        arquivos = list(Path(caminho_origem_split).rglob("*.*"))
        arquivos = [f for f in arquivos if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        for caminho_img in tqdm(arquivos):
            try:
                # ex: caminho_img = classificacao/treino_kfold/benigno/img1.png
                # nome_sem_ext = img1
                nome_sem_ext = caminho_img.stem
                
                # Descobre a classe (pasta pai imediata)
                classe = caminho_img.parent.name # 'benigno' ou 'maligno'
                
                # Busca JSON correspondente
                caminho_json = os.path.join(DIR_ANOTACOES, f"{nome_sem_ext}.json")
                
                # Se não tem JSON, não tem como saber onde está a lesão -> Pula
                if not os.path.exists(caminho_json):
                    continue
                
                # Ler Imagem e JSON
                img = cv2.imread(str(caminho_img))
                if img is None: continue
                
                with open(caminho_json, 'r') as f:
                    dados = json.load(f)
                
                shapes = dados.get('shapes', [])
                if not shapes: continue
                
                # Calcular Crop
                bbox = calcular_bbox_geral(shapes)
                if bbox is None: continue
                
                x1, y1, x2, y2 = bbox
                h_img, w_img = img.shape[:2]
                w_box, h_box = x2 - x1, y2 - y1
                
                # Aplicar Padding
                pad_x = int(w_box * PADDING)
                pad_y = int(h_box * PADDING)
                
                nx1 = max(0, x1 - pad_x)
                ny1 = max(0, y1 - pad_y)
                nx2 = min(w_img, x2 + pad_x)
                ny2 = min(h_img, y2 + pad_y)
                
                crop = img[ny1:ny2, nx1:nx2]
                
                # --- O PULO DO GATO: SALVAR NA ESTRUTURA CERTA ---
                # Destino: Fase2/classificacao_crops/treino_kfold/benigno/img1.png
                pasta_destino = os.path.join(BASE_DESTINO, pasta_split, classe)
                os.makedirs(pasta_destino, exist_ok=True)
                
                cv2.imwrite(os.path.join(pasta_destino, caminho_img.name), crop)
                total_sucesso += 1
                
            except Exception as e:
                print(f"Erro ao processar {caminho_img.name}: {e}")

    print(f"\nConcluído! Total de crops gerados: {total_sucesso}")
    print(f"Estrutura criada em: {BASE_DESTINO}")
    print("Agora as subpastas 'treino_kfold' e 'teste_final' existem e batem com o CSV.")

if __name__ == "__main__":
    processar_crops_completo()