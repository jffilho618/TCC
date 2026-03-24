import os
import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURAÇÕES =================
# Pasta onde estão as imagens organizadas atualmente
DIR_IMAGENS_ORIGEM = os.path.join('classificacao', 'treino_kfold')

# Pasta onde estão os JSONs brutos (LabelMe)
DIR_ANOTACOES = os.path.join('BTXRD', 'Annotations')

# Onde salvar os recortes
DIR_DESTINO = os.path.join('classificacao_crops', 'treino_kfold')

# Margem de segurança ao redor da lesão (0.15 = 15%)
PADDING = 0.15
# =================================================

def calcular_bbox_geral(shapes):
    """
    Encontra a bounding box que engloba TODOS os shapes da imagem.
    Retorna (x_min, y_min, x_max, y_max)
    """
    all_points = []
    
    for shape in shapes:
        points = shape.get('points', [])
        # LabelMe rectangle: [[x1, y1], [x2, y2]]
        # LabelMe polygon: [[x1, y1], [x2, y2], [x3, y3]...]
        for p in points:
            all_points.append(p)
            
    if not all_points:
        return None
        
    all_points = np.array(all_points)
    
    x_min = np.min(all_points[:, 0])
    y_min = np.min(all_points[:, 1])
    x_max = np.max(all_points[:, 0])
    y_max = np.max(all_points[:, 1])
    
    return int(x_min), int(y_min), int(x_max), int(y_max)

def processar_recortes():
    print(f"--- INICIANDO GERAÇÃO DE CROPS VIA JSON ---")
    print(f"Origem Imagens: {DIR_IMAGENS_ORIGEM}")
    print(f"Origem JSONs:   {DIR_ANOTACOES}")
    print(f"Destino:        {DIR_DESTINO}")
    
    # Contadores
    processados = 0
    sem_json = 0
    sem_shapes = 0
    erros_leitura = 0
    
    # Percorrer recursivamente a pasta de imagens (inclui subpastas benigno/maligno)
    arquivos_imagem = list(Path(DIR_IMAGENS_ORIGEM).rglob("*.*"))
    # Filtra apenas extensões de imagem comuns
    arquivos_imagem = [f for f in arquivos_imagem if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
    
    print(f"Total de imagens encontradas: {len(arquivos_imagem)}")
    
    for caminho_img in tqdm(arquivos_imagem):
        try:
            # 1. Identificar caminhos
            nome_arquivo = caminho_img.name
            nome_sem_ext = caminho_img.stem
            
            # Caminho relativo (ex: maligno/imagem01.jpg) para manter estrutura
            caminho_relativo = caminho_img.relative_to(DIR_IMAGENS_ORIGEM)
            pasta_pai = caminho_relativo.parent # ex: maligno
            
            # Caminho do JSON esperado
            caminho_json = os.path.join(DIR_ANOTACOES, f"{nome_sem_ext}.json")
            
            # 2. Verificar existência do JSON
            if not os.path.exists(caminho_json):
                # print(f"AVISO: JSON não encontrado para {nome_arquivo}")
                sem_json += 1
                continue
                
            # 3. Ler JSON e Imagem
            with open(caminho_json, 'r') as f:
                dados_json = json.load(f)
            
            img = cv2.imread(str(caminho_img))
            if img is None:
                print(f"ERRO: Não foi possível ler a imagem {caminho_img}")
                erros_leitura += 1
                continue
                
            h_img, w_img = img.shape[:2]
            shapes = dados_json.get('shapes', [])
            
            if not shapes:
                # print(f"AVISO: JSON sem anotações (shapes) para {nome_arquivo}")
                sem_shapes += 1
                continue
                
            # 4. Calcular BBox
            bbox = calcular_bbox_geral(shapes)
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = bbox
            w_box = x2 - x1
            h_box = y2 - y1
            
            # 5. Aplicar Margem (Padding)
            pad_x = int(w_box * PADDING)
            pad_y = int(h_box * PADDING)
            
            new_x1 = max(0, x1 - pad_x)
            new_y1 = max(0, y1 - pad_y)
            new_x2 = min(w_img, x2 + pad_x)
            new_y2 = min(h_img, y2 + pad_y)
            
            # 6. Recortar
            crop = img[new_y1:new_y2, new_x1:new_x2]
            
            # 7. Salvar na nova estrutura
            pasta_destino_final = os.path.join(DIR_DESTINO, pasta_pai)
            os.makedirs(pasta_destino_final, exist_ok=True)
            
            caminho_salvar = os.path.join(pasta_destino_final, nome_arquivo)
            cv2.imwrite(caminho_salvar, crop)
            
            processados += 1
            
        except Exception as e:
            print(f"CRASH no arquivo {caminho_img}: {e}")
            erros_leitura += 1

    print("\n" + "="*40)
    print("RELATÓRIO FINAL DE CROPS")
    print("="*40)
    print(f"Imagens processadas com sucesso: {processados}")
    print(f"Imagens sem JSON correspondente: {sem_json} (Ignoradas)")
    print(f"JSONs vazios (sem shapes):       {sem_shapes} (Ignorados)")
    print(f"Erros de leitura/corrupção:      {erros_leitura}")
    print(f"Pasta gerada: {os.path.abspath(DIR_DESTINO)}")
    print("="*40)
    
    # Copia o CSV de controle antigo para a pasta nova (se existir), 
    # para que os scripts de treino continuem funcionando
    csv_controle = os.path.join('classificacao', 'dataset_kfold_controle.csv')
    if os.path.exists(csv_controle):
        os.makedirs('classificacao_crops', exist_ok=True)
        shutil.copy(csv_controle, os.path.join('classificacao_crops', 'dataset_kfold_controle.csv'))
        print("CSV de controle copiado para a pasta de crops.")

if __name__ == "__main__":
    processar_recortes()