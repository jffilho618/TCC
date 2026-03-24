import os
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_ORIGEM = 'classificacao'
DIR_ANOTACOES = os.path.join('..', 'BTXRD', 'Annotations')

def calcular_bbox_geral(shapes):
    all_points = []
    for shape in shapes:
        points = shape.get('points', [])
        for p in points: all_points.append(p)
    if not all_points: return None
    all_points = np.array(all_points)
    return int(np.min(all_points[:,0])), int(np.min(all_points[:,1])), int(np.max(all_points[:,0])), int(np.max(all_points[:,1]))

def aplicar_clahe(img):
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        final = clahe.apply(img)
    return final

def processar_exemplo():
    # Encontra uma imagem válida com anotação no treino maligno (aleatoriamente para não repetir)
    pasta_busca = os.path.join(BASE_ORIGEM, 'treino_kfold', 'maligno')
    img_escolhida = None
    json_path_escolhido = None
    
    arquivos = [arq for arq in os.listdir(pasta_busca) if arq.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(arquivos)
    
    for arquivo in arquivos:
        caminho_img = os.path.join(pasta_busca, arquivo)
        nome_sem_ext = os.path.splitext(arquivo)[0]
        caminho_json = os.path.join(DIR_ANOTACOES, f"{nome_sem_ext}.json")
        if os.path.exists(caminho_json):
            img_escolhida = caminho_img
            json_path_escolhido = caminho_json
            break
                
    if not img_escolhida:
        print("Nenhuma imagem com anotação encontrada na pasta classificacao/treino_kfold/maligno.")
        return

    # 1. Imagem Original
    img_orig = cv2.imread(img_escolhida)
    img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    res_1 = f"{img_orig.shape[1]}x{img_orig.shape[0]}"
    
    # Lendo bbox
    with open(json_path_escolhido, 'r') as f: data = json.load(f)
    bbox = calcular_bbox_geral(data.get('shapes', []))
    x1, y1, x2, y2 = bbox
    
    # 2. Lesão Cropada Exacta
    crop_exato = img_orig_rgb[y1:y2, x1:x2]
    res_2 = f"{crop_exato.shape[1]}x{crop_exato.shape[0]}"
    
    # 3. Crop com 40% Padding
    PADDING = 0.40
    h, w = img_orig.shape[:2]
    pad_x, pad_y = int((x2-x1)*PADDING), int((y2-y1)*PADDING)
    nx1, ny1 = max(0, x1-pad_x), max(0, y1-pad_y)
    nx2, ny2 = min(w, x2+pad_x), min(h, y2+pad_y)
    crop_padding = img_orig_rgb[ny1:ny2, nx1:nx2]
    res_3 = f"{crop_padding.shape[1]}x{crop_padding.shape[0]}"
    
    # 4. Crop 40% Padding + CLAHE
    crop_clahe_bgr = aplicar_clahe(img_orig[ny1:ny2, nx1:nx2])
    crop_clahe = cv2.cvtColor(crop_clahe_bgr, cv2.COLOR_BGR2RGB)
    res_4 = f"{crop_clahe.shape[1]}x{crop_clahe.shape[0]}"
    
    # 5. Crop 40% CLAHE + Resize 384
    FINAL_SIZE = 384
    crop_resized = cv2.resize(crop_clahe, (FINAL_SIZE, FINAL_SIZE), interpolation=cv2.INTER_LINEAR)
    res_5 = f"{FINAL_SIZE}x{FINAL_SIZE}"
    
    # Plotando Grade 2x3 (2 linhas, 3 colunas)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    axes_flat = axes.flatten()
    
    etapas = [
        ("1. Original Completa", img_orig_rgb, res_1),
        ("2. Crop Exato (YOLO)", crop_exato, res_2),
        ("3. Crop Expandido (40% Pad)", crop_padding, res_3),
        ("4. Pad + Filtro CLAHE", crop_clahe, res_4),
        ("5. Resize Final (Treino)", crop_resized, res_5)
    ]
    
    # Remover o 6º plot (vazio)
    axes_flat[5].axis('off')
    
    for ax, (titulo, img, res) in zip(axes_flat, etapas):
        ax.imshow(img)
        ax.set_title(f"{titulo}\nRes: {res}", fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
        
    plt.tight_layout()
    output_path = 'etapas_pre_processamento.jpg'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grade salva com sucesso em {output_path} (Amostra: {os.path.basename(img_escolhida)})!")

if __name__ == '__main__':
    processar_exemplo()
