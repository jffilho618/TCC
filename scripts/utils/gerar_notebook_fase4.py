import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(text):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(text):
    lines = [line + '\n' for line in text.split('\n')]
    if lines: lines[-1] = lines[-1].rstrip('\n')
    notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})

# -----------------
# BUILD THE NOTEBOOK
# -----------------

add_markdown("# Fase 4: Pipeline Ponta-a-Ponta das CNNs\nNeste notebook repetiremos rigorosamente os testes de arquitetura da Fase 3, mantendo estrita isolação dos dados. As 3 CNNs serão ensinadas nas mesmas condições e submetidas às mesmas extrações (incluindo o novíssimo check de Precisão e as Matrizes de Confusão Visuais).")

add_code('''# Instalações necessárias (se não estiverem instaladas)
# !pip install timm rich matplotlib seaborn pandas opencv-python pytorch-grad-cam albumentations
''')

add_markdown("## 1. Módulo 1: Delineamento e Isolamento Hídrico (Cropping)\nCélula responsável por resgatar os polígonos, efetuar o corte (40% de padding) e separar em 3 caixas intocáveis (`treino`, `val`, `teste`) seguindo fielmente a aleatorização da Fase 3 (o mesmo Fold).")

add_code(r'''import os
import json
import cv2
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

# Definições Unificadas da Fase 4
BASE_FASE4 = '.'
DIR_DATASET = os.path.join(BASE_FASE4, 'dataset')
PASTAS = ['treino', 'val', 'teste']
CLASSES = ['benigno', 'maligno']

DIR_IMAGENS_ORIGINAIS = r'C:\Users\jffil\OneDrive\Documentos\TCC2\TCC2\data\images'
DIR_ANOTACOES = r'C:\Users\jffil\OneDrive\Documentos\TCC2\TCC2\data\Annotations'
CSV_ORIGINAL = r'C:\Users\jffil\OneDrive\Documentos\TCC2\TCC2\data\metadados\dataset_kfold_controle.csv'
CSV_CONTROLE_FASE4 = os.path.join(BASE_FASE4, 'dataset_controle_fase4.csv')

PADDING = 0.40

def calcular_bbox_geral(shapes):
    all_points = []
    for shape in shapes:
        for p in shape.get('points', []): all_points.append(p)
    if not all_points: return None
    all_points = np.array(all_points)
    return int(np.min(all_points[:,0])), int(np.min(all_points[:,1])), int(np.max(all_points[:,0])), int(np.max(all_points[:,1]))

def preparar_pastas():
    for p in PASTAS:
        for c in CLASSES:
            os.makedirs(os.path.join(DIR_DATASET, p, c), exist_ok=True)

def gerar_crops_e_separar():
    preparar_pastas()
    df = pd.read_csv(CSV_ORIGINAL)
    novas_linhas = []
    
    print("Gerando e fatiando os Crops de osso...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cls_name = row['class_name']
        img_id = row['image_id']
        split = row['split_group']
        fold = row['fold']
        
        # O Ponto Vital: Separação em 3 pastas!
        if split == 'teste_final': pasta_destino = 'teste'
        elif fold == 0: pasta_destino = 'val'
        else: pasta_destino = 'treino'
            
        caminho_img = os.path.join(DIR_IMAGENS_ORIGINAIS, img_id)
        nome_sem_ext = os.path.splitext(img_id)[0]
        caminho_json = os.path.join(DIR_ANOTACOES, f"{nome_sem_ext}.json")
        
        if not os.path.exists(caminho_img) or not os.path.exists(caminho_json):
            continue
            
        with open(caminho_json, 'r') as f: data = json.load(f)
        img = cv2.imread(caminho_img)
        if img is None: continue
            
        bbox = calcular_bbox_geral(data.get('shapes', []))
        if bbox is None: continue
        x1, y1, x2, y2 = bbox
        
        h, w = img.shape[:2]
        pad_x, pad_y = int((x2-x1)*PADDING), int((y2-y1)*PADDING)
        nx1, ny1 = max(0, x1-pad_x), max(0, y1-pad_y)
        nx2, ny2 = min(w, x2+pad_x), min(h, y2+pad_y)
        
        crop = img[ny1:ny2, nx1:nx2]
        cv2.imwrite(os.path.join(DIR_DATASET, pasta_destino, cls_name, img_id), crop)
        
        nova_linha = row.copy()
        nova_linha['split_fase4'] = pasta_destino
        novas_linhas.append(nova_linha)
        
    df_fase4 = pd.DataFrame(novas_linhas)
    df_fase4.to_csv(CSV_CONTROLE_FASE4, index=False)
    print(f"Crops enclausurados e controle gravado: {CSV_CONTROLE_FASE4}")

# Descomente para rodar
# gerar_crops_e_separar()
''')

add_markdown("## 2. Módulo 2: O Filtro X-Ray (CLAHE Offline)\nAo invés de processar por epoch, cravamos a equalização contrastiva a limiar diretamente nos JPGs recortados no passo anterior para acelerar a fase de treinamento.")

add_code(r'''def aplicar_clahe_offline():
    print("Aplicando Adaptação Limitada ao Histograma (CLAHE)...")
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    
    for pasta in PASTAS:
        for cls_name in CLASSES:
            caminho = os.path.join(DIR_DATASET, pasta, cls_name)
            if not os.path.exists(caminho): continue
                
            arquivos = os.listdir(caminho)
            for arq in tqdm(arquivos, desc=f"CLAHE ({pasta} | {cls_name})"):
                path_arq = os.path.join(caminho, arq)
                img = cv2.imread(path_arq)
                if img is None: continue
                    
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                
                # OVERWRITE direto na imagem. Offline final state.
                cv2.imwrite(path_arq, final)
    print("Aprimoramento X-Ray Offline 100% gravado!")

# Descomente para aplicar CLAHE (rode 1x apenas)
# aplicar_clahe_offline()
''')

add_markdown("## 3. Módulo 3: O Treinamento CNN (O Tira-Teima)\nReprodução simétrica da Fase 3 para as CNNs `EfficientNetV2-L`, `ResNet50` e `DenseNet121` validando sob isolamento estrito de treino/validação antes da matriz cega final de Teste.")

add_code(r'''import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
import timm
import random
import time

SEED = 42
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 60
PATIENCE = 12
LR = 1e-4
IMG_SIZE = 384
LABEL_SMOOTHING = 0.1
DROP_PATH_RATE = 0.2
ACC_STEPS = 2

CNN_MODELS = [
    {'name': 'tf_efficientnetv2_l', 'batch': 6, 'radimagenet': False},
    {'name': 'resnet50', 'batch': 24, 'radimagenet': True},
    {'name': 'densenet121', 'batch': 20, 'radimagenet': True}
]

DIR_MODELOS_SALVOS = os.path.join(BASE_FASE4, 'modelos_fase4')
os.makedirs(DIR_MODELOS_SALVOS, exist_ok=True)
PASTA_PESOS_RAD_GLOBAL = r'C:\Users\jffil\OneDrive\Documentos\TCC2\TCC2\modelos\pesos_radimagenet'

# Transforms Puros (Apenas geométrico, já que CLAHE é offline)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def carregar_pesos_rad_global(model, model_name):
    nome_arquivo = ''
    if 'resnet50' in model_name: nome_arquivo = 'RadImageNet-ResNet50_notop.pth'
    elif 'densenet121' in model_name: nome_arquivo = 'DenseNet121.pt'
    
    caminho = os.path.join(PASTA_PESOS_RAD_GLOBAL, nome_arquivo)
    if not os.path.exists(caminho):
        caminho_alt = os.path.join(PASTA_PESOS_RAD_GLOBAL, f"{model_name}.pt")
        if os.path.exists(caminho_alt): caminho = caminho_alt
    
    if os.path.exists(caminho):
        state_dict = torch.load(caminho, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f" -> [!] Pesos do RadImageNet carregados com sucesso da pasta global!")
        return True
    return False

def treinar_modelo_cnn(config):
    model_name = config['name']
    batch_size = config['batch']
    usar_rad = config['radimagenet']
    print(f"\n{'='*50}\n-> INICIANDO REDE: {model_name} (Batch: {batch_size})\n{'='*50}")
    
    train_dataset = datasets.ImageFolder(os.path.join(DIR_DATASET, 'treino'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DIR_DATASET, 'val'), transform=val_transforms)
    
    class_counts = np.bincount(train_dataset.targets)
    weights = 1. / class_counts
    samples_weights = weights[train_dataset.targets]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=2, drop_path_rate=DROP_PATH_RATE)
    except:
        model = timm.create_model(model_name, pretrained=True, num_classes=2)
        
    if usar_rad:
        carregar_pesos_rad_global(model, model_name)
        
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    best_loss = float('inf')
    patience_count = 0
    save_path = os.path.join(DIR_MODELOS_SALVOS, f"{model_name}_fase4.pth")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(imgs)
                loss = criterion(outputs, lbls) / ACC_STEPS
            
            scaler.scale(loss).backward()
            if (i + 1) % ACC_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += loss.item() * ACC_STEPS
            
        avg_train = train_loss / len(train_loader)
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                val_loss += criterion(outputs, lbls).item()
        
        avg_val = val_loss / len(val_loader)
        print(f"Época {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}", end=' ')
        
        if avg_val < best_loss:
            best_loss = avg_val
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(" [★ SALVO ✓]")
        else:
            patience_count += 1
            print(f" [Patience: {patience_count}/{PATIENCE}]")
            if patience_count >= PATIENCE:
                print("\n[!!!] Early Stopping Disparado [!!!]")
                break

# Descomente para treinar
# for config in CNN_MODELS:
#     treinar_modelo_cnn(config)
''')

add_markdown("## 4. Módulo 4: A Prova dos Nove (Teste Cego & Matriz de Confusão)\nAvaliação no conjunto blind de `teste`. Além da Acurácia/AUC/F1/Recall, foi incluída a **Precisão**, e uma geração fotográfica cruzada (Calor de acertos vs erros via Confusão).")

add_code(r'''from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

CSV_RESULTADOS_FINAIS = os.path.join(BASE_FASE4, 'RESULTADOS_TESTE_FASE4.csv')

def validar_teste_cego():
    test_dataset = datasets.ImageFolder(os.path.join(DIR_DATASET, 'teste'), transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    resultados = []
    
    # Prepara Lousa para as Matrizes de Confusão
    fig, axes = plt.subplots(1, len(CNN_MODELS), figsize=(6*len(CNN_MODELS), 5))
    if len(CNN_MODELS) == 1: axes = [axes]
    
    for idx, config in enumerate(CNN_MODELS):
        model_name = config['name']
        save_path = os.path.join(DIR_MODELOS_SALVOS, f"{model_name}_fase4.pth")
        
        if not os.path.exists(save_path):
            print(f"Modelo [{model_name}] não existe!")
            continue
            
        print(f"\nRodando matrizes de fuzilamento para -> {model_name}")
        try:
            model = timm.create_model(model_name, num_classes=2, drop_path_rate=DROP_PATH_RATE)
        except:
            model = timm.create_model(model_name, num_classes=2)
            
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        all_preds, all_lbls, all_probs = [], [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_lbls.extend(lbls.numpy())
                all_probs.extend(probs.cpu().numpy())
                
        # Extração de Score (Inclusão da Precisão)
        acc = accuracy_score(all_lbls, all_preds)
        f1 = f1_score(all_lbls, all_preds)
        rec = recall_score(all_lbls, all_preds)
        prec = precision_score(all_lbls, all_preds)
        auc = roc_auc_score(all_lbls, all_probs)
        
        resultados.append({
            'Rede': model_name,
            'Acurácia': round(acc, 4),
            'AUC': round(auc, 4),
            'F1-Score': round(f1, 4),
            'Precisão': round(prec, 4),
            'Recall (Sensibilidade)': round(rec, 4)
        })
        
        # Desenhar a Matriz no Eixo correspondente
        cm = confusion_matrix(all_lbls, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[idx], cbar=False,
                    xticklabels=['Predito: Benigno', 'Predito: Maligno'], 
                    yticklabels=['Real: Benigno', 'Real: Maligno'])
        axes[idx].set_title(f"Confusion Matrix: {model_name}")
        
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_FASE4, 'Matrizes_Confusao_CNNs.png'), dpi=300)
    plt.show()
    
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(CSV_RESULTADOS_FINAIS, index=False)
    print("\n--- TESTE FINAL CONCLUÍDO ---")
    display(df_res)

# Descomente para testar e printar os gráficos
# validar_teste_cego()
''')

add_markdown("## 5. Módulo 5: Explicabilidade e Prova de Conceito (Grad-CAM)\nAprovação médica visual. O bloco insere a imagem dentro do modelo e lê o fluxo de gradiente convolucional reverso retornando exatamente em qual textura térmica a CNN focou para determinar o câncer.")

add_code(r'''from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

DIR_GRADCAM = os.path.join(BASE_FASE4, 'GradCAM_Heatmaps')
os.makedirs(DIR_GRADCAM, exist_ok=True)

def espionar_visao_modelo(model_name='tf_efficientnetv2_l', imagens_por_classe=3):
    print(f"\nIniciando espectrômetro Grad-CAM no modelo da Fase 4 -> {model_name}")
    save_path = os.path.join(DIR_MODELOS_SALVOS, f"{model_name}_fase4.pth")
    if not os.path.exists(save_path): return
        
    model = timm.create_model(model_name, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()
    
    if 'densenet' in model_name: target_layers = [model.features[-1]]
    elif 'resnet' in model_name: target_layers = [model.layer4[-1]]
    else: target_layers = [model.conv_head]
        
    cam = GradCAM(model=model, target_layers=target_layers)
    
    for cls in CLASSES:
        pasta_alvo = os.path.join(DIR_DATASET, 'teste', cls)
        # Sorteia algumas imagens aleatórias ou pega as primeiras
        arquivos = os.listdir(pasta_alvo)[:imagens_por_classe]
        
        fig, axes = plt.subplots(len(arquivos), 2, figsize=(10, 4*len(arquivos)))
        if len(arquivos) == 1: axes = [axes]
        
        for idx, arq in enumerate(arquivos):
            path_img = os.path.join(pasta_alvo, arq)
            img_cv = cv2.imread(path_img)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_norm = np.float32(img_rgb) / 255
            
            input_tensor = val_transforms(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
            targets = [ClassifierOutputTarget(1)] # Olhar pela ótica Maligna
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)
            
            axes[idx][0].imshow(img_rgb)
            axes[idx][0].set_title(f"Amostra: {cls.upper()} - {arq}")
            axes[idx][0].axis('off')
            
            axes[idx][1].imshow(cam_image)
            axes[idx][1].set_title("Onde a CNN procurou câncer")
            axes[idx][1].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_GRADCAM, f'Visao_{model_name}_{cls}.png'))
        plt.show()

# Descomente para gerar o laudo de interpretabilidade das 3 CNNs
# for cnn in CNN_MODELS:
#     espionar_visao_modelo(cnn['name'])
''')

# Salva arquivo Jupyter Notebook (JSON)
with open("c:/Users/jffil/OneDrive/Documentos/TCC2/TCC2/fases/Fase4/Pipeline_Fase4_Master.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook magistral da Fase 4 criado!")
