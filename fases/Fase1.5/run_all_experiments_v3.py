import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
import time
import gc

# ================= CONFIGURAÇÕES GERAIS =================
# Obtém o diretório onde este script está localizado
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_CONTROLE = os.path.join(SCRIPT_DIR, 'classificacao_crops', 'dataset_kfold_controle.csv')
PASTA_BASE = os.path.join(SCRIPT_DIR, 'classificacao_crops')
ARQUIVO_RESULTADOS = os.path.join(SCRIPT_DIR, 'RESULTADOS_FINAIS_FASE_1.5.csv')
PASTA_MODELOS = os.path.join(SCRIPT_DIR, 'modelos_salvos_crops')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 2

# Configurações de Treino
EPOCHS = 100
PATIENCE = 15
LR = 1e-4

# Lista de Campeões (Batch otimizado para RTX 4060 Ti)
MODELOS_PARA_TESTAR = [
    # --- CNNs ---
    {'name': 'tf_efficientnetv2_s', 'batch': 32, 'img_size': 224, 'family': 'CNN'},
    {'name': 'densenet201',         'batch': 32, 'img_size': 224, 'family': 'CNN'},
    
    # --- ViTs ---
    {'name': 'vit_base_patch16_224',         'batch': 16, 'img_size': 224, 'family': 'ViT'}, 
    {'name': 'deit_base_distilled_patch16_224', 'batch': 32, 'img_size': 224, 'family': 'ViT'},
    {'name': 'beit_base_patch16_224',        'batch': 16, 'img_size': 224, 'family': 'ViT'},

    # --- Híbridos ---
    {'name': 'swinv2_tiny_window16_256',    'batch': 32, 'img_size': 256, 'family': 'Hybrid'},
    {'name': 'maxvit_rmlp_tiny_rw_256',      'batch': 16, 'img_size': 256, 'family': 'Hybrid'},
    {'name': 'coatnet_2_rw_224',            'batch': 16, 'img_size': 224, 'family': 'Hybrid'},
]
# ========================================================

class BoneDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.class_map = {'benigno': 0, 'maligno': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['split_group'], row['class_name'], row['image_id'])
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
        
        label = self.class_map[row['class_name']]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def treinar_um_modelo(config_modelo):
    model_name = config_modelo['name']
    batch_size = config_modelo['batch']
    img_size = config_modelo['img_size']
    
    # Define o caminho final onde o modelo será salvo
    caminho_salvamento = os.path.join(PASTA_MODELOS, f"{model_name}_best.pth")
    
    print(f"\n{'#'*60}")
    print(f"INICIANDO: {model_name} (Família: {config_modelo['family']})")
    print(f"Salvando pesos em: {caminho_salvamento}")
    print(f"{'#'*60}")
    
    start_time = time.time()
    
    # 1. Dados
    df = pd.read_csv(CSV_CONTROLE)
    df_train = df[(df['split_group'] == 'treino_kfold') & (df['fold'] != 0)]
    df_val = df[(df['split_group'] == 'treino_kfold') & (df['fold'] == 0)]
    df_test = df[df['split_group'] == 'teste_final']

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(BoneDataset(df_train, PASTA_BASE, train_transforms), batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(BoneDataset(df_val, PASTA_BASE, val_transforms), batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(BoneDataset(df_test, PASTA_BASE, val_transforms), batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # 2. Modelo
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=2)
        model = model.to(DEVICE)
    except Exception as e:
        print(f"ERRO CRÍTICO: {e}")
        return None

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # Conta quantos exemplos de cada classe existem no treino atual
    n_benignos = len(df_train[df_train['target'] == 0])
    n_malignos = len(df_train[df_train['target'] == 1])
    n_total = n_benignos + n_malignos
    
    # Calcula pesos inversamente proporcionais
    # A classe com menos exemplos ganha um peso maior
    weight_0 = n_total / (2 * n_benignos)  # Peso para Benigno (~0.6)
    weight_1 = n_total / (2 * n_malignos)  # Peso para Maligno (~2.7)
    
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(DEVICE)
    print(f"Pesos de Classe aplicados: Benigno={weight_0:.2f} | Maligno={weight_1:.2f}")

    # Passa os pesos para a função de erro
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # ============================================
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    best_epoch_log = 0
    patience_counter = 0
    
    # 3. Loop de Treino
    for epoch in range(EPOCHS):
        model.train()
        for imgs, lbls in tqdm(train_loader, desc=f"Epoca {epoch+1}/{EPOCHS}", leave=False):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_log = epoch + 1
            patience_counter = 0
            # AQUI ESTÁ A MUDANÇA: SALVAMENTO PERMANENTE
            torch.save(model.state_dict(), caminho_salvamento)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early Stopping acionado na época {epoch+1}")
                break
    
    # 4. Teste Final
    if os.path.exists(caminho_salvamento):
        print(f"Carregando melhor modelo de: {caminho_salvamento}")
        model.load_state_dict(torch.load(caminho_salvamento))
    else:
        print("Aviso: Nenhum modelo salvo.")
        
    model.eval()
    preds, targets, probs = [], [], []
    
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            prob = torch.softmax(out, dim=1)[:, 1]
            _, pred = torch.max(out, 1)
            preds.extend(pred.cpu().numpy())
            targets.extend(lbls.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    rec = recall_score(targets, preds)
    try: auc = roc_auc_score(targets, probs)
    except: auc = 0
    total_time = (time.time() - start_time) / 60

    result = {
        'ID_Exp': 'Auto',
        'Familia': config_modelo['family'],
        'Modelo': model_name,
        'Resolucao': f"{img_size}x{img_size}",
        'Best_Epoch': best_epoch_log,
        'Val_Loss_Final': round(best_val_loss, 4),
        'Tempo_Total_min': round(total_time, 2),
        'Acuracia': round(acc, 4),
        'F1_Score': round(f1, 4),
        'Recall': round(rec, 4),
        'AUC': round(auc, 4),
        'Batch_Size': batch_size
    }
    
    return result

def main():
    os.makedirs(PASTA_MODELOS, exist_ok=True) # Cria a pasta de modelos
    resultados = []
    
    if os.path.exists(ARQUIVO_RESULTADOS):
        try: resultados = pd.read_csv(ARQUIVO_RESULTADOS).to_dict('records')
        except: pass
    
    for config in MODELOS_PARA_TESTAR:
        if any(r['Modelo'] == config['name'] for r in resultados):
            print(f"PULANDO {config['name']} (Já existe).")
            continue

        try:
            gc.collect()
            torch.cuda.empty_cache()
            
            res = treinar_um_modelo(config)
            if res:
                resultados.append(res)
                df_res = pd.DataFrame(resultados)
                df_res.to_csv(ARQUIVO_RESULTADOS, index=False)
                print(f"--> Resultado salvo! Modelo guardado em '{PASTA_MODELOS}'")
                
        except Exception as e:
            print(f"CRASH no modelo {config['name']}: {e}")
            continue

if __name__ == "__main__":
    main()