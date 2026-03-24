import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import cv2
import random
import time
import gc
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

# --- IMPORTS DE UI (RICH) ---
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich import box

# Inicializa o console
console = Console()

# ================= CONFIGURAÇÕES FASE 2.5 (TUNADA) =================
# Caminhos relativos ao diretório do script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASE2_DIR = SCRIPT_DIR

# Verificar onde está classificacao_crops
PASTA_BASE_FASE2 = os.path.join(SCRIPT_DIR, 'classificacao_crops')
PASTA_BASE_FASE15 = os.path.join(os.path.dirname(SCRIPT_DIR), 'Fase1.5', 'classificacao_crops')

if os.path.exists(PASTA_BASE_FASE2):
    PASTA_BASE = PASTA_BASE_FASE2
elif os.path.exists(PASTA_BASE_FASE15):
    PASTA_BASE = PASTA_BASE_FASE15
else:
    PASTA_BASE = 'classificacao_crops'

CSV_CONTROLE = os.path.join(PASTA_BASE, 'dataset_kfold_controle.csv')
PASTA_PESOS_RAD = os.path.join(FASE2_DIR, 'pesos_radimagenet')
PASTA_SAIDA_MODELOS = os.path.join(FASE2_DIR, 'modelos_fase2_ultimate') # Pasta nova para não misturar
ARQUIVO_LOG = os.path.join(FASE2_DIR, 'RESULTADOS_FASE2.5_ULTIMATE.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0
SEED = 42

# --- NOVOS HIPERPARÂMETROS ---
EPOCHS = 60
PATIENCE = 12
LR = 1e-4   
IMG_SIZE = 384 
LABEL_SMOOTHING = 0.1 # <--- MELHORIA 1: Suavização de Rótulos
DROP_PATH_RATE = 0.2  # <--- MELHORIA 2: Stochastic Depth (Drop Path)

MODELOS_PARA_TREINO = [
    {'name': 'resnet50', 'batch': 16, 'use_radimagenet': True},
    {'name': 'tf_efficientnetv2_l', 'batch': 4, 'use_radimagenet': False},
    {'name': 'swin_base_patch4_window12_384', 'batch': 4, 'use_radimagenet': False},
    {'name': 'beit_base_patch16_224', 'batch': 8, 'use_radimagenet': False},
]
# ===================================================================

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

class ApplyCLAHE(object):
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            final = clahe.apply(img_np)
            final = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(final)

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
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        label = self.class_map[row['class_name']]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def carregar_pesos_radimagenet(model, model_name):
    nomes_possiveis = ['RadImageNet-ResNet50_notop.pth', 'ResNet50.pt']
    caminho_final = None
    
    for nome in nomes_possiveis:
        p = os.path.join(PASTA_PESOS_RAD, nome)
        if os.path.exists(p):
            caminho_final = p
            break
            
    if 'resnet50' in model_name and caminho_final:
        console.print(f"[cyan] -> 🩻 Encontrado RadImageNet: {os.path.basename(caminho_final)}[/]")
        try:
            state_dict = torch.load(caminho_final, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            console.print(f"[bold green] -> ✓ Pesos Médicos Carregados![/]")
            return True
        except Exception as e:
            console.print(f"[bold red] -> ✗ Falha ao carregar: {e}[/]")
            return False
    else:
        if 'resnet50' in model_name:
            console.print(f"[yellow] -> ⚠ RadImageNet não encontrado. Usando ImageNet padrão.[/]")
        return False

def criar_tabela_dashboard():
    table = Table(title="Histórico de Treinamento", box=box.ROUNDED)
    table.add_column("Época", justify="center", style="cyan", no_wrap=True)
    table.add_column("Train Loss", justify="right", style="magenta")
    table.add_column("Val Loss", justify="right", style="magenta")
    table.add_column("Val F1", justify="right", style="green")
    table.add_column("Status", justify="center")
    return table

def treinar_ciclo(config):
    model_name = config['name']
    batch_size = config['batch']
    usar_rad = config['use_radimagenet']
    
    # Detecção automática de tamanho
    if 'beit' in model_name:
        display_size = 224
    elif 'swin' in model_name and '256' in model_name:
        display_size = 256
    else:
        display_size = IMG_SIZE

    console.print(Panel.fit(
        f"[bold white]Modelo:[/][bold yellow] {model_name}[/]\n"
        f"[bold white]Batch:[/][cyan] {batch_size}[/] | [bold white]Size:[/][green] {display_size}px[/]\n"
        f"[bold white]Smoothing:[/][magenta] {LABEL_SMOOTHING}[/] | [bold white]DropPath:[/][magenta] {DROP_PATH_RATE}[/]",
        title="🚀 Iniciando Experimento (Fase 2.5 Ultimate)",
        border_style="blue"
    ))

    with console.status("[bold green]Preparando pipeline...", spinner="dots"):
        if 'beit' in model_name:
            current_size = 224
        elif 'swin' in model_name and '256' in model_name:
            current_size = 256
        else:
            current_size = IMG_SIZE
        
        # --- MELHORIA 3: Random Resized Crop (Zoom Inteligente) ---
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(current_size, scale=(0.85, 1.0)), # Zoom de 85% a 100%
            ApplyCLAHE(clip_limit=4.0),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize((current_size, current_size)),
            ApplyCLAHE(clip_limit=4.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Loaders
        df = pd.read_csv(CSV_CONTROLE)
        df_train = df[(df['split_group'] == 'treino_kfold') & (df['fold'] != 0)]
        df_val = df[(df['split_group'] == 'treino_kfold') & (df['fold'] == 0)]
        df_test = df[df['split_group'] == 'teste_final']

        targets = df_train['class_name'].map({'benigno': 0, 'maligno': 1}).values
        class_counts = np.bincount(targets)
        weights = 1. / class_counts
        samples_weights = weights[targets]
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

        train_loader = DataLoader(BoneDataset(df_train, PASTA_BASE, train_transforms), 
                                  batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS)
        val_loader = DataLoader(BoneDataset(df_val, PASTA_BASE, val_transforms), 
                                batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(BoneDataset(df_test, PASTA_BASE, val_transforms), 
                                 batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

        # --- MELHORIA 2: Stochastic Depth (Drop Path) no Modelo ---
        model = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=2,
            drop_path_rate=DROP_PATH_RATE # Aplica em todos os modelos suportados
        )
        
        if usar_rad and 'resnet' in model_name:
            carregar_pesos_radimagenet(model, model_name)
        model = model.to(DEVICE)

        # --- MELHORIA 1: Label Smoothing na Loss ---
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scaler = torch.amp.GradScaler('cuda')

    # Training Loop
    best_loss = float('inf')
    patience_count = 0
    save_path = os.path.join(PASTA_SAIDA_MODELOS, f"{model_name}_fase2_ult.pth")
    os.makedirs(PASTA_SAIDA_MODELOS, exist_ok=True)
    start_time = time.time()
    
    dashboard_table = criar_tabela_dashboard()
    
    # Barra de progresso para as épocas
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )
    
    with Live(Panel(dashboard_table, title="Monitoramento em Tempo Real", border_style="green"), refresh_per_second=4) as live:
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            
            # --- MELHORIA 4: Barra de Carregamento (Já estava, agora garantida) ---
            task_id = progress.add_task(f"[cyan]Epoca {epoch+1}/{EPOCHS}", total=len(train_loader))
            
            for i, (imgs, lbls) in enumerate(train_loader):
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                
                if i % 5 == 0:
                    title_str = f"Monitoramento | Epoca {epoch+1} | Batch {i}/{len(train_loader)}"
                    live.update(Panel(dashboard_table, title=title_str, border_style="green"))

            progress.remove_task(task_id)
            scheduler.step(epoch + train_loss/len(train_loader))

            # Validação
            model.eval()
            val_loss = 0
            all_preds, all_lbls = [], []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    outputs = model(imgs)
                    val_loss += criterion(outputs, lbls).item()
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_lbls.extend(lbls.cpu().numpy())

            val_loss /= len(val_loader)
            avg_train_loss = train_loss/len(train_loader)
            val_f1 = f1_score(all_lbls, all_preds, average='macro')

            status_icon = ""
            if val_loss < best_loss:
                best_loss = val_loss
                patience_count = 0
                torch.save(model.state_dict(), save_path)
                status_icon = "[bold green]★ SALVO[/]"
            else:
                patience_count += 1
                status_icon = f"[yellow]Wait {patience_count}/{PATIENCE}[/]"

            dashboard_table.add_row(
                f"{epoch+1}",
                f"{avg_train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{val_f1:.4f}",
                status_icon
            )
            live.update(Panel(dashboard_table, title="Monitoramento em Tempo Real", border_style="green"))

            if patience_count >= PATIENCE:
                dashboard_table.add_row("STOP", "-", "-", "-", "[bold red]EARLY STOP[/]")
                live.update(Panel(dashboard_table, border_style="red"))
                break

    # Avaliação Final
    console.print("\n[bold cyan]--- AVALIANDO NO TESTE FINAL ---[/]")
    with console.status("Calculando métricas finais...", spinner="earth"):
        model.load_state_dict(torch.load(save_path))
        model.eval()
        test_preds, test_targets, test_probs = [], [], []
        
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(lbls.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(test_targets, test_preds)
    f1 = f1_score(test_targets, test_preds)
    rec = recall_score(test_targets, test_preds)
    try: auc = roc_auc_score(test_targets, test_probs)
    except: auc = 0.0

    return {
        'Modelo': model_name,
        'RadImageNet': 'Sim' if (usar_rad and 'resnet' in model_name) else 'Nao',
        'Resolucao': f"{current_size}x{current_size}",
        'Acuracia': round(acc, 4),
        'F1_Score': round(f1, 4),
        'Recall': round(rec, 4),
        'AUC': round(auc, 4),
        'Tempo_min': round((time.time() - start_time)/60, 2)
    }

def main():
    console.rule("[bold blue]SISTEMA DE TREINAMENTO DE TCC v2.5 (ULTIMATE)[/]")

    console.print(f"[dim]PASTA_BASE: {PASTA_BASE}[/]")
    
    if os.path.exists(ARQUIVO_LOG):
        resultados = pd.read_csv(ARQUIVO_LOG).to_dict('records')
    else:
        resultados = []

    total_models = len(MODELOS_PARA_TREINO)
    
    for i, config in enumerate(MODELOS_PARA_TREINO):
        nome_mod = config['name']
        console.print(f"\n[bold white on blue] MODELO {i+1}/{total_models} [/]")
        
        if any(r['Modelo'] == nome_mod for r in resultados):
            console.print(f"[dim]PULANDO {nome_mod} (Já consta nos registros).[/]")
            continue
            
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            res = treinar_ciclo(config)
            resultados.append(res)
            pd.DataFrame(resultados).to_csv(ARQUIVO_LOG, index=False)
            
            console.print(Panel(
                f"Acurácia: [bold blue]{res['Acuracia']}[/]\n"
                f"Recall:   [bold green]{res['Recall']}[/]\n"
                f"AUC:      [bold magenta]{res['AUC']}[/]",
                title=f"🏆 Resultado: {nome_mod}",
                expand=False
            ))
            
        except Exception as e:
            console.print_exception()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrompido pelo usuário![/]")