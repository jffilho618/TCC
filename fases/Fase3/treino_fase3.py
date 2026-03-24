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
import random
import time
import gc
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

# --- UI IMPORTS ---
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich import box

console = Console()

# ================= CONFIGURAÇÕES FASE 3 (DEFINITIVA) =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Diretórios Gerados pelo script de crops
DIR_CNN = os.path.join(SCRIPT_DIR, 'classificacao_crops_cnn')
DIR_VIT = os.path.join(SCRIPT_DIR, 'classificacao_crops_vit')
DIR_TESTE = os.path.join(SCRIPT_DIR, 'classificacao_crops_teste') # Crops Limpos para Teste

# CSVs
CSV_CNN = os.path.join(DIR_CNN, 'dataset_kfold_controle.csv')
CSV_VIT = os.path.join(DIR_VIT, 'dataset_kfold_controle_vit.csv')
PASTA_PESOS_RAD = os.path.join(SCRIPT_DIR, 'pesos_radimagenet')
PASTA_SAIDA_MODELOS = os.path.join(SCRIPT_DIR, 'modelos_fase3')
ARQUIVO_LOG = os.path.join(SCRIPT_DIR, 'RESULTADOS_FASE3.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4  # Otimizado para Windows
SEED = 42

# Hiperparâmetros Globais
EPOCHS = 60
PATIENCE = 12
LR = 1e-4
IMG_SIZE = 384
LABEL_SMOOTHING = 0.1
DROP_PATH_RATE = 0.2
ACCUMULATION_STEPS = 2  # Gradient Accumulation: simula batch 2x maior  

# --- LISTA DE MODELOS FASE 3 ---
# Definimos o 'tipo' para saber qual pasta de dados usar
MODELOS_PARA_TREINO = [
    # # 1. ResNet50 (RadImageNet) - O Baseline Forte
    {'name': 'resnet50', 'batch': 24, 'tipo': 'cnn', 'radimagenet': True},  # Otimizado: 16 → 24

    # 2. DenseNet121 (RadImageNet) - O Detalhista (Novo!)
    {'name': 'densenet121', 'batch': 20, 'tipo': 'cnn', 'radimagenet': True},  # Otimizado: 12 → 20

    # # 3. EfficientNetV2-L (ImageNet) - O Moderno
    {'name': 'tf_efficientnetv2_l', 'batch': 6, 'tipo': 'cnn', 'radimagenet': False},  # Otimizado: 4 → 6

    # 5. BEiT v2 Base (ImageNet-22k) - O Semântico (Novo!)
    {'name': 'beitv2_base_patch16_224.in1k_ft_in22k', 'batch': 12, 'tipo': 'vit', 'radimagenet': False},  # Otimizado: 8 → 12

    # 4. SwinV2 Base (ImageNet-22k) - O Monstro (Novo!)
    # Atenção: swinv2_base_window12to24_192to384_22kft1k é pesado. Batch baixo.
    {'name': 'swinv2_base_window12to24_192to384_22kft1k', 'batch': 2, 'tipo': 'vit', 'radimagenet': False},  # Otimizado: 4 → 6

]


# =====================================================================

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False  # Otimizado para performance
    torch.backends.cudnn.benchmark = True  # Otimizado: 20-30% mais rápido!

seed_everything(SEED)

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
        # Caminho: root/treino_kfold/benigno/img.png
        img_path = os.path.join(self.root_dir, row['split_group'], row['class_name'], row['image_id'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback seguro
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
            
        label = self.class_map[row['class_name']]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def carregar_pesos_radimagenet(model, model_name):
    """Carrega pesos médicos para ResNet e DenseNet."""
    nome_arquivo = ''
    if 'resnet50' in model_name:
        nome_arquivo = 'RadImageNet-ResNet50_notop.pth'
    elif 'densenet121' in model_name:
        nome_arquivo = 'DenseNet121.pt'
    
    # Tenta achar o arquivo (aceita .pth ou .pt)
    caminho = os.path.join(PASTA_PESOS_RAD, nome_arquivo)
    if not os.path.exists(caminho):
        # Tenta versão alternativa curta
        caminho_alt = os.path.join(PASTA_PESOS_RAD, f"{model_name}.pt")
        if os.path.exists(caminho_alt): caminho = caminho_alt
    
    if os.path.exists(caminho):
        console.print(f"[cyan] -> 🩻 RadImageNet detectado: {os.path.basename(caminho)}[/]")
        try:
            state_dict = torch.load(caminho, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            console.print(f"[bold green] -> ✓ Pesos Médicos Carregados![/]")
            return True
        except Exception as e:
            console.print(f"[bold red] -> ✗ Erro ao carregar: {e}[/]")
    else:
        console.print(f"[yellow] -> ⚠ RadImageNet não encontrado em {PASTA_PESOS_RAD}. Usando ImageNet.[/]")
    return False

def criar_tabela_dashboard():
    table = Table(title="Histórico de Treinamento", box=box.ROUNDED)
    table.add_column("Ep", justify="center", style="cyan", no_wrap=True)
    table.add_column("Train Loss", justify="right", style="magenta")
    table.add_column("Val Loss", justify="right", style="magenta")
    table.add_column("Val F1", justify="right", style="green")
    table.add_column("Status", justify="center")
    return table

def treinar_ciclo(config):
    model_name = config['name']
    batch_size = config['batch']
    tipo_modelo = config['tipo'] # 'cnn' ou 'vit'
    usar_rad = config['radimagenet']
    
    # 1. Definir Pastas e CSVs baseados no tipo
    if tipo_modelo == 'cnn':
        pasta_treino = DIR_CNN
        arquivo_csv = CSV_CNN
        info_dados = "Original (Crops Limpos)"
    else:
        pasta_treino = DIR_VIT
        arquivo_csv = CSV_VIT
        info_dados = "Expandido (10x Augmentation)"

    # 2. Resolução Dinâmica
    if 'beit' in model_name: current_size = 224
    else: current_size = IMG_SIZE # 384 padrão

    console.print(Panel.fit(
        f"[bold white]Modelo:[/][bold yellow] {model_name}[/]\n"
        f"[bold white]Tipo:[/][cyan] {tipo_modelo.upper()}[/] | [bold white]Dados:[/][cyan] {info_dados}[/]\n"
        f"[bold white]Batch:[/][green] {batch_size}[/] | [bold white]Size:[/][green] {current_size}px[/]",
        title="🚀 Configuração Fase 3", border_style="blue"
    ))

    with console.status("[bold green]Preparando pipeline...", spinner="dots"):
        
        # Transforms (CLAHE já foi aplicado no pré-processamento)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(current_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((current_size, current_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Loaders
        if not os.path.exists(arquivo_csv):
            raise FileNotFoundError(f"CSV não encontrado: {arquivo_csv}")

        df = pd.read_csv(arquivo_csv)
        
        # Filtros rigorosos
        # Treino: Fold 1, 2, 3, 4 (Pode ter augments se for ViT)
        df_train = df[(df['split_group'] == 'treino_kfold') & (df['fold'] != 0)]
        
        # Validação: Fold 0 (SEMPRE DADOS REAIS, o script de geração garantiu isso)
        df_val = df[(df['split_group'] == 'treino_kfold') & (df['fold'] == 0)]
        
        # Teste: Carrega da pasta de TESTE LIMPO
        # Vamos ler os arquivos da pasta de teste fisicamente para garantir
        test_images = []
        for cls_name in ['benigno', 'maligno']:
            path_cls = os.path.join(DIR_TESTE, cls_name)
            if os.path.exists(path_cls):
                for f in os.listdir(path_cls):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):  # ✅ Adicionado .jpeg
                        test_images.append({
                            'split_group': '', # Dummy, pois vamos passar root direto
                            'class_name': cls_name,
                            'image_id': f
                        })
        df_test = pd.DataFrame(test_images)

        # ✅ Validação crítica
        if len(df_test) == 0:
            raise ValueError(f"❌ ERRO: Nenhuma imagem de teste encontrada em {DIR_TESTE}!\nExecute 'python gerar_crops_fase3.py' primeiro.")

        # Sampler para Treino
        targets = df_train['class_name'].map({'benigno': 0, 'maligno': 1}).values
        class_counts = np.bincount(targets)
        weights = 1. / class_counts
        samples_weights = weights[targets]
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

        # Configuração Otimizada para Windows
        loader_kwargs = {
            'num_workers': NUM_WORKERS,
            'pin_memory': True,          # Acelera a cópia RAM -> VRAM (GPU)
            'persistent_workers': True if NUM_WORKERS > 0 else False  # Mantém os workers vivos entre épocas
        }

        # DataLoaders
        # Treino e Val usam a pasta específica (CNN ou VIT)
        train_loader = DataLoader(
            BoneDataset(df_train, pasta_treino, train_transforms),
            batch_size=batch_size,
            sampler=sampler,
            **loader_kwargs
        )

        val_loader = DataLoader(
            BoneDataset(df_val, pasta_treino, val_transforms),
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs
        )

        # Teste usa SEMPRE a pasta de TESTE LIMPO
        # Nota: BoneDataset espera estrutura split/classe/img.
        # Como DIR_TESTE já tem classe/img, vamos ajustar o BoneDataset temporariamente ou hackear o path
        # Hack: Passamos DIR_TESTE como root, e no df 'split_group' é vazio.
        # Assim path vira: DIR_TESTE + "" + benigno + img.png -> Correto.
        test_loader = DataLoader(
            BoneDataset(df_test, DIR_TESTE, val_transforms),
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs
        )

        # Modelo (com tratamento para arquiteturas antigas)
        # DenseNet e algumas CNNs antigas não suportam drop_path_rate
        try:
            model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=2,
                drop_path_rate=DROP_PATH_RATE
            )
        except TypeError:
            # Fallback para modelos que não suportam drop_path_rate (ex: DenseNet)
            console.print(f"[yellow]⚠ {model_name} não suporta drop_path_rate, criando sem ele...[/]")
            model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=2
            )
        
        if usar_rad:
            carregar_pesos_radimagenet(model, model_name)

        model = model.to(DEVICE)

        # LR ajustado para ViT (mais sensíveis)
        lr_atual = LR / 2 if tipo_modelo == 'vit' else LR

        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=lr_atual, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # Desabilita AMP para ViT (pode causar overflow/NaN)
        usar_amp = tipo_modelo != 'vit'
        scaler = torch.amp.GradScaler('cuda', enabled=usar_amp)

    # Loop
    best_loss = float('inf')
    patience_count = 0
    save_path = os.path.join(PASTA_SAIDA_MODELOS, f"{model_name}_fase3.pth")
    checkpoint_path = os.path.join(PASTA_SAIDA_MODELOS, f"{model_name}_checkpoint.pth")
    os.makedirs(PASTA_SAIDA_MODELOS, exist_ok=True)
    start_time = time.time()
    start_epoch = 0

    # ✅ RETOMAR TREINO SE CHECKPOINT EXISTIR
    if os.path.exists(checkpoint_path):
        console.print(f"[yellow]📥 Checkpoint encontrado! Retomando treino...[/]")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1
        patience_count = checkpoint['patience_count']
        console.print(f"[green]✓ Retomando da época {start_epoch}/{EPOCHS} | Best Loss: {best_loss:.4f}[/]")
    
    dashboard_table = criar_tabela_dashboard()
    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeRemainingColumn())

    with Live(Panel(dashboard_table, title="Monitoramento Fase 3", border_style="green"), refresh_per_second=4) as live:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            train_loss = 0
            task_id = progress.add_task(f"[cyan]Ep {epoch+1}", total=len(train_loader))
            optimizer.zero_grad()  # Inicializar gradiente no começo da época

            for i, (imgs, lbls) in enumerate(train_loader):
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

                with torch.amp.autocast('cuda', enabled=usar_amp):
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls) / ACCUMULATION_STEPS  # Dividir loss pela acumulação

                # Proteção contra NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    console.print(f"[bold red]⚠ NaN/Inf detectado no batch {i}! Pulando...[/]")
                    continue

                scaler.scale(loss).backward()

                # Gradient Accumulation: só atualiza a cada ACCUMULATION_STEPS
                if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * ACCUMULATION_STEPS  # Recuperar loss original

                if i % 10 == 0:
                    live.update(Panel(dashboard_table, title=f"Fase 3 | Ep {epoch+1} | Batch {i}/{len(train_loader)}", border_style="green"))

            progress.remove_task(task_id)

            # Proteção contra NaN no scheduler
            avg_train_loss = train_loss / len(train_loader)
            if not (torch.isnan(torch.tensor(avg_train_loss)) or torch.isinf(torch.tensor(avg_train_loss))):
                scheduler.step(epoch + avg_train_loss)
            else:
                console.print(f"[bold red]⚠ Train loss inválido na época {epoch+1}! Pulando scheduler...[/]")
                scheduler.step(epoch)

            # Val
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
            val_f1 = f1_score(all_lbls, all_preds, average='macro')
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_count = 0
                torch.save(model.state_dict(), save_path)
                status_icon = "[bold green]★ SALVO[/]"
            else:
                patience_count += 1
                status_icon = f"[yellow]{patience_count}/{PATIENCE}[/]"

            # ✅ Salvar checkpoint a cada época
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_loss': best_loss,
                'patience_count': patience_count,
            }, checkpoint_path)

            dashboard_table.add_row(f"{epoch+1}", f"{train_loss/len(train_loader):.4f}", f"{val_loss:.4f}", f"{val_f1:.4f}", status_icon)
            live.update(Panel(dashboard_table, title="Monitoramento Fase 3", border_style="green"))

            if patience_count >= PATIENCE:
                dashboard_table.add_row("STOP", "-", "-", "-", "[bold red]EARLY STOP[/]")
                # ✅ Limpar checkpoint ao terminar
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                break

    # Avaliação Final (TESTE LIMPO)
    console.print("\n[bold cyan]--- AVALIANDO NO TESTE FINAL (CROPS LIMPOS) ---[/]")
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

    # ✅ Validação final antes de calcular métricas
    if len(test_targets) == 0:
        console.print("[bold red]❌ ERRO: Test loader vazio! Verifique a pasta de teste.[/]")
        return {
            'Modelo': model_name,
            'Tipo': tipo_modelo.upper(),
            'RadImageNet': 'Sim' if usar_rad else 'Nao',
            'Resolucao': f"{current_size}x{current_size}",
            'Val_Loss': round(best_loss, 4),
            'Acuracia': 0.0,
            'F1_Score': 0.0,
            'Recall': 0.0,
            'AUC': 0.0,
            'Tempo_min': round((time.time() - start_time)/60, 2)
        }

    return {
        'Modelo': model_name,
        'Tipo': tipo_modelo.upper(),
        'RadImageNet': 'Sim' if usar_rad else 'Nao',
        'Resolucao': f"{current_size}x{current_size}",
        'Val_Loss': round(best_loss, 4),
        'Acuracia': round(accuracy_score(test_targets, test_preds), 4),
        'F1_Score': round(f1_score(test_targets, test_preds), 4),
        'Recall': round(recall_score(test_targets, test_preds), 4),
        'AUC': round(roc_auc_score(test_targets, test_probs), 4) if len(set(test_targets)) > 1 else 0,
        'Tempo_min': round((time.time() - start_time)/60, 2)
    }

def main():
    console.rule("[bold blue]SISTEMA FASE 3: O GRANDE EXPERIMENTO[/]")
    resultados = []
    
    if os.path.exists(ARQUIVO_LOG):
        resultados = pd.read_csv(ARQUIVO_LOG).to_dict('records')

    for config in MODELOS_PARA_TREINO:
        nome_mod = config['name']
        if any(r['Modelo'] == nome_mod for r in resultados):
            console.print(f"PULANDO {nome_mod} (Já feito).")
            continue
            
        gc.collect()
        torch.cuda.empty_cache()
        try:
            res = treinar_ciclo(config)
            resultados.append(res)
            pd.DataFrame(resultados).to_csv(ARQUIVO_LOG, index=False)
            console.print(Panel(f"Recall: [bold green]{res['Recall']}[/] | AUC: [bold magenta]{res['AUC']}[/]", title=f"Fim: {nome_mod}"))
        except Exception as e:
            console.print_exception()

if __name__ == "__main__":
    main()