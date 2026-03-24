import csv

# Ler o CSV existente
with open(r'c:\Users\jffil\OneDrive\Área de Trabalho\TCC2\analise_trabalhos_relacionados.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Definir resumos metodológicos para cada trabalho
resumos = {
    'TR001': 'Pipeline sequencial em duas etapas usando ResNet34: (1) DETECÇÃO - separa imagens normais de imagens com tumor (Normal vs Tumor) com 5-fold CV alcançando 99.8% acc e AUC=0.998; (2) CLASSIFICAÇÃO - separa tumores benignos de malignos (Benign vs Malignant) usando validação combinatorial C(5,2)=10 alcançando 71.2% acc e AUC=0.62. Preprocessing: conversão DICOM→JPEG e grayscale. Sem data augmentation. Transfer learning ImageNet padrão.',

    'TR002': 'Classificação multi-tarefa com EfficientNet-B0 fine-tuned: avalia TRÊS tarefas distintas (Benign vs Not-Benign, Malignant vs Not-Malignant, 3-way full) usando ensemble de 3 modelos. REQUISITO CRÍTICO: cropping manual da ROI por radiologista antes do treinamento. Preprocessing: DICOM→JPEG, manual ROI crop, padding, resize 512x512, normalização ImageNet. Augmentation: HFlip, Affine, Contrast. Validação: 5-fold CV paciente-level + external test (institutions 4-5). RAdam optimizer, Categorical Cross-Entropy loss, 20 epochs, LR=3e-4, batch=96, Cosine annealing.',

    'TR003': 'Data descriptor BTXRD com validação técnica baseline usando YOLOv8s em TRÊS variantes: (1) YOLOv8s detection (localização com bounding boxes: mAP@0.5=0.707); (2) YOLOv8s-seg (segmentação pixel-level: mAP@0.5=0.691); (3) YOLOv8s-cls (classificação 3-way Normal/Benign/Malignant: 90.4%/89.9%/96.5%). Preprocessing: DICOM→PNG com janelamento adaptativo 16-bit→8-bit, resize 640x640, normalização [0,1], padding. Augmentation: HFlip, VFlip, Rotation 90°, Mosaic, Mixup. Transfer learning: COCO pretrained. Split: 80% train, 10% val, 10% test estratificado. 300 epochs (detection/seg), 100 epochs (classification).',

    'TR004': 'Classificação binária (Normal vs Cancer) usando k-NN TRADICIONAL modificado (ELBCP: Elevated Learning based Bone Cancer Prediction) com mutual information statistics. NÃO é deep learning. Feature extraction MANUAL: GLCM texture features (contrast, correlation, energy, homogeneity). Preprocessing: median filter 3×3, grayscale, sharpening. SEM data augmentation. SEM transfer learning. Validação: simple train/test split com testing count 5-25 imagens. Baseline: SVM (85.09% acc). Resultado: 96.59% acc (testing count=25), SEM AUC/F1 reportados. Implementação MATLAB.',

    'TR005': 'Detecção binária (Normal vs Malignant) usando CNN genérica NÃO especificada. Preprocessing: resize 256x256, normalização [0,1], median filter. Augmentation: random rotations + HFlip. Validação: train/val/test split (proporções NÃO especificadas). Resultado: 90% accuracy, >90% recall. SEM AUC, SEM F1, SEM especificação de arquitetura CNN, SEM hiperparâmetros reportados, SEM transfer learning. Dataset privado 500 imagens (250 Normal, 250 Malignant). Trabalho SUPERFICIAL com descrição genérica.',

    'TR006': 'Detecção two-stage usando Faster R-CNN: (1) CNN customizada (NÃO especificada) classifica Normal vs Tumor (98.73% test acc); (2) RPN (Region Proposal Network) detecta bounding boxes (96.28% mAP@0.5). Transfer learning INTERNO: pesos da CNN customizada transferidos para RPN. Preprocessing: grayscale, median/gaussian filters, one-hot encoding. Augmentation: data augmentation + early stopping (não detalhado). Anotação: CVAT tool. Optimizer: Adam (LR=0.0001). Loss: BCE (CNN) + Multi-task Loss (Faster R-CNN). RPN IoU threshold: ≥0.7 positivo, ≤0.3 negativo. Pipeline end-to-end automático.',

    'TR007': 'Detecção binária (Tumor vs Normal) COMPARANDO 4 modelos: (1) CNN customizada (FALHOU: 49.1% acc, AUC=0.45); (2) SDG/SGD (88.3% acc, AUC=0.64); (3) XGBoost ML (VENCEU: 94% acc, AUC=0.72); (4) ResNet-50 (88% acc, AUC=0.81). Segmentação: CNN + Canny Edge Detection com resize 64×64 (muito baixo). Preprocessing: grayscale, Gaussian filter, normalização. Validação: 80/20 train/test split. Dataset: Figshare 2001 imagens balanceado. GUI Tkinter para real-time. 15 epochs. PROBLEMA METODOLÓGICO: CNN catastrófica indica issue no treinamento. XGBoost (ML tradicional) superou deep learning.',

    'TR008': 'Classificação HÍBRIDA CNN+ViT+MLP para 4-classes (NT, NVT, VT, NVR - necrose tumoral em HISTOPATOLOGIA): (1) CNN customizada extrai features locais (1024); (2) ViT extrai features globais (150,528); (3) Concatenação de features (151,552 total); (4) MLP classifier. Preprocessing: resize 128×128, normalização, class weighting. Augmentation: rotation ±15°, HFlip, VFlip, brightness/contrast (minority classes). Transfer learning: ImageNet (ResNet-50, ViT). Optimizer: Adam (β1=0.9, β2=0.999, LR=1e-4). Loss: Cross-Entropy + class weighting. Validação: 60/15/25 train/val/test + early stopping. Resultado: 99.08% (4-class), 99.56% (3-class). MODALIDADE DIFERENTE: histopatologia H&E, NÃO Raio-X.',

    'TR009': 'Detecção + Classificação end-to-end COMPARANDO Transformer vs CNN: (1) DINO (Transformer-based): 85.7% detection, 77.0% classification B vs M, 70.5% precision, 84.3% sensitivity malignant; (2) YOLO-v8x (CNN): 80.1% detection (métricas classification NÃO reportadas). DINO SUPERIOR (p=0.014). Pipeline: full-field limb radiographs → object detection → classificação Benign vs Malignant automática (SEM cropping manual). Preprocessing: resize 512×512, flipping left→right (exceto hip). Augmentation: rotation + cropping. Treinamento: from scratch (SEM transfer learning especificado). Validação: 5-fold CV (80% train, 20% val), test=initial visit only. IoU threshold: 0.1. Comparação com MÉDICOS (3 oncologistas + 3 cirurgiões). DINO attention mechanism captura relações globais (bone shapes, predilection sites).',

    'TR010': 'Classificação HÍBRIDA SqueezeNet+LSTM com dupla otimização para binária (Healthy vs Cancerous): (1) Preprocessing: Bilateral Filtering (edge-preserving smoothing); (2) Feature extraction: SqueezeNet (Fire modules: squeeze 1×1 conv + expand 1×1/3×3 parallel) com hiperparâmetros otimizados por GSO (Golden Search Optimization); (3) Classification: LSTM (forget/input/output gates) com hiperparâmetros otimizados por ICS (Improved Cuckoo Search: adaptive ELR + adaptive egg numbers). Resultado: 95.52% train acc, 94.79% test acc, 95.30% precision, 95.00% recall/F1, 97.00% specificity. Inference: 54 seconds. SEM data augmentation. SEM transfer learning especificado. Validação: train/test split NÃO detalhado. Dataset: 200 imagens (100 Healthy, 100 Cancerous) fonte NÃO especificada.',

    'TR011-REVIEW': 'SURVEY/REVIEW PAPER sem metodologia experimental própria. Revisa técnicas de ML para bone cancer detection (2022): (1) SEGMENTAÇÃO: K-means clustering, Region growing, Edge-based; (2) FEATURE EXTRACTION: GLCM (Gray Level Co-occurrence Matrix) manual, HOG; (3) CLASSIFICAÇÃO: SVM, Random Forest, KNN, Decision Tree, Genetic Algorithm, CNNs genéricas. TABELA COMPARATIVA (pág 3-4): resume 10 trabalhos experimentais com técnicas, datasets (X-ray, MRI, CT), performance. IDENTIFICA DESAFIOS: falta standardização CNNs para bone cancer, necessidade plataforma comum multi-modal, external validation/generalizability/dataset size críticos. COMPARA MODALIDADES: X-ray (baixa densidade, não mostra soft tissue) vs CT (detalhes ósseos, limitação <5mm) vs MRI (soft tissue, skip metastases) vs Bone Scan (radioativo, metástases). Contextualiza ML tradicional pré-2022.'
}

# Adicionar a coluna Resumo_Metodológico
fieldnames = list(rows[0].keys()) + ['Resumo_Metodológico']

# Escrever novo CSV
with open(r'c:\Users\jffil\OneDrive\Área de Trabalho\TCC2\analise_trabalhos_relacionados_com_resumo.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        row_id = row['ID']
        row['Resumo_Metodológico'] = resumos.get(row_id, 'N/A')
        writer.writerow(row)

print("✅ Novo CSV criado: analise_trabalhos_relacionados_com_resumo.csv")
print(f"✅ Coluna 'Resumo_Metodológico' adicionada com sucesso!")
print(f"✅ Total de trabalhos: {len(rows)}")
