# 📋 Checklist de Trabalhos Relacionados Analisados

**Objetivo:** Rastrear quais PDFs já foram analisados para evitar duplicação

---

## ✅ Trabalhos Analisados

### Detecção e Classificação (ambos)

- [x] **TR001** - `Deng_2024_Normal_vs_Tumor_e_Benigno_vs_Maligno.pdf`
  - **Original:** `1-s2.0-S2212137424001283-main.pdf`
  - **Localização:** `Trabalhos relacionados/Detecção e Classificação/`
  - **Tipo:** Detecção (Normal vs Tumor) + Classificação (Benigno vs Maligno)
  - **Data análise:** 2026-01-03
  - **Status CSV:** ✅ Registrado

### Classificação

- [x] **TR002** - `He_2020_3way_Classification_Benign_Intermediate_Malignant.pdf`
  - **Original:** `1-s2.0-S2352396420304977-main.pdf`
  - **Localização:** `Trabalhos relacionados/Classificação/`
  - **Tipo:** Classificação (3 tarefas: B vs Not-B, M vs Not-M, 3-way)
  - **Data análise:** 2026-01-03
  - **Status CSV:** ✅ Registrado
  - **Observação:** REQUER CROPPING MANUAL (não é end-to-end)

- [x] **TR003** - `A radiograph dataset.pdf`
  - **Original:** `A radiograph dataset.pdf`
  - **Localização:** `Trabalhos relacionados/Classificação/`
  - **Tipo:** DATA DESCRIPTOR + Validação Técnica (Localização + Segmentação + Classificação 3-way)
  - **Data análise:** 2026-01-03
  - **Status CSV:** ✅ Registrado
  - **Observação:** **DATASET BTXRD** (mesmo usado no MEU TCC!) - Baseline YOLOv8s para validação técnica

- [x] **TR004** - `A_Deep_Evaluation_of_Digital_Image_based_Bone_Cancer_Prediction_using_Modified_Machine_Learning_Strategy.pdf`
  - **Localização:** `Trabalhos relacionados/Classificação/`
  - **Tipo:** Classificação Binária (Normal vs Cancer) - k-NN modificado
  - **Data análise:** 2026-01-03
  - **Status CSV:** ✅ Registrado
  - **Observação:** ⚠️ **BAIXA QUALIDADE** - Dataset NÃO especificado, método antiquado (k-NN, NÃO deep learning), título enganoso, validação fraca

---

## 📂 Trabalhos Pendentes

### Classificação

- [ ] Nenhum pendente no momento

### Detecção

- [x] **TR005** - `Applying_CNNs_for_the_Early_Detection_of_Bone_Malignant_Tumors_in_X-Ray_Images.pdf`
  - **Localização:** `Trabalhos relacionados/Detecção/`
  - **Tipo:** Detecção Binária (Normal vs Malignant)
  - **Data análise:** 2026-01-03
  - **Status CSV:** ✅ Registrado
  - **Observação:** ⚠️ **BAIXA QUALIDADE** - Arquitetura CNN NÃO especificada, dataset pequeno (500), hiperparâmetros ausentes

- [x] **TR011-REVIEW** - `Bone_Cancer_Detection_techniques_using_Machine_Learning.pdf`
  - **Localização:** `Trabalhos relacionados/Detecção/`
  - **Tipo:** **SURVEY/REVIEW PAPER** - Taxonomia de técnicas ML para bone cancer detection
  - **Data análise:** 2026-01-03
  - **Status CSV:** ✅ Registrado como TR011-REVIEW
  - **Observação:** 📚 **REVIEW ÚTIL** - ICCMSO 2022. Tabela comparativa com 10 trabalhos (técnicas: K-means, SVM, RF, KNN, CNN, GA). Taxonomia de algoritmos ML. Comparação X-ray vs CT vs MRI vs Bone Scan. Desafios identificados: falta standardização, external validation, dataset size.
- [x] **TR006** - `Bone_Tumor_Detection_using_Faster_R-CNN.pdf`
  - **Localização:** `Trabalhos relacionados/Detecção/`
  - **Tipo:** Detecção (Normal vs Tumor) + Localização com Bounding Boxes
  - **Data análise:** 2026-01-03
  - **Status CSV:** ✅ Registrado
  - **Observação:** Faster R-CNN com CNN customizada, 98.73% test acc, 96.28% mAP. Transfer learning interno CNN→RPN
- [x] **TR007** - `Bone_Tumor_Detection_Using_X-Ray_Images.pdf`
  - **Localização:** `Trabalhos relacionados/Detecção/`
  - **Tipo:** Detecção Binária (Tumor vs Normal)
  - **Data análise:** 2026-01-03
  - **Status CSV:** ✅ Registrado
  - **Observação:** ⚠️ HÍBRIDO ML vs DL - XGBoost VENCEU (94%), CNN FALHOU (49.1%!). Dataset Figshare 2001 imgs. GUI Tkinter

### Detecção e Classificação

- [x] **TR008** - `fmed-12-1555907.pdf`
  - **Localização:** `Trabalhos relacionados/` (pasta raiz)
  - **Tipo:** Classificação 4-classes (NT, NVT, VT, NVR) - **HISTOPATOLOGIA (não Raio-X!)**
  - **Data análise:** 2026-01-03
  - **Status CSV:** ✅ Registrado
  - **Observação:** ⭐ EXCELENTE! Híbrido CNN+ViT, 99.08% acc 4-class. **MODALIDADE DIFERENTE (histopatologia vs Raio-X)** - Frontiers Medicine 2025
- [x] **TR009** - `BJR-14-2046-3758.149.BJR-2024-0505.R1.pdf`
  - **Localização:** `Trabalhos relacionados/Detecção e Classificação/`
  - **Tipo:** Detecção + Classificação Binária (Benign vs Malignant)
  - **Data análise:** 2026-01-03
  - **Status CSV:** ✅ Registrado
  - **Observação:** ⭐ EXCELENTE! Transformer DINO vs YOLO, end-to-end, multi-institucional (3 centros), 40 tipos tumores, comparação com médicos - Bone Joint Res 2025
- [x] **TR010** - `dalai-et-al-2024-automated-bone-cancer-detection-using-deep-learning-on-x-ray-images.pdf`
  - **Localização:** `Trabalhos relacionados/Detecção e Classificação/`
  - **Tipo:** Detecção + Classificação Binária (Healthy vs Cancerous)
  - **Data análise:** 2026-01-03
  - **Status CSV:** ✅ Registrado
  - **Observação:** ⚠️ **QUALIDADE MÉDIA** - Dataset MUITO pequeno (200), fonte NÃO especificada (irreproducível), híbrido SqueezeNet+LSTM com GSO/ICS - Surgical Innovation 2025

---

## 📊 Estatísticas

- **Total analisados:** 11 ✅ (10 experimentais + 1 review)
- **Detecção:** 3 (TR005, TR006, TR007)
- **Classificação:** 3 (TR002, TR003, TR004)
- **Ambos (Detecção+Classificação):** 4 (TR001, TR008, TR009, TR010)
- **Review/Survey Papers:** 1 (TR011-REVIEW - Taxonomia ML)
- **Data Descriptors:** 1 (TR003 - BTXRD)
- **Histopatologia (não Raio-X):** 1 (TR008 - modalidade diferente!)
- **Baixa Qualidade:** 2 (TR004 - k-NN; TR005 - CNN não especificada)
- **Qualidade Média:** 1 (TR010 - dataset pequeno 200, não especificado)
- **Issues Metodológicos:** 1 (TR007 - CNN 49% acc catastrófica)
- **Alta Qualidade / SOTA:** 2 (TR008 - 99% acc CNN+ViT Frontiers Medicine; TR009 - DINO Bone Joint Res)
- **Pendentes Classificação:** 0 ✅
- **Pendentes Detecção:** 0 ✅
- **Pendentes Detecção+Classificação:** 0 ✅
- **Total Pendentes:** 0 ✅✅✅

---

## 🔑 Insights Importantes

### Resumo dos 4 Trabalhos:
- **TR001:** Dataset pequeno (130 imgs), ResNet34, baixo desempenho B vs M (71.2% acc, AUC 0.62)
- **TR002:** Dataset grande (2899 imgs), EfficientNet-B0, melhor desempenho mas **REQUER CROPPING MANUAL**
- **TR003:** **BTXRD DATASET PAPER** (mesmo dataset do MEU TCC!) - Baseline YOLOv8s: Normal=90.4%, Benign=89.9%, Malignant=96.5%
- **TR004:** ⚠️ **BAIXA QUALIDADE** - k-NN antiquado (NÃO deep learning), dataset não especificado, apenas 96.59% acc sem AUC

### Vantagens do MEU TCC:
1. ✅ **End-to-end automático** (vs TR002 que requer cropping manual)
2. ✅ **Dataset público BTXRD** (vs privados TR001/TR002/TR004)
3. ✅ **Técnicas avançadas** (CLAHE, WeightedSampler, offline aug 10x, Label Smoothing, DropPath)
4. ✅ **Transfer learning médico** (RadImageNet vs COCO do baseline TR003, nenhum em TR004)
5. ✅ **Melhor performance B vs M** (93.95% acc + AUC=0.9749 vs 71-73% TR001/TR002)
6. ✅ **Arquiteturas state-of-the-art** (EfficientNetV2-L, Swin-L, MaxViT vs ResNet34/EfficientNet-B0/YOLOv8s-cls/k-NN)
7. ✅ **Deep learning REAL** (vs k-NN antiquado do TR004)
8. ✅ **Reproduzibilidade total** (vs TR004 que não especifica dataset)

### Destaques:
- 🎯 **TR003 CRUCIAL:** É o paper do dataset que eu uso! Permite comparar com baseline oficial
- ⚠️ **TR004 ALERTA:** Exemplo de trabalho de baixa qualidade metodológica (título enganoso, dataset não especificado, método antiquado)
- 📈 **Qualidade dos trabalhos:** TR002 > TR001 > TR003 (baseline) > TR004 (baixa qualidade)

---

---

## 🎉 ANÁLISE COMPLETA!

**Todos os 11 trabalhos relacionados foram analisados e registrados no CSV!**

### Resumo Final:
- ✅ **11 trabalhos analisados** (TR001-TR011)
  - **10 experimentais** (TR001-TR010)
  - **1 review/survey** (TR011-REVIEW)
- ✅ **3 categorias experimentais:** Detecção (3), Classificação (3), Detecção+Classificação (4)
- ✅ **Qualidade variada:** 2 SOTA, 2 Baixa, 1 Média, 5 Razoável
- ✅ **Modalidades:** 9 Raio-X, 1 Histopatologia
- ✅ **Dataset BTXRD:** TR003 é o paper do dataset usado no TCC!
- ✅ **Review útil:** TR011 documenta técnicas ML tradicionais pré-2022

### Destaques dos últimos papers (TR009-TR011):
- **TR009 ⭐ EXCELENTE:** Transformer DINO vs YOLO, multi-institucional (3 centros), 40 tipos tumores, comparação com médicos (77% acc B vs M, 85.7% detection) - Bone Joint Research 2025
- **TR010 ⚠️ QUALIDADE MÉDIA:** Híbrido SqueezeNet+LSTM com GSO/ICS, dataset MUITO pequeno (200), fonte NÃO especificada (irreproducível), 94.79% acc Healthy vs Cancerous - Surgical Innovation 2024
- **TR011-REVIEW 📚 ÚTIL:** Survey ICCMSO 2022 - Tabela comparativa 10 trabalhos, taxonomia ML (SVM, RF, KNN, Decision Tree, GA), comparação modalidades (X-ray vs CT vs MRI), desafios identificados (standardização, external validation, dataset size)

### Valor do TR011-REVIEW para o TCC:
- **Contextualiza** estado-da-arte ML tradicional (2022) que o TCC supera com deep learning moderno (2024-2025)
- **Documenta** técnicas clássicas: K-means, GLCM features, SVM, Random Forest, KNN
- **Identifica desafios** que o TCC aborda: external validation ✅, generalizability ✅, dataset size ✅
- **Mostra evolução**: ML tradicional (até 2022) → Deep Learning SOTA (TCC 2024-2025)

---

**Última atualização:** 2026-01-03 - **ANÁLISE 100% COMPLETA ✅**
