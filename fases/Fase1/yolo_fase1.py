import os
import shutil
import pandas as pd
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.utils import resample
import glob
import random

# ================= CONFIGURAÇÕES =================
ORIGEM_TREINO = os.path.join('classificacao', 'treino_kfold')
ORIGEM_TESTE_FINAL = os.path.join('classificacao', 'teste_final')
PASTA_TEMP_YOLO = 'temp_yolo_data_balanceado'
ARQUIVO_CSV = 'RESULTADOS_FINAIS_FASE1.csv'
MODELO_NOME = 'yolov8n-cls.pt' # Nano (leve). Use 'yolov8s-cls.pt' se quiser um pouco mais forte.
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 32
# =================================================

def preparar_pastas_yolo_balanceado():
    print("--- 1. ORGANIZANDO DADOS PARA O YOLO (COM BALANCEAMENTO) ---")
    
    if os.path.exists(PASTA_TEMP_YOLO):
        try:
            shutil.rmtree(PASTA_TEMP_YOLO)
        except:
            print(f"Aviso: Não foi possível apagar a pasta {PASTA_TEMP_YOLO}. Verifique se está aberta.")
    
    classes = ['benigno', 'maligno']
    
    # Dicionários para guardar os caminhos antes de copiar
    dados_treino = {'benigno': [], 'maligno': []}
    dados_val = {'benigno': [], 'maligno': []}

    # 1. Leitura e Divisão (Split)
    for cls in classes:
        src_dir = os.path.join(ORIGEM_TREINO, cls)
        imagens = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
        
        # Divide 80% Treino / 20% Validação
        train_imgs, val_imgs = train_test_split(imagens, test_size=0.2, random_state=42)
        
        dados_treino[cls] = train_imgs
        dados_val[cls] = val_imgs

    # 2. Balanceamento (Oversampling no TREINO apenas)
    # Descobre qual classe tem mais imagens no treino
    qtd_benigno = len(dados_treino['benigno'])
    qtd_maligno = len(dados_treino['maligno'])
    maior_qtd = max(qtd_benigno, qtd_maligno)
    
    print(f"Contagem Original Treino: Benigno={qtd_benigno}, Maligno={qtd_maligno}")
    print(f"Alvo para Balanceamento: {maior_qtd} imagens por classe.")

    for cls in classes:
        imgs_atuais = dados_treino[cls]
        
        # Se essa classe tiver menos que a maior, faz resampling (duplica imagens aleatórias)
        if len(imgs_atuais) < maior_qtd:
            print(f"--> Balanceando classe '{cls}' (Upsampling)...")
            imgs_balanceadas = resample(imgs_atuais, 
                                        replace=True,     # Permite repetir imagens
                                        n_samples=maior_qtd, 
                                        random_state=42)
            dados_treino[cls] = imgs_balanceadas
        else:
            dados_treino[cls] = imgs_atuais

    # 3. Copiar Arquivos Físicos
    for split_name, dados_split in [('train', dados_treino), ('val', dados_val)]:
        for cls in classes:
            dest_dir = os.path.join(PASTA_TEMP_YOLO, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Para evitar sobrescrever arquivos com mesmo nome no Oversampling,
            # vamos adicionar um prefixo único se for treino
            for i, img_path in enumerate(dados_split[cls]):
                nome_original = os.path.basename(img_path)
                
                if split_name == 'train':
                    # Prefixo numérico para garantir que cópias tenham nomes diferentes
                    novo_nome = f"aug_{i}_{nome_original}"
                else:
                    novo_nome = nome_original
                    
                shutil.copy(img_path, os.path.join(dest_dir, novo_nome))

    print("Estrutura YOLO balanceada criada com sucesso!")

def treinar_e_avaliar():
    print("\n--- 2. INICIANDO TREINAMENTO YOLO ---")
    model = YOLO(MODELO_NOME)
    
    # Treina apontando para a pasta balanceada
    results = model.train(
        data=PASTA_TEMP_YOLO,
        epochs=EPOCHS,
        imgsz=224,
        batch=BATCH_SIZE,
        project='YOLO_Resultados',
        name='experimento_yolo_balanceado',
        patience=PATIENCE,
        device=0,
        verbose=False,
        exist_ok=True # Sobrescreve se rodar 2 vezes
    )
    
    print("\n--- 3. AVALIAÇÃO FINAL NO COFRE DE TESTE ---")
    # Avaliação manual no teste final para métricas precisas
    
    preds_yolo = []
    targets_yolo = []
    probs_yolo = []
    
    test_benigno = glob.glob(os.path.join(ORIGEM_TESTE_FINAL, 'benigno', '*'))
    test_maligno = glob.glob(os.path.join(ORIGEM_TESTE_FINAL, 'maligno', '*'))
    
    todas_imagens = test_benigno + test_maligno
    labels_reais = [0]*len(test_benigno) + [1]*len(test_maligno)
    
    print(f"Processando {len(todas_imagens)} imagens de teste...")
    
    # Inferência
    results_test = model.predict(todas_imagens, verbose=False, device=0)
    
    for i, r in enumerate(results_test):
        probs = r.probs.data.cpu().numpy() # [prob_classe0, prob_classe1]
        
        # Verifica mapeamento de classes do YOLO (ele ordena alfabeticamente)
        # names: {0: 'benigno', 1: 'maligno'} -> Confirmação
        # Se benigno é 0 e maligno é 1:
        pred_class = np.argmax(probs)
        
        preds_yolo.append(pred_class)
        probs_yolo.append(probs[1]) # Probabilidade de ser Maligno
        targets_yolo.append(labels_reais[i])

    # Métricas
    acc = accuracy_score(targets_yolo, preds_yolo)
    f1 = f1_score(targets_yolo, preds_yolo)
    rec = recall_score(targets_yolo, preds_yolo)
    try:
        auc = roc_auc_score(targets_yolo, probs_yolo)
    except:
        auc = 0.0

    print(f"\nRESULTADOS YOLO (Balanceado):")
    print(f"Acurácia: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"AUC:      {auc:.4f}")

    # --- 4. SALVAR NO CSV ---
    novo_resultado = {
        'ID_Exp': 'Auto_YOLO_Bal',
        'Familia': 'CNN (YOLO)',
        'Modelo': MODELO_NOME + ' (Balanceado)',
        'Resolucao': '224x224',
        'Best_Epoch': 'Auto',
        'Val_Loss_Final': 'N/A',
        'Tempo_Total_min': 'N/A',
        'Acuracia': round(acc, 4),
        'F1_Score': round(f1, 4),
        'Recall': round(rec, 4),
        'AUC': round(auc, 4),
        'Batch_Size': BATCH_SIZE
    }
    
    if os.path.exists(ARQUIVO_CSV):
        df = pd.read_csv(ARQUIVO_CSV)
        df_novo = pd.DataFrame([novo_resultado])
        df_final = pd.concat([df, df_novo], ignore_index=True)
        df_final.to_csv(ARQUIVO_CSV, index=False)
        print(f"\nSalvo com sucesso em {ARQUIVO_CSV}!")
    else:
        # Se for o primeiro a rodar (improvável, mas possível)
        pd.DataFrame([novo_resultado]).to_csv(ARQUIVO_CSV, index=False)
        print(f"Arquivo {ARQUIVO_CSV} criado.")

if __name__ == '__main__':
    preparar_pastas_yolo_balanceado()
    treinar_e_avaliar()