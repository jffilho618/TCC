import os
import pandas as pd

xlsx_path = os.path.join('BTXRD', 'dataset.xlsx')
images_dir = os.path.join('BTXRD', 'images')

print('Lendo:', xlsx_path)
try:
    df = pd.read_excel(xlsx_path, engine='openpyxl')
except Exception as e:
    print('Erro lendo xlsx:', e)
    raise

print('Linhas no XLSX:', len(df))
if 'class_name' in df.columns:
    print('\nContagem por class_name:')
    print(df['class_name'].value_counts(dropna=False))
if 'tumor' in df.columns:
    print('\nTumor column counts:')
    print(df['tumor'].value_counts(dropna=False))

print('\nColunas do arquivo:')
print(list(df.columns))

# contar imagens na pasta
img_count = 0
if os.path.isdir(images_dir):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for root,_,files in os.walk(images_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                img_count += 1
print('\nArquivos de imagem em BTXRD/images:', img_count)

# mostrar contagem por label se existirem colunas benigno/maligno
for col in ['benigno','maligno','benign','malignant']:
    if col in df.columns:
        print(f"\nContagem pela coluna {col}:")
        print(df[col].value_counts(dropna=False))

# show unique values of some relevant columns
for col in ['target','class_name','split_group']:
    if col in df.columns:
        print(f"\nValores únicos em {col}: {df[col].unique()[:50]}")

print('\nPronto')
