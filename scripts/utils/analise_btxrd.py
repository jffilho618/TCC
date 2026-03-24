import os
import argparse
import pandas as pd
import json
from collections import Counter


def load_data(path=None):
    # Preferir o XLSX completo em BTXRD quando existir
    candidates = []
    if path:
        candidates.append(path)

    # Preferência explícita: o arquivo XLSX dentro de BTXRD é o principal (contém normais + tumores)
    candidates.extend([
        os.path.join('BTXRD', 'dataset.xlsx'),
        os.path.join('BTXRD', 'dataset(total).xlsx'),
        os.path.join('BTXRD', 'dataset_kfold_controle.csv'),
        'dataset_kfold_controle.csv',
    ])

    for p in candidates:
        if not p:
            continue
        if not os.path.exists(p):
            continue
        try:
            if p.lower().endswith('.csv'):
                df = pd.read_csv(p)
            elif p.lower().endswith('.xlsx') or p.lower().endswith('.xls'):
                df = pd.read_excel(p, engine='openpyxl')
            else:
                try:
                    df = pd.read_csv(p)
                except Exception:
                    df = pd.read_excel(p, engine='openpyxl')
            print(f"Carregado: {p} -> {len(df)} linhas")
            return df
        except Exception as e:
            print(f"Falha ao ler {p}: {e}")
    raise FileNotFoundError('Não foi possível encontrar um arquivo de dataset válido nas localizações esperadas.')


def summarize(df, btxrd_root=None, check_jsons=False):
    total = len(df)

    # Tumor / Normal
    tumor_count = None
    if 'tumor' in df.columns:
        tumor_count = int(df['tumor'].astype(int).sum())
        normal_count = total - tumor_count
    else:
        # try infer from class_name (common values: benigno/maligno/normal)
        if 'class_name' in df.columns:
            cn = df['class_name'].str.lower()
            tumor_count = int((cn != 'normal') & (cn != 'normalizado') & (cn != 'nao').sum())
            normal_count = total - tumor_count
        else:
            tumor_count = None
            normal_count = None

    # Benign / Malign
    benign_count = None
    malignant_count = None
    if 'benign' in df.columns or 'benigno' in df.columns:
        # prefer binary columns
        if 'benign' in df.columns:
            benign_count = int(df['benign'].astype(int).sum())
        if 'benigno' in df.columns:
            benign_count = int(df['benigno'].astype(int).sum())
    if 'malignant' in df.columns or 'maligno' in df.columns:
        if 'malignant' in df.columns:
            malignant_count = int(df['malignant'].astype(int).sum())
        if 'maligno' in df.columns:
            malignant_count = int(df['maligno'].astype(int).sum())

    # fallback: count by class_name values
    class_name_counts = None
    if 'class_name' in df.columns:
        class_name_counts = df['class_name'].value_counts(dropna=False).to_dict()
        # attempt to extract benign/malign from class_name
        if benign_count is None and malignant_count is None:
            b = 0
            m = 0
            for k, v in class_name_counts.items():
                kl = str(k).lower()
                if 'benign' in kl or 'benigno' in kl or 'benignos' in kl:
                    b += v
                elif 'malign' in kl or 'maligno' in kl:
                    m += v
            if b + m > 0:
                benign_count = b
                malignant_count = m

    # Gender, Age distribution (if present)
    gender_counts = df['gender'].value_counts(dropna=False).to_dict() if 'gender' in df.columns else None
    age_stats = None
    if 'age' in df.columns:
        try:
            ages = pd.to_numeric(df['age'], errors='coerce')
            age_stats = {
                'count': int(ages.count()),
                'min': float(ages.min()) if not ages.isna().all() else None,
                'max': float(ages.max()) if not ages.isna().all() else None,
                'mean': float(ages.mean()) if not ages.isna().all() else None,
            }
        except Exception:
            age_stats = None

    # split / fold
    split_counts = df['split_group'].value_counts(dropna=False).to_dict() if 'split_group' in df.columns else None
    fold_counts = df['fold'].value_counts(dropna=False).to_dict() if 'fold' in df.columns else None

    # annotations existence check (optional)
    jsons_with_annotation = None
    if check_jsons:
        if btxrd_root is None:
            btxrd_root = 'BTXRD'
        ann_dir = os.path.join(btxrd_root, 'Annotations')
        if os.path.isdir(ann_dir):
            has = 0
            no = 0
            for idx, row in df.iterrows():
                name = str(row.get('image_id') or row.get('image') or row.get('image_name'))
                base = os.path.splitext(name)[0]
                json_path = os.path.join(ann_dir, base + '.json')
                if os.path.exists(json_path):
                    has += 1
                else:
                    no += 1
            jsons_with_annotation = {'with_json': has, 'without_json': no}
        else:
            jsons_with_annotation = {'error': f'annotations dir not found: {ann_dir}'}

    return {
        'total_images': total,
        'tumor_count': tumor_count,
        'normal_count': normal_count,
        'benign_count': benign_count,
        'malignant_count': malignant_count,
        'class_name_counts': class_name_counts,
        'gender_counts': gender_counts,
        'age_stats': age_stats,
        'split_counts': split_counts,
        'fold_counts': fold_counts,
        'jsons_with_annotation': jsons_with_annotation,
    }


def print_summary(s):
    print('\n=== RESUMO EXPLORATÓRIO BTXRD ===\n')
    print(f"Total imagens: {s['total_images']}")
    if s['tumor_count'] is not None:
        print(f"Tumor (doente): {s['tumor_count']} ({s['tumor_count']/s['total_images']:.2%})")
        print(f"Normal: {s['normal_count']} ({s['normal_count']/s['total_images']:.2%})")
    if s['benign_count'] is not None:
        print(f"Benigno: {s['benign_count']} ({s['benign_count']/s['total_images']:.2%})")
    if s['malignant_count'] is not None:
        print(f"Maligno: {s['malignant_count']} ({s['malignant_count']/s['total_images']:.2%})")

    if s['class_name_counts']:
        print('\nContagem por `class_name`:')
        for k, v in s['class_name_counts'].items():
            print(f" - {k}: {v}")

    if s['gender_counts']:
        print('\nDistribuição por gênero:')
        for k, v in s['gender_counts'].items():
            print(f" - {k}: {v}")

    if s['age_stats']:
        print('\nEstatísticas de idade:')
        for k, v in s['age_stats'].items():
            print(f" - {k}: {v}")

    if s['split_counts']:
        print('\nSplit groups:')
        for k, v in s['split_counts'].items():
            print(f" - {k}: {v}")

    if s['fold_counts']:
        print('\nFold distribution:')
        for k, v in s['fold_counts'].items():
            print(f" - {k}: {v}")

    if s['jsons_with_annotation'] is not None:
        print('\nAnotações (JSONs):')
        for k, v in s['jsons_with_annotation'].items():
            print(f" - {k}: {v}")


def save_summary_csv(path, summary):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('key,value\n')
        for k, v in summary.items():
            # serialize nested structures as JSON string
            if isinstance(v, (dict, list)):
                v_str = json.dumps(v, ensure_ascii=False)
            else:
                v_str = str(v)
            f.write(f"{k},{v_str}\n")


def save_summary_json(path, summary):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Análise exploratória do dataset BTXRD')
    parser.add_argument('--path', '-p', help='Caminho para o CSV ou XLSX do dataset')
    parser.add_argument('--check-jsons', action='store_true', help='Verifica existência de JSONs em BTXRD/Annotations')
    parser.add_argument('--btxrd-root', default='BTXRD', help='Caminho raiz do diretório BTXRD (usado para checar JSONs)')
    parser.add_argument('--save-json', help='Caminho para salvar o resumo em JSON')
    parser.add_argument('--save-csv', help='Caminho para salvar o resumo em CSV (key,value)')
    args = parser.parse_args()

    df = load_data(args.path)
    summary = summarize(df, btxrd_root=args.btxrd_root, check_jsons=args.check_jsons)
    print_summary(summary)

    # salvar resultados quando solicitado
    if args.save_json:
        save_summary_json(args.save_json, summary)
        print(f"Resumo salvo em JSON: {args.save_json}")
    if args.save_csv:
        save_summary_csv(args.save_csv, summary)
        print(f"Resumo salvo em CSV: {args.save_csv}")


if __name__ == '__main__':
    main()
