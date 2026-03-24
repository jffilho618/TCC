import csv

BASE_DIR = r'c:\Users\jffil\OneDrive\Área de Trabalho\TCC2'
INPUT_CSV = BASE_DIR + r'\analise_trabalhos_relacionados.csv'
OUTPUT_CSV = BASE_DIR + r'\analise_trabalhos_relacionados_paginas.csv'

# Ler CSV completo
with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    all_fieldnames = reader.fieldnames

print(f"✅ Lido: {len(rows)} trabalhos, {len(all_fieldnames)} colunas")

# Definir colunas por PÁGINA
paginas = {
    'IDENTIFICAÇÃO': ['ID', 'Título', 'Autores', 'Ano', 'Tipo_Tarefa', 'Modalidade'],

    'DATASET': ['ID', 'Dataset_Tamanho', 'Dataset_Composição', 'Validação'],

    'ARQUITETURA': ['ID', 'Arquitetura', 'Modelo_Específico', 'Pesos_Pretrain'],

    'TÉCNICAS': ['ID', 'Técnicas_Preprocessing', 'Técnicas_Augmentation', 'Otimizador', 'Loss_Function'],

    'RESULTADOS_DETECÇÃO': ['ID', 'Detecção_Acurácia', 'Detecção_PPV', 'Detecção_NPV',
                             'Detecção_AUC', 'Detecção_Recall', 'Detecção_F1Score'],

    'RESULTADOS_CLASSIFICAÇÃO': ['ID', 'Classificação_Acurácia', 'Classificação_PPV', 'Classificação_NPV',
                                  'Classificação_AUC', 'Classificação_Recall', 'Classificação_F1Score'],

    'ANÁLISE_QUALITATIVA': ['ID', 'Observações_Importantes', 'Limitações', 'Pontos_Fortes', 'Comparação_Com_Meu_TCC'],

    'RESUMO_METODOLÓGICO': ['ID', 'Resumo_Metodológico']
}

# Criar CSV com páginas
with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)

    for pagina_nome, colunas in paginas.items():
        # Escrever separador de página
        writer.writerow([])
        writer.writerow([f'===== PÁGINA: {pagina_nome} ====='])
        writer.writerow([])

        # Escrever header da página
        writer.writerow(colunas)

        # Escrever dados
        for row in rows:
            row_data = [row.get(col, 'N/A') for col in colunas]
            writer.writerow(row_data)

        # Espaçamento entre páginas
        writer.writerow([])

print(f"\n✅ CSV reorganizado criado: analise_trabalhos_relacionados_paginas.csv")
print(f"\n📄 Páginas criadas:")
for i, (pagina_nome, colunas) in enumerate(paginas.items(), 1):
    print(f"   {i}. {pagina_nome} ({len(colunas)-1} colunas + ID)")
print(f"\n💡 Abra o arquivo CSV e navegue pelas seções separadas por '===== PÁGINA: ... =====' ")