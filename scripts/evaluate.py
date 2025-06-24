"""
evaluate.py
- Avalia o modelo (baseline ou treinado)
- Executa queries e compara resultados
- Gera relatório de acurácia
"""
import os
import pandas as pd
import torch
from tqdm import tqdm
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def verificar_execucao(query_gerada, query_correta, db_path):
    if not isinstance(query_gerada, str) or not query_gerada.strip():
        return False
    conexao = None
    try:
        conexao = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conexao.cursor()
        cursor.execute(query_correta)
        resultado_correto = cursor.fetchall()
        cursor.execute(query_gerada)
        resultado_gerado = cursor.fetchall()
        resultado_gerado_str = [tuple(str(item) for item in row) for row in resultado_gerado]
        resultado_correto_str = [tuple(str(item) for item in row) for row in resultado_correto]
        return sorted(resultado_gerado_str) == sorted(resultado_correto_str)
    except sqlite3.Error:
        return False
    finally:
        if conexao:
            conexao.close()

def main():
    ARQUIVO_DE_RESULTADOS = "llama3_local_baseline_resultados.csv"  # ou "llama3_treinado_v1_baseline.csv"
    try:
        df_resultados = pd.read_csv(ARQUIVO_DE_RESULTADOS)
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{ARQUIVO_DE_RESULTADOS}' não encontrado.")
        return
    sucessos = 0
    total = len(df_resultados)
    resultados_detalhados = []
    print(f"Iniciando a verificação por execução para {total} exemplos...")
    for index, row in tqdm(df_resultados.iterrows(), total=total, desc="Verificando Resultados"):
        status = verificar_execucao(row['query_gerada'], row['query_correta'], row['db_path'])
        if status:
            sucessos += 1
            resultados_detalhados.append('Sucesso')
        else:
            resultados_detalhados.append('Falha')
    df_resultados['status_execucao'] = resultados_detalhados
    df_resultados.to_csv(ARQUIVO_DE_RESULTADOS.replace('.csv', '_verificados.csv'), index=False)
    print("\n--- Relatório de Baseline (Execução Local) ---")
    print(f"Total de Amostras Avaliadas: {total}")
    print(f"Número de Sucessos (Execution Accuracy): {sucessos}")
    print(f"Número de Falhas: {total - sucessos}")
    if total > 0:
        taxa_de_sucesso = (sucessos / total) * 100
        print(f"Taxa de Sucesso Bruta: {taxa_de_sucesso:.2f}%")
    print(f"\nResultados detalhados foram salvos em '{ARQUIVO_DE_RESULTADOS.replace('.csv', '_verificados.csv')}'")

if __name__ == "__main__":
    main()
