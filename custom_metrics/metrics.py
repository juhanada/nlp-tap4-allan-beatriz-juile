"""
metrics.py
Funções para métricas customizadas de avaliação de queries SQL geradas por LLMs.
Inclui:
- Comparação textual (exata)
- Comparação semântica por execução
- Cálculo de Execution Accuracy para um DataFrame de resultados
"""
import sqlite3
import pandas as pd

def exact_match(query_generated, query_gold):
    """Retorna True se as queries forem exatamente iguais (ignorando espaços em branco)."""
    return query_generated.strip() == query_gold.strip()

def execution_match(query_generated, query_gold, db_path):
    """Compara o resultado da execução das duas queries no banco de dados."""
    if not isinstance(query_generated, str) or not query_generated.strip():
        return False
    conexao = None
    try:
        conexao = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conexao.cursor()
        cursor.execute(query_gold)
        result_gold = cursor.fetchall()
        cursor.execute(query_generated)
        result_gen = cursor.fetchall()
        result_gold_str = [tuple(str(item) for item in row) for row in result_gold]
        result_gen_str = [tuple(str(item) for item in row) for row in result_gen]
        return sorted(result_gold_str) == sorted(result_gen_str)
    except sqlite3.Error:
        return False
    finally:
        if conexao:
            conexao.close()

def execution_accuracy(df, query_col='query_gerada', gold_col='query_correta', db_col='db_path'):
    """
    Calcula a Execution Accuracy para um DataFrame de resultados.
    Retorna a taxa de sucesso (float entre 0 e 1) e adiciona uma coluna 'status_execucao'.
    """
    successes = 0
    status_list = []
    for _, row in df.iterrows():
        status = execution_match(row[query_col], row[gold_col], row[db_col])
        if status:
            successes += 1
            status_list.append('Sucesso')
        else:
            status_list.append('Falha')
    df['status_execucao'] = status_list
    accuracy = successes / len(df) if len(df) > 0 else 0.0
    return accuracy
