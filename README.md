# Análise Quantitativa do Trade-off entre Especialização e Generalização em LLMs via Fine-Tuning

Este projeto investiga o trade-off entre especialização e generalização em Large Language Models (LLMs) por meio de fine-tuning, utilizando datasets como Spider e MMLU.

## Estrutura do Projeto
- `NLP_TP4_Allan_Beatriz_Juile.ipynb` e `NLP_TP4_Allan_Beatriz_Juile_FINALIZADO.ipynb`: Notebooks principais com todo o pipeline de análise, avaliação e experimentos.
- `custom_metrics/`: Métricas customizadas para avaliação dos modelos.
- `scripts/`: Scripts Python (em desenvolvimento) para modularizar o pipeline do notebook.
- `requirements.txt`: Lista de dependências do projeto.

## Dependências
Instale todas as dependências necessárias com:
```bash
pip install -r requirements.txt
```

## Como Executar
1. **Baixe os dados**: O notebook faz download automático dos dados necessários (Spider, MMLU, etc).
2. **Execute os scripts**: Após a conversão dos notebooks para scripts, utilize os scripts em `scripts/` para rodar as etapas de pré-processamento, avaliação e análise.
3. **Reproduza os experimentos**: Siga as células dos notebooks ou scripts para reproduzir os resultados.

## Objetivo
Avaliar quantitativamente como o fine-tuning em tarefas específicas afeta a capacidade de generalização dos LLMs, utilizando benchmarks padronizados e métricas de execução real de queries SQL.

## Autores
- Allan Carvalho de Aguiar
- Beatriz Emily Silva Aguiar
- Juíle Yoshie Sarkis Hanada

## Orientação
Professores: Altigran Soares e André Carvalho

## Observações
- O projeto utiliza modelos da Hugging Face e requer token de acesso para alguns downloads.
- Scripts e notebooks foram desenvolvidos para rodar em ambientes como Google Colab e Linux local.

Link com os adaptadores Lora V1 e V2:
https://drive.google.com/drive/folders/1xdK1ABNrKQdwjNQ1HXzjb1XQVVBpfnmj?usp=sharing

Arquivos_finais.zip contém os csv com os resultados extraidos do modelo
