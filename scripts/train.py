"""
train.py
- Carrega modelo e tokenizador
- Configura LoRA
- Formata dataset
- Executa fine-tuning
"""
import os
import gc
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def formatar_schema(exemplo_spider, schemas_map):
    db_id = exemplo_spider['db_id']
    if db_id not in schemas_map:
        return f"ERRO: Esquema para o db_id '{db_id}' não encontrado."
    schema_info = schemas_map[db_id]
    table_names = schema_info['table_names_original']
    column_names = schema_info['column_names_original']
    column_types = schema_info['column_types']
    schema_parts = []
    for table_idx, table_name in enumerate(table_names):
        cols_da_tabela = []
        for i in range(len(column_names)):
            if column_names[i][0] == table_idx:
                col_name = column_names[i][1]
                col_type = column_types[i]
                cols_da_tabela.append(f"{col_name} {col_type}")
        schema_parts.append(f"CREATE TABLE {table_name} ({', '.join(cols_da_tabela)})")
    return "\n".join(schema_parts)

def formatar_exemplo_para_treino(exemplo, schemas_map):
    schema = formatar_schema(exemplo, schemas_map)
    system_prompt = "Você é um especialista em SQL. Sua tarefa é gerar uma consulta SQL precisa a partir de uma pergunta em linguagem natural e o esquema do banco de dados fornecido. Não adicione nenhuma explicação, apenas o código SQL."
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n### Esquema do Banco de Dados:\n# {schema}\n### Pergunta:\n# {exemplo['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{exemplo['query']}<|eot_id|>"""
    return {"text": prompt}

def main():
    gc.collect()
    torch.cuda.empty_cache()
    # Carregar dados
    spider_dataset = load_dataset("spider")
    train_spider = spider_dataset['train']
    # Carregar schemas
    import json
    with open('spider_data/tables.json', 'r') as f:
        schemas_data = json.load(f)
    schemas_map = {db['db_id']: db for db in schemas_data}
    # Formatar dataset
    formatted_dataset = train_spider.map(lambda ex: formatar_exemplo_para_treino(ex, schemas_map))
    # Configuração LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # --- Treinamento 1: Configuração V1 ---
    training_arguments_v1 = SFTConfig(
        output_dir="./resultados/llama3-lora-exp1",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        optim="paged_adamw_32bit",
        logging_steps=25,
        save_strategy="epoch",
        fp16=True,
        push_to_hub=False,
        report_to="none",
    )
    # Carregar modelo/tokenizador
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    COMMIT_HASH = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    hf_token = os.environ.get("HF_TOKEN")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=COMMIT_HASH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, COMMIT_HASH, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    # Treinador
    os.environ["WANDB_DISABLED"] = "true"
    print("Iniciando o fine-tuning com LoRA (Configuração V1)...")
    trainer1 = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_arguments_v1,
    )
    trainer1.train()
    trainer1.save_model(training_arguments_v1.output_dir)
    print(f"Adaptador LoRA V1 salvo em: {training_arguments_v1.output_dir}")

    # --- Treinamento 2: Configuração V2 ---
    training_arguments_v2 = SFTConfig(
        output_dir="./resultados/llama3-lora-exp2",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        optim="paged_adamw_32bit",
        logging_steps=25,
        save_strategy="epoch",
        fp16=True,
        push_to_hub=False,
        # early_stopping_callback pode ser adicionado se desejado
    )
    print("Iniciando o fine-tuning com LoRA (Configuração V2)...")
    # Recomenda-se recarregar o modelo base para o segundo experimento
    model_v2 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=COMMIT_HASH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model_v2.config.use_cache = False
    trainer2 = SFTTrainer(
        model=model_v2,
        train_dataset=formatted_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_arguments_v2,
    )
    trainer2.train()
    trainer2.save_model(training_arguments_v2.output_dir)
    print(f"Adaptador LoRA V2 salvo em: {training_arguments_v2.output_dir}")

if __name__ == "__main__":
    main()
