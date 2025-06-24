"""
preprocessing.py
- Baixa e prepara os dados Spider e MMLU
- Cria a suíte de avaliação composta
"""
import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets, DatasetDict

def download_and_prepare_data():
    if not os.path.exists("spider_data"):
        import gdown
        gdown.download("1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J", output="spider_data.zip", quiet=False)
        os.system("unzip -q spider_data.zip")
        os.remove("spider_data.zip")

    spider_dataset = load_dataset("spider")
    mmlu_dataset = load_dataset("cais/mmlu", "all")
    return spider_dataset, mmlu_dataset

def create_evaluation_suite():
    NUM_QUESTIONS_PER_CATEGORY = 50
    CATEGORIES = {
        "stem": "college_computer_science",
        "humanities": "philosophy",
        "social_sciences": "econometrics"
    }
    selected_datasets = []
    for category, mmlu_subject in CATEGORIES.items():
        current_subject_dataset_dict = load_dataset("cais/mmlu", mmlu_subject)
        dataset = current_subject_dataset_dict['test']
        shuffled_dataset = dataset.shuffle(seed=42).select(range(NUM_QUESTIONS_PER_CATEGORY))
        selected_datasets.append(shuffled_dataset)
    final_evaluation_dataset = concatenate_datasets(selected_datasets).shuffle(seed=42)
    evaluation_suite = DatasetDict({"evaluation": final_evaluation_dataset})
    df = evaluation_suite["evaluation"].to_pandas()
    print(df['subject'].value_counts())
    return evaluation_suite

if __name__ == "__main__":
    spider_dataset, mmlu_dataset = download_and_prepare_data()
    evaluation_suite = create_evaluation_suite()
    print("Dados prontos para uso.")
