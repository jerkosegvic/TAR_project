try:
    from dataset import MyDataset
except ImportError:
    pass

from typing import List
import os
import torch
import pandas as pd
from torch.utils.data import random_split

english_speaking_countries = ['gb']

#DATA_DIR = "/kaggle/input/data-all"
DATA_DIR = "data/orientation"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-cased"
OUTPUT_DIR = "data/torch/orientation"

def load_data(file_path: str):
    '''
    Loads specified dataset and returns lists of columns
    '''
    country_code = file_path.split('-')[1]  # Extract country code from filename
    try:
        df = pd.read_csv(file_path, delimiter='\t', quoting=3, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()
    
    if country_code in english_speaking_countries:
        df['text_combined'] = df['text']
    else:
        df['text_combined'] = df['text_en']
    # Drop rows where 'text_combined' is NaN or empty
    df = df.dropna(subset=['text_combined'])
    df = df[df['text_combined'] != '']
    df['file_path'] = file_path  # Add file path information
    return list(df['id']), list(df['speaker']), list(df['sex']), list(df['text']), list(df['text_combined']), list(df['label'])

def train_val_test_split_country(data: MyDataset, val_size:float = 0.1, test_size:float = 0.1, random_state:int = 42):
    train_size = 1 - test_size - val_size
    train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size], \
                                                   generator=torch.Generator().manual_seed(random_state))
    return train_data, val_data, test_data


train_dataset, val_dataset, test_dataset = [], [], []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".tsv"):
        file_path = os.path.join(DATA_DIR, filename)
        ids, speakers, sexes, texts, texts_en, labels = load_data(file_path)
        df = MyDataset(
            ids=ids,
            speakers=speakers,
            sexes=sexes,
            texts=texts,
            texts_en=texts_en,
            labels=labels,
            device=DEVICE,
            model_name=MODEL_NAME
        )
        train_df, val_df, test_df = train_val_test_split_country(df)
        train_dataset.append(train_df)
        val_dataset.append(val_df)
        test_dataset.append(test_df)
        torch.save(train_df, os.path.join(OUTPUT_DIR, f"train_dataset_{filename.replace('-train.tsv', '.pt')}"))
        torch.save(val_df, os.path.join(OUTPUT_DIR, f"val_dataset_{filename.replace('-train.tsv', '.pt')}"))
        torch.save(test_df, os.path.join(OUTPUT_DIR, f"test_dataset_{filename.replace('-train.tsv', '.pt')}"))
        breakpoint()
        print(f"Processed {filename}, created train, val, and test datasets of size {len(train_df)}, {len(val_df)}, and {len(test_df)} respectively.")

train_dataset = torch.utils.data.ConcatDataset(train_dataset)
val_dataset = torch.utils.data.ConcatDataset(val_dataset)
test_dataset = torch.utils.data.ConcatDataset(test_dataset)

torch.save(train_dataset, os.path.join(OUTPUT_DIR, "train_dataset_all.pt"))
torch.save(val_dataset, os.path.join(OUTPUT_DIR, "val_dataset_all.pt"))
torch.save(test_dataset, os.path.join(OUTPUT_DIR, "test_dataset_all.pt"))

print(f"Processed all files, created train, val, and test datasets of size {len(train_dataset)}, {len(val_dataset)}, and {len(test_dataset)} respectively.")