import os.path

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import pandas as pd
import torch
import random
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.n = int(os.path.basename(file_path).replace(".csv","").split("_")[-2])
        self.max_length = ((self.n+1)*2+3)
        self.rng = np.random.RandomState(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Randomly choose a starting point
        start_index = self.rng.randint(0, self.n)
        # Match s_i as input and s_{i+1} as the target
        s1 = row[f's{start_index}']
        s2 = row[f's{start_index + 1}']

        input_text = s1 + ' > ' + s2 + self.tokenizer.eos_token
        # Encode sentences with the format [s1, s2]
        encoding = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding='max_length')
        s1_encoded = self.tokenizer.encode(s1)
        #print(s1_encoded)

        # Create labels with the target sequence (s2) masked for autoregressive training
        labels = encoding["input_ids"].copy()
        labels[:len(s1_encoded)] = [-100]*len(s1_encoded)  # Mask `s1` tokens

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(labels)
        }


class EvalDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.n = int(os.path.basename(file_path).replace(".csv","").split("_")[-2])
        self.max_length = ((self.n+1)*2+3)
        self.rng = np.random.RandomState(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        start_index = 0
        input_text = row[f's{start_index}']
        target_text = row[f's{self.n}']

        return {
            'input_text': input_text,
            'target_text': target_text
        }

if __name__ == '__main__':
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    dataset = TrainDataset('data/data_4_0.csv', tokenizer)
    for item in dataset:
        for key, value in item.items():
            if key == 'input_ids': #or key == 'labels':
                raw = tokenizer.decode(value)
            else:
                raw = "0"
            print(f"{key}: {raw} : {value}")  # Adjust processing logic as necessary
