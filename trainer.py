from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import TrainerCallback
from mydataset import TrainDataset

from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np

# Custom Dataset


model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

dataset = TrainDataset('data/data_3_0.csv', tokenizer)
dataset2 = TrainDataset('data/data_3_0.csv', tokenizer)
#for item in dataset:
#    for key, value in item.items():
#        print(f"{key}: {value}")  # Adjust processing logic as necessary

model = GPT2LMHeadModel.from_pretrained(model_name)

# Load dataset

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    save_steps=50,
    save_total_limit=3,
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01
)

# Initialize Trainer

class StringOutputEvaluator(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        device = args.device
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Example: Just evaluating the first example in the eval dataset
        input_ids = self.eval_dataset[0]['input_ids'].unsqueeze(0).to(device)
        attention_mask = self.eval_dataset[0]['input_ids'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            predicted_ids = outputs.logits.argmax(dim=-1)
            #predicted_max = outputs.logits.max(dim=2)

            print(f"input:{tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
            print(f"predicted:{tokenizer.decode(predicted_ids[0], skip_special_tokens=True)}")
            print(f"predicted_max:{outputs.logits.sum()}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    callbacks=[StringOutputEvaluator(dataset2, tokenizer)]
)

# Train the model
trainer.train()
