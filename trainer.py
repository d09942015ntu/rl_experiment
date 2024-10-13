from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import TrainerCallback
from mydataset import TrainDataset, EvalDataset
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np

class StringOutputEvaluator(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        device = args.device
        # TODO

def main():
    parser = argparse.ArgumentParser(description='Train a GPT-2 model.')
    parser.add_argument('--model_name', type=str, default='gpt2',  help='Pre-trained model name or path')
    parser.add_argument('--dataset_train', type=str, default='/Users/markchang/code/RLSTaR/data/data_4_0.csv',  help='Path to the training dataset')
    parser.add_argument('--dataset_eval', type=str, default='/Users/markchang/code/RLSTaR/data/data_4_1.csv',  help='Path to the evaluation dataset')
    parser.add_argument('--output_dir', type=str, default='/Users/markchang/code/RLSTaR/results',  help='Path to output directory')

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    dataset_train = TrainDataset(args.dataset_train, tokenizer)
    dataset_eval = EvalDataset(args.dataset_eval, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        save_steps=50,
        save_total_limit=3,
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        tokenizer=tokenizer,
        callbacks=[StringOutputEvaluator(dataset_eval, tokenizer)]
    )

    # Train the model
    trainer.train()
# Initialize Trainer


if __name__ == '__main__':
    main()