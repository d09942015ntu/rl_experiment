from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from mydataset import EvalDataset
from torch.utils.data import DataLoader

import argparse

from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np

# Custom Dataset

def repr_to_int(x):
    try:
        x = int(''.join(filter(str.isdigit, x)))
    except:
        x = 0
    return x

def generate_one_cycle(batch_text, model, tokenizer, max_length):
    num_return_sequences = 1
    #batch_text = np.array(batch['input_text']).transpose()
    batch_size = len(batch_text)
    batch_text = [x+" >" for x in batch_text]
    batch_text = [x.split(" ") for x in batch_text]
    max_input_length = max([len(x) for x in batch_text])
    batch_text = [[tokenizer.pad_token]*(max_input_length-len(x))+x for x in batch_text]
    batch_text = np.concatenate(batch_text)
    input_ids = tokenizer.encode(batch_text.tolist(), return_tensors='pt')
    #print(len(input_ids.flatten()))
    input_ids = input_ids.reshape(batch_size, int(len(input_ids.flatten())/batch_size))
    #print(input_ids)
    # Generate text continuation
    with torch.no_grad():
        #input_ids = tokenizer.encode(batch_text[i] + " >", return_tensors='pt')
        output = model.generate(
            input_ids,
            max_length=max_length*2+1,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            top_p=0.95,
            temperature=0.7
        )
        output_decoded = [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in output]
        output_decoded = [x.split(">")[1].split("<")[0].strip() for x in output_decoded]
        #print(output_decoded)
        return output_decoded

    # Example usage


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    incorrect = 0
    with torch.no_grad():
        for batch in dataloader:
            states = [batch['input_text']]
            for _ in range(dataloader.dataset.n):
                si = generate_one_cycle(states[-1], model, dataloader.dataset.tokenizer, dataloader.dataset.max_length)
                states.append(si)
            predicted = np.array([repr_to_int(x) for x in states[-1]])
            ground_truth = ([repr_to_int(x) for x in batch['target_text']])
            correct+=np.sum(ground_truth == predicted)
            incorrect+=np.sum(ground_truth != predicted)
            print(f"predicted:{predicted}")
            print(f"ground_truth:{ground_truth}")
            print(f"accuracy:{100*correct/(correct+incorrect):.3f}%")
    return correct/(correct+incorrect)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with checkpoint and dataset paths.')
    parser.add_argument('--ckpt_path', type=str, default="/Users/markchang/code/RLSTaR/checkpoints/checkpoint-7815",
                        help='Path to the checkpoint file.')
    parser.add_argument('--dataset_path', type=str, default='/Users/markchang/code/RLSTaR/data/data_4_0.csv',
                        help='Path to the dataset file.')

    args = parser.parse_args()

    # Access the parsed arguments
    checkpoint_path = args.ckpt_path
    dataset_path = args.dataset_path

    # Assuming EvalDataset is defined elsewhere in your code

    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    dataset = EvalDataset(dataset_path, tokenizer)

    # Load the dataset and create the dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False) # Multiple Batch Size requires paddings

    # Evaluate the model
    evaluate(model, dataloader)


if __name__ == '__main__':
    main()
