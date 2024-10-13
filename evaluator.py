from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from mydataset import EvalDataset
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np

# Custom Dataset
def generate(model, tokenizer, max_length):


    num_return_sequences=10
    input_ids = tokenizer.encode("76 85 49 x >", return_tensors='pt')
    print(input_ids)
    # Generate text continuation
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            top_p=0.95,
            temperature=0.7
        )
    # Decode and return the generated text
    print([tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in output])

    # Example usage
    prompt = "Person A: Hi, how are you doing today?\nPerson B:"


def evaluate(model, tokenizer, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            print(1)
            #input_ids = batch['input_ids'].squeeze().to(device)


            #attention_mask = batch['attention_mask'].squeeze().to(device)

            #outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            #predicted_ids = outputs.logits.argmax(dim=-1)
            #print(1)
            #for i in range(input_ids.shape[0]):
            #    print(f"input:{tokenizer.decode(input_ids[i], skip_special_tokens=True)}")
            #    print(f"predicted:{tokenizer.decode(predicted_ids[i], skip_special_tokens=True)}")

            #loss = outputs.loss
            #print(f"loss: {loss.item()}")
            #total_loss += loss.item()



def main():

    #checkpoint_path = "/Users/markchang/code/RLSTaR/checkpoints/checkpoint-220"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    #model0 = GPT2LMHeadModel.from_pretrained("/Users/markchang/code/RLSTaR/checkpoints/checkpoint-20")
    model = GPT2LMHeadModel.from_pretrained("/Users/markchang/code/RLSTaR/checkpoints/checkpoint-7815")

    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    dataset = EvalDataset('data/data_3_0.csv', tokenizer)

    # Load the dataset and create the dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Evaluate the model
    evaluate(model, tokenizer, dataloader, device)


if __name__ == '__main__':
    main()
