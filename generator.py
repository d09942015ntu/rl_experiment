import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = 'gpt2'  # You can choose 'gpt2-medium', 'gpt2-large', 'gpt2-xl' if needed

checkpoint_path = "/Users/markchang/code/RLSTaR/checkpoints/checkpoint-250"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(checkpoint_path)

# Function to generate a response from the model
def generate_conversation(prompt, max_length=100, num_return_sequences=1):
    # Encode the input prompt
    #input_ids = tokenizer.encode(prompt, return_tensors='pt')


    input_ids = tokenizer.encode("76 85 49 x >", return_tensors='pt')
    # tensor([[31373,   703,   389,   345,    30]], device='mps:0')
    # tensor([[31373,   703,   389,   345,    30]])
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
    return [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in output]

# Example usage
prompt = "Person A: Hi, how are you doing today?\nPerson B:"
responses = generate_conversation(prompt)

for idx, response in enumerate(responses):
    print(f"Conversation {idx + 1}:\n{response}\n")
