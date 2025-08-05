import urllib.request
import re
import torch
from GPTDataSetV1 import GPTDataSetV1
from torch.utils.data import DataLoader
from importlib.metadata import version
import tiktoken

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")

file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)
tokenizer = tiktoken.get_encoding("gpt2")


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

def create_dataloader_v1(txt, batch_size=4, max_length = 256, stride = 128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2") 
    dataset = GPTDataSetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, #drops the last batch if it is smaller than batch_size
        num_workers=num_workers #num_workers is the number of subprocesses to use for data loading, 0 means that the data will be loaded in the main process
    )
    return dataloader

vocab_size = 50257 #tokens of gpt2 vocab
output_dim = 256 #embedding dimension 
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) #create token embedding layer

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print ("Token IDs:\n", inputs)
print ("\nInputs Shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs) #takes your batch of inputs and looks up their corresponding embeddings
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
post_embeddings = pos_embedding_layer(torch.arange(context_length))
print(post_embeddings.shape)


input_embeddings = token_embeddings + post_embeddings
print(input_embeddings.shape)

#3 Coding Attention Mechanisms
#1 Simplified Self-Attention #2 Self-Attention #3 Causal Attention #4 Multi-Head Attention

