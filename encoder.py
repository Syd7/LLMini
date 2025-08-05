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

#test
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
second_batch = next(data_iter)
print(first_batch)
print(second_batch)

#exercise 2.2
#max_length is just like the number of tokens in a sequence, batch_size is the number of sequences in a batch

dataloader2 = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
print("max_length:", 4)
data_iter = iter(dataloader2)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("Targets:\n", targets)

#now we have to do token embeddings, e.g converting token IDs to embedding vectors. 
#input text -> tokenized text -> token IDs -> embeddings
#creating token embeddings

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6 #6 rows
output_dim = 3 #3length

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim) #six ros for six tokens (vocab_size), 3 columns for output_dim
print (embedding_layer.weight)
print(embedding_layer(torch.tensor(3)))  # shows u the 4th row of the embedding matrix, which corresponds to token ID 3 (cuz 0 idex)
print(embedding_layer(torch.tensor([2, 3, 5, 1])))  # shows the embeddings for all tokens in input_ids





