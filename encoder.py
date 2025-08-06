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
#Today I will write #1 and #2 and maybe #3 if im feeling quirky

#Self-Attention is a mechanism that allows the model to evaluate each position in the input sequence
#and the relevancy of other positions in the same sequence when computing the representation
#one of the hardest aspects of creating an llm is the self-attention
#Without Training Weights

#simple self_attention without weights
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], #Your  x^1
     [0.55, 0.87, 0.66], #Journey x^2
     [0.57, 0.85, 0.64], #Starts x^3
     [0.22, 0.58, 0.33], #With x^4
     [0.77, 0.25, 0.10], #A x^5
     [0.05, 0.80, 0.55]] #Step x^6
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0]) #this will hold the attention scores
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) #dot product of query and each input vector
#Dot Vectors are not only used to convert two vectors into a scalar, but also to measure the similarity between them
print(attn_scores_2)
#now lets try and normalize these vectors 
attn_weights_2_tmp = attn_scores_2/attn_scores_2.sum() #normalize the attention scores just like in physics
print("Attention Weights", attn_weights_2_tmp)
print("Sum: ", attn_weights_2_tmp.sum())

#try to use softmax approach
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention Weights (Naive Softmax):", attn_weights_2_naive)
print("Sum (Naive Softmax):", attn_weights_2_naive.sum()) #alternative way of normalizing attention sore but it may suffer from
#overflow or underflow



