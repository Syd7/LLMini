import urllib.request
import re
import torch
from GPTDataSetV1 import GPTDataSetV1
from SelfAttention_V1 import SelfAttention_v1
from SelfAttention_V2 import SelfAttention_v2
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
attn_scores_2 = torch.empty(inputs.shape[0]) #this will hold the attention scores since its 0 it'll just be a row
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) #dot product of query and each input vector
#Dot Vectors are not only used to convert two vectors into a scalar, but also to measure the similarity between them
print("HIIII", attn_scores_2)
#now lets try and normalize these vectors 
res = 0 
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]

print(res)
print(torch.dot(inputs[0], query)) #this is the same as the above line


attn_weights_2_tmp = attn_scores_2/attn_scores_2.sum() #normalize the attention scores just like in physics
#print("Attention Weights", attn_weights_2_tmp)
#print("Sum: ", attn_weights_2_tmp.sum())

# GOOD SO FAR ^^^

#try to use softmax approach
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2 = softmax_naive(attn_scores_2)
#print("Attention Weights Naive:", attn_weights_2)
#print("Sum: ", attn_weights_2.sum())

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i #multiply each input vector by its corresponding attention weight
#print(context_vec_2) #GOOD


attn_scores = inputs @ inputs.T #this is the dot product loop but done with transposition and matrix multiplication 
#print(attn_scores) #GOOD

#normalize each row
attn_weights = torch.softmax(attn_scores, dim=1) #softmax is a function that normalizes the attention scores
#print(attn_weights) #GOOD


#use these attention weights to compute context vector via matrix mutliplication
all_context_vecs = attn_weights @ inputs #multiply the attention weights by the inputs to get the context vector
print(all_context_vecs)
print("Previous 2nd Context Vector:", context_vec_2)
#implementing self attention with weights

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query #
key_2 = x_2 @ W_key #key vector
value_2 = x_2 @ W_value #value vector
#print("QUERY 2", query_2) #good na

keys = inputs @ W_key #key vectors
values = inputs @ W_value #value vectors
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

keys_2 = keys[1]
attn_scores_22 = query_2.dot(keys_2)
print(attn_scores_22)

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) #scaled dot product attention
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values
print("Context Vector:", context_vec_2)

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

#now we want to hide future words with causal attention
#we want it to only consider words before the current position
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length)) #lower triangular matrix
print(mask_simple)

masked_simple = attn_weights * mask_simple #apply the mask to the attention scores (also recall that matrix multiplication is NOT commutative)
print(masked_simple)

#normalize the masked attention weights
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

#we can implement this more efficient masking trick with creating a mask of 1s above the diagonal and then replacing it with negative infinity.
mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf) #replace the upper triangular part with -inf
print(masked)

#apply softmax to these masked values
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1) #scaled dot product attention
print(attn_weights)

#Mask the additional weights with dropout
#Technique of dropping randomly selected neurons during training to prevent overfitting
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) #dropout with 50% probability (50% probability of dropping it to zero)
example = torch.ones(6, 6)
print(dropout(example))

torch.manual_seed(123)
print(dropout(attn_weights))