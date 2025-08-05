import urllib.request
import re
from importlib.metadata import version
import tiktoken
from SimpleTokenizerV1 import SimpleTokenizerV1

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")

file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)
tokenizer = tiktoken.get_encoding("gpt2")


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print (f"y: {y}")

#for loop

for i in range (1, context_size+1):
    context = enc_sample[:i] #get all of the context up to i so it gets bigger and bigger
    desired = enc_sample[i] #the next element is what we desire, this is what we will use for learning
    print(context, "---->", desired)
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired])) #decode is expecting a list of ids, so if we [desired] it will pass a list of one element
#input-target pairs for LLM Training -^


print(len(enc_text))
