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


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print ("Total number of characters:", len(raw_text))
print(raw_text[:99])


#basic Tokenizer
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

#sort all unique tokens
all_words_sorted = sorted(set(preprocessed))
vocab_size = len(preprocessed)
print(vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words_sorted)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|unk|>", "<|endoftext|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))
tokenizer = SimpleTokenizerV1(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|endoftext|> ".join((text1, text2))
print(text)

encoded = tokenizer.encode(text)
print(tokenizer.decode(encoded))

print("tiktoken version" + version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)

#Experimenting with tiktoken's tokenizer and how it works with unknown strings 
TestingBrokenText = "Akwir ier"
integer2 = tokenizer.encode(TestingBrokenText, allowed_special={"<|endoftext|>"})
print(integer2)
string2 = tokenizer.decode(integer2)
print(string2)
