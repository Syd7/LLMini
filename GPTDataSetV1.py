import torch 
from torch.utils.data import Dataset, DataLoader
class GPTDataSetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride): 
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text) #tokenizes the entire text

        for i in range (0, len(token_ids) - max_length, stride): #2 chunk the book into sequences of max_length
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self): #returns total rows in data set
        return len(self.input_ids)
    
    def __getitem__(self, idx):  #returns a single row from data set
        return self.input_ids[idx], self.target_ids[idx]