import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.vocab = vocab                        
        self.id_vocab = {i: t for t, i in vocab.items()}  

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.vocab else "<|unk|>" for item in preprocessed]
        ids = [self.vocab[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.id_vocab[i] for i in ids])
        text = re.sub(r"\s+([,.:;?_!'\"()])", r"\1", text)
        return text
