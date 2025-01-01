import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()] 
all_words = sorted(set(preprocessed)) 
vocab = {token:integer for integer,token in enumerate(all_words)}

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab # vocab is dect containing word to token ID relation
        self.int_to_str = {i:s for s,i in vocab.items()} # just reverse the dictionary
    def encode(self, text): 
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids): 
        text =  " ".join([self.int_to_str[i] for i in ids]) 
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) 
        return text # for array of id's
    def decode_v2(self, id):
        text_v2 = self.int_to_str[id]
        return text_v2 # for single id

tokenizer = SimpleTokenizerV2(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
# print(ids)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

# for i,item in enumerate(list(vocab.items())[-5:]):
#     print(item)
        
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
# print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
nums = tokenizer.encode(text)
print(tokenizer.decode(nums))



