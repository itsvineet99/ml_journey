import re

# text = "Hello, world. This, is a test."
# result = re.split(r'([,.]|\s)', text)
# print(result)

# result = [item for item in result if item.strip()]
# print(result)

# text = "Hello, world. Is this-- a test?"
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# result = [item.strip() for item in result if item.strip()]
# print(result)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    #print("Total number of character:", len(raw_text))
    #print(raw_text[:99])
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()] # removes empty srings and white spaces
# print(len(preprocessed))
# print(preprocessed[:30])
all_words = sorted(set(preprocessed)) # set removes duplicate words
# vocab_size = len(all_words)
# print(vocab_size)
vocab = {token:integer for integer,token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab # vocab is dect containing word to token ID relation
        self.int_to_str = {i:s for s,i in vocab.items()} # just reverse the dictionary
    def encode(self, text): 
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids): 
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) 
        return text

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
        