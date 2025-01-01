import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# text = (
# "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
# "of someunknownPlace."
# )
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)


# text = (
# "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
# )


# text = "AKwirw ier"
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
# for i in integers:
#     print(tokenizer.decode_single_token_bytes(i))
# strings = tokenizer.decode(integers)
# print(strings)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4 # context size tell how many tokens are included into input
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]
# print(f"x: {x}")
# print(f"y: {y}")
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))