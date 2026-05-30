"""
Example exercise mentioned by Raschka
"""

import tiktoken


from importlib.metadata import version


print(f"Tiktoken version: {version('tiktoken')}")

tokenizer = tiktoken.get_encoding("gpt2")

txt_1 = "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."

ints_1 = tokenizer.encode(txt_1, allowed_special={"<|endoftext|>"})
print(ints_1)


strs_1 = tokenizer.decode(ints_1)
print(strs_1)


### Exercise
print("AKWIRW IER EXERCISE\n" + "=" * 40)

txt = "Akwirw ier"

ints = tokenizer.encode(txt)

mapping = [(i, tokenizer.decode([i])) for i in ints]

print(f"MAPPING: {mapping}")


decoded = tokenizer.decode(ints)

print(f"DECODED: {decoded}")
