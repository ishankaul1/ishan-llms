import tiktoken

from utils.constants import THE_VERDICT_URL
from utils.fetch_text import fetch_text

txt = fetch_text(THE_VERDICT_URL)

tokenizer = tiktoken.get_encoding("gpt2")


enc_txt = tokenizer.encode(txt)


print(f"num tokens: {len(enc_txt)}")

enc_sample = enc_txt[:50]


"""
Scratch dataset example.

We're basically grabbing ctx window size, then target is that shifted +1.
We get 1 pc of data per ctx window size per sliding example.

Eg -- this is just the first sample of 4 out of the 50 sample; we could grab 46 more.
"""

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]

print(f"x: {x}")
print(f"y:      {y}")


for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"{tokenizer.decode(context)} -----> {tokenizer.decode([desired])}")
