import torch
import torch.nn as nn
from dataclasses import dataclass
from layer_norm import LayerNorm


from config import GPTConfig

from attention import MultiHeadAttention
from feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            ctx_len=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            kqv_bias=cfg.qkv_bias
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(emb_dim=cfg.emb_dim)
        self.norm2 = LayerNorm(emb_dim=cfg.emb_dim)

        self.drop_shortcut = nn.Dropout(cfg.drop_rate)


    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        # NOTE - why does norm after lead to worse training dynamics? Just empirical or a real reaason?
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        # Mapping from tok id -> trained embedding vec
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)

        # 1 vector per context lenght position;
        # QUESTION/NOTE - Seems like an "absolute" impl; would be interesting to use a more
        # sophisticated one like RoPE etc.
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)

        # QUESTION - Why is it an emb?
        # Answered later I think -- it is not; jsut apply dropout to the embeddings themselves
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        # Actual blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)

        # Final one -- take each activation & produce a number per token
        # This represents the models 'output distribution' on what to say next?
        
        # NOTE -- actual original GPT-2 just reuses the token emb layer (weight tying)!
        # But Raschka says it is strictly worse on training performance so we will skip.
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx):
        # NOTE -- assumes already tokenized. Eg B x S token ids
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds

        x = self.drop_emb(x)
        x = self.trf_blocks(x)

        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits


import tiktoken

GPT_CONFIG_124M = GPTConfig()

print("Block")
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)


print("/n/nGPT:")
tokenizer = tiktoken.get_encoding("gpt2")

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# Note -- super brittle; would have to fill to context length probably
# And at this point im pretty sure that's what you need to predict the next token as well yea?
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))


batch = torch.stack(batch, dim=0)
print("batch", batch)

torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)

logits = model(batch)
print("Output shape", logits.shape)
print(logits)

# --- Param Counts ----

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parms: {total_params}")

print("For fun -- Named params!")
for name, param in model.named_parameters():
    print(f"{name:<50} -> {list(param.shape)}")


"""
Exercise 4.1:

"Calculate and compare # parameters in the feed fwd vs multi head attn modules"

class GPTConfig:
    vocab_size: int = 50257  # Vocabulary size (equivalent to BPE tokenizer words)
    context_length: int = 1024  # Context length
    emb_dim: int = 768  # Embedding dimension
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = False  # Query-Key-Value bias

_ per layer_:

ATTN:
3 x emb_dim * emb_dim trainable params
= 3 x 768^2 = 1.769 million

(if you consider out_proj):
    768^2 * 4 -> 2.4m

MLP:
(768 * 768 * 4 * 2) = 4.719 million


Total:
Attn: ~21-22m params; 28 including out_proj
MLP: ~56m params


Now for rest of breakdown estimate:

 = emb_dim * vocab_size;
768 * 50257 = ~40m
* 2 if not reusing tok_emb; 80m;

Adds up to ~120-160; just as Raschka said!!

"""

# --- Memory --- 

total_size_bytes = total_params * 4 # assume fp32
total_size_mb = total_size_bytes / (1024 **2)

print(f"Total size of the model: {total_size_mb:.2f} MB")


# CONTINUE: Ex 4.2 Bigger model & attribution
# And -- 4.7 generation :) 