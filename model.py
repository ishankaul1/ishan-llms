import torch
import torch.nn as nn
from dataclasses import dataclass

from config import GPTConfig



GPT_CONFIG_124M = GPTConfig()


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class DummyGPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__ ()

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
            *[DummyTransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = DummyLayerNorm(cfg.emb_dim)

        # Final one -- take each activation & produce a number per token
        # This represents the models 'output distribution' on what to say next?
        self.out_head = nn.Linear(
            cfg.emb_dim, cfg.vocab_size, bias=False
        )

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

model = DummyGPTModel(GPT_CONFIG_124M)

logits = model(batch)
print("Output shape", logits.shape)
print(logits)

# NOTE -- ouptut shape is [2, 4, vocab]
# Does not predict "next token" in this case -- just literally gives you a number for each position 
# for each batch

# Next -- 4.2 Normalizing Activations with LayerNorm