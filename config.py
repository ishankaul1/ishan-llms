from dataclasses import dataclass, field


@dataclass(frozen=True)
class DropoutConfig:
    attn_drop: float = 0.1
    shortcut_drop: float = 0.1
    emb_drop: float = 0.1

    @classmethod
    def uniform(cls, rate: float = 0.1) -> "DropoutConfig":
        return cls(attn_drop=rate, shortcut_drop=rate, emb_drop=rate)


@dataclass
class GPTConfig:
    vocab_size: int = 50257  # Vocabulary size (equivalent to BPE tokenizer words)
    context_length: int = 1024  # Context length
    emb_dim: int = 768  # Embedding dimension
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_config: DropoutConfig = field(default_factory=DropoutConfig)
    qkv_bias: bool = False  # Query-Key-Value bias
