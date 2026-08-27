import torch
from torch import nn

from config import GPTConfig

# NOTE - GELU approximates x * phi(x); where phi is CDF of standard Gaussian
# TODO - would be good to understand the math here & why it's better than ReLU
# As well as try out swiglu

# Computationally cheaper approximation was apparently found during curve-fitting


# Shape is essentially a smoother version of ReLU, and with small non-zero values
# and gradient for almost all negative values; EG negatives can still contribute
# to training
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )

# NOTE --
# 1) Pretty intuitive, but projection onto 4 x larger space, gelu, then contract back
# allows the model to explore a much richer representation space for each tok
# 2) same in/out dim makes for easier scaling. Make sense since shard math is the same for
# all layers

class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)