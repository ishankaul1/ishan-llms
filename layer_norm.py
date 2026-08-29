import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        # Prevent div by 0 error on off chance
        self.eps = 1e-5 

        # Trainable params to allow model to scale & shift the outputs at this layer itself
        # NOTE -- these operate on each value of the dim _separately_
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

        # Threads to pull -- 
        # 1) Mathematically, why does setting mean 0 variance 1 remove the exploding/vanishing gradient problem?
        # 2) Does allowing the model to scale/shift now reintroduce the possibility of vanishing gradient/shifted values?
        # How/why did they decide they needed to do that?


        # NOTE -- They mention batch normalization as a counterpart that wasn't chosen here
        # due to layernorm having higher flexibility & less computational effort
        # Flexibility makes sense; I also wonder if batch norm causes weird/unstable behavior at inference time
        # They mention distributed training as well -- I guess makes sense because it would cut the amount of comms
        # Needed to compute the final mean/variance? Eg an all reduce or similar? I should map out what tha might look like

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


if __name__ == "__main__":
    # Example
    torch.manual_seed(123)

    batch_example = torch.randn(2, 5)
    # NOTE -- RELU just sets all negative inputs to 0.
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)

    # NOTE -- an insight is that the output of ReLU by nature almost never has a mean 0 / 
    # is always imbalanced positive since you 0 all negatives

    print(out)

    # NOTE - keepdim jsut keeps it as 2 x 1 instead of just making it 2,
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)

    print("Mean:\n", mean)
    print("Variance:\n", var)

    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)


    print("Normalized layer outputs:\n", out_norm)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    ln = LayerNorm(emb_dim = 6)
    out_ln = ln(out)

    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)


    # Next -- Implement feed forward with GELU
