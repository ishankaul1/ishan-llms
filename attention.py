import torch

import torch.nn as nn


class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        self.w_q = nn.Parameter(torch.rand(d_in, d_out))
        self.w_k = nn.Parameter(torch.rand(d_in, d_out))
        self.w_v = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input -- assume seq_len * in_dims (at least as dims[-2:])
        # NOTE/TODO -
        # 1) qkv projections are all sequential/not parallel; how would I concat them to parallelize?
        # 2) This impl breaks in 3d; how would i restructure to be able to fan on batches as well?

        # Project input onto q k v

        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        # print(f"q shape: {q.shape}")
        # print(f"k.T shape: {k.mT.shape}")

        # Get attn scores for all
        attn_scores = q @ k.mT

        # Scaled dot product attn for weights
        attn_weights = torch.softmax(attn_scores / (self.d_out**0.5), dim=-1)

        # Get final output v
        return attn_weights @ v


if __name__ == "__main__":
    torch.manual_seed(123)

    SEQ_LEN = 500
    IN_DIM = 100

    sa_v1 = SelfAttentionV1(d_in=IN_DIM, d_out=IN_DIM)


    # 2D Example
    inputs_2d = torch.rand(SEQ_LEN, IN_DIM)

    outputs_2d = sa_v1(inputs_2d)

    print(f"Outputs 2d shape: {outputs_2d.shape}")

    # 3D Example 
    BATCH_SIZE = 20
    inputs_3d = torch.rand(BATCH_SIZE, SEQ_LEN, IN_DIM)

    outputs_3d = sa_v1(inputs_3d)
    
    print(f"Outputs 3d shape: {outputs_3d.shape}")

