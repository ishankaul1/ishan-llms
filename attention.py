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
        # TODO -
        # 1) qkv projections are all sequential/not parallel; how would I concat them to parallelize?

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

    def set_weights(
        self, new_q: torch.Tensor, new_k: torch.Tensor, new_v: torch.Tensor
    ):
        self.w_q.data.copy_(new_q)
        self.w_k.data.copy_(new_k)
        self.w_v.data.copy_(new_v)


class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input -- assume seq_len * in_dims (at least as dims[-2:])
        # TODO -
        # 1) qkv projections are all sequential/not parallel; how would I concat them to parallelize?

        # Project input onto q k v

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # print(f"q shape: {q.shape}")
        # print(f"k.T shape: {k.mT.shape}")

        # Get attn scores for all
        attn_scores = q @ k.mT

        # Scaled dot product attn for weights
        attn_weights = torch.softmax(attn_scores / (self.d_out**0.5), dim=-1)

        # Get final output v
        return attn_weights @ v

    def get_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.w_q.weight.T, self.w_k.weight.T, self.w_v.weight.T)


if __name__ == "__main__":
    torch.manual_seed(123)

    SEQ_LEN = 500
    IN_DIM = 100
    OUT_DIM = 200

    sa_v2 = SelfAttentionV2(d_in=IN_DIM, d_out=OUT_DIM)

    # 2D Example
    inputs_2d = torch.rand(SEQ_LEN, IN_DIM)

    outputs_2d = sa_v2(inputs_2d)

    print(f"Outputs 2d shape: {outputs_2d.shape}")

    # 3D Example
    BATCH_SIZE = 20
    inputs_3d = torch.rand(BATCH_SIZE, SEQ_LEN, IN_DIM)

    outputs_3d = sa_v2(inputs_3d)

    print(f"Outputs 3d shape: {outputs_3d.shape}")

    # Ex 3.1 - Weight transfer

    sa_v1 = SelfAttentionV1(d_in=IN_DIM, d_out=OUT_DIM)

    v2_weights = sa_v2.get_weights()
    print(f"Weight shapes: {[w.shape for w in v2_weights]}")

    sa_v1.set_weights(*v2_weights)

    print(f"Output of v1 3d:\n{sa_v1(inputs_3d)}")
    print(f"Output of v2 3d:\n{sa_v2(inputs_3d)}")
