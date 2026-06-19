from this import d
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


class CausalAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        max_ctx_len,
        dropout,
        qkv_bias=False,
    ):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        self.max_ctx_len = max_ctx_len

        self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(self.max_ctx_len, self.max_ctx_len), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input -- assume seq_len * in_dims (at least as dims[-2:])

        x_dim = len(x.shape)

        if x_dim >= 3:
            b, num_tokens, dims = (
                x.shape
            )  # extract num tokens just in case not always same seq len
        else:
            num_tokens, dims = x.shape

        # Project input onto q k v

        # TODO (BONUS)
        # qkv projections are all sequential/not parallel; how would I concat them to parallelize?
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Get attn scores for all
        attn_scores = q @ k.mT  # NOTE - .transpose(1, 2) is the same thing

        # Mask future positions so they don't contribute to the final context vector;
        # Essentially want _none_ of the value vectors after current pos to contribute to the final sum;
        # Zero out any position in weight matrix where post(y) > pos(x) (eg the upper triangular)

        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )  # Trailing underscore runs in-place

        # Scaled dot product attn for weights
        attn_weights = torch.softmax(attn_scores / (self.d_out**0.5), dim=-1)

        # NOTE - should we avoid dropout if in "inference" mode?
        attn_weights = self.dropout(attn_weights)

        # Get final output v
        return attn_weights @ v

    def get_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.w_q.weight.T, self.w_k.weight.T, self.w_v.weight.T)


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, ctx_len, dropout, num_heads, kqv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                # NOTE - Karpathy's impl divides the model dims by num heads & splits them out;
                # this impl seems to multiply d_in by the amount of heads.
                # Is this the reason Raschka chooses d_in separate from d_out? (eg - is he going to make d_in = d_out / n_heads?)
                CausalAttention(d_in, d_out, ctx_len, dropout, kqv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        # Run each head separately on the input, then concatenate on the last dim;
        # EG - we really pull the outputs back together on the final model dimension

        # Also NOTE - this is fully sequential processing
        return torch.cat([head(x) for head in self.heads], dim=-1)


# TODO - Rascka gives you MHA for free; do it yourself without looking!


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, d_out, ctx_len, dropout, num_heads, kqv_bias=False):
        super().__init__()

        # How to get multiple heads?
        # Start with normal w_k, w_q, w_v;
        # Split into num_heads separate mats before the projections
        # Then run normal attn over each; except rolled over separate matrixes in each head

        self.d_in = d_in
        self.d_out = d_out
        self.ctx_len = ctx_len

        self.dropout = dropout
        self.num_heads = num_heads
        self.kqv_bias = kqv_bias

        self.w_q = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=kqv_bias)

        self.head_dims = self.d_out // self.num_heads

        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(d_out, d_out)

        self.register_buffer(
            "mask", torch.triu(torch.ones(self.ctx_len, self.ctx_len), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        # First, need to do projections

        # Projections matrices stay the same, we just want _attention itself_ run on separate heads
        queries = self.w_q(x)
        keys = self.w_k(x)
        values = self.w_v(x)

        # Assume always 3d now
        batch, seq_len, d_in = x.shape

        # Now expand the inner dim into num_heads separate;
        queries = queries.view(batch, seq_len, self.num_heads, , self.head_dims).transpose(1, 2)
        keys = keys.view(batch, seq_len, self.num_heads, self.head_dims).transpose(1, 2)
        values = values.view(batch, seq_len, self.num_heads, self.head_dims).transpose(1, 2) 

        # NOTE - very important to split along the inner dim _first_ for view, _then_ transpose for the attention scores
        # Need to ensure the inner dim is actually the full token tha can be split into heads

        # Now we need to make keys run every pos against every pos; same as last time

        attn_scores = queries @ keys.mT

        # Mask for causal

        mask_bool = self.mask.bool()[:seq_len, :seq_len]

        attn_scores.masked_fill(mask_bool, -torch.inf)

        # Softmax

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # (b x nh x seq x seq)

        # Context vector from values
        ctx_vec = attn_weights @ values
        # (b x nh x seq x head_dim)

        # NOTE - transpose(1, 2) is the same as view(batch, seq_len, self.num_heads, self.head_dims)
        # NOTE - contiguous() is necessary to ensure the tensor is contiguous in memory; transpose() does not guarantee contiguousness; and view() will use contiguous memory to fill.
        # NOTE - view(batch, seq_len, self.d_out) is the same as view(batch, seq_len, self.num_heads * self.head_dims)
        ctx_vec = ctx_vec.transpose(1, 2).contiguous().view(batch, seq_len, self.d_out)

        ctx_vec = self.out_proj(ctx_vec)
        return ctx_vec # (b x seq x d_out)

# TODO as a follow-up - 
# Play around with torch.view / matmuls/contiguous() separately (load bearing for final understanding)

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

    # print(f"Output of v1 3d:\n{sa_v1(inputs_3d)}")
    # print(f"Output of v2 3d:\n{sa_v2(inputs_3d)}")

    causal_attn = CausalAttention(
        d_in=IN_DIM, d_out=OUT_DIM, max_ctx_len=SEQ_LEN, dropout=0.2
    )

    causal_outputs = causal_attn(inputs_3d)

    print(f"Causal Outputs: {causal_outputs.shape}")

    mha = MultiHeadAttentionWrapper(IN_DIM, OUT_DIM, SEQ_LEN, 0.2, 8)

    mha_out = mha(inputs_3d)

    print(f"MHA Output Shape: {mha_out.shape}")
    # Prediction - mha is the same along first 2 dim (batch = 20, seq len = 500), but has OUT_DIM * n_heads = 1600 on output dim

    mha_out_2d = mha(inputs_2d)  # 500 X 1600
    print(f"MHA Out Shape: {mha_out_2d.shape}")


    mha_2 = MultiHeadedAttention(IN_DIM, OUT_DIM, SEQ_LEN, 0.2, 8)
    mha_2_out = mha_2(inputs_3d)
    print(f"MHA 2 Output Shape: {mha_2_out.shape}")

    mha_2_out_3d = mha_2(inputs_3d)
    print(f"MHA 2 Output Shape 3d: {mha_2_out_3d.shape}")
