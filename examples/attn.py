"""
Attention examples; probably the most important secsion of the whole book.
"""

import torch

CTX_LEN = 6
NUM_DIMS = 3

inputs = torch.rand(CTX_LEN, NUM_DIMS)
# 6 tokens embed dim 3
print(f"INPUTS:\n{inputs}")


"""
Part 1: Compute the "context vector" for a single token position
"""


def _run_single(inputs: torch.Tensor):
    # Example -- calculate context vec at position 2

    q = inputs[1]

    attn_scores = torch.empty(inputs.shape[0])

    for i, x_i in enumerate(inputs):
        attn_scores[i] = torch.dot(x_i, q)

    # Vec -- how relevant each position is to this one.
    # We are simulating softmax q @ k here
    print(f"Raw attn scores for this pos: {attn_scores}")

    attn_weights = torch.softmax(attn_scores, dim=-1)

    print(f"Attn weights softmaxed: {attn_weights}")
    print(f"Sum: {attn_weights.sum()}")

    # final step --> "context" vector at position x is the sum of all value vectores
    # for each position, multiplied by their attn scalar for that position

    q = inputs[1]
    context_vec = torch.zeros(q.shape)

    for i, x_i in enumerate(inputs):
        context_vec += attn_weights[i] * x_i

    print(f"Context vec: {context_vec}")


def _run_multi(inputs: torch.Tensor):
    # Now we need attn scores between _all_ positions; eg so that the weighted sum
    # @ each position for the final ctx vector can be calculated

    attn_scores = torch.empty(CTX_LEN, CTX_LEN)

    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i][j] = torch.dot(x_i, x_j)
        # NOTE - this does 2x as much computation as necessary!

    print(f"All Attn Scores:\n {attn_scores}")

    """
    NOTE - why transpose?

    Our goal here is to get every vector's dot product with every other vector.
    Any matmul lines up rows if A with col's of b;
    If our goal is to get row i's dot with column j at i, j, we've got to transpose input in the second
    matrix.
    """

    attn_scores = inputs @ inputs.T

    print(f"All Attn Scores w/ matmul:\n {attn_scores}")

    # Softmax them against the innermost dim
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(f"Attn Weights:\n {attn_weights}")

    # Confirm they all sum to 1
    print(f"All row sums:\n {torch.sum(attn_weights, dim=-1)}")

    """
    NOTE - what are we actually doing here?

    Attn weights - 6 token pos x 6 attn weights (one for each pos)

    Inputs - 6 token pos x 3 values in output dim
        - Each row is a token pos, each col is an output dim
    
    At i, j I want the output dim j of token i; eg each row of i is just its context vector
    You get that with essentially the weighted sum of each output dim of each input token
    multiplied by its attn weight, hence why the weighted sum "just works" without the
    transpose.
    
    Along the inner dim - the 6 attn weights (one per tok) y vec in mat A line up to splay across each  
    of the 6 token pos in vec B to compute one weighted score per dim per token.
    

    """
    all_ctx_vecs = attn_weights @ inputs
    print(f"All ctx vecs:\n {all_ctx_vecs}")


def _run_trainable_weights(inputs: torch.Tensor):
    D_IN = inputs.shape[1]
    D_OUT = 2  # Just for learning purposes

    # Project -> output dim along the D_In dimension
    # NOTE - matmul for (m x n) (n x o) matrix does
    # m x n x o multiplications; produces m x o output vector, where
    #
    w_q = torch.nn.Parameter(torch.rand(D_IN, D_OUT), requires_grad=False)
    w_k = torch.nn.Parameter(torch.rand(D_IN, D_OUT), requires_grad=False)
    w_v = torch.nn.Parameter(torch.rand(D_IN, D_OUT), requires_grad=False)

    # Target - context vector @ pos 2

    x_2 = inputs[1]  # vector size D_IN

    q_2 = x_2 @ w_q
    # k_2 = x_2 @ w_k
    # v_2 = x_2 @ w_v

    print(f"w_q:\n {w_q}")
    print(f"x_1:\n {x_2}")

    # NOTE -> we just projected into 1 x 2 vectory of output_dim size
    print(f"q_2:\n {q_2}")

    keys = inputs @ w_k
    vals = inputs @ w_v

    print(f"keys.shape: {keys.shape}")
    print(f"vals.shape: {vals.shape}")

    # Dot the query @ 2 with each key vector for each token (need to splay them so token pos runs
    # column-wise)
    attn_scores_2 = q_2 @ keys.T
    print(f"attn_scores_2:\n{attn_scores_2}")

    # NOTE -- Raschka kind of just throws in the sqrt but explains it later.
    # TLDR - large embed dims w/ large dot products often result in 
    # very small gradients during backprop once softmaxed. 

    # The reason being -- large dims -> large attn scores, then softmax causes outputs to peak near one-hot.
    # Backprop over softmax for near-one hot tends to shift both the low and high values barely.

    # NOTE - Why sqrt(d)?
    # TLDR - Dot product std scales ~ sqrt(d), so divide scores by sqrt(d) to keep softmax inputs O(1)
    attn_weights_2 = torch.softmax(attn_scores_2 / D_OUT**0.5, dim=-1)
    print(f"attn_weights_2:\n {attn_weights_2}")

    ctx_vec_2 = attn_weights_2 @ vals
    print(f"ctx_vec_2: {ctx_vec_2}")


def _softmax_grad_norm(scores: torch.Tensor, values: torch.Tensor) -> float:
    """
    Use a fake values tensor shape(num_keys) to mimic gradient flowing back through attention via
    weighted sum.

    """
    scores = scores.detach().clone().requires_grad_(True)
    weights = torch.softmax(scores, dim=-1)
    context = weights @ values
    context.backward()
    return scores.grad.norm().item()


def _repro_softmax_scale(num_keys: int = 8, seed: int = 42):
    """
    Need a *vector* of scores (one query vs many keys). Softmax on a scalar is
    always [1.0] with zero grad — not useful for this demo.
    """
    torch.manual_seed(seed)
    d_iter = range(64, 1025, 64)

    print(f"{'D':>6}  {'max_unscaled':>12}  {'max_scaled':>12}  {'grad_unscaled':>14}  {'grad_scaled':>14}")
    print("-" * 66)

    values = torch.randn(num_keys)  # fixed "value" vector for all D; (num_keys)

    for d in d_iter:
        q = torch.randn(d)  # one query
        k = torch.randn(num_keys, d)  # many keys → vector of dot products

        scores = q @ k.T  # shape [num_keys] (1, num_keys)

        weights_unscaled = torch.softmax(scores, dim=-1)
        weights_scaled = torch.softmax(scores / d**0.5, dim=-1)

        grad_unscaled = _softmax_grad_norm(scores, values)
        grad_scaled = _softmax_grad_norm(scores / d**0.5, values)

        print(
            f"{d:>6}  {weights_unscaled.max().item():>12.4f}  "
            f"{weights_scaled.max().item():>12.4f}  "
            f"{grad_unscaled:>14.6f}  {grad_scaled:>14.6f}"
        )




_run_trainable_weights(inputs)

if __name__ == "__main__":
    print("\n--- softmax scale repro ---")
    _repro_softmax_scale()
