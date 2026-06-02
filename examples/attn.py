"""
Attention examples; probably the most important secsion of the whole book.
"""

from sympy.polys.specialpolys import _w_2
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
    # The tldr for why - large embed dims w/ large dot products often result in 
    # very small gradients during backprop once softmaxed;
    # Small gradients result in slow/stagnated learning. 
    # Scaling down the values before softmax helps mitigate

    # TODO - how did they learn to scale it by sqrt of embed dim though? Vs just embed dim or 
    # some other derivative value of it?
    attn_weights_2 = torch.softmax(attn_scores_2 / D_OUT**0.5, dim=-1)
    print(f"attn_weights_2:\n {attn_weights_2}")

    ctx_vec_2 = attn_weights_2 @ vals
    print(f"ctx_vec_2: {ctx_vec_2}")


_run_trainable_weights(inputs)
