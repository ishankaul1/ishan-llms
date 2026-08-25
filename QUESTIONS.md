# Questions to dig into

Personal backlog from working through *LLMs from Scratch* (Raschka). Each item links back to the original note.

Ranked for **engineer intuition** — enough to debug a training run, read a modern impl, and explain a design choice — not research-depth. Some of Raschka’s “this works well” claims do not have a theorem waiting at the bottom; stop when you can tell the story and move on.

---

## Do these

Work top to bottom. 1–4 are one story (how a deep net actually trains). 5–6 are attention. 7 is the cost model.

### 1. Why do gradients vanish in deep nets? (the math)

[examples/nn_residual.py](examples/nn_residual.py#L21)

The residual example makes it *observable*: without shortcuts, earlier layers get tiny gradient means. The "why" is still open.

**To learn**

- Chain rule through many layers: how Jacobian singular values / activation derivatives multiply into something ≈ 0.
- Why GELU (and ReLU) make this worse than a linear stack would.
- What "exploding" is as the other side of the same product.

### 2. Why LayerNorm *and* residuals? Aren't they solving the same thing?

[examples/nn_residual.py](examples/nn_residual.py#L80)

This is the one to get right. LN does **not** "remove" exploding/vanishing by itself.

**To learn**

- Residual: identity path so gradient can skip the block (`∂L/∂x += ∂L/∂F`). That's the highway.
- LayerNorm: re-center/re-scale features so nonlinearities don't saturate, so the residual *plus* the block stay well-behaved.
- Why transformers need both, and what Pre-LN vs Post-LN changes (add this when you hit §4.6).

### 3. Why mean 0 / var 1 — and don't scale & shift just undo it?

[layer_norm.py](layer_norm.py#L47)

Two notes, one question. Centering and unit-variance on the last dim, then `y = scale * x_norm + shift`.

**To learn**

- Why that keeps activations (and gradients) in a stable range — think "don't saturate GELU/softmax," not "proof that gradients can't vanish."
- How this relates to the ReLU "outputs are biased positive, never mean-0" observation a few lines above.
- If the model can freely rescale, why that doesn't fully undo the guarantee. Why the affine was added at all. What happens if you freeze `scale=1`, `shift=0`.

### 4. Why LayerNorm instead of BatchNorm?

[layer_norm.py](layer_norm.py#L52)

Raschka: more flexibility, less compute. The interesting part is systems, not the Wikipedia definition.

**To learn**

- BatchNorm stats are per-batch, across examples. What breaks at inference (running vs batch stats) and at batch size 1.
- Distributed: BatchNorm needs an all-reduce of mean/var across ranks. LayerNorm is per-token — no cross-device reduction.
- Why token-wise stats fit variable-length sequences.

### 5. Why divide attention scores by √d?

[examples/attn.py](examples/attn.py#L143)

Already sketched: large `d` → large dots → softmax near one-hot → backprop through softmax barely moves. Dot-product std scales ~√d, so dividing keeps softmax inputs O(1). Repro in `_repro_softmax_scale`.

**To learn** (you're close — finish the derivation)

- If q, k have i.i.d. entries with variance 1, `Var(q·k) = d`.
- Why a peaked softmax has a vanishing Jacobian.
- What happens if you don't scale (or use RMSNorm / QK-norm, as some later models do).

### 6. Why GELU over ReLU? What's SwiGLU?

[feed_forward.py](feed_forward.py#L6)

GELU ≈ `x * Φ(x)` (Φ = standard Gaussian CDF). Smooth ReLU: tiny non-zero values/grads on the negative side. Also see the ReLU mean-0 note in [layer_norm.py](layer_norm.py#L8).

**Stop at "I can explain this in two minutes."** There isn't a clean theorem. Smoothness + negatives still send a gradient, then SwiGLU won ablations (Llama etc.).

**To learn**

- Why multiply by the Gaussian CDF (the probabilistic story is enough).
- Why that helps vs hard-zero ReLU.
- What SwiGLU *is* and why modern LLMs switched. Don't wait for a proof.

### 7. How many FLOPs per attention head? When does attn dominate the FFN?

[attention.py](attention.py#L353)

GPT-2-sized forward timing experiment is just above this note. This is the "will this fit / why is decode slow" question.

**To learn**

- FLOPs for QKV proj, scores (`seq² · d_head`), attn·V, output proj.
- Attention is `O(n² d)`, FFN is `O(n d²)` — which dominates at which context length.
- How that changes with GQA / MQA / sliding window (skim, don't implement all three).

### 8. Why does the dataset return `(x, y)` windows, not unrolled next-token pairs?

[data.py](data.py#L16) · output shape note in [model.py](model.py#L103)

Each example is a `max_length` chunk and the same chunk shifted by one. Logits are `[batch, seq, vocab]`, not a single next-token.

**To learn**

- Teacher forcing: one forward pass is `max_length` training signals.
- Why that shape is the training objective, and why padding / packing / stride matter once sequences aren't fixed length.

---

## Hands-on, not a study session

These are real, but you learn them by breaking code for 20 minutes, not by reading papers. Don't put them next to vanishing gradients.

### PyTorch "unrolling" (broadcasting)

[examples/pos_embedding_example.py](examples/pos_embedding_example.py#L52)

`tok_embed` is `8 × 4 × 256`, `pos_embed` is `4 × 256`, and the add just works — you called this unrolling; the library name is broadcasting. Align dims from the right; size-1 or missing dims get repeated. Print shapes, force a couple of errors, you're done.

### `view` / `transpose` / `contiguous`

[attention.py](attention.py#L229)

Load-bearing for MHA. Split inner dim *first*, then transpose so the matmul is over `(seq, head_dim)` per head. `transpose` is not contiguous; `view` requires it. Play with the ops in isolation until `transpose(1, 2).view(...)` throwing makes sense.

Matmul orientation (why `inputs @ inputs.T` for scores, why *no* transpose for `weights @ values`) is already written down in [examples/attn.py](examples/attn.py#L65).

### Fuse Q, K, V into one matmul

[attention.py](attention.py#L20) (same note at [V2](attention.py#L62) and [CausalAttention](attention.py#L128))

Do it once: concat the three weights, `x @ W_qkv`, split. One GEMM vs three. Then you know.

### Attention dropout at inference

[attention.py](attention.py#L148)

`nn.Dropout` already no-ops in `eval()`. Confirm `model.eval()` for generation, know that this dropout is on the *weights* not the values, move on.

---

## Nice to haves

Park these until the list above feels solid, or until the book/code forces the question.

### Why expand 4× in the FFN?

[feed_forward.py](feed_forward.py#L33)

Position-wise MLP over the residual stream; same in/out dim makes stacking easy. **4× is a GPT-2 convention**, not a law — later models change it. Worth knowing what the layer *does*; not worth a derivation of the constant.

### Sequential wrapper vs true multi-head

[attention.py](attention.py#L163)

You already implemented the real one. Skim param count (`n_heads` separate `d_in → d_out` vs one proj then split) and why `out_proj` exists after concat. Don't rebuild the wrapper.

### Absolute pos embeddings vs RoPE

[model.py](model.py#L36) · [examples/pos_embedding_example.py](examples/pos_embedding_example.py#L40)

Why add (not concat)? Why a lookup table vs sinusoid vs rotary? Come back when you care about **long context** or you're reading Llama-style code. Implementing RoPE is a good later project, not blocking for finishing the GPT-2-shaped model.

### Word2Vec: real corpus, OOV, "are these embeddings actually good?"

[side_quests/word_2_vec.py](side_quests/word_2_vec.py#L33) · [bonus notes](side_quests/word_2_vec.py#L216)

Fun side quest, not on the transformer critical path. Train on real text, handle missing vocab, check that similar words are actually close.
