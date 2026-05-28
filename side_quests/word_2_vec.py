"""
Working thru quick primer on embeddings

Partial follow along from - https://jaketae.github.io/study/word2vec/
Except I converted it to pytorch for some reps.

Intuition -

First layer: 1hot tokens (input) -> embeddng (or just tok idx -> embedding for pyt)
(word)
[batch x len_corpus] @ [len_corpus dot dims] => batch x dims
[10000] dot [21975]T -> [2] (that's your embedding; assuming 1d)

Second layer: embedding -> word space

[batch x dims] @ [dims x len_corpus] -> [batch x len_corpus]

cross entropy loss on the final --> gives you backprop on input -> target prediction

Generation just softmax
"""

from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

import torch

from tqdm import tqdm

import re


# TODO - Pull a real dataset!!


def tokenize(text):
    pattern = re.compile(r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*")
    return pattern.findall(text.lower())


def get_lookups(tokens: list[str]):
    word_to_idx = {}
    idx_to_word = {}

    for i, token in enumerate(set(tokens)):
        word_to_idx[token] = i
        idx_to_word[i] = token

    return word_to_idx, idx_to_word


class Word2VecRepro(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.w1 = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.w2 = nn.Linear(embed_dim, vocab_size)

    # NOTE: X should be just a batch list of integer ix'es
    def forward(self, X: Tensor) -> Tensor:
        y = self.w1(X)
        y = self.w2(y)
        return y  # Tensor batch x vocab_size


def _build_pairs(lst: list, window: int) -> list[tuple[int, int]]:

    assert window >= 1, "Window must be positive"

    pairs = []  # list of tup(int, int)
    for i_ctr in range(len(lst)):
        # Range from (ctr - window -> ctr + window, excluding ctr)
        after_rng = range(
            min(i_ctr + 1, len(lst)), min(i_ctr + window + 1, len(lst))
        )  # NOTE - how to prevent going outside of range?

        pairs.extend([(i_ctr, i_ctx) for i_ctx in [*after_rng]])

    return pairs


class Word2VecDataset(Dataset):
    # NOTE: word_to_idx is a property of the entire vocabulary
    def __init__(self, tokens: list[str], word_to_idx: dict, window: int):
        # Generate and store all (center, context) pairs here

        # Build pairs
        token_pairs = _build_pairs(tokens, window)

        # Map each pair to token idx

        as_vocab_pairs = []
        for t_p in token_pairs:
            i_ctr, i_ctx = t_p

            tok_ctr = tokens[i_ctr]
            tok_ctx = tokens[i_ctx]

            vocab_ix_ctr = word_to_idx[tok_ctr]
            vocab_ix_ctx = word_to_idx[tok_ctx]

            as_vocab = (vocab_ix_ctr, vocab_ix_ctx)
            as_vocab_pairs.append(as_vocab)

        # Store

        self.data = torch.tensor(as_vocab_pairs, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tensor:
        return self.data[index]


def train(
    model: Word2VecRepro,
    dataloader: DataLoader,
    n_epochs: int,
    lr: float,
    weight_decay: float,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    pbar = tqdm(range(n_epochs), desc="training")

    for epoch in pbar:
        for batch in dataloader:
            # print(f"BATCH: {batch}, shape: {batch.shape}")
            X = batch[:, 0]
            y = batch[:, 1]
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})


def test(model: Word2VecRepro, loader: DataLoader):
    model.eval()

    total, n = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            X = batch[:, 0]
            y = batch[:, 1]
            logits = model(X)

            total += nn.functional.cross_entropy(logits, y).item()
            n += 1

        print(f"Test Loss: {total / n:.4f}")


if __name__ == "__main__":
    """
    Main train-test loop for this exercise.

    Steps -
        1. Load in a large-ish representative dataset/corpus
        2. Build the lookups & dataset/dataloader
        3. Train the model
        4. Evaluate on some held out section of the corpus


    NOTE - will just use lookups from the whole dataset to prevent having to handle
    missing lookups, but that is a real issue for this model!!

    """
    EMBED_DIMS = 20
    N_EPOCHS = 50
    LR = 1e-3
    WEIGHT_DECAY = 0.01
    WINDOW = 3

    FAKE_DATA = """Machine learning is the study of computer algorithms that \
    improve automatically through experience. It is seen as a \
    subset of artificial intelligence. Machine learning algorithms \
    build a mathematical model based on sample data, known as \
    training data, in order to make predictions or decisions without \
    being explicitly programmed to do so. Machine learning algorithms \
    are used in a wide variety of applications, such as email filtering \
    and computer vision, where it is difficult or infeasible to develop \
    conventional algorithms to perform the needed tasks."""

    tokens = tokenize(FAKE_DATA)
    word_to_ix, ix_to_word = get_lookups(tokens)

    dataset = Word2VecDataset(tokens=tokens, word_to_idx=word_to_ix, window=WINDOW)

    split_ix = int(0.8 * len(tokens))

    train_ds = Word2VecDataset(
        tokens=tokens[:split_ix], word_to_idx=word_to_ix, window=WINDOW
    )
    test_ds = Word2VecDataset(
        tokens=tokens[split_ix:], word_to_idx=word_to_ix, window=WINDOW
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = Word2VecRepro(vocab_size=len(word_to_ix), embed_dim=EMBED_DIMS)

    train(model, train_loader, n_epochs=N_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY)

    test(model, test_loader)


"""
NOTE - This worked pretty well for getting the general shape/structure of pytorch training from scratch.

An interesting follow up here would be - 

    - Train/test loop on a much larger corpus (a few wikipedia docs/reddit threads, etc.). More data processing
    boilerplate but good practice. Probably can AI-code that part
    - Implement the actual encode() function and a few tests for e2e behavior after training (eg -- check against a few 
    actual words that should be close versus far.)


Moving along with the rest of Rashcka in the meantime!

"""