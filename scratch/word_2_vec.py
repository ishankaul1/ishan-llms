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
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
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
        return y # Tensor batch x vocab_size


class Word2VecDataset(Dataset):
    # NOTE: word_to_idx is a property of the entire vocabulary
    def __init__(self, tokens: list[str], word_to_idx: dict, window: int):
        assert window >= 1, "Window must be positive"
        # Generate and store all (center, context) pairs here
        valid_ixs = [] # list of tup(int, int)
        for i_ctr, tok in enumerate(tokens):
            



    
    def __len__(self):
        pass

    def __getitem__(self, index) -> Tensor:
        # Return a single pair as tensor


def train(model: Word2VecRepro, dataloader: DataLoader, n_epochs: int, lr: float, weight_decay: float):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    pbar = tqdm(range(n_epochs), desc="training")

    for epoch in pbar:
        for X, y in dataloader:
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})



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
    weight_decay = .01


    FAKE_DATA = '''Machine learning is the study of computer algorithms that \
    improve automatically through experience. It is seen as a \
    subset of artificial intelligence. Machine learning algorithms \
    build a mathematical model based on sample data, known as \
    training data, in order to make predictions or decisions without \
    being explicitly programmed to do so. Machine learning algorithms \
    are used in a wide variety of applications, such as email filtering \
    and computer vision, where it is difficult or infeasible to develop \
    conventional algorithms to perform the needed tasks.'''



