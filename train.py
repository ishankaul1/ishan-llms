# Lightweight training impl

# TODO -- try the Project Gutenberg training example!!

import tiktoken
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data import create_dataloader_v1
from model import GPT_CONFIG_124M, GPTModel
from utils.constants import THE_VERDICT_URL
from utils.fetch_text import fetch_text


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    # This really just runs the model & flattens the outputs/targets lined up to pass
    # to x-entropy
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    # Guess -- logits are B x S x V
    # Targets -- B x S
    # Flatten across seq bc each position is its own data point to train on.
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    # this must be the averaging he simulated earlier
    return total_loss / num_batches


# NEXT TODO -- 5.2 Actual training loop!!! Pay close attention here. And make sure we follow up with appendix D at the end!

def train_model_simple(
    model: GPTModel,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: tiktoken.Encoding,
) -> tuple[list[float], list[float], list[int]]:
    train_losses: list[float] = []
    val_losses: list[float] = []
    track_tokens_seen: list[int] = []

    tokens_seen, global_step = 0, -1

    # Each epoch pulls a set of batches and trains on them
    # Note -- whats the real importance of a difference between batches and epoch? Like not just keep pulling batches?
    for epoch in range(num_epochs):
        model.train()


if __name__ == "__main__":
    train_ratio = 0.9

    text = fetch_text(THE_VERDICT_URL)

    split_idx = int(train_ratio * len(text))

    train_data = text[:split_idx]

    val_data = text[:split_idx]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M.context_length,
        stride=GPT_CONFIG_124M.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M.context_length,
        stride=GPT_CONFIG_124M.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    # NOTE -- loader outputs target, token tensor pairs fully alilgned.
    # eg -- x and y anre bot 2, 1024 and y is shifted 1 past x in the actual data/prose

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(cfg=GPT_CONFIG_124M)
    model.to(device)

    with torch.no_grad():  # disable training/gradient tracking for now
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
