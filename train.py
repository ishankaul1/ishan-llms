# Lightweight training impl

# TODO -- try the Project Gutenberg training example!!

import argparse

import tiktoken
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config import GPTConfig
from data import create_dataloader_v1
from model import GPTModel
from token_utils import text_to_token_ids, token_ids_to_text
from utils.constants import THE_VERDICT_URL
from utils.fetch_text import fetch_text

# Book ch5: shorter context so the tiny verdict split still has several windows.
TRAIN_CFG = GPTConfig(context_length=256)


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    # Note -- these comes out as:
    # input_batch: seq of tokens
    # target_batch: same # of tokens, shifted up 1

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    # Logits are now B x S x V -- we just ran the model for _all_ positions in its seq
    # Targets are still lined up B x S -- the "correct" token prediction for each pos in each sequence

    # snazzy!! just flatten along S now to pass to cross-entropy
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


def evaluate_model(
    model: GPTModel,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:

    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    return train_loss, val_loss


def generate_and_print_sample(
    model: GPTModel,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
    start_context: str,
) -> None:
    model.eval()

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = model.generate_text_simple(
            idx=encoded, max_new_tokens=50, context_size=context_size
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)

    print(f"generate_and_print decoded: {decoded_text}")

    model.train()


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
        # "train" mode -- modules like Dropout auto respond
        model.train()

        for input_batch, target_batch in train_loader:
            # Reset loss grads from prev batch. Kinda funny that you still have to do this after stepping
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss.backward()

            optimizer.step()

            tokens_seen += (
                input_batch.numel()
            )  # Literally the # token ids in the input/trained on

            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )

                # Eval step bookkeeping
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_verdict_dataloaders(
    cfg: GPTConfig = TRAIN_CFG,
    train_ratio: float = 0.9,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    text = fetch_text(THE_VERDICT_URL)
    split_idx = int(train_ratio * len(text))
    train_data = text[:split_idx]
    val_data = text[split_idx:]

    # NOTE -- loader outputs target, token tensor pairs fully aligned.
    # x and y are both [batch, ctx] and y is shifted 1 past x in the prose.
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def loader_loss_example() -> None:
    train_loader, val_loader = load_verdict_dataloaders()

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    device = get_device()
    print(f"Using device: {device}")

    model = GPTModel(cfg=TRAIN_CFG)
    model.to(device)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


def run_train_example() -> None:
    torch.manual_seed(123)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = load_verdict_dataloaders()
    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(TRAIN_CFG)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )
    print("Final train losses:", train_losses)
    print("Final val losses:", val_losses)
    print("Tokens seen:", tokens_seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loader-loss",
        action="store_true",
        help="Run the no-grad loader loss check instead of training.",
    )
    args = parser.parse_args()

    if args.loader_loss:
        loader_loss_example()
    else:
        run_train_example()