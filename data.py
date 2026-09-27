import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        txt: str,
        tokenizer: tiktoken.Encoding,
        max_length: int,
        stride: int,
    ) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        # First, tokenize the text fully
        token_ids = tokenizer.encode(txt)

        # Generate input -> output tensors
        # NOTE - these can be passed as-is to model() and validated against target.
        # Target is shifted one up, so prediction at that postion aligns directly with the next token,
        # which is how we have built this dataset

        # TODO -- for large scale datasets, do you have to cache/discard them on disk/in-mem?
        for i_start in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i_start : i_start + max_length]
            target_chunk = token_ids[i_start + 1 : i_start + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(
        txt=txt, tokenizer=tokenizer, max_length=max_length, stride=stride
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


if __name__ == "__main__":
    import requests

    txt = requests.get(
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    ).text

    loader = create_dataloader_v1(
        txt=txt, batch_size=8, max_length=4, stride=4, shuffle=False
    )

    data_iter = iter(loader)
    
    inputs, targets = next(data_iter)

    print(f"Inputs:\n{inputs}")
    print(f"Targets:\n{targets}")
