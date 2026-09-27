import tiktoken
import torch

from config import GPTConfig
from model import GPTModel


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


if __name__ == "__main__":
    GPT_CONFIG_124M = GPTConfig(context_length=256)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = model.generate_text_simple(
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M.context_length,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    