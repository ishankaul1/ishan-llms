import torch

from data import create_dataloader_v1

from utils.constants import THE_VERDICT_URL
from utils.fetch_text import fetch_text


MAX_LEN = 4
VOCAB_SIZE = 50257
OUTPUT_DIM = 256


txt = fetch_text(THE_VERDICT_URL)


loader = create_dataloader_v1(
    txt, batch_size=8, max_length=MAX_LEN, stride=MAX_LEN, shuffle=False
)

data_iter = iter(loader)

inputs, targets = next(data_iter)


print("Token IDs:\n", inputs)
print(
    "\nInputs shape:\n",
    inputs.shape,
)
# Our batch is 8x4 (8 batches of 4 token ids)


tok_embed_layer = torch.nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)

tok_embed = tok_embed_layer(inputs)
print(f"Tok embed shape:\n {tok_embed.shape}")
# -> 8 x 4 x 256 (for each token id snag the embedding value)

# Pos embed approach - create another embed layer with the _same_
# ouptut dims as the original layer;
# However -- you only need _one_ embedding per possible position (eg max context len)

pos_embed_layer = torch.nn.Embedding(MAX_LEN, OUTPUT_DIM)
pos_embed = pos_embed_layer(
    torch.arange(MAX_LEN)
)  # just 4 x output_dim; extracting values
print(f"Pos embed shape:\n {pos_embed.shape}")


input_embed = tok_embed + pos_embed
# NOTE -
# This just works; each of the 8 samples in the batch
# get 1 embed for each token pos added to them; 
# pytorch automatically 'unrolls'
# TODO - would be interesting to learn undertsand deeply _how_
# pytorch unrolls in general. Eg what the algorithm of choice is, 
# any weird edge cases or gotchas, etc.

print(f"Input Embed:\n {input_embed.shape}")
