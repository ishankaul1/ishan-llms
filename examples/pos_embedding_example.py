import torch

from data import create_dataloader_v1

from utils.constants import THE_VERDICT_URL
from utils.fetch_text import fetch_text


MAX_LEN = 4
VOCAB_SIZE = 50257
OUTPUT_DIM = 256


txt = fetch_text(THE_VERDICT_URL)



loader = create_dataloader_v1(
    txt, batch_size=8, max_length = MAX_LEN, stride=MAX_LEN, shuffle=False
)

data_iter = iter(loader)

inputs, targets = next(data_iter)


print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
# Our batch is 8x4 (8 batches of 4 token ids)



tok_embed_layer = torch.nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)

tok_embed = tok_embed_layer(inputs)
# -> 8 x 4 x 256 (for each token id snag the embedding value)

# Pos embed approach - create another embed layer with the _same_ 
# ouptut dims as the original layer;
# However -- you only need _one_ embedding per possible position (eg max context len)

pos_embed_layer = torch.nn.Embedding(MAX_LEN, OUTPUT_DIM)
pos_embed = pos_embed_layer(torch.arange(MAX_LEN)) # just 4 x output_dim; extracting values
print(pos_embed.shape)


