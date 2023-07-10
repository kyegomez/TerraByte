import torch
from TerraByte import MEGABYTE

model = MEGABYTE(
    num_tokens = 16000,             # number of tokens
    dim = (512, 256),               # transformer model dimension (512 for coarsest, 256 for fine in this example)
    max_seq_len = (1024, 4),        # sequence length for global and then local. this can be more than 2
    depth = (6, 4),                 # number of layers for global and then local. this can be more than 2, but length must match the max_seq_len's
    dim_head = 64,                  # dimension per head
    heads = 8,                      # number of attention heads
    flash_attn = True               # use flash attention
)

x = torch.randint(0, 16000, (1, 1024, 4))

loss = model(x, return_loss = True)
loss.backward()

# then after much training

logits = model(x)

# and sample from the logits accordingly
# or you can use the generate function

sampled = model.generate(temperature = 0.9, filter_thres = 0.9) # (1, 1024, 4)