import torch
from TerraByte import DilatedMegabyte

model = DilatedMegabyte(
    num_tokens = 16000,
    dim = (512, 256),   
    dim_head=64,
    dilation_rate=4, 
    segment_size=2,
    max_seq_len = (1024, 4), 
    depth = (6, 4),          
    dim_head = 64,           
    heads = 8,  
)

x = torch.randint(0, 16000, (1, 1024, 4))

loss = model(x, return_loss = True)
loss.backward()

# then after much training
logits = model(x)

# and sample from the logits accordingly
# or you can use the generate function
sampled = model.generate(temperature = 0.9, filter_thres = 0.9) # (1, 1024, 4)