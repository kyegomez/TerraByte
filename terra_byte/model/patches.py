from typing import Tuple

import torch
from beartype.typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class PatchEmbeddings(nn.Module):
    def __init__(self, dim_in, dim_out, seq_len):
        super().__init__()
        self.embedding = nn.Sequential(
            Rearrange('... rd -> ... (r d)'),
            nn.LayerNorm(seq_len * dim_in),
            nn.Linear(seq_len * dim_in, dim_out),
            nn.LayerNorm(dim_out),
        )
    
    def forward(self, x):
        return self.embedding(x)
    

class UniversalPatchEmbedder(nn.Module):

    #Universal modality patch embdders => process all modalities
    """In this implementation, we create a UniversalPatchEmbedder class that takes a tuple of input dimensions,
    an output dimension, and a patch size as arguments. The class contains a list of embedders and modality embeddings. 
    In the forward method, we select the appropriate embedder based on the
    modality and apply it to the input. We then add the modality embeddings to the output.
    """
    def __init__(
        self, 
        input_dims: Tuple[int], 
        output_dim: int, 
        patch_size: int
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embedders = nn.ModuleList([nn.Linear(dim, output_dim) for dim in input_dims])
        self.modality_embeddings = nn.Embedding(len(input_dims), output_dim)


    def forward(self, x: Tensor, modality: Tensor) -> Tensor:
        #determine the input shape of x
        input_shape = x.shape

        print(f'Input shape: {input_shape}')

        #reshape x into a common shape (batch_size, input_dim)
        x = x.view(input_shape[0], -1)
        print(f'x reshaped: {x}')

        #select the most optimal embedder for modality
        embedder = self.embedders[modality]
        print(f"Embedder: {embedder}")

        #apply selected embedder
        x = rearrange(x, 'b (p1 p2) d -> b p1 p2 d', p1 = self.patch_size)
        x = embedder(x)

        #modality embeddings
        modality_emb = self.modality_embeddings(torch.tensor(modality).to(x.device))
        print(f"Modality embedder: {modality_emb}")

        x = x + modality_emb

        print(f"X shape: {x}")
        
        return x
    