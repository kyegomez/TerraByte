import torch.nn as nn
from mgqa.attention import MGQA as Attention

from terra_byte.model.helpers import (
    FeedForward,
    RMSNorm,
    RotaryEmbedding,
    exists,
    token_shift,
)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        rel_pos = True,
        flash_attn = False
    ):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim_head) if rel_pos else None
        self.layers = nn.ModuleList([])

        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim = dim, 
                    dim_head = dim_head, 
                    heads = heads, 
                    dropout = attn_dropout, 
                    flash = flash_attn
                ),
                
                FeedForward(
                    dim = dim, 
                    mult = ff_mult, 
                    dropout = ff_dropout
                )
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x):
        n = x.shape[-2]
        rotary_emb = self.rotary_emb(n) if exists(self.rotary_emb) else None

        for attn, ff in self.layers:
            x = attn(token_shift(x)) + x
            x = ff(token_shift(x)) + x

        return self.norm(x)



