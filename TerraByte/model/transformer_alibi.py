# class Transformer(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         layers,
#         dim_head = 64,
#         heads = 8,
#         attn_dropout = 0.,
#         ff_dropout = 0.,
#         ff_mult = 4,
#         rel_pos_bias = True,
#         flash_attn = True,
#     ):
#         super().__init__()
#         self.alibi = Alibi(heads = heads) if rel_pos_bias else None
#         self.layers = nn.ModuleList([])

#         for _ in range(layers):
#             self.layers.append(nn.ModuleList([
#                 Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn),
#                 FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
#             ]))

#         self.norm = RMSNorm(dim)

#     def forward(self, x):
#         n = x.shape[-2]
#         attn_bias = self.alibi(n, n, device = x.device) if exists(self.alibi) else None

#         for attn, ff in self.layers:
#             x = attn(token_shift(x), attn_bias = attn_bias) + x
#             x = ff(token_shift(x)) + x

#         return self.norm(x)
    

