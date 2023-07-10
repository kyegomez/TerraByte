import math
import functools
from itertools import zip_longest

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Tuple, Union

from TerraByte.model.attend import Attend

from tqdm import tqdm

from torch import Tensor
from typing import Tuple, Union
from einops import rearrange, repeat

from zeta import DilatedAttention2


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def reduce_mult(nums):
    return functools.reduce(lambda x, y: x * y, nums, 1)

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# token shift, from Peng et al of RWKV

def token_shift(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1))
    return torch.cat((t, t_shift), dim = -1)

# positional bias

class Alibi(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, i, j, device):
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :j]

        bias = torch.arange(j, device = device)
        bias = rearrange(bias, 'j -> 1 1 j')
        bias = bias * self.slopes

        self.register_buffer('bias', bias, persistent = False)
        return self.bias

# norm

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# helper classes

def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.attend = Attend(
            causal = True,
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, attn_bias = None):
        h, device = self.heads, x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        out = self.attend(q, k, v, attn_bias = attn_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

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
        rel_pos_bias = True,
        flash_attn = False,
    ):
        super().__init__()
        self.alibi = Alibi(heads = heads) if rel_pos_bias else None
        self.layers = nn.ModuleList([])

        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                #swap for dilated attention
                # DilatedAttention2(d_model=dim, num_heads = heads, dropout=attn_dropout),
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x):
        n = x.shape[-2]
        attn_bias = self.alibi(n, n, device = x.device) if exists(self.alibi) else None

        for attn, ff in self.layers:
            x = attn(token_shift(x), attn_bias = attn_bias) + x
            x = ff(token_shift(x)) + x

        return self.norm(x)
    

class DilatedTransformer(nn.Module):
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
        rel_pos_bias = True,
        dilation_rate = None,
        segment_size = None, 
        use_xpos = False,
        use_rel_pos_bias = False,
    ):
        super().__init__()


        assert dilation_rate is not None, "You must specify a 'dilation_rate' "
        assert segment_size is not None,"You must specify a 'segment_size' "


        self.alibi = Alibi(heads = heads) if rel_pos_bias else None
        self.layers = nn.ModuleList([])

        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                #swap for dilated attention
                DilatedAttention2(
                d_model= dim,
                num_heads=heads, 
                dilation_rate=dilation_rate,
                segment_size=segment_size,
                dropout=attn_dropout,
                casual=False,
                use_xpos=use_xpos,
                use_rel_pos_bias=use_rel_pos_bias
                ),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x):
        n = x.shape[-2]
        attn_bias = self.alibi(n, n, device = x.device) if exists(self.alibi) else None

        for attn, ff in self.layers:
            x = attn(x) + x  # Removed the token_shift since DilatedAttention2 is taking care of offsetting
            x = ff(x) + x

        return self.norm(x)


#Universal modality patch embdders => process all modalities
"""In this implementation, we create a UniversalPatchEmbedder class that takes a tuple of input dimensions,
an output dimension, and a patch size as arguments. The class contains a list of embedders and modality embeddings. 
In the forward method, we select the appropriate embedder based on the
modality and apply it to the input. We then add the modality embeddings to the output.

Verison2
* Use Imagebind for embedders

"""

class UniversalPatchEmbedder(nn.Module):
    def __init__(self, input_dims: Tuple[int], output_dim: int, patch_size: int):
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
    

class PatchEmbeddings(nn.Module):
    def __init__(self, dim_in, dim_out, seq_len):
        super().__init__()
        self.embedding = nn.Sequential(
            Rearrange('... rd -> ... (r d)')
            nn.LayerNorm(seq_len * dim_in),
            nn.Linear(seq_Len * dim_in, dim_out),
            nn.LayerNorm(dim_out)
        )
    
    def forward(self, x):
        return self.embedding(x)
    




# main class
class OmniMEGABYTE(nn.Module):

    @beartype
    def __init__(
        self,
        *,
        num_tokens,
        dim: Union[Tuple, int],
        depth: Tuple,
        max_seq_len: Tuple,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0,
        rel_pos_bias = True,
        flash_attn = False
    ):
        super().__init__()

        # simplified configuration for each stage of the hierarchy
        # depth = (2, 2, 4) would translate to depth 2 at first stage, depth 2 second stage, depth 4 third
        # max_seq_len = (16, 8, 4) would translate to max sequence length of 16 at first stage, length of 8 at second stage, length of 4 for last

        assert isinstance(depth, tuple) and isinstance(max_seq_len, tuple)
        assert len(depth) == len(max_seq_len)

        self.stages = len(depth)
        dim = cast_tuple(dim, self.stages)

        assert len(dim) == self.stages

        coarsest_dim, *_, fine_dim = dim

        self.token_emb = nn.Embedding(num_tokens, fine_dim)

        self.max_seq_len = max_seq_len

        self.start_tokens = nn.ParameterList([nn.Parameter(torch.randn(h_dim)) for h_dim, seq_len in zip(dim, max_seq_len)])
        self.pos_embs = nn.ModuleList([nn.Embedding(seq_len, h_dim) for h_dim, seq_len in zip(dim, max_seq_len)])

        # self.patch_embedders = nn.ModuleList([nn.Sequential(
        #     Rearrange('... r d -> ... (r d)'),
        #     nn.LayerNorm(seq_len * dim_in),
        #     nn.Linear(seq_len * dim_in, dim_out),
        #     nn.LayerNorm(dim_out)
        # ) for dim_in, dim_out, seq_len in zip(dim[1:], dim[:-1], max_seq_len[1:])])

        #v2
        input_dims = (dim[1], dim[0], max_seq_len[1])
        self.patch_embedders = UniversalPatchEmbedder(input_dims, dim[0], max_seq_len[1])

        #------->

        self.transformers = nn.ModuleList([])
        self.to_next_transformer_projections = nn.ModuleList([])

        for h_dim, next_h_dim, stage_depth, next_seq_len in zip_longest(dim, dim[1:], depth, max_seq_len[1:]):
            self.transformers.append(Transformer(
                dim = h_dim,
                layers = stage_depth,
                dim_head = dim_head,
                heads = heads,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                ff_mult = ff_mult,
                rel_pos_bias = rel_pos_bias,
                flash_attn = flash_attn
            ))

            proj = nn.Identity()

            if exists(next_h_dim) and next_h_dim != dim:
                proj = nn.Sequential(
                    nn.Linear(h_dim, next_h_dim * next_seq_len),
                    Rearrange('... (n d) -> (...) n d', n = next_seq_len)
                )

            self.to_next_transformer_projections.append(proj)

        self.to_logits = nn.Linear(fine_dim, num_tokens)
        self.pad_id = pad_id

    def generate(self, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        total_seq_len = reduce_mult(self.max_seq_len)
        device = next(self.parameters()).device

        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        seq = prime
        batch = seq.shape[0]

        for _ in tqdm(range(total_seq_len - seq.shape[-1])):
            logits = self.forward(seq)[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)

        return seq.reshape(batch, *self.max_seq_len)

    def forward_empty(self, batch_size):
        # take care of special case
        # where you sample from input of 0 (start token only)

        prev_stage_tokens_repr = None

        for stage_start_tokens, transformer, proj in zip(self.start_tokens, self.transformers, self.to_next_transformer_projections):
            tokens = repeat(stage_start_tokens, 'd -> b 1 d', b = batch_size)

            if exists(prev_stage_tokens_repr):
                tokens = tokens + prev_stage_tokens_repr[..., :tokens.shape[-2], :]

            tokens = transformer(tokens)
            prev_stage_tokens_repr = proj(tokens)

        return self.to_logits(tokens)

    def forward(self, ids, modality,  return_loss = False):
        batch = ids.shape[0]

        print(f'ids shape: {ids.shape[0]}')

        assert ids.ndim in {2, self.stages + 1}
        print(f"self stages: {self.stages}")

        flattened_dims = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dims:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_seq_len[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = ids.reshape(batch, -1, *self.max_seq_len[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        # check some dimensions

        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'

        # get token embeddings

        tokens = self.token_emb(ids)

        # get tokens for all hierarchical stages, reducing by appropriate dimensions
        # and adding the absolute positional embeddings

        tokens_at_stages = []
        reduced_tokens = tokens

        patch_embedders_list = [self.patch_embedders]

        for ind, pos_emb, patch_emb in zip(range(len(prec_dims)), reversed(self.pos_embs), reversed(patch_embedders_list)):
            is_first = ind == 0

            if not is_first:
                reduced_tokens = patch_emb(reduced_tokens, modality)

            positions = pos_emb(torch.arange(reduced_tokens.shape[-2], device=device))
            tokens_with_position = reduced_tokens + positions
            tokens_at_stages.insert(0, tokens_with_position)

        # the un-pixelshuffled representations of the previous hierarchy, starts with None

        prev_stage_tokens_repr = None

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions        

        for stage_start_tokens, stage_tokens, transformer, proj in zip(self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections):
            stage_tokens, ps = pack_one(stage_tokens, '* n d')

            stage_start_tokens = repeat(stage_start_tokens, 'f -> b 1 f', b=stage_tokens.shape[0])

            #update the dimensions of the stage_start_tokens tensor
            stage_start_tokens = stage_start_tokens[..., :stage_tokens.shape[-1]]

            # Print the shapes of the tensors before concatenating
            print(f"stage_start_tokens shape: {stage_start_tokens.shape}")
            print(f"stage_tokens shape: {stage_tokens.shape}")

            # concat start token

            stage_tokens = torch.cat((
                stage_start_tokens,
                stage_tokens,
            ), dim=-2)

            # sum the previous hierarchy's representation

            if exists(prev_stage_tokens_repr):
                prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value=0.)
                stage_tokens = stage_tokens + prev_stage_tokens_repr

            attended = transformer(stage_tokens)

            attended = unpack_one(attended, ps, '* n d')

            # project for next stage in the hierarchy

            prev_stage_tokens_repr = proj(attended[..., :-1, :])

        # project to logits

        logits = self.to_logits(attended)

        start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None))]
        start_tokens = rearrange(start_tokens, 'b d -> b 1 d')

        logits = logits[..., 1:, :]

        if not return_loss:

            if flattened_dims:
                logits = rearrange(logits, 'b ... c -> b (...) c')
                logits = logits[:, :seq_len]

            return logits

        logits = rearrange(logits, 'b ... c -> b (...) c')
        logits = torch.cat((start_tokens, logits), dim = -2)

        preds = rearrange(logits, 'b n c -> b c n')
        labels = rearrange(ids, 'b ... -> b (...)')

        loss = F.cross_entropy(
            preds[..., :-1],
            labels,
            ignore_index = self.pad_id
        )

        return loss



#regular megabyte no universal patch embedder
class MEGABYTE(nn.Module):

    @beartype
    def __init__(
        self,
        *,
        num_tokens,
        dim: Union[Tuple, int],
        depth: Tuple,
        max_seq_len: Tuple,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0,
        rel_pos = False,
        pos_emb = False,
        flash_attn = False
    ):
        super().__init__()

        # simplified configuration for each stage of the hierarchy
        # depth = (2, 2, 4) would translate to depth 2 at first stage, depth 2 second stage, depth 4 third
        # max_seq_len = (16, 8, 4) would translate to max sequence length of 16 at first stage, length of 8 at second stage, length of 4 for last

        assert isinstance(depth, tuple) and isinstance(max_seq_len, tuple)
        assert len(depth) == len(max_seq_len)

        self.stages = len(depth)
        dim = cast_tuple(dim, self.stages)

        assert len(dim) == self.stages

        coarsest_dim, *_, fine_dim = dim

        self.max_seq_len = max_seq_len

        self.start_tokens = nn.ParameterList([nn.Parameter(torch.randn(h_dim)) for h_dim, seq_len in zip(dim, max_seq_len)])
        self.pos_embs = nn.ModuleList([nn.Embedding(seq_len, h_dim) for h_dim, seq_len in zip(dim, max_seq_len)]) if pos_emb else None

        self.token_embs = nn.ModuleList([])

        patch_size = 1
        self.token_embs.append(nn.Embedding(num_tokens, fine_dim))

        for dim_out, seq_len in zip(reversed(dim[:-1]), reversed(max_seq_len[1:])):
            patch_size *= seq_len

            self.token_embs.append(nn.Sequential(
                nn.Embedding(num_tokens, fine_dim),
                Rearrange('... r d -> ... (r d)'),
                nn.LayerNorm(patch_size * fine_dim),
                nn.Linear(patch_size * fine_dim, dim_out),
                nn.LayerNorm(dim_out)
            ))

        self.transformers = nn.ModuleList([])
        self.to_next_transformer_projections = nn.ModuleList([])

        for h_dim, next_h_dim, stage_depth, next_seq_len in zip_longest(dim, dim[1:], depth, max_seq_len[1:]):
            self.transformers.append(Transformer(
                dim = h_dim,
                layers = stage_depth,
                dim_head = dim_head,
                heads = heads,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                ff_mult = ff_mult,
                rel_pos = rel_pos,
                flash_attn = flash_attn
            ))

            proj = nn.Identity()

            if exists(next_h_dim) and next_h_dim != dim:
                proj = nn.Sequential(
                    Rearrange('b ... d -> b (...) d'),
                    nn.Linear(h_dim, next_h_dim * next_seq_len),
                    Rearrange('b m (n d) -> (b m) n d', n = next_seq_len)
                )

            self.to_next_transformer_projections.append(proj)

        self.to_logits = nn.Linear(fine_dim, num_tokens)
        self.pad_id = pad_id

    def generate(self, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        total_seq_len = reduce_mult(self.max_seq_len)
        device = next(self.parameters()).device

        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        seq = prime
        batch = seq.shape[0]

        for _ in tqdm(range(total_seq_len - seq.shape[-1])):
            logits = self.forward(seq)[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)

        return seq.reshape(batch, *self.max_seq_len)

    def forward_empty(self, batch_size):
        # take care of special case
        # where you sample from input of 0 (start token only)

        prev_stage_tokens_repr = None

        for stage_start_tokens, transformer, proj in zip(self.start_tokens, self.transformers, self.to_next_transformer_projections):
            tokens = repeat(stage_start_tokens, 'd -> b 1 d', b = batch_size)

            if exists(prev_stage_tokens_repr):
                tokens = tokens + prev_stage_tokens_repr[..., :tokens.shape[-2], :]

            tokens = transformer(tokens)
            prev_stage_tokens_repr = proj(tokens)

        return self.to_logits(tokens)

    def forward(self, ids, return_loss = False):
        batch = ids.shape[0]

        assert ids.ndim in {2, self.stages + 1}
        flattened_dims = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dims:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_seq_len[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = ids.reshape(batch, -1, *self.max_seq_len[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        # check some dimensions

        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'

        # get tokens for all hierarchical stages, reducing by appropriate dimensions
        # and adding the absolute positional embeddings

        tokens_at_stages = []
        pos_embs = default(self.pos_embs, (None,))

        for ind, pos_emb, token_emb in zip_longest(range(len(prec_dims)), pos_embs, self.token_embs):
            is_first = ind == 0

            tokens = token_emb(ids)

            if exists(pos_emb):
                positions = pos_emb(torch.arange(tokens.shape[-2], device = device))
                tokens = tokens + positions

            tokens_at_stages.insert(0, tokens)

            if is_first:
                continue

            ids = rearrange(ids, '... m n -> ... (m n)')

        # the un-pixelshuffled representations of the previous hierarchy, starts with None

        prev_stage_tokens_repr = None

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions        

        for stage_start_tokens, stage_tokens, transformer, proj in zip(self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections):
            stage_tokens, ps = pack_one(stage_tokens, '* n d')
            stage_start_tokens = repeat(stage_start_tokens, 'f -> b 1 f', b = stage_tokens.shape[0])

            # concat start token

            stage_tokens = torch.cat((
                stage_start_tokens,
                stage_tokens,
            ), dim = -2)

            # sum the previous hierarchy's representation

            if exists(prev_stage_tokens_repr):
                prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value = 0.)
                stage_tokens = stage_tokens + prev_stage_tokens_repr

            attended = transformer(stage_tokens)

            attended = unpack_one(attended, ps, '* n d')

            # project for next stage in the hierarchy

            prev_stage_tokens_repr = proj(attended[..., :-1, :])

        # project to logits

        logits = self.to_logits(attended)

        start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None))]
        start_tokens = rearrange(start_tokens, 'b d -> b 1 d')

        logits = logits[..., 1:, :]

        if not return_loss:

            if flattened_dims:
                logits = rearrange(logits, 'b ... c -> b (...) c')
                logits = logits[:, :seq_len]

            return logits

        logits = rearrange(logits, 'b ... c -> b (...) c')
        logits = torch.cat((start_tokens, logits), dim = -2)

        preds = rearrange(logits, 'b n c -> b c n')
        labels = rearrange(ids, 'b ... -> b (...)')

        loss = F.cross_entropy(
            preds[..., :-1],
            labels,
            ignore_index = self.pad_id
        )

        return loss

class DilatedMegabyte(nn.Module):

    @beartype
    def __init__(
        self,
        *,
        num_tokens,
        dim: Union[Tuple, int],
        depth: Tuple,
        max_seq_len: Tuple,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0,
        rel_pos_bias = True,
        dilation_rate = None,
        segment_size = None,
        use_xpos = False,
        use_rel_pos_bias = False,
        flash_attn = False
    ):
        super().__init__()

        # simplified configuration for each stage of the hierarchy
        # depth = (2, 2, 4) would translate to depth 2 at first stage, depth 2 second stage, depth 4 third
        # max_seq_len = (16, 8, 4) would translate to max sequence length of 16 at first stage, length of 8 at second stage, length of 4 for last

        assert isinstance(depth, tuple) and isinstance(max_seq_len, tuple)
        assert len(depth) == len(max_seq_len)

        self.stages = len(depth)
        dim = cast_tuple(dim, self.stages)

        assert len(dim) == self.stages

        coarsest_dim, *_, fine_dim = dim

        self.max_seq_len = max_seq_len

        self.start_tokens = nn.ParameterList([nn.Parameter(torch.randn(h_dim)) for h_dim, seq_len in zip(dim, max_seq_len)])
        self.pos_embs = nn.ModuleList([nn.Embedding(seq_len, h_dim) for h_dim, seq_len in zip(dim, max_seq_len)]) if pos_emb else None

        self.token_embs = nn.ModuleList([])

        patch_size = 1
        self.token_embs.append(nn.Embedding(num_tokens, fine_dim))

        for dim_out, seq_len in zip(reversed(dim[:-1]), reversed(max_seq_len[1:])):
            patch_size *= seq_len

            self.token_embs.append(nn.Sequential(
                nn.Embedding(num_tokens, fine_dim),
                Rearrange('... r d -> ... (r d)'),
                nn.LayerNorm(patch_size * fine_dim),
                nn.Linear(patch_size * fine_dim, dim_out),
                nn.LayerNorm(dim_out)
            ))

        self.transformers = nn.ModuleList([])
        self.to_next_transformer_projections = nn.ModuleList([])

        for h_dim, next_h_dim, stage_depth, next_seq_len in zip_longest(dim, dim[1:], depth, max_seq_len[1:]):
            self.transformers.append(DilatedTransformer(
                dim = h_dim,
                layers = stage_depth,
                dim_head = dim_head,
                heads = heads,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                ff_mult = ff_mult,
                rel_pos_bias = rel_pos_bias,
                dilation_rate = dilation_rate,
                segment_size = segment_size,
                use_xpos = use_xpos,
                use_rel_pos_bias = use_rel_pos_bias,
                flash_attn = flash_attn
            ))

            proj = nn.Identity()

            if exists(next_h_dim) and next_h_dim != dim:
                proj = nn.Sequential(
                    Rearrange('b ... d -> b (...) d'),
                    nn.Linear(h_dim, next_h_dim * next_seq_len),
                    Rearrange('b m (n d) -> (b m) n d', n = next_seq_len)
                )

            self.to_next_transformer_projections.append(proj)

        self.to_logits = nn.Linear(fine_dim, num_tokens)
        self.pad_id = pad_id

    def generate(self, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        total_seq_len = reduce_mult(self.max_seq_len)
        device = next(self.parameters()).device

        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        seq = prime
        batch = seq.shape[0]

        for _ in tqdm(range(total_seq_len - seq.shape[-1])):
            logits = self.forward(seq)[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)

        return seq.reshape(batch, *self.max_seq_len)

    def forward_empty(self, batch_size):
        # take care of special case
        # where you sample from input of 0 (start token only)

        prev_stage_tokens_repr = None

        for stage_start_tokens, transformer, proj in zip(self.start_tokens, self.transformers, self.to_next_transformer_projections):
            tokens = repeat(stage_start_tokens, 'd -> b 1 d', b = batch_size)

            if exists(prev_stage_tokens_repr):
                tokens = tokens + prev_stage_tokens_repr[..., :tokens.shape[-2], :]

            tokens = transformer(tokens)
            prev_stage_tokens_repr = proj(tokens)

        return self.to_logits(tokens)

    def forward(self, ids, return_loss = False):
        batch = ids.shape[0]

        assert ids.ndim in {2, self.stages + 1}
        flattened_dims = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dims:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_seq_len[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = ids.reshape(batch, -1, *self.max_seq_len[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        # check some dimensions

        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'

        # get tokens for all hierarchical stages, reducing by appropriate dimensions
        # and adding the absolute positional embeddings

        tokens_at_stages = []
        pos_embs = default(self.pos_embs, (None,))

        for ind, pos_emb, token_emb in zip_longest(range(len(prec_dims)), pos_embs, self.token_embs):
            is_first = ind == 0

            tokens = token_emb(ids)

            if exists(pos_emb):
                positions = pos_emb(torch.arange(tokens.shape[-2], device = device))
                tokens = tokens + positions

            tokens_at_stages.insert(0, tokens)

            if is_first:
                continue

            ids = rearrange(ids, '... m n -> ... (m n)')

        # the un-pixelshuffled representations of the previous hierarchy, starts with None

        prev_stage_tokens_repr = None

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions        

        for stage_start_tokens, stage_tokens, transformer, proj in zip(self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections):
            stage_tokens, ps = pack_one(stage_tokens, '* n d')
            stage_start_tokens = repeat(stage_start_tokens, 'f -> b 1 f', b = stage_tokens.shape[0])

            # concat start token

            stage_tokens = torch.cat((
                stage_start_tokens,
                stage_tokens,
            ), dim = -2)

            # sum the previous hierarchy's representation

            if exists(prev_stage_tokens_repr):
                prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value = 0.)
                stage_tokens = stage_tokens + prev_stage_tokens_repr

            attended = transformer(stage_tokens)

            attended = unpack_one(attended, ps, '* n d')

            # project for next stage in the hierarchy

            prev_stage_tokens_repr = proj(attended[..., :-1, :])

        # project to logits

        logits = self.to_logits(attended)

        start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None))]
        start_tokens = rearrange(start_tokens, 'b d -> b 1 d')

        logits = logits[..., 1:, :]

        if not return_loss:

            if flattened_dims:
                logits = rearrange(logits, 'b ... c -> b (...) c')
                logits = logits[:, :seq_len]

            return logits

        logits = rearrange(logits, 'b ... c -> b (...) c')
        logits = torch.cat((start_tokens, logits), dim = -2)

        preds = rearrange(logits, 'b n c -> b c n')
        labels = rearrange(ids, 'b ... -> b (...)')

        loss = F.cross_entropy(
            preds[..., :-1],
            labels,
            ignore_index = self.pad_id
        )

        return loss
    
class DilatedOmniMegabyte(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim: Union[Tuple, int],
        depth: Tuple,
        max_seq_len: Tuple,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0,
        rel_pos_bias = True,
        dilation_rate = None,
        segment_size = None,
        use_xpos = False,
        use_rel_pos_bias = False,
        flash_attn = False
    ):
        super().__init__()

        # simplified configuration for each stage of the hierarchy
        # depth = (2, 2, 4) would translate to depth 2 at first stage, depth 2 second stage, depth 4 third
        # max_seq_len = (16, 8, 4) would translate to max sequence length of 16 at first stage, length of 8 at second stage, length of 4 for last

        assert isinstance(depth, tuple) and isinstance(max_seq_len, tuple)
        assert len(depth) == len(max_seq_len)

        self.stages = len(depth)
        dim = cast_tuple(dim, self.stages)

        assert len(dim) == self.stages

        coarsest_dim, *_, fine_dim = dim

        self.token_emb = nn.Embedding(num_tokens, fine_dim)

        self.max_seq_len = max_seq_len

        self.start_tokens = nn.ParameterList([nn.Parameter(torch.randn(h_dim)) for h_dim, seq_len in zip(dim, max_seq_len)])
        self.pos_embs = nn.ModuleList([nn.Embedding(seq_len, h_dim) for h_dim, seq_len in zip(dim, max_seq_len)])

        #v2
        input_dims = (dim[1], dim[0], max_seq_len[1])
        self.patch_embedders = UniversalPatchEmbedder(input_dims, dim[0], max_seq_len[1])

        #------->

        self.transformers = nn.ModuleList([])
        self.to_next_transformer_projections = nn.ModuleList([])

        for h_dim, next_h_dim, stage_depth, next_seq_len in zip_longest(dim, dim[1:], depth, max_seq_len[1:]):
            self.transformers.append(DilatedTransformer(
                dim = h_dim,
                layers = stage_depth,
                dim_head = dim_head,
                heads = heads,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                ff_mult = ff_mult,
                rel_pos_bias = rel_pos_bias,
                dilation_rate = dilation_rate,
                segment_size = segment_size,
                use_xpos = use_xpos,
                use_rel_pos_bias = use_rel_pos_bias,
                flash_attn = flash_attn
            ))

            proj = nn.Identity()

            if exists(next_h_dim) and next_h_dim != dim:
                proj = nn.Sequential(
                    nn.Linear(h_dim, next_h_dim * next_seq_len),
                    Rearrange('... (n d) -> (...) n d', n = next_seq_len)
                )

            self.to_next_transformer_projections.append(proj)

        self.to_logits = nn.Linear(fine_dim, num_tokens)
        self.pad_id = pad_id

    def generate(self, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        total_seq_len = reduce_mult(self.max_seq_len)
        device = next(self.parameters()).device

        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        seq = prime
        batch = seq.shape[0]

        for _ in tqdm(range(total_seq_len - seq.shape[-1])):
            logits = self.forward(seq)[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)

        return seq.reshape(batch, *self.max_seq_len)

    def forward_empty(self, batch_size):
        # take care of special case
        # where you sample from input of 0 (start token only)

        prev_stage_tokens_repr = None

        for stage_start_tokens, transformer, proj in zip(self.start_tokens, self.transformers, self.to_next_transformer_projections):
            tokens = repeat(stage_start_tokens, 'd -> b 1 d', b = batch_size)

            if exists(prev_stage_tokens_repr):
                tokens = tokens + prev_stage_tokens_repr[..., :tokens.shape[-2], :]

            tokens = transformer(tokens)
            prev_stage_tokens_repr = proj(tokens)

        return self.to_logits(tokens)

    def forward(self, ids, modality,  return_loss = False):
        batch = ids.shape[0]

        print(f'ids shape: {ids.shape[0]}')

        assert ids.ndim in {2, self.stages + 1}
        print(f"self stages: {self.stages}")

        flattened_dims = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dims:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_seq_len[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = ids.reshape(batch, -1, *self.max_seq_len[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        # check some dimensions

        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'

        # get token embeddings

        tokens = self.token_emb(ids)

        # get tokens for all hierarchical stages, reducing by appropriate dimensions
        # and adding the absolute positional embeddings

        tokens_at_stages = []
        reduced_tokens = tokens

        patch_embedders_list = [self.patch_embedders]

        for ind, pos_emb, patch_emb in zip(range(len(prec_dims)), reversed(self.pos_embs), reversed(patch_embedders_list)):
            is_first = ind == 0

            if not is_first:
                reduced_tokens = patch_emb(reduced_tokens, modality)

            positions = pos_emb(torch.arange(reduced_tokens.shape[-2], device=device))
            tokens_with_position = reduced_tokens + positions
            tokens_at_stages.insert(0, tokens_with_position)

        # the un-pixelshuffled representations of the previous hierarchy, starts with None

        prev_stage_tokens_repr = None

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions        

        for stage_start_tokens, stage_tokens, transformer, proj in zip(self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections):
            stage_tokens, ps = pack_one(stage_tokens, '* n d')

            stage_start_tokens = repeat(stage_start_tokens, 'f -> b 1 f', b=stage_tokens.shape[0])

            #update the dimensions of the stage_start_tokens tensor
            stage_start_tokens = stage_start_tokens[..., :stage_tokens.shape[-1]]

            # Print the shapes of the tensors before concatenating
            print(f"stage_start_tokens shape: {stage_start_tokens.shape}")
            print(f"stage_tokens shape: {stage_tokens.shape}")

            # concat start token

            stage_tokens = torch.cat((
                stage_start_tokens,
                stage_tokens,
            ), dim=-2)

            # sum the previous hierarchy's representation

            if exists(prev_stage_tokens_repr):
                prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value=0.)
                stage_tokens = stage_tokens + prev_stage_tokens_repr

            attended = transformer(stage_tokens)

            attended = unpack_one(attended, ps, '* n d')

            # project for next stage in the hierarchy

            prev_stage_tokens_repr = proj(attended[..., :-1, :])

        # project to logits

        logits = self.to_logits(attended)

        start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None))]
        start_tokens = rearrange(start_tokens, 'b d -> b 1 d')

        logits = logits[..., 1:, :]

        if not return_loss:

            if flattened_dims:
                logits = rearrange(logits, 'b ... c -> b (...) c')
                logits = logits[:, :seq_len]

            return logits

        logits = rearrange(logits, 'b ... c -> b (...) c')
        logits = torch.cat((start_tokens, logits), dim = -2)

        preds = rearrange(logits, 'b n c -> b c n')
        labels = rearrange(ids, 'b ... -> b (...)')

        loss = F.cross_entropy(
            preds[..., :-1],
            labels,
            ignore_index = self.pad_id
        )

        return loss













#==========================> v2 
# class Attention(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         dim_head=64,
#         heads=8,
#         dropout=0.,
#         flash=False
#     ):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         inner_dim = dim_head * heads

#         self.attend = Attend(
#             causal=True,
#             flash=flash,
#             dropout=dropout
#         )

#         self.dropout = nn.Dropout(dropout)
#         self.norm = RMSNorm(dim)
#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)

#     def forward(self, x, attn_bias=None):
#         h, device = self.heads, x.device

#         x = self.norm(x)
#         q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
#         q = rearrange(q, 'b n (h d) -> b h n d', h=h)

#         out = self.attend(q, k, v, attn_bias=attn_bias)

#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

# class Transformer(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         layers,
#         dim_head=64,
#         heads=8,
#         attn_dropout=0.,
#         ff_mult=4,
#         ff_dropout=0.,
#         rel_pos_bias=True,
#         flash_attn=False
#     ):
#         super().__init__()
#         self.alibi = Alibi(heads=heads) if rel_pos_bias else None
#         self.layers = nn.ModuleList([])

#         for _ in range(layers):
#             self.layers.append(nn.ModuleList([
#                 Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn),
#                 FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
#             ]))

#         self.norm = RMSNorm(dim)

#     def forward(self, x):
#         n = x.shape[-2]
#         attn_bias = self.alibi(n, n, device=x.device) if exists(self.alibi) else None

#         for attn, ff in self.layers:
#             x = attn(token_shift(x), attn_bias=attn_bias) + x
#             x = ff(token_shift(x)) + x

#         return self.norm(x)


# class UniversalPatchEmbedder(nn.Module):
#     def __init__(self, input_dims: Tuple[Tuple[int, int], ...], output_dim: int, patch_size: int):
#         super().__init__()
#         self.patch_size = patch_size
#         self.embedders = nn.ModuleList([nn.Linear(dim[0] * dim[1], output_dim) for dim in input_dims if dim is not None])
#         self.modality_embeddings = nn.Embedding(len(input_dims), output_dim)

#     def forward(self, x: Tensor, modality: Tensor) -> Tensor:
#         input_shape = x.shape
#         x = x.view(input_shape[0], -1)
#         embedder = self.embedders[modality]
#         x = rearrange(x, 'b (p1 p2) d -> b p1 p2 d', p1=self.patch_size)
#         x = embedder(x)
#         modality_emb = self.modality_embeddings(torch.tensor(modality).to(x.device))
#         x = x + modality_emb
#         return x


# class ModalityEmbedder(nn.Module):
#     def __init__(self, input_dim: int, output_dim: int):
#         super().__init__()
#         self.embedding = nn.Linear(input_dim, output_dim)

#     def forward(self, x: Tensor) -> Tensor:
#         return self.embedding(x)


# class ModalityPositionalEmbedding(nn.Module):
#     def __init__(self, seq_len: int, dim: int):
#         super().__init__()
#         self.embedding = nn.Embedding(seq_len, dim)

#     def forward(self, x: Tensor) -> Tensor:
#         return self.embedding(x)


# class MEGABYTE(nn.Module):
#     def __init__(
#         self,
#         *,
#         num_tokens,
#         dim: Union[Tuple, int],
#         depth: Tuple,
#         max_seq_len: Tuple,
#         dim_head=64,
#         heads=8,
#         attn_dropout=0.,
#         ff_mult=4,
#         ff_dropout=0.,
#         pad_id=0,
#         rel_pos_bias=True,
#         flash_attn=False
#     ):
#         super().__init__()

#         assert isinstance(depth, tuple) and isinstance(max_seq_len, tuple)
#         assert len(depth) == len(max_seq_len)

#         self.stages = len(depth)
#         dim = cast_tuple(dim, self.stages)

#         assert len(dim) == self.stages

#         coarsest_dim, *_, fine_dim = dim

#         self.token_emb = nn.Embedding(num_tokens, fine_dim)

#         self.max_seq_len = max_seq_len

#         self.start_tokens = nn.ParameterList([nn.Parameter(torch.randn(h_dim)) for h_dim, seq_len in zip(dim, max_seq_len)])
#         self.pos_embs = nn.ModuleList([nn.Embedding(seq_len, h_dim) for h_dim, seq_len in zip(dim, max_seq_len)])

#         input_dims = (dim[1:], dim[0], max_seq_len[1])
#         self.patch_embedders = UniversalPatchEmbedder(input_dims, dim[0], max_seq_len[1])

#         self.transformers = nn.ModuleList([])
#         self.to_next_transformer_projections = nn.ModuleList([])

#         for h_dim, next_h_dim, stage_depth, next_seq_len in zip_longest(dim, dim[1:], depth, max_seq_len[1:]):
#             self.transformers.append(Transformer(
#                 dim=h_dim,
#                 layers=stage_depth,
#                 dim_head=dim_head,
#                 heads=heads,
#                 attn_dropout=attn_dropout,
#                 ff_dropout=ff_dropout,
#                 ff_mult=ff_mult,
#                 rel_pos_bias=rel_pos_bias,
#                 flash_attn=flash_attn
#             ))

#             proj = nn.Identity()

#             if exists(next_h_dim) and next_h_dim != dim:
#                 proj = nn.Sequential(
#                     nn.Linear(h_dim, next_h_dim * next_seq_len),
#                     Rearrange('... (n d) -> (...) n d', n=next_seq_len)
#                 )

#             self.to_next_transformer_projections.append(proj)

#         self.to_logits = nn.Linear(fine_dim, num_tokens)
#         self.pad_id = pad_id

#     def generate(self, prime=None, filter_thres=0.9, temperature=1., default_batch_size=1):
#         total_seq_len = reduce_mult(self.max_seq_len)
#         device = next(self.parameters()).device

#         if not exists(prime):
#             prime = torch.empty((default_batch_size, 0), dtype=torch.long, device=device)

#         seq = prime
#         batch = seq.shape[0]

#         for _ in tqdm(range(total_seq_len - seq.shape[-1])):
#             logits = self.forward(seq)[:, -1]
#             logits = top_k(logits, thres=filter_thres)
#             sampled = gumbel_sample(logits, dim=-1, temperature=temperature)
#             seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim=-1)

#         return seq.reshape(batch, *self.max_seq_len)

#     def forward_empty(self, batch_size):
#         prev_stage_tokens_repr = None

#         for stage_start_tokens, transformer, proj in zip(self.start_tokens, self.transformers, self.to_next_transformer_projections):
#             tokens = repeat(stage_start_tokens, 'd -> b 1 d', b=batch_size)

#             if exists(prev_stage_tokens_repr):
#                 tokens = tokens + prev_stage_tokens_repr[..., :tokens.shape[-2], :]

#             tokens = transformer(tokens)
#             prev_stage_tokens_repr = proj(tokens)

#         return self.to_logits(tokens)

#     def forward(self, ids, modality, return_loss=False):
#         batch = ids.shape[0]

#         assert ids.ndim in {2, self.stages + 1}

#         flattened_dims = ids.ndim == 2
#         ids_orig_ndim = ids.ndim

#         if ids.numel() == 0:
#             return self.forward_empty(ids.shape[0])

#         if flattened_dims:
#             seq_len = ids.shape[-1]
#             multiple_of = reduce_mult(self.max_seq_len[1:])
#             padding = remainder_to_mult(seq_len, multiple_of)
#             ids = F.pad(ids, (0, padding), value=self.pad_id)
#             ids = ids.reshape(batch, -1, *self.max_seq_len[1:])

#         b, *prec_dims, device = *ids.shape, ids.device

#         tokens = self.token_emb(ids)

#         tokens_at_stages = []
#         reduced_tokens = tokens

#         patch_embedders_list = [self.patch_embedders]

#         for ind, pos_emb, patch_emb in zip(range(len(prec_dims)), reversed(self.pos_embs), reversed(patch_embedders_list)):
#             is_first = ind == 0

#             if not is_first:
#                 reduced_tokens = patch_emb(reduced_tokens, modality)

#             positions = pos_emb(torch.arange(reduced_tokens.shape[-2], device=device))
#             tokens_with_position = reduced_tokens + positions
#             tokens_at_stages.insert(0, tokens_with_position)

#         prev_stage_tokens_repr = None

#         for stage_start_tokens, stage_tokens, transformer, proj in zip(self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections):
#             stage_tokens, ps = pack_one(stage_tokens, '* n d')

#             stage_start_tokens = repeat(stage_start_tokens, 'f -> b 1 f', b=stage_tokens.shape[0])
#             stage_start_tokens = stage_start_tokens[..., :stage_tokens.shape[-1]]

#             stage_tokens = torch.cat((
#                 stage_start_tokens,
#                 stage_tokens,
#             ), dim=-2)

#             if exists(prev_stage_tokens_repr):
#                 prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value=0.)
#                 stage_tokens = stage_tokens + prev_stage_tokens_repr

#             attended = transformer(stage_tokens)

#             attended = unpack_one(attended, ps, '* n d')

#             prev_stage_tokens_repr = proj(attended[..., :-1, :])

#         logits = self.to_logits(attended)

#         start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None))]
#         start_tokens = rearrange(start_tokens, 'b d -> b 1 d')

#         logits = logits[..., 1:, :]

#         if not return_loss:

#             if flattened_dims:
#                 logits = rearrange(logits, 'b ... c -> b (...) c')
#                 logits = logits[:, :seq_len]

#             return logits

#         logits = rearrange(logits, 'b ... c -> b (...) c')
#         logits = torch.cat((start_tokens, logits), dim=-2)

#         preds = rearrange(logits, 'b n c -> b c n')
#         labels = rearrange(ids, 'b ... -> b (...)')

#         loss = F.cross_entropy(
#             preds[..., :-1],
#             labels,
#             ignore_index=self.pad_id
#         )

#         return loss