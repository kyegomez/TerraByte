
from itertools import zip_longest
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Tuple, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from tqdm import tqdm

from terra_byte.model.helpers import (
    cast_tuple,
    exists,
    gumbel_sample,
    pack_one,
    reduce_mult,
    remainder_to_mult,
    top_k,
    unpack_one,
)
from terra_byte.model.patches import UniversalPatchEmbedder
from terra_byte.model.transformer import Transformer


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

