# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
from timm.models.layers.helpers import to_2tuple


def _get_conv2d_weights(in_channels, out_channels, kernel_size):
    weight = torch.empty(out_channels, in_channels, *kernel_size)
    return weight


def _get_conv2d_biases(out_channels):
    bias = torch.empty(out_channels)
    return bias



class ParallelVarPatchEmbed(nn.Module):
    """Variable to Patch Embedding with multiple variables in a single kernel."""

    def __init__(self, max_vars: int, img_size, patch_size, embed_dim, norm_layer=None, flatten=True):
        super().__init__()
        self.max_vars = max_vars
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        grouped_weights = torch.stack(
            [_get_conv2d_weights(1, embed_dim, self.patch_size) for _ in range(max_vars)], dim=0
        )
        self.proj_weights = nn.Parameter(grouped_weights)
        grouped_biases = torch.stack([_get_conv2d_biases(embed_dim) for _ in range(max_vars)], dim=0)
        self.proj_biases = nn.Parameter(grouped_biases)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(self.max_vars):
            nn.init.kaiming_uniform_(self.proj_weights[idx], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.proj_weights[idx])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.proj_biases[idx], -bound, bound)

    def forward(self, x, vars=None):
        B, C, H, W = x.shape
        if vars is None:
            vars = range(self.max_vars)
        weights = self.proj_weights[vars].flatten(0, 1)
        biases = self.proj_biases[vars].flatten(0, 1)

        groups = len(vars)
        proj = F.conv2d(x, weights, biases, groups=groups, stride=self.patch_size)
        if self.flatten:
            proj = proj.reshape(B, groups, -1, *proj.shape[-2:])
            proj = proj.flatten(3).transpose(2, 3)

        proj = self.norm(proj)
        return proj


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)  # 修改这里
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    grid_h = np.arange(grid_size_h, dtype=float)  # 修改这里
    grid_w = np.arange(grid_size_w, dtype=float)  # 修改这里
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class ClimaX(nn.Module):
    def __init__(
        self,
        in_chans,
        img_size=[32, 64],
        patch_size=6,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        upscaling_factor=4,
        parallel_patch_embed=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.parallel_patch_embed = parallel_patch_embed

        if self.parallel_patch_embed:
            self.token_embeds = ParallelVarPatchEmbed(in_chans, img_size, patch_size, embed_dim)
            self.num_patches = self.token_embeds.num_patches
        else:
            self.token_embeds = nn.ModuleList(
                [PatchEmbed(img_size, patch_size, 1, embed_dim) for _ in range(in_chans)]
            )
            self.num_patches = self.token_embeds[0].num_patches

        self.var_embed = nn.Parameter(torch.zeros(1, in_chans, embed_dim), requires_grad=True)
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop_path=dpr[i], norm_layer=nn.LayerNorm, drop=drop_rate)
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, in_chans * patch_size**2))
        self.head = nn.Sequential(*self.head)
        self.to_img = nn.Sequential(
            nn.Conv2d(in_chans, in_chans * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(self.in_chans))
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        if self.parallel_patch_embed:
            for i in range(len(self.token_embeds.proj_weights)):
                w = self.token_embeds.proj_weights[i].data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        else:
            for i in range(len(self.token_embeds)):
                w = self.token_embeds[i].proj.weight.data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        p = self.patch_size
        c = self.in_chans
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)
        x = x.squeeze()
        x = x.unflatten(dim=0, sizes=(b, l))
        return x

    def forward_encoder(self, x: torch.Tensor):
        if self.parallel_patch_embed:
            var_ids = torch.arange(self.in_chans, device=x.device)
            x = self.token_embeds(x, var_ids)
        else:
            embeds = [self.token_embeds[i](x[:, i : i + 1]) for i in range(self.in_chans)]
            x = torch.stack(embeds, dim=1)

        x = x + self.var_embed.unsqueeze(2)
        x = self.aggregate_variables(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    

    def forward(self, x):
        out_transformers = self.forward_encoder(x)
        preds = self.head(out_transformers)
        preds = self.unpatchify(preds)
        preds = self.to_img(preds)
        return preds
    
def create_net():
    net = ClimaX(in_chans=10,img_size=[180,360])
    return net

if __name__ == "__main__":
    input = torch.randn(1, 1, 32, 64)
    model = ClimaX(in_chans=1)
    y = model(input)
    print(y.shape)
