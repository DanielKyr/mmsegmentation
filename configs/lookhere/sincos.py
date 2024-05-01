import torch
import numpy as np
import torch.nn.functional as F
from .vision_transformer import VisionTransformer, resample_abs_pos_embed
from einops import rearrange
from mmseg.registry import MODELS
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.logging import MMLogger, print_log
import math

@MODELS.register_module()
class SinCos(VisionTransformer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.pos_embed.requires_grad = False  # fixed
    
    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if not (self.pretrained is None):
            checkpoint = CheckpointLoader.load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            new_position_embeddings = resample_abs_pos_embed(
                    posemb=checkpoint['pos_embed'],
                    new_size=self.grid_size,
                    old_size=(14,14),
                )
            checkpoint['pos_embed'] = new_position_embeddings
            load_state_dict(self, checkpoint, strict=False, logger=logger)
        else:
            self.apply(self._init_weights)
            pos_embed = get_2d_sincos_pos_embed(self.num_features, self.grid_size)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
            self.pos_embed.data = pos_embed

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb