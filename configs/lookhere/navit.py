from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from .vision_transformer import VisionTransformer
from mmseg.registry import MODELS
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.logging import MMLogger, print_log
import math

@MODELS.register_module()
class NaViT(VisionTransformer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.pos_embed_height = nn.Parameter(torch.randn(self.grid_size[0], self.embed_dim))
        self.pos_embed_width = nn.Parameter(torch.randn(self.grid_size[1], self.embed_dim))
        pos = torch.stack(
            torch.meshgrid(
                (torch.arange(self.grid_size[0]), torch.arange(self.grid_size[1])), indexing="ij"
            ),
            dim=-1,
        )
        patch_positions = rearrange(pos, "h w c -> (h w) c")
        self.h_indices, self.w_indices = patch_positions.unbind(dim=-1)

    def forward_features(self, x):
        x = self.patch_embed(x)
        # add position embeddings
        h_pos = self.pos_embed_height[self.h_indices,...]
        w_pos = self.pos_embed_width[self.w_indices,...]
        x = x + h_pos + w_pos
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x
    
    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if not (self.pretrained is None):
            checkpoint = CheckpointLoader.load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            new_pos_embed_height = resample_1d_pos_embed(
                posemb=checkpoint['pos_embed_height'],
                new_size=self.grid_size[0]
            )
            new_pos_embed_width = resample_1d_pos_embed(
                    posemb=checkpoint['pos_embed_width'],
                    new_size=self.grid_size[1]
                )
            checkpoint['pos_embed_height'] = new_pos_embed_height
            checkpoint['pos_embed_width'] = new_pos_embed_width
            load_state_dict(self, checkpoint, strict=False, logger=logger)
        else:
            self.apply(self._init_weights)

def resample_1d_pos_embed(
    posemb,
    new_size
):
    if posemb.shape[1] == new_size:
        return posemb

    # do the interpolation
    posemb = rearrange(posemb, "w c -> () c w")
    posemb = F.interpolate(posemb, size=new_size, mode='linear')
    posemb = rearrange(posemb, "() c w -> w c")

    return posemb