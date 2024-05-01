import torch
import torch.nn as nn
import torch.utils.checkpoint
from .vision_transformer import VisionTransformer
from einops import rearrange, repeat
import numpy as np
from mmseg.registry import MODELS
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.logging import MMLogger

class LearnableFourierFeatures(nn.Module):
    def __init__(self, f_dim, h_dim, d_dim):
        super(LearnableFourierFeatures, self).__init__()

        enc_f_dim = int(f_dim / 2)
        self.Wr = nn.Linear(2, enc_f_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(f_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, d_dim)
        )
        self.div_term = np.sqrt(f_dim)

    def forward(self, pos):
        XWr = self.Wr(pos)
        F = torch.cat([torch.cos(XWr), torch.sin(XWr)], dim=-1) / self.div_term
        return self.mlp(F)

@MODELS.register_module()
class LFFViT(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lff = LearnableFourierFeatures(f_dim=128, h_dim=256, d_dim=768)

        h_pos = torch.arange(self.grid_size[0], dtype=torch.float16) / self.grid_size[0]
        w_pos = torch.arange(self.grid_size[1], dtype=torch.float16) / self.grid_size[1]

        all_h_pos = repeat(h_pos, 'h -> (h w)', w=self.grid_size[1])
        all_w_pos = repeat(w_pos, 'w -> (h w)', h=self.grid_size[0])

        all_pos = torch.stack([all_h_pos, all_w_pos], dim=-1)
        self.all_pos = rearrange(all_pos, 'l d -> 1 l d')


    def forward_features(self, x):
        x = self.patch_embed(x)
        # add position embeddings
        lff_embeds = self.lff(self.all_pos.repeat(x.shape[0], 1, 1).type_as(x))
        x = x + lff_embeds
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if not (self.pretrained is None):
            checkpoint = CheckpointLoader.load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            load_state_dict(self, checkpoint, strict=False, logger=logger)
        else:
            self.apply(self._init_weights)