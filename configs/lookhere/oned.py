from typing import Any, Mapping
import torch
import torch.nn as nn
from .vision_transformer import VisionTransformer, resample_abs_pos_embed
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.logging import MMLogger,print_log
import math

@MODELS.register_module()
class OneD(VisionTransformer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

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

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x