from typing import Callable, List, Optional, Sequence, Tuple, Union
from scipy import interpolate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from .vision_transformer import Attention, Block, VisionTransformer
from mmseg.registry import MODELS
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.logging import MMLogger

class AttentionWithBEIT(Attention):
    def __init__(self, grid_size, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.num_relative_distance = (2 * self.grid_size[0] - 1) * (
            2 * self.grid_size[1] - 1
        ) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        self.register_buffer(
            "relative_position_index",
            gen_relative_position_index(self.grid_size),
            persistent=False,
        )

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.grid_size[0] * self.grid_size[1] + 1,
            self.grid_size[0] * self.grid_size[1] + 1,
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        rel_pos_bias = self._get_rel_pos_bias()

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=rel_pos_bias,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class BlockwithBEIT(Block):
    def __init__(self, grid_size, **kwargs):
        super().__init__(**kwargs)
        self.attn = AttentionWithBEIT(grid_size,
            dim=kwargs["dim"],
            num_heads=kwargs["num_heads"],
        )

@MODELS.register_module()
class BEIT(VisionTransformer):
    def __init__(self, **kwargs):
        kwargs["block_fn"] = lambda **kwargs: BlockwithBEIT(self.grid_size, **kwargs)
        super().__init__(**kwargs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x
    
    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if not (self.pretrained is None):
            checkpoint = CheckpointLoader.load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            for i, block in enumerate(self.blocks):
                new_embeds = resize_rel_pos_bias_table(
                    checkpoint[f'blocks.{i}.attn.relative_position_bias_table'],  # old relative bias embeddings
                    new_window_size=self.grid_size,
                    new_bias_shape=block.attn.relative_position_bias_table.shape,
                )
                checkpoint[f'blocks.{i}.attn.relative_position_bias_table'] = new_embeds
 
            load_state_dict(self, checkpoint, strict=False, logger=logger)
        else:
            self.apply(self._init_weights)

def resize_rel_pos_bias_table(
    rel_pos_bias,
    new_window_size: Tuple[int, int],
    new_bias_shape: Tuple[int, ...],
):
    """Resize relative position bias table using more advanced interpolation.

    Modified from code in Microsoft Unilm (https://github.com/microsoft/unilm) repo (BeiT, BeiT-v2, etc).

    https://github.com/microsoft/unilm/blob/5255d52de86dad642810f5849dd357769346c1d7/beit/run_class_finetuning.py#L351

    Args:
        rel_pos_bias:
        new_window_size:
        new_bias_shape:

    Returns:

    """

    dst_size = (new_window_size[0] * 2 - 1, new_window_size[1] * 2 - 1)
    if rel_pos_bias.ndim == 3:
        # TF maxvit style (num_heads, H, W) bias shape, no extra tokens currently supported
        num_extra_tokens = 0
        _, dst_h, dst_w = new_bias_shape
        assert dst_h == dst_size[0] and dst_w == dst_size[1]
        num_attn_heads, src_h, src_w = rel_pos_bias.shape
        src_size = (src_h, src_w)
        has_flat_shape = False
    else:
        assert rel_pos_bias.ndim == 2
        # (num_pos, num_heads) (aka flat) bias shape
        dst_num_pos, _ = new_bias_shape
        src_num_pos, num_attn_heads = rel_pos_bias.shape
        num_extra_tokens = dst_num_pos - (dst_size[0] * dst_size[1])
        src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
        src_size = (src_size, src_size)
        has_flat_shape = True

    if src_size[0] != dst_size[0] or src_size[1] != dst_size[1]:
        # print("Interpolating position from %dx%d to %dx%d" % (src_size[0], src_size[1], dst_size[0], dst_size[1]))
        if num_extra_tokens:
            extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
            rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
        else:
            extra_tokens = None

        def geometric_progression(a, r, n):
            return a * (1.0 - r**n) / (1.0 - r)

        def _calc(src, dst):
            left, right = 1.01, 1.5
            while right - left > 1e-6:
                q = (left + right) / 2.0
                gp = geometric_progression(1, q, src // 2)
                if gp > dst // 2:
                    right = q
                else:
                    left = q

            dis = []
            cur = 1
            for i in range(src // 2):
                dis.append(cur)
                cur += q ** (i + 1)
            r_ids = [-_ for _ in reversed(dis)]
            return r_ids + [0] + dis

        y = _calc(src_size[0], dst_size[0])
        x = _calc(src_size[1], dst_size[1])
        yx = [torch.tensor(y), torch.tensor(x)]
        # print("Original positions = %s" % str(x))

        ty = dst_size[0] // 2.0
        tx = dst_size[1] // 2.0
        dy = torch.arange(-ty, ty + 0.1, 1.0)
        dx = torch.arange(-tx, tx + 0.1, 1.0)
        dyx = torch.meshgrid([dy, dx])
        # print("Target positions = %s" % str(dx))

        all_rel_pos_bias = []
        for i in range(num_attn_heads):
            if has_flat_shape:
                z = rel_pos_bias[:, i].view(src_size[0], src_size[1]).float()
            else:
                z = rel_pos_bias[i, :, :].float()

            # Original beit code uses scipy w/ cubic interpolation
            f = interpolate.interp2d(x, y, z.cpu().numpy(), kind="cubic")
            r = torch.Tensor(f(dx, dy)).contiguous().to(rel_pos_bias.device)

            if has_flat_shape:
                r = r.view(-1, 1)
            all_rel_pos_bias.append(r)

        if has_flat_shape:
            rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
        else:
            rel_pos_bias = torch.cat(all_rel_pos_bias, dim=0)

        if extra_tokens is not None:
            assert has_flat_shape
            rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)

    return rel_pos_bias


def gen_relative_position_index(window_size: Tuple[int, int]) -> torch.Tensor:
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    # cls to token & token 2 cls & cls to cls
    # get pair-wise relative position index for each token inside the window
    window_area = window_size[0] * window_size[1]
    coords = torch.stack(
        torch.meshgrid([torch.arange(window_size[0]), torch.arange(window_size[1])])
    )  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = (
        coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = torch.zeros(
        size=(window_area + 1,) * 2, dtype=relative_coords.dtype
    )
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


