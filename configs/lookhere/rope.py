import torch
from .vision_transformer import Attention, Block, VisionTransformer
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.logging import MMLogger

class RoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, D, 2).float().to(device) / D)
            )
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert (
            tokens.size(3) % 2 == 0
        ), "number of dimensions should be a multiple of two"
        D = tokens.size(3) // 2
        assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2
        cos, sin = self.get_cos_sin(
            D, int(positions.max()) + 1, tokens.device, tokens.dtype
        )
        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        tokens = torch.cat((y, x), dim=-1)
        return tokens


class PositionGetter(object):
    """return positions of patches"""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if not (h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h, w] = torch.cartesian_prod(y, x)  # (h, w, 2)
        pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()
        return pos


class AttentionWithRoPE(Attention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, pos, rope_fn):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q_cls = q[:, :, 0, :].unsqueeze(dim=-2)  # (bsz, num_heads, 1, head_dim)
        k_cls = k[:, :, 0, :].unsqueeze(dim=-2)  # (bsz, num_heads, 1, head_dim)

        q = rope_fn(q[:, :, 1:, :], pos)  # (bsz, num_heads, grid_size^2, head_dim)
        k = rope_fn(k[:, :, 1:, :], pos)  # (bsz, num_heads, grid_size^2, head_dim)
        q = torch.cat([q_cls, q], dim=2)  # (bsz, num_heads, grid_size^2+1, head_dim)
        k = torch.cat([k_cls, k], dim=2)  # (bsz, num_heads, grid_size^2+1, head_dim)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class BlockwithRoPE(Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.attn = AttentionWithRoPE(
            dim=kwargs["dim"],
            num_heads=kwargs["num_heads"],
        )

    def forward(self, x, pos, rope_fn):
        x = x + self.drop_path1(self.ls1((self.attn(self.norm1(x), pos, rope_fn))))
        x = x + self.drop_path2(self.ls2((self.mlp(self.norm2(x)))))
        return x

@MODELS.register_module()
class RoPEViT(VisionTransformer):
    def __init__(self, **kwargs):
        kwargs["block_fn"] = BlockwithRoPE

        super().__init__(**kwargs)
        self.position_getter = PositionGetter()
        self.rope = RoPE2D()

    def forward_features(self, x):
        x = self.patch_embed(x)
        h = w = int(x.shape[1] ** 0.5)
        pos = self.position_getter(x.shape[0], h, w, x.device)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        for block in self.blocks:
            x = block(x, pos, self.rope)

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