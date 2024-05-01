import torch
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
from itertools import groupby
from .vision_transformer import Attention, Block, VisionTransformer
import torch.nn.functional as F
from einops import rearrange
from mmseg.registry import MODELS
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.logging import MMLogger

class AttentionWithBiases(Attention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_bias_map(self, bias_map):
        self.register_buffer("bias_map", bias_map)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias_map.type_as(q))

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class BlockwithBiases(Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn = AttentionWithBiases(
            dim=kwargs["dim"],
            num_heads=kwargs["num_heads"],
        )

@MODELS.register_module()
class Alibi(VisionTransformer):
    def __init__(self, alibi_config, **kwargs):
        kwargs["block_fn"] = BlockwithBiases

        super().__init__(**kwargs)
        self.alibi_config = alibi_config

        grid_size = self.grid_size
        num_patch = int(grid_size[0] * grid_size[1])
        bias_maps = create_alibi_bias_tensor(
            self.alibi_config, grid_size, self.depth, self.num_heads)
        zeros_1 = torch.zeros(
            size=(self.depth, self.num_heads, num_patch, 1),
            dtype=torch.float) 
        bias_maps = torch.cat([zeros_1, bias_maps], dim=-1) 
        zeros_2 = torch.zeros(
            size=(self.depth, self.num_heads, 1, (num_patch + 1)),
            dtype=torch.float) 
        bias_maps = torch.cat([zeros_2, bias_maps], dim=-2) 

        for i, block in enumerate(self.blocks):
            block.attn.set_bias_map(bias_maps[i])
    
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
            load_state_dict(self, checkpoint, strict=False, logger=logger)
        else:
            self.apply(self._init_weights)


############################################## ALIBI code #################################################################
def get_none_slopes(n):
    slopes = [0.25, 0.0625, 0.015625, 0.00390625]
    if n == 2:
        return slopes[:2]
    elif n == 4:
        return slopes
    else:
        return np.repeat(slopes, n // 4)


def get_slopes(n):
    return np.repeat(np.exp2(-np.arange(1, n / 4 + 1)), 4)


def create_alibi_bias_tensor(alibi_config, grid_size, depth, num_heads):
    global_slope = alibi_config["global_slope"]
    layer_slopes = np.linspace(
        alibi_config["layer_slopes"][0], alibi_config["layer_slopes"][1], depth
    )
    head_directions = alibi_config["head_directions"]

    if type(head_directions) is list:
        alibi = []
        for direction in head_directions:
            alibi.append(create_alibi_layer_tensor(grid_size, direction, num_heads))
        alibi = np.stack(alibi, axis=0)
    else:
        alibi = create_alibi_layer_tensor(grid_size, head_directions, num_heads)

    # replace inf with max
    layer_slopes = np.expand_dims(layer_slopes, (1, 2, 3))
    alibi = global_slope * layer_slopes * alibi
    alibi = np.nan_to_num(alibi, posinf=torch.finfo(torch.bfloat16).max)
    alibi = torch.tensor(alibi, dtype=torch.bfloat16) * -1.0

    return alibi


def create_alibi_layer_tensor(grid_size, direction, num_heads):
    dir_short_hand = {
        "none": [[]] * 4,
        "zero": [["zero"]] * 4,
        "none1zero3": [[]] + [["zero"]] * 3,
        "none2zero2": [[]] * 2 + [["zero"]] * 2,
        "none3zero1": [[]] * 3 + [["zero"]],
        "center": [["center"]] * 4,
        "halves": [["up"], ["down"], ["left"], ["right"]],
        "corners": [
            ["up", "left"],
            ["up", "right"],
            ["down", "left"],
            ["down", "right"],
        ],
        "triangles": [["tri_up"], ["tri_down"], ["tri_left"], ["tri_right"]],
        "diagonals": [
            ["diag_up_right"],
            ["diag_up_left"],
            ["diag_down_right"],
            ["diag_down_left"],
        ],
        "eighths1": [
            ["tri_up", "left"],
            ["tri_down", "left"],
            ["tri_left", "up"],
            ["tri_right", "up"],
        ],
        "eighths2": [
            ["tri_up", "right"],
            ["tri_down", "right"],
            ["tri_left", "down"],
            ["tri_right", "down"],
        ],
        "zerohalves": [
            ["zero", "up"],
            ["zero", "down"],
            ["zero", "left"],
            ["zero", "right"],
        ],
        "zerocorners": [
            ["zero", "up", "left"],
            ["zero", "up", "right"],
            ["zero", "down", "left"],
            ["zero", "down", "right"],
        ],
        "zerotriangles": [
            ["zero", "tri_up"],
            ["zero", "tri_down"],
            ["zero", "tri_left"],
            ["zero", "tri_right"],
        ],
    }

    # get configs
    sh = direction.split("-")
    dirs = [dir_short_hand[s] for s in sh]

    # number of options
    if num_heads == 6:
        head_names = [sh[0]] * 4 + [sh[1]] * 2
        head_directions = dirs[0] + dirs[1][:2]
    elif len(sh) == 1:
        head_names = [sh[0]] * num_heads
        head_directions = dirs[0] * (num_heads // 4)
    elif len(sh) == 2 and num_heads == 12:
        head_names = [sh[0]] * 8 + [sh[1]] * 4
        head_directions = dirs[0] * 2 + dirs[1]
    elif len(sh) == 2 and num_heads == 16:
        head_names = [sh[0]] * 8 + [sh[1]] * 8
        head_directions = dirs[0] * 2 + dirs[1] * 2
    elif len(sh) == 3 and num_heads == 12:
        head_names = [sh[0]] * 4 + [sh[1]] * 4 + [sh[2]] * 4
        head_directions = dirs[0] + dirs[1] + dirs[2]
    elif len(sh) == 3 and num_heads == 16:
        head_names = [sh[0]] * 8 + [sh[1]] * 4 + [sh[2]] * 4
        head_directions = dirs[0] * 2 + dirs[1] + dirs[2]
    elif len(sh) == 4 and num_heads == 16:
        head_names = [sh[0]] * 4 + [sh[1]] * 4 + [sh[2]] * 4 + [sh[3]] * 4
        head_directions = dirs[0] + dirs[1] + dirs[2] + dirs[3]

    # calculate slopes
    head_slopes = []
    for k, v in groupby(head_names):
        if ("zero" in k) or ("center" in k) or ("none" in k):
            head_slopes.append(get_none_slopes(len(list(v))))
        elif (
            ("halves" in k)
            or ("corners" in k)
            or ("triangles" in k)
            or ("eighths1" in k)
            or ("eighths2" in k)
            or ("diagonals" in k)
        ):
            head_slopes.append(get_slopes(len(list(v))))
    head_slopes = np.concatenate(head_slopes)

    # create list of grid coordinates
    grid_y, grid_x = grid_size
    num_patch = grid_x * grid_y
    grid_locs = np.meshgrid(range(grid_x), range(grid_y))
    coords = np.stack(grid_locs, axis=-1).reshape((-1, 2))

    # bias map for center direction
    center_location = np.array([(grid_x - 1) / 2, (grid_y - 1) / 2], ndmin=2)
    center_map = np.repeat(
        cdist(center_location, coords, "euclidean"), num_patch, axis=0
    )
    zero_map = np.zeros_like(
        center_map
    )  # map of zeros means no positional information whatsoever

    # create mask for direction types
    if (
        ("halves" in head_names)
        or ("corners" in head_names)
        or ("eighths1" in head_names)
        or ("eighths2" in head_names)
        or ("zerohalves" in head_names)
        or ("zerocorners" in head_names)
    ):
        compare_x = np.expand_dims(np.arange(grid_x), (1, 2))
        compare_y = np.expand_dims(np.arange(grid_y), (1, 2))
        direction_mask = {  # means the direction we are looking
            "right": np.less(grid_locs[0], compare_x)[coords[:, 0]],
            "left": np.greater(grid_locs[0], compare_x)[coords[:, 0]],
            "up": np.greater(grid_locs[1], compare_y)[coords[:, 1]],
            "down": np.less(grid_locs[1], compare_y)[coords[:, 1]],
        }

    # for triangles
    if (
        ("triangles" in head_names)
        or ("zerotriangles" in head_names)
        or ("eighths1" in head_names)
        or ("eighths2" in head_names)
        or ("diagonals" in head_names)
    ):
        # get the k for each coordinate (k = 0 is the main diagonal, while k < 0 is below it, and k > 0 is above)
        diag_coord_ud = []
        diag_label_ud = []
        diag_coord_lr = []
        diag_label_lr = []
        max_dia = np.max((grid_x, grid_y))
        for i in range(-max_dia + 1, max_dia):
            d = np.vstack([np.diag(grid_locs[0], i), np.diag(grid_locs[1], i)])
            diag_label_ud.append(i * np.ones(d.shape[1]))
            diag_coord_ud.append(d)
            d = np.vstack(
                [
                    np.diag(np.fliplr(grid_locs[0]), i),
                    np.diag(np.fliplr(grid_locs[1]), i),
                ]
            )
            diag_label_lr.append(i * np.ones(d.shape[1]))
            diag_coord_lr.append(d)

        diag_coord_ud = np.concatenate(diag_coord_ud, axis=1).T
        diag_label_ud = np.concatenate(diag_label_ud).T
        diag_coord_lr = np.concatenate(diag_coord_lr, axis=1).T
        diag_label_lr = np.concatenate(diag_label_lr).T

        # for each coordinate determine the associated triangle in each direction
        triangles = {
            dir: np.ones((num_patch, num_patch), dtype=bool)
            for dir in ["up", "down", "left", "right"]
        }
        for i in range(coords.shape[0]):
            diag_pos = diag_label_ud[
                np.all(coords[[i], :] == diag_coord_ud[:, :2], axis=1)
            ]  # positive diagonal
            diag_neg = diag_label_lr[
                np.all(coords[[i], :] == diag_coord_lr[:, :2], axis=1)
            ]  # negative diagonal
            trid = np.tri(grid_y, grid_x, k=diag_pos[0], dtype=bool)
            triu = ~np.tri(grid_y, grid_x, k=diag_pos[0] - 1, dtype=bool)
            trir = np.fliplr(np.tri(grid_y, grid_x, k=diag_neg[0], dtype=bool))
            tril = ~np.fliplr(np.tri(grid_y, grid_x, k=diag_neg[0] - 1, dtype=bool))
            triangles["up"][i, :] = ~(triu & tril).flatten()
            triangles["down"][i, :] = ~(trid & trir).flatten()
            triangles["right"][i, :] = ~(triu & trir).flatten()
            triangles["left"][i, :] = ~(trid & tril).flatten()

    # calculate dist matrix
    distvec = pdist(coords, "euclidean")
    m = squareform(distvec)
    alibi = np.tile(m, (num_heads, 1, 1))

    # apply directions
    for h in range(num_heads):
        dir_mask = np.zeros_like(m, dtype=bool)
        for dir in head_directions[h]:
            if dir == "center":
                alibi[h, :] = center_map
            elif dir == "zero":
                alibi[h, :] = zero_map
            elif "tri" in dir:
                tri_type = dir.split("_")[1]
                dir_mask = dir_mask | triangles[tri_type]
            elif "diag" in dir:
                tri_type_1 = dir.split("_")[1]
                tri_type_2 = dir.split("_")[2]
                first_mask = triangles[tri_type_1] & triangles[tri_type_2]
                dir_mask = dir_mask | first_mask
            else:
                dir_mask = dir_mask | direction_mask[dir].reshape((num_patch, -1))
            alibi[h, dir_mask] = np.inf

    # apply slopes to alibi
    head_slopes = np.expand_dims(head_slopes, (1, 2))
    alibi = head_slopes * alibi

    return alibi