"""
Point Transformer - V3 Mode1 (Jittor Version)
Converted from PyTorch by ChatGPT

Author: Xiaoyang Wu (original), converted by ChatGPT for Jittor
"""

import sys
import math
from functools import partial
from collections import OrderedDict

import jittor as jt
from jittor import nn
import numpy as np
import jsparse.nn as spnn
from jsparse import SparseTensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        if self.drop_prob == 0.0 or not self.is_training():
            return x
        keep_prob = 1 - self.drop_prob
        # x.shape[0] 是 batch size
        shape = [x.shape[0]] + [1] * (x.ndim - 1)  # [B, 1, 1, ...]
        random_tensor = jt.rand(shape).add(keep_prob).floor()
        output = x / keep_prob * random_tensor
        return output

from .serialization import encode

@jt.no_grad()
def offset2bincount(offset):
    """返回 offset 中每个 batch 的元素数量"""
    offset = offset.int32()
    diff = offset[1:] - offset[:-1]
    first = offset[0:1]
    bincount = jt.concat([first, diff], dim=0)
    return bincount

@jt.no_grad()
def offset2batch(offset):
    """将 offset（每个 batch 开始索引）转换为 batch index 张量"""
    bincount = offset2bincount(offset)
    batch_ids = [jt.full((c.item(),), i, dtype=jt.int32) for i, c in enumerate(bincount)]
    return jt.concat(batch_ids)

@jt.no_grad()
def batch2offset(batch):
    """将 batch index 转换为 offset 索引"""
    unique = jt.unique(batch)
    counts = jt.zeros(unique.shape[0], dtype=jt.int32)
    for i in range(unique.shape[0]):
        counts[i] = (batch == unique[i]).sum()
    return jt.cumsum(counts, dim=0)

class Point(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self and "offset" in self:
            self["batch"] = offset2batch(self["offset"])
        elif "offset" not in self and "batch" in self:
            self["offset"] = batch2offset(self["batch"])

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        assert "batch" in self
        if "grid_coord" not in self:
            assert {"grid_size", "coord"}.issubset(self.keys())
            min_coord = jt.min(self["coord"], dim=0, keepdims=True)
            self["grid_coord"] = ((self["coord"] - min_coord) / self["grid_size"]).floor().int32()

        if depth is None:
            depth = int(jt.max(self["grid_coord"]).item()).bit_length()

        self["serialized_depth"] = depth

        # 确保 depth * num_dims + batch_bits <= 63 (针对64位整数)
        # 这里的 num_dims 应该是 3，而不是 self["grid_coord"].shape[1]
        # 因为我们通常处理3D坐标
        assert depth * self["grid_coord"].shape[1] + len(self["offset"]).bit_length() <= 63 # 修正为用实际维度
        assert depth <= 16

        # 这里不再需要 batch = self["batch"].reshape(-1, 1)，
        # 因为 encode 不接受 batch 参数，且 grid_coord 已经是所有点的扁平列表

        # 获取实际的维度数，通常是3
        num_dims_actual = self["grid_coord"].shape[1]

        code = [
            encode(self["grid_coord"], self["batch"], depth, order=order_)
            for order_ in order
        ]
        code = jt.stack(code)  # (k, n)
        order_idx = jt.argsort(code, dim=1)
        inverse = jt.zeros_like(order_idx)
        for i in range(code.shape[0]):
            # Jittor 的 scatter_ 使用的是索引赋值，这里是正确的
            inverse[i].scatter_(0, order_idx[i], jt.arange(code.shape[1]))

        if shuffle_orders:
            perm = jt.randperm(code.shape[0])
            code = code[perm]
            order_idx = order_idx[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order_idx
        self["serialized_inverse"] = inverse
        
    def sparsify(self, pad=96):
        """
        Convert Point data to Jittor SparseTensor format for sparse convolution.
        
        Relies on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        pad: padding size for sparse shape (optional, not directly used here but can be kept for compatibility).
        """

        assert {"feat", "batch"}.issubset(self.keys())

        # 如果没有grid_coord，则计算grid_coord（离散坐标）
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = (self["coord"] - self["coord"].min(0)[0]) // self["grid_size"]
            self["grid_coord"] = self["grid_coord"].int()

        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = (jt.max(self["grid_coord"], dim=0) + pad).tolist()
        coords = jt.concat([self["batch"].unsqueeze(-1).int(), self["grid_coord"].int()], dim=1)

        sparse_tensor = SparseTensor(
            values=self["feat"],
            indices=coords,
            voxel_size=self["grid_size"],
            quantize=True,
        )

        self["sparse_shape"] = sparse_shape
        self["sparse_tensor"] = sparse_tensor

class PointModule(nn.Module):
    """
    PointModule
    Placeholder base class — all module subclasses will take Point dict as input
    and be used in PointSequential.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PointSequential(PointModule):
    """
    A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.__setattr__(key, module)
        else:
            for idx, module in enumerate(args):
                self.__setattr__(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self.__dict__:
                raise ValueError("name exists.")
            self.__setattr__(name, module)

    def __getitem__(self, idx):
        keys = list(self.__dict__.keys())
        values = list(self.__dict__.values())
        if not (-len(values) <= idx < len(values)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(values)
        return values[idx]

    def __len__(self):
        return len(self.__dict__)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self.__dict__))
        if name in self.__dict__:
            raise KeyError("name exists")
        self.__setattr__(name, module)

    def execute(self, input):
        for name in self.__dict__:
            module = getattr(self, name)
            if isinstance(module, PointModule):
                input = module(input)
            elif hasattr(module, 'execute'):
                # General nn.Module in Jittor
                if isinstance(input, dict) and 'feat' in input:
                    input['feat'] = module(input['feat'])
                else:
                    input = module(input)
            else:
                raise NotImplementedError(f"Unsupported module type: {type(module)}")
        return input

class PDNorm(nn.Module):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive

        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer(num_features)

        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def execute(self, point):
        assert "feat" in point and "condition" in point

        # 处理 condition（支持字符串或长度为1的列表）
        if isinstance(point["condition"], str):
            condition = point["condition"]
        else:
            condition = point["condition"][0]

        if self.decouple:
            assert condition in self.conditions
            idx = self.conditions.index(condition)
            norm = self.norm[idx]
        else:
            norm = self.norm

        point["feat"] = norm(point["feat"])

        if self.adaptive:
            assert "context" in point
            shift, scale = self.modulation(point["context"]).chunk(2, dim=1)
            point["feat"] = point["feat"] * (1.0 + scale) + shift

        return point

class RPE(nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1

        # 初始化参数 rpe_table
        self.rpe_table = jt.empty((3 * self.rpe_num, num_heads))
        self.rpe_table = nn.Parameter(self.rpe_table)
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def execute(self, coord):
        # coord: (N, K, K, 3)
        coord = jt.clamp(coord, -self.pos_bnd, self.pos_bnd)
        offset = jt.array([0, 1, 2]).reshape(1, 1, 1, 3) * self.rpe_num
        idx = coord + self.pos_bnd + offset  # shape: (N, K, K, 3)
        idx_flat = idx.reshape(-1).int()  # (N * K * K * 3,)

        # index_select 替代：gather 方式构造
        out = self.rpe_table[idx_flat]  # (N * K * K * 3, num_heads)
        out = out.reshape(*idx.shape, self.num_heads).sum(dim=3)  # (N, K, K, num_heads)
        out = out.permute(0, 3, 1, 2)  # (N, H, K, K)
        return out

class SerializedAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.enable_rpe = enable_rpe

        self.patch_size_max = patch_size
        self.patch_size = 0

        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @jt.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point["grid_coord"][order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @jt.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point["offset"]
            bincount = offset2bincount(offset)
            bincount_pad = (
                ((bincount + self.patch_size - 1) // self.patch_size)
                * self.patch_size
            )
            mask_pad = bincount > self.patch_size
            bincount_pad = (~mask_pad) * bincount + mask_pad * bincount_pad
            _offset = jt.concat([jt.zeros(1, dtype=offset.dtype), offset])
            _offset_pad = jt.concat([jt.zeros(1, dtype=offset.dtype), jt.cumsum(bincount_pad, 0)])

            pad = jt.arange(_offset_pad[-1].item(), dtype=jt.int32)
            unpad = jt.arange(_offset[-1].item(), dtype=jt.int32)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    start = _offset_pad[i + 1] - self.patch_size + (bincount[i] % self.patch_size)
                    end = _offset_pad[i + 1]
                    pad[start:end] = pad[start - self.patch_size:end - self.patch_size]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(jt.arange(_offset_pad[i], _offset_pad[i + 1], step=self.patch_size, dtype=jt.int32))

            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = jt.concat(cu_seqlens).concat(jt.array([_offset_pad[-1]]))

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def execute(self, point):
        # 计算patch_size取最小值
        self.patch_size = min(offset2bincount(point["offset"]).min().item(), self.patch_size_max)

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point["serialized_order"][self.order_index][pad]
        inverse = unpad[point["serialized_inverse"][self.order_index]]

        qkv = self.qkv(point["feat"])[order]

        # reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
        qkv = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = jt.matmul(q * self.scale, k.transpose(0, 1, 3, 2))  # (N', H, K, K)

        if self.enable_rpe:
            attn += self.rpe(self.get_rel_pos(point, order))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        feat = jt.matmul(attn, v).transpose(0, 1, 3, 2).reshape(-1, C)

        feat = feat[inverse]

        feat = self.proj(feat)
        feat = self.proj_drop(feat)

        point["feat"] = feat
        return point

class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spnn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels)
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            # enable_flash=enable_flash,
            # upcast_attention=upcast_attention,
            # upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )

        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def execute(self, point):
        shortcut = point["feat"]
        point = self.cpe(point)
        point["feat"] = shortcut + point["feat"]
        shortcut = point["feat"]
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point["feat"] = shortcut + point["feat"]
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point["feat"]
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point["feat"] = shortcut + point["feat"]
        if not self.pre_norm:
            point = self.norm2(point)
        point["sparse_conv_feat"] = point["sparse_conv_feat"].replace_feature(point["feat"])
        return point


import jittor as jt

def segment_csr(data, segment_ids, num_segments, reduce="sum"):
    """
    data: (N, C) 输入特征
    segment_ids: (N,) 有序的分段ID，如 [0,0,0,1,1,2,2,2,2]
    num_segments: int，总分段数
    reduce: 聚合方式，sum/mean/min/max

    return: (num_segments, C)
    """
    N, C = data.shape
    assert segment_ids.shape[0] == N
    assert reduce in ["sum", "mean", "min", "max"]

    # 找每段起始index
    # segment_ids有序，可以用diff找边界
    boundaries = jt.concat([jt.array([0]), jt.nonzero(segment_ids[1:] != segment_ids[:-1]).reshape(-1)+1, jt.array([N])])
    # boundaries[i] 到 boundaries[i+1] 是第 i 个 segment 的范围

    out = jt.zeros((num_segments, C), dtype=data.dtype)

    for i in range(num_segments):
        start = boundaries[i].item()
        end = boundaries[i+1].item()
        segment_data = data[start:end]  # 该段所有数据

        if segment_data.shape[0] == 0:
            # 如果该段没有点，返回0（或合适的初始化值）
            continue

        if reduce == "sum":
            out[i] = segment_data.sum(dim=0)
        elif reduce == "mean":
            out[i] = segment_data.mean(dim=0)
        elif reduce == "min":
            out[i], _ = segment_data.min(dim=0)
        elif reduce == "max":
            out[i], _ = segment_data.max(dim=0)

    return out


class SerializedPooling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

    def execute(self, point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point["serialized_depth"]:
            pooling_depth = 0

        assert hasattr(point, "serialized_code")
        code = point["serialized_code"] >> (pooling_depth * 3)
        code_flat = code[0]  # 取 batch 维度第 0 个

        # Jittor的unique_with_counts返回 (unique, inverse, counts)
        unique_code, inverse_indices, counts = jt.unique_with_counts(code_flat, return_inverse=True)

        # 依据 inverse_indices 排序，获得排序后的索引
        sort_indices = jt.argsort(inverse_indices)
        
        # 计算 idx_ptr 边界，用于分段聚合
        counts_np = counts.data.numpy().tolist()
        idx_ptr = [0]
        for c in counts_np:
            idx_ptr.append(idx_ptr[-1] + c)
        idx_ptr = jt.array(idx_ptr, dtype="int32")

        # 取每个 segment 的第一个点索引
        head_indices = sort_indices[idx_ptr[:-1]]

        # 生成新的序列化信息
        code = code[:, head_indices]
        order = jt.argsort(code, dim=1)
        inverse = jt.zeros_like(order)
        # 逆序索引构造，Jittor不支持scatter_赋值，改用循环赋值
        for b in range(order.shape[0]):
            inverse[b][order[b]] = jt.arange(order.shape[1])

        if self.shuffle_orders:
            perm = jt.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # 特征投影和聚合
        feat_proj = self.proj(point["feat"])
        # 使用自定义 segment_csr 聚合特征和坐标
        feat = segment_csr(feat_proj[sort_indices], inverse_indices[sort_indices], unique_code.shape[0], self.reduce)
        coord = segment_csr(point["coord"][sort_indices], inverse_indices[sort_indices], unique_code.shape[0], "mean")

        # 构造结果字典
        point_dict = {
            "feat": feat,
            "coord": coord,
            "grid_coord": point["grid_coord"][head_indices] >> pooling_depth,
            "serialized_code": code,
            "serialized_order": order,
            "serialized_inverse": inverse,
            "serialized_depth": point["serialized_depth"] - pooling_depth,
            "batch": point["batch"][head_indices],
        }

        if hasattr(point, "condition"):
            point_dict["condition"] = point["condition"]
        if hasattr(point, "context"):
            point_dict["context"] = point["context"]

        if self.traceable:
            point_dict["pooling_inverse"] = inverse_indices
            point_dict["pooling_parent"] = point

        point_new = Point(point_dict)

        if self.norm is not None:
            point_new["feat"] = self.norm(point_new["feat"])
        if self.act is not None:
            point_new["feat"] = self.act(point_new["feat"])

        point_new.sparsify()
        return point_new

class Embedding(jt.nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        voxel_size=0.02,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.voxel_size = voxel_size

        layers = []
        layers.append(
            spnn.Conv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                stride=1,
                # padding=1,
                bias=False,
            )
        )
        if norm_layer is not None:
            layers.append(norm_layer(embed_channels))
        if act_layer is not None:
            layers.append(act_layer())
        self.stem = jt.nn.Sequential(*layers)

    def forward(self, point: SparseTensor):
        """
        输入:
          point: SparseTensor, 包含 coords (N,4) 和 features (N,C_in)

        返回:
          SparseTensor, features 经过卷积、归一化和激活
        """
        out = self.stem(point)
        return out

class PointTransformerV3(nn.Module):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        transformer_block=None,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)

        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm

        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        enc_drop_path = jt.linspace(0, drop_path, sum(enc_depths)).tolist()
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                BlockClass = transformer_block if transformer_block is not None else Block
                enc.add(
                    BlockClass(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

    def execute(self, vertices: jt.Var):
        data_dict = {
            "feat": vertices,
            "batch": jt.zeros(vertices.shape[0], dtype=jt.int32),
            "coord": vertices[:, :3],  # 假设前3列是坐标
            "grid_size": jt.array([0.02]),  # 设置一个默认的网格大小
            "offset": jt.array([0, vertices.shape[0]], dtype=jt.int32),  # 添加 offset 字段
        }
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        return point["feat"]