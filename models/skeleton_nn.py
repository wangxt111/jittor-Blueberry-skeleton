import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys


# ==================================================================
# |                                                                |
# |      Section 1: PointTransformerV3 and Preprocessing           |
# | (The new, complete Ptv3 implementation, required for ptv3_simple) |
# |                                                                |
# ==================================================================

# --- Z-Order Serialization Functions ---
def _part1by2_10bit(n):
    n = n & 0x000003ff
    n = (n ^ (n << 16)) & 0xff0000ff
    n = (n ^ (n << 8))  & 0x0300f00f
    n = (n ^ (n << 4))  & 0x030c30c3
    n = (n ^ (n << 2))  & 0x09249249
    return n

def _compute_morton_code_3d_10bit(coords):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]
    return _part1by2_10bit(x) | (_part1by2_10bit(y) << 1) | (_part1by2_10bit(z) << 2)

def serialize_point_cloud_z_order(xyz):
    xyz_permuted = xyz.permute(0, 2, 1)
    xyz_min= xyz_permuted.min(dim=1, keepdims=True)
    xyz_max= xyz_permuted.max(dim=1, keepdims=True)
    xyz_range = xyz_max - xyz_min
    xyz_range[xyz_range < 1e-8] = 1e-8
    xyz_normalized = (xyz_permuted - xyz_min) / xyz_range
    bits = 10
    grid_max_val = (1 << bits) - 1
    int_coords = (xyz_normalized * grid_max_val).int64()
    morton_codes = _compute_morton_code_3d_10bit(int_coords)
    _, sort_indices = jt.argsort(morton_codes, dim=1)
    sort_indices_expanded = sort_indices.unsqueeze(1).expand_as(xyz)
    serialized_xyz = jt.gather(xyz, dim=2, index=sort_indices_expanded)
    return serialized_xyz

# --- PointTransformerV3 Model Definition ---
class SA_Layer_V3(nn.Module):
    def __init__(self, channels):
        super(SA_Layer_V3, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    def execute(self, x):
        x_q, x_k, x_v = self.q_conv(x).permute(0, 2, 1), self.k_conv(x), self.v_conv(x)
        energy = nn.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = nn.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        return x + x_r

class PointTransformerV3(nn.Module):
    def __init__(self, output_channels=256):
        super(PointTransformerV3, self).__init__()
        channels = 64
        self.conv1, self.conv2 = nn.Conv1d(3, channels, 1, bias=False), nn.Conv1d(channels, channels, 1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm1d(channels), nn.BatchNorm1d(channels)
        self.cpe_conv = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm1d(channels), nn.ReLU())
        self.sa1, self.sa2, self.sa3, self.sa4 = SA_Layer_V3(channels), SA_Layer_V3(channels), SA_Layer_V3(channels), SA_Layer_V3(channels)
        self.conv_fuse = nn.Sequential(nn.Conv1d(channels * 4, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.feat_output = nn.Linear(1024, output_channels)
        self.relu = nn.ReLU()
    def execute(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.cpe_conv(x)
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = self.conv_fuse(concat((x1, x2, x3, x4), dim=1))
        x = jt.max(x, dim=2).view(x.shape[0], -1)
        return self.feat_output(x)
# --- NEW Ptv3-based model ---

class SimpleSkeletonModel_Ptv3(nn.Module):
    """
    A Ptv3-based version of SimpleSkeletonModel.
    It uses the PointTransformerV3 backbone and handles the necessary preprocessing.
    """
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_channels = output_channels
        
        # Use the PointTransformerV3 backbone
        self.transformer = PointTransformerV3(output_channels=feat_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        serialized_vertices = serialize_point_cloud_z_order(vertices)
        
        # Pass preprocessed data to the Ptv3 backbone
        x = self.transformer(serialized_vertices)
        return self.mlp(x)

# ==================================================================
# |                                                                |
# |           Section 3: Updated Model Factory                     |
# |                                                                |
# ==================================================================

def create_model(model_name='ptv3_simple', output_channels=66, **kwargs):
    if model_name == "ptv3_simple":
        print("---INFO: Creating 'ptv3_simple' model with PointTransformerV3 backbone.---")
        return SimpleSkeletonModel_Ptv3(feat_dim=256, output_channels=output_channels)
        
    raise NotImplementedError(f"Model '{model_name}' is not implemented.")