import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys

# Import the PCT model components
# 假设 pct_n.py 文件与此文件在同一目录下或在Python路径中
from PCT.networks.cls.pct_n import Point_Transformer_Big, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group, Point_Transformer_Symmetric
from PCT.networks.cls.pct import Point_Transformer_Multi,Point_Transformer


class SimpleSkeletonModel(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        
        self.transformer = Point_Transformer(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)

class SimpleSkeletonModel2(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        self.transformer = Point_Transformer2(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )

    def execute(self, vertices: jt.Var):
        clean_vertices = jt.array(vertices)
        x = self.transformer(clean_vertices)
        return self.mlp(x)

class SimpleSkeletonModel_Symmetric(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim        = feat_dim
        self.output_channels = output_channels

        self.transformer = Point_Transformer(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )
        self.symmetric_pairs = [
            (9,13), (8,12), (7,11), (6,10),
            (14,18), (15,19), (16,20), (17,21),
        ]
    
    def mirror_x(self, coords):
        # coords: (B, J, 3)
        mirrored = coords.clone()
        mirrored[..., 0] = -mirrored[..., 0]
        return mirrored

    def symmetric_average(self, joints):
        # joints: (B, J, 3)
        out = joints.clone()
        for i, j in self.symmetric_pairs:
            pi, pj = joints[:, i, :], joints[:, j, :]
            pj_mirror = self.mirror_x(pj)
            pi_mirror = self.mirror_x(pi)

            avg_i = (pi + pj_mirror) / 2
            avg_j = (pj + pi_mirror) / 2

            out[:, i, :] = avg_i
            out[:, j, :] = avg_j
        return out

    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        pred = self.mlp(x)  # (B, J*3)

        # reshape为 (B, J, 3)
        B = pred.shape[0]
        J = self.output_channels // 3
        joints = pred.view(B, J, 3)

        joints_sym = self.symmetric_average(joints)
        return joints_sym.reshape(B, -1)

class MultiSkeletonModel(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        
        self.transformer = Point_Transformer_Multi(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)

class SymSkeletonModel(nn.Module):

    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim

        self.direct_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17]
        self.symmetric_pairs = [
            (9, 13), (8, 12), (7, 11), (6, 10),
            (14, 18), (15, 19), (16, 20), (17, 21),
        ]

        self.transformer = Point_Transformer(output_channels=feat_dim)

        self.mlp_joint = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, len(self.direct_joints)*3),  # 14 joints × 3 coords
        )

        self.mlp_plane = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # (a, b, c, d)
        )

    def reflect_points(self, points: jt.Var, plane: jt.Var):
        """
        points: [B, N, 3]
        plane:  [B, 4]  -> (a,b,c,d)
        returns: [B, N, 3] reflected across plane
        """
        normal = plane[:, :3]  # [B, 3]
        d = plane[:, 3:]       # [B, 1]
        normal = normal / (jt.norm(normal, dim=1, keepdim=True) + 1e-8)

        dist = (points * normal.unsqueeze(1)).sum(dim=2, keepdims=True) + d.unsqueeze(1)

        reflected = points - 2 * dist * normal.unsqueeze(1)
        return reflected

    def execute(self, vertices: jt.Var):
        """
        vertices: [B, N, 3]
        return: joints [B, 66]
        """
        x = self.transformer(vertices.permute(0, 2, 1)) # [B, feat_dim]
        joint_pred = self.mlp_joint(x)               # [B, 42]
        plane_pred = self.mlp_plane(x)               # [B, 4]

        B = joint_pred.shape[0]
        joints14 = joint_pred.reshape(B, 14, 3)       # [B, 14, 3]

        joints22 = jt.zeros((B, 22, 3))
        for i, idx in enumerate(self.direct_joints):
            joints22[:, idx, :] = joints14[:, i, :]

        to_reflect = []
        for left, right in self.symmetric_pairs:
            if right not in self.direct_joints:
                to_reflect.append(left)

        src = jt.stack([joints22[:, i, :] for i in to_reflect], dim=1)
        refl = self.reflect_points(src, plane_pred)

        for i, (left, right) in enumerate(self.symmetric_pairs):
            if right not in self.direct_joints:
                joints22[:, right, :] = refl[:, i, :]

        return joints22.reshape(B, -1)  # [B, 66]


class SymSkeletonModelnc(nn.Module):

    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim

        self.direct_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17]
        self.symmetric_pairs = [
            (9, 13), (8, 12), (7, 11), (6, 10),
            (14, 18), (15, 19), (16, 20), (17, 21),
        ]

        self.transformer = Point_Transformer(output_channels=feat_dim)

        self.mlp_joint = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, len(self.direct_joints)*3),
        )

        self.mlp_plane = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # (a, b, c, d)
        )

    def reflect_points(self, points: jt.Var, plane: jt.Var):
        """
        points: [B, N, 3]
        plane:  [B, 4]  -> (a,b,c,d)
        returns: [B, N, 3] reflected across plane
        """
        normal = plane[:, :3]
        d = plane[:, 3:]
        normal = normal / (jt.norm(normal, dim=1, keepdim=True) + 1e-8)
        dist = (points * normal.unsqueeze(1)).sum(dim=2, keepdims=True) + d.unsqueeze(1)
        reflected = points - 2 * dist * normal.unsqueeze(1)
        return reflected

    def execute(self, vertices: jt.Var):
        """
        vertices: [B, N, 3]
        return: joints [B, 66]
        """
        # FIX: Point_Transformer expects input shape (B, C, N).
        x = self.transformer(vertices.permute(0, 2, 1))
        joint_pred = self.mlp_joint(x)
        plane_pred = self.mlp_plane(x)

        B = joint_pred.shape[0]
        joints14 = joint_pred.reshape(B, 14, 3)

        joints22 = jt.zeros((B, 22, 3))
        for i, idx in enumerate(self.direct_joints):
            joints22[:, idx, :] = joints14[:, i, :]

        to_reflect = []
        for left, right in self.symmetric_pairs:
            if right not in self.direct_joints:
                to_reflect.append(left)
        
        src = jt.stack([joints22[:, i, :] for i in to_reflect], dim=1)
        refl = self.reflect_points(src, plane_pred)

        for i, (left, right) in enumerate(self.symmetric_pairs):
            if right not in self.direct_joints:
                joints22[:, right, :] = refl[:, i, :]

        return joints22.reshape(B, -1), plane_pred

# Factory function to create models
def create_model(model_name='pct', output_channels=66, **kwargs):
    if model_name == "pct":
        return SimpleSkeletonModel(feat_dim=256, output_channels=output_channels)
    # 注册新的 SimpleSkeletonModel2
    elif model_name == "pct2":
        print("Using SimpleSkeletonModel2 with Point_Transformer2")
        # 不再需要处理 input_channels
        return SimpleSkeletonModel2(feat_dim=256, output_channels=output_channels)
    elif model_name == "sym":
        return SymSkeletonModel(feat_dim=256)
    elif model_name == "symnc":
        return SymSkeletonModelnc(feat_dim=256)
    elif model_name == "force_pct":
        return SimpleSkeletonModel_Symmetric(feat_dim=256, output_channels=output_channels)
    elif model_name == "multihead":
        print("Using MultiSkeletonModel")
        return MultiSkeletonModel(feat_dim=256, output_channels=output_channels)
    raise NotImplementedError()