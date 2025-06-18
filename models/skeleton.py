import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer_Big, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group

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
        normal = normal / (jt.norm(normal, dim=1, keepdim=True) + 1e-8)  # normalize

        # 点到平面的有向距离
        # signed_dist = (P ⋅ n + d)
        dist = (points * normal.unsqueeze(1)).sum(dim=2, keepdims=True) + d.unsqueeze(1)  # [B, N, 1]

        # 反射点公式: P' = P - 2 * dist * n
        reflected = points - 2 * dist * normal.unsqueeze(1)  # [B, N, 3]
        return reflected

    def execute(self, vertices: jt.Var):
        """
        vertices: [B, N, 3]
        return: joints [B, 66]
        """
        x = self.transformer(vertices)               # [B, feat_dim]
        joint_pred = self.mlp_joint(x)               # [B, 42]
        plane_pred = self.mlp_plane(x)               # [B, 4]

        B = joint_pred.shape[0]
        joints14 = joint_pred.reshape(B, 14, 3)       # [B, 14, 3]

        # 构建完整22个关节
        joints22 = jt.zeros((B, 22, 3))
        for i, idx in enumerate(self.direct_joints):
            joints22[:, idx, :] = joints14[:, i, :]

        # 反射生成剩下8个点
        to_reflect = []
        for left, right in self.symmetric_pairs:
            if right not in self.direct_joints:
                to_reflect.append(left)

        # 被反射的原点
        src = jt.stack([joints22[:, i, :] for i in to_reflect], dim=1)  # [B, 8, 3]
        refl = self.reflect_points(src, plane_pred)                     # [B, 8, 3]

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
        normal = normal / (jt.norm(normal, dim=1, keepdim=True) + 1e-8)  # normalize

        # 点到平面的有向距离
        # signed_dist = (P ⋅ n + d)
        dist = (points * normal.unsqueeze(1)).sum(dim=2, keepdims=True) + d.unsqueeze(1)  # [B, N, 1]

        # 反射点公式: P' = P - 2 * dist * n
        reflected = points - 2 * dist * normal.unsqueeze(1)  # [B, N, 3]
        return reflected

    def execute(self, vertices: jt.Var):
        """
        vertices: [B, N, 3]
        return: joints [B, 66]
        """
        x = self.transformer(vertices)               # [B, feat_dim]
        joint_pred = self.mlp_joint(x)               # [B, 42]
        plane_pred = self.mlp_plane(x)               # [B, 4]

        B = joint_pred.shape[0]
        joints14 = joint_pred.reshape(B, 14, 3)       # [B, 14, 3]

        # 构建完整22个关节
        joints22 = jt.zeros((B, 22, 3))
        for i, idx in enumerate(self.direct_joints):
            joints22[:, idx, :] = joints14[:, i, :]

        # 反射生成剩下8个点
        to_reflect = []
        for left, right in self.symmetric_pairs:
            if right not in self.direct_joints:
                to_reflect.append(left)

        # 被反射的原点
        src = jt.stack([joints22[:, i, :] for i in to_reflect], dim=1)  # [B, 8, 3]
        refl = self.reflect_points(src, plane_pred)                     # [B, 8, 3]

        for i, (left, right) in enumerate(self.symmetric_pairs):
            if right not in self.direct_joints:
                joints22[:, right, :] = refl[:, i, :]

        return joints22.reshape(B, -1), plane_pred

# Factory function to create models
def create_model(model_name='pct', output_channels=66, **kwargs):
    if model_name == "pct":
        return SimpleSkeletonModel(feat_dim=256, output_channels=output_channels)
    elif model_name == "sym":
        # print("Using SymSkeletonModel")
        return SymSkeletonModel(feat_dim=256)
    elif model_name == "symnc":
        # print("Using SymSkeletonModelnc")
        return SymSkeletonModelnc(feat_dim=256)
    raise NotImplementedError()
