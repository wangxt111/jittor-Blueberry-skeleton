import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys
from math import sqrt

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer_Big, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    
    def execute(self, x):
        B = x.shape[0]
        return self.encoder(x.reshape(-1, self.input_dim)).reshape(B, -1, self.output_dim)

class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    
    def execute(self, x):
        B, N, _ = x.shape
        x = x.reshape(-1, x.shape[-1])  # (B*N, input_dim)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(B, N, -1)
        return x

class SimpleSkinModel(nn.Module):

    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        self.pct = Point_Transformer(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim) # 关节
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim) # 顶点
        self.relu = nn.ReLU()
    
    def execute(self, vertices: jt.Var, joints: jt.Var):
        # (B, latents)
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))

        # (B, N, latents)
        vertices_latent = (
            self.vertex_mlp(concat([vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)], dim=-1))
        )

        # (B, num_joints, latents)
        joints_latent = (
            self.joint_mlp(concat([joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1))
        )

        # (B, N, num_joints)
        res = nn.softmax(vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), dim=-1)
        assert not jt.isnan(res).any()

        return res


# Factory function to create models
def create_model(model_name='pct', feat_dim=256, **kwargs):
    if model_name == "pct":
        return SimpleSkinModel(feat_dim=feat_dim, num_joints=22)
    raise NotImplementedError()