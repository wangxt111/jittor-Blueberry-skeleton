import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group


class SimpleSkeletonModelWithPrior(nn.Module):
    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints

        self.backbone = Point_Transformer(output_channels=feat_dim)

        self.joint_mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_joints * 3),
        )

    def execute(self, vertices):
        # vertices: (B, N, 3)
        features = self.backbone(vertices)  # (B, N, feat_dim)

        # Pooling to get global features for joint prediction
        global_feat = jt.mean(features, dim=1)  # (B, feat_dim)
        pred_joints = self.joint_mlp(global_feat).reshape((-1, self.num_joints, 3))

        # Point-wise weight prediction
        pred_weights = self.weight_mlp(features)  # (B, N, J)
        pred_weights = nn.softmax(pred_weights, dim=-1)  # 保证权重为概率分布

        return pred_joints, pred_weights
