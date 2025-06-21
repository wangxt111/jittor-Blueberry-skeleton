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
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def execute(self, x):
        B = x.shape[0]
        return self.encoder(x.reshape(-1, self.input_dim)).reshape(B, -1, self.output_dim)


def relative_encoding1(vertices, joints):
    """
    vertices: (B, N, 3)
    joints: (B, J, 3)
    return: (B, N, J, 4) relative offset + distance
    """
    rel_pos = vertices.unsqueeze(2) - joints.unsqueeze(1)  # (B, N, J, 3)
    distance = jt.norm(rel_pos, dim=-1, keepdim=True)      # (B, N, J, 1)
    return concat([rel_pos, distance], dim=-1)             # (B, N, J, 4)

class SkinModel_Fusion_MLP(nn.Module):
    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        self.pct = Point_Transformer(output_channels=feat_dim)

        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)
        self.fusion_mlp = MLP(feat_dim * 2 + 4, feat_dim)  # vertex + joint + relative pos

        self.relu = nn.ReLU()

    def execute(self, vertices: jt.Var, joints: jt.Var):
        B, N, _ = vertices.shape
        J = self.num_joints
        # (B, C, N) -> (B, N, feat_dim)
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))

        # Vertex embedding: (B, N, feat_dim)
        v_input = concat([vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)], dim=-1)
        v_latent = self.vertex_mlp(v_input)

        # Joint embedding: (B, J, feat_dim)
        j_input = concat([joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1)
        j_latent = self.joint_mlp(j_input)

        # Relative positional encoding: (B, N, J, 4)
        rel_encoding = relative_encoding1(vertices, joints)

        # Broadcast features: (B, N, J, feat_dim)
        v_feat = v_latent.unsqueeze(2).repeat(1, 1, J, 1)
        j_feat = j_latent.unsqueeze(1).repeat(1, N, 1, 1)

        # Fuse all info: (B, N, J, feat_dim)
        fused_input = concat([v_feat, j_feat, rel_encoding], dim=-1)
        fused_input_flat = fused_input.reshape(-1, fused_input.shape[-1])  # (B*N*J, input_dim)
        fused_feat = self.fusion_mlp(fused_input_flat)  # 输出 (B*N*J, feat_dim)
        fused_feat = fused_feat.reshape(B, N, J, -1)

        distance = rel_encoding[..., -1]
        decay_weight = 1.0 / (distance + 1e-6)

        # Attention score: (B, N, J)
        score = fused_feat.sum(dim=-1) * decay_weight
        weights = nn.softmax(score, dim=-1)

        assert not jt.isnan(weights).any()
        return weights
    
class SimpleAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def execute(self, query, key, value):
        # query/key/value: (B, N, J, C)
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # (B*N, J, C)
        B, N, J, C = Q.shape
        Q = Q.reshape(B * N, J, C)
        K = K.reshape(B * N, J, C)
        V = V.reshape(B * N, J, C)

        scores = jt.matmul(Q, K.transpose(0, 2, 1)) * self.scale  # (B*N, J, J)
        attn = nn.softmax(scores, dim=-1)
        out = jt.matmul(attn, V)  # (B*N, J, C)
        out = out.reshape(B, N, J, C)
        return self.out_proj(out)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = SimpleAttention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def execute(self, x):
        x = x + self.attn(x, x, x)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

def relative_encoding(vertices, joints):
    """
    vertices: (B, N, 3)
    joints: (B, J, 3)
    return: (B, N, J, 6)
    """
    B, N, _ = vertices.shape
    _, J, _ = joints.shape
    diff = vertices.unsqueeze(2) - joints.unsqueeze(1)  # (B, N, J, 3)
    dist = jt.norm(diff, dim=-1, keepdim=True)         # (B, N, J, 1)
    unit_vec = diff / (dist + 1e-6)                     # (B, N, J, 3)
    inv_dist = 1.0 / (dist + 1e-6)
    log_dist = jt.log(dist + 1e-6)
    return concat([unit_vec, log_dist, inv_dist , dist], dim=-1)  # (B, N, J, 6)

class SkeletonPropagation(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        from dataset.format import parents
        self.parents = parents
        self.update_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def execute(self, j_latent):
        B, J, C = j_latent.shape
        j_new = j_latent.clone()
        for j in range(J):
            p = self.parents[j]
            if p != None:
                child_feat = j_latent[:, j, :]    # (B, C)
                parent_feat = j_latent[:, p, :]   # (B, C)
                concat_feat = concat([child_feat, parent_feat], dim=-1)  # (B, 2C)
                j_new[:, j, :] = self.update_mlp(concat_feat)            # (B, C)
        return j_new
    
class SkinModel(nn.Module):
    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        self.pct = Point_Transformer(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)

        self.joint_embed = nn.Embedding(num_joints, feat_dim) 
        self.joint_prop = SkeletonPropagation(feat_dim)

        self.fusion = ResidualAttentionBlock(feat_dim)
        self.fuse_proj = nn.Linear(feat_dim * 2 + 6, feat_dim)

        self.relu = nn.ReLU()

    def execute(self, vertices: jt.Var, joints: jt.Var):
        B, N, _ = vertices.shape
        J = self.num_joints

        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))  # (B, N, feat_dim)

        v_input = concat([vertices, shape_latent.unsqueeze(1).repeat(1, N, 1)], dim=-1)
        v_latent = self.vertex_mlp(v_input)  # (B, N, feat_dim)

        j_input = concat([joints, shape_latent.unsqueeze(1).repeat(1, J, 1)], dim=-1)
        j_latent = self.joint_mlp(j_input)   # (B, J, feat_dim)

        j_latent = j_latent + self.joint_embed(jt.arange(J).unsqueeze(0).repeat(B, 1))
        j_latent = self.joint_prop(j_latent)

        rel_encoding = relative_encoding(vertices, joints)  # (B, N, J, 6)

        v_feat = v_latent.unsqueeze(2).repeat(1, 1, J, 1)  # (B, N, J, feat_dim)
        j_feat = j_latent.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, J, feat_dim)

        fused_input = concat([v_feat, j_feat, rel_encoding], dim=-1)  # (B, N, J, 2*feat + 6)
        fused_input = self.fuse_proj(fused_input)  # (B, N, J, feat_dim)

        attn_feat = self.fusion(fused_input)

        score = attn_feat.sum(dim=-1)  # (B, N, J)
        weights = nn.softmax(score, dim=-1)
        assert not jt.isnan(weights).any()
        return weights

class SkinModelNoJoints(nn.Module):
    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        self.pct = Point_Transformer(output_channels=feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)

        self.joint_embed = nn.Embedding(num_joints, feat_dim)
        self.joint_prop = SkeletonPropagation(feat_dim)

        self.fusion = ResidualAttentionBlock(feat_dim)

        self.fuse_proj = nn.Linear(feat_dim * 2, feat_dim)

        self.relu = nn.ReLU()

    def execute(self, vertices: jt.Var, joints: jt.Var):
        B, N, _ = vertices.shape
        J = self.num_joints

        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))  # (B, N, feat)

        v_input = concat([vertices, shape_latent.unsqueeze(1).repeat(1, N, 1)], dim=-1)
        v_latent = self.vertex_mlp(v_input)  # (B, N, feat)

        j_latent = self.joint_embed(jt.arange(J).unsqueeze(0).repeat(B, 1))  # (B, J, feat)
        j_latent = self.joint_prop(j_latent)

        v_feat = v_latent.unsqueeze(2).repeat(1, 1, J, 1)  # (B, N, J, feat)
        j_feat = j_latent.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, J, feat)

        fused_input = concat([v_feat, j_feat], dim=-1)  # (B, N, J, 2*feat)
        fused_input = self.fuse_proj(fused_input)       # (B, N, J, feat)

        attn_feat = self.fusion(fused_input)            # (B, N, J, feat)

        score = attn_feat.sum(dim=-1)                   # (B, N, J)
        weights = nn.softmax(score, dim=-1)             # (B, N, J)

        assert not jt.isnan(weights).any()
        return weights

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
    elif model_name == "pct_fusion_mlp":
        print("pct_fusion_mlp")
        return SkinModel_Fusion_MLP(feat_dim=feat_dim, num_joints=22)
    elif model_name == "skin":
        print("skinnn")
        return SkinModel(feat_dim=feat_dim, num_joints=22)
    elif model_name == "skin_no_joints":
        print("skin_no_joints")
        return SkinModelNoJoints(feat_dim=feat_dim, num_joints=22)
    raise NotImplementedError()