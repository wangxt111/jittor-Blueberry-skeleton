import jittor as jt
import numpy as np
import os
import argparse
import time
import random

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from models.skeleton import create_model
from dataset.format import parents

from models.metrics import J2J

# Set Jittor flags
jt.flags.use_cuda = 1
symmetric_pairs = [
    (9,13),
    (8,12),
    (7,11),
    (6,10),
    (14,18),
    (15,19),
    (16,20),
    (17,21),
]

def reflect_across_plane(points, plane_point, plane_normal):
    """
    将点 points 绕平面 (plane_point, plane_normal) 做反射。
    points: (B, 3)
    plane_point: (B, 3)
    plane_normal: (B, 3)
    """
    vec = points - plane_point         # (B, 3)
    proj = (vec * plane_normal).sum(dim=-1, keepdims=True)  # (B,1)
    reflected = points - 2 * proj * plane_normal            # (B,3)
    return reflected

def find_reflaection_plane_colomna(target, pairs):
    """
    通过脊柱节点(0,1,2,3,4,5)计算人体的对称平面
    target: (B, N, 3) 关节点坐标
    pairs: 对称点对列表
    返回：c (B, 3), n (B, 3) 表示过点c、法向量为n的对称平面
    """
    # 获取脊柱节点
    spine_nodes = target[:, 0:6, :]  # (B, 6, 3)
    
    # 1. 计算脊柱节点的中心点作为平面中心
    c = spine_nodes.mean(dim=1)  # (B, 3)
    
    # 2. 计算脊柱方向向量
    spine_start = spine_nodes[:, 0, :]  # (B, 3)
    spine_end = spine_nodes[:, -1, :]   # (B, 3)
    spine_dir = spine_end - spine_start  # (B, 3)
    spine_dir = spine_dir / (jt.norm(spine_dir, dim=-1, keepdims=True) + 1e-6)
    
    # 3. 计算垂直于脊柱方向的向量作为平面法向量
    # 使用脊柱节点拟合一个平面
    spine_centered = spine_nodes - c.unsqueeze(1)  # (B, 6, 3)
    
    # 计算协方差矩阵
    cov = jt.matmul(spine_centered.transpose(1, 2), spine_centered)  # (B, 3, 3)
    
    # 计算特征值和特征向量
    # 使用SVD分解
    U, S, V = jt.linalg.svd(cov)
    
    # 取最小特征值对应的特征向量作为平面法向量
    n = U[:, :, -1]  # (B, 3)
    
    # 确保法向量与脊柱方向垂直
    dot_product = (n * spine_dir).sum(dim=-1, keepdims=True)
    n = jt.where(dot_product > 0, -n, n)
    
    # 单位化法向量
    n = n / (jt.norm(n, dim=-1, keepdims=True) + 1e-6)
    
    return c, n

def symmetry_loss(pred, target, pairs=None):
    """
    每个样本独立推断对称平面，根据反射点计算对称损失
    """
    if pairs is None:
        return jt.zeros(1)

    B = pred.shape[0]
    plane_c, plane_n = find_reflaection_plane_colomna(target, pairs)  # (B, 3), (B, 3)
    loss = 0.0

    for left, right in pairs:
        left_joint = pred[:, left, :]         # (B, 3)
        right_joint = pred[:, right, :]       # (B, 3)
        right_mirrored = reflect_across_plane(right_joint, plane_c, plane_n)  # (B, 3)
        loss += jt.norm(left_joint - right_mirrored, dim=-1).mean()

    return loss / len(pairs)

npz_path = "newdata/train/mixamo/387_aug.npz"
data = np.load(npz_path)
joints = data["joints"]  # shape: (B, 22, 3)

# 如果joints是二维的(N, 3)，则增加一个批次维度，使其变为(1, N, 3)
if joints.ndim == 2:
    joints = np.expand_dims(joints, axis=0)

# 转为 Jittor 变量
joints_jt = jt.array(joints)

# 打印所有骨骼位置
print("所有骨骼位置 (joints_jt):", joints_jt)

# 计算对称平面中心和平面法向量
c, n = find_reflaection_plane_colomna(joints_jt, symmetric_pairs)

# 打印每个样本的对称平面法向量
print("对称平面中心 (c):", c)
print("对称平面法向量 (n):", n)

# -----------------------------------------------------
# 新增：判断骨骼的pairs是否就平面对称，并输出误差
print("\n对称性误差检查:")
for batch_idx in range(joints_jt.shape[0]): # 遍历批次中的每个样本
    print(f"  样本 {batch_idx}:")
    batch_c = c[batch_idx] # 获取当前样本的平面中心
    batch_n = n[batch_idx] # 获取当前样本的平面法向量
    
    for left_idx, right_idx in symmetric_pairs:
        left_joint_pos = joints_jt[batch_idx, left_idx, :] # 获取左侧骨骼位置
        right_joint_pos = joints_jt[batch_idx, right_idx, :] # 获取右侧骨骼位置
        
        # 将右侧骨骼位置相对于对称平面进行反射
        # 注意：reflect_across_plane期望(B,3)形状，所以需要先unsqueeze(0)再squeeze(0)
        right_mirrored_pos = reflect_across_plane(
            right_joint_pos.unsqueeze(0), 
            batch_c.unsqueeze(0), 
            batch_n.unsqueeze(0)
        ).squeeze(0)
        
        # 计算左侧骨骼与反射后的右侧骨骼之间的欧几里得距离作为误差
        symmetry_error = jt.norm(left_joint_pos - right_mirrored_pos).item()
        
        print(f"    对 ({left_idx}, {right_idx}): 误差 = {symmetry_error:.6f}")
# -----------------------------------------------------