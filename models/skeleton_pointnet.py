import jittor as jt
from jittor import nn
import numpy as np

def square_distance(src, dst):
    """
    计算两个点集之间的平方距离
    src: (B, N, C)
    dst: (B, M, C)
    return: (B, N, M)
    """
    B, N, C = src.shape
    _, M, _ = dst.shape
    dist = -2 * jt.matmul(src, dst.permute(0, 2, 1))
    dist += jt.sum(src**2, dim=2, keepdims=True)
    dist += jt.sum(dst**2, dim=2).unsqueeze(1)
    return dist

def farthest_point_sample(xyz, npoint):
    """
    FPS采样
    xyz: (B, N, 3)
    return: (B, npoint)采样点索引
    """
    B, N, _ = xyz.shape
    centroids = jt.zeros((B, npoint), dtype='int32')
    distance = jt.ones((B, N)) * 1e10
    farthest = jt.zeros(B, dtype='int32')
    batch_indices = jt.arange(B)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = jt.stack([batch_indices, farthest], dim=1)  # (B, 2)
        centroid_xyz = xyz[batch_indices, farthest]  # (B, 3)
        dist = jt.sum((xyz - centroid_xyz.unsqueeze(1)) ** 2, dim=-1)  # (B, N)
        mask = dist < distance
        distance = jt.where(mask, dist, distance)
        farthest = jt.argmax(distance, dim=1)
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    球邻域搜索
    xyz: (B, N, 3) 所有点
    new_xyz: (B, S, 3) 中心点
    return: idx (B, S, nsample) 每个中心点的邻居索引
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    group_idx = jt.argsort(sqrdists, dim=-1)[:, :, :nsample]  # 取最近的nsample个点
    group_dist = jt.gather(sqrdists, 2, group_idx)
    mask = group_dist > radius ** 2
    group_idx = jt.where(mask, jt.zeros_like(group_idx), group_idx)
    return group_idx

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        last_channel = in_channel + 3  # 加上坐标差
        for out_channel in mlp_channels:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def execute(self, xyz, points):
        """
        xyz: (B, N, 3) 输入点的坐标
        points: (B, C, N) 输入点的特征
        """
        B, N, _ = xyz.shape

        # 1. 采样点
        if self.npoint is None:
            new_xyz = jt.zeros((B, 1, 3))
            new_points = points.unsqueeze(2)  # (B, C, N) -> (B, C, 1, N)
        else:
            idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
            batch_indices = jt.arange(B).unsqueeze(1).repeat(1, self.npoint)
            new_xyz = xyz[batch_indices, idx]  # (B, npoint, 3)

            # 2. 球邻域搜索
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (B, npoint, nsample)
            grouped_xyz = xyz.unsqueeze(1).repeat(1, self.npoint, 1, 1)  # (B, npoint, N, 3)
            grouped_xyz = jt.gather(xyz.unsqueeze(1).repeat(1, self.npoint, 1, 1), 2, group_idx.unsqueeze(-1).repeat(1,1,1,3))  # (B, npoint, nsample, 3)
            grouped_xyz -= new_xyz.unsqueeze(2)  # 相对坐标

            if points is not None:
                grouped_points = jt.gather(points.permute(0,2,1), 2, group_idx)  # (B, C, npoint, nsample)
                grouped_points = grouped_points.permute(0,2,3,1)  # (B, npoint, nsample, C)
                new_points = jt.concat([grouped_xyz, grouped_points], dim=-1)  # (B, npoint, nsample, C+3)
            else:
                new_points = grouped_xyz  # (B, npoint, nsample, 3)

            new_points = new_points.permute(0,3,1,2)  # (B, C+3, npoint, nsample)

        # 3. MLP提取特征 + max pooling
        new_points = self.mlp(new_points)  # (B, out_channel, npoint, nsample)
        new_points = jt.max(new_points, dim=-1)  # (B, out_channel, npoint)
        return new_xyz, new_points

class PointNetPP(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=0, mlp_channels=[64,64,128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp_channels=[128,128,256])
        
        self.fc1 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_classes)

    def execute(self, xyz):
        """
        xyz: (B, N, 3)
        """
        B, N, _ = xyz.shape
        points = None
        l1_xyz, l1_points = self.sa1(xyz, points)  # 第一层抽象 (B, 512, 3), (B, 128, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # 第二层抽象 (B, 128, 3), (B, 256, 128)

        # 池化全局特征
        x = jt.max(l2_points, dim=2)  # (B, 256)

        x = self.drop1(self.bn1(self.fc1(x)))
        x = self.drop2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
