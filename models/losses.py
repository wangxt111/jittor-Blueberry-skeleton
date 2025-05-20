import jittor as jt
from jittor import nn
from typing import Dict, List, Tuple

class SkeletonLoss(nn.Module):
    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()
        # 默认损失权重
        self.loss_weights = loss_weights or {
            'mse': 1.0,
            'bone_length': 0.1,
            'symmetry': 0.1,
            'angle': 0.1,
            'smoothness': 0.05,
            # 'chamfer': 0.5  # 新增点云Chamfer损失权重
        }
        # 验证权重
        for weight in self.loss_weights.values():
            assert weight >= 0, "Loss weights must be non-negative"
        self.mse = nn.MSELoss()

    def forward(self, pred_joints: jt.Var, target_joints: jt.Var, parents: jt.Var) -> jt.Var:
        """
        计算总损失
        Args:
            pred_joints: [B, N, 3] 预测的关节点
            target_joints: [B, N, 3] 目标关节点
            parents: [N] 父节点索引
            pred_points: [B, M, 3] 预测的点云（可选）
            target_points: [B, M, 3] 目标点云（可选）
        Returns:
            total_loss: 总损失
        """
        total_loss = 0.0

        # 1. MSE损失
        mse_loss = self.mse(pred_joints, target_joints)
        total_loss += self.loss_weights['mse'] * mse_loss

        # 2. 骨骼长度约束
        bone_length_loss = self.bone_length_loss(pred_joints, target_joints, parents)
        total_loss += self.loss_weights['bone_length'] * bone_length_loss

        # 3. 对称性约束
        symmetry_loss = self.symmetry_loss(pred_joints)
        total_loss += self.loss_weights['symmetry'] * symmetry_loss

        # 4. 关节角度约束
        angle_loss = self.joint_angle_loss(pred_joints, parents)
        total_loss += self.loss_weights['angle'] * angle_loss

        return total_loss

    def bone_length_loss(self, pred_joints: jt.Var, target_joints: jt.Var, parents: jt.Var) -> jt.Var:
        """
        计算骨骼长度约束损失
        Args:
            pred_joints: [B, N, 3] 预测的关节点
            target_joints: [B, N, 3] 目标关节点
            parents: [N] 父节点索引
        Returns:
            loss: 骨骼长度损失
        """
        # 忽略根节点（父节点为-1）
        valid_mask = parents != -1
        valid_parents = parents[valid_mask]
        
        # 计算骨骼向量
        pred_bones = pred_joints[:, valid_mask] - pred_joints[:, valid_parents]
        target_bones = target_joints[:, valid_mask] - target_joints[:, valid_parents]

        # 计算骨骼长度
        pred_lengths = jt.norm(pred_bones, dim=2)
        target_lengths = jt.norm(target_bones, dim=2)

        # 计算长度差异
        length_diff = jt.abs(pred_lengths - target_lengths)
        return jt.mean(length_diff)

    def symmetry_loss(self, joints: jt.Var, symmetry_axis: str = 'x') -> jt.Var:
        """
        计算对称性约束损失
        Args:
            joints: [B, N, 3] 关节点
            symmetry_axis: 对称轴 ('x', 'y', 'z')
        Returns:
            loss: 对称性损失
        """
        # 定义左右对称的关节点对
        left_right_pairs = [
            (1, 2),   # 左右肩
            (4, 5),   # 左右肘
            (7, 8),   # 左右腕
            (10, 11), # 左右髋
            (13, 14), # 左右膝
            (16, 17)  # 左右踝
        ]

        symmetry_loss = 0.0
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[symmetry_axis]

        for left_idx, right_idx in left_right_pairs:
            left_joints = joints[:, left_idx]
            right_joints = joints[:, right_idx]

            # 翻转对称轴
            right_joints_flipped = right_joints.clone()
            right_joints_flipped[:, axis_idx] = -right_joints_flipped[:, axis_idx]

            # 计算对称损失
            pair_loss = jt.mean(jt.abs(left_joints - right_joints_flipped))
            symmetry_loss += pair_loss

        return symmetry_loss / max(len(left_right_pairs), 1)

    def joint_angle_loss(self, joints: jt.Var, parents: jt.Var) -> jt.Var:
        """
        计算关节角度约束损失
        Args:
            joints: [B, N, 3] 关节点
            parents: [N] 父节点索引
        Returns:
            loss: 关节角度损失
        """
        # 计算骨骼向量
        valid_mask = parents != -1
        valid_parents = parents[valid_mask]
        bones = joints[:, valid_mask] - joints[:, valid_parents]

        # 计算关节角度
        angles = self.compute_joint_angles(bones)

        # 定义关节角度限制
        angle_limits = {
            'shoulder': (-90, 180),
            'elbow': (0, 150),
            'hip': (-90, 90),
            'knee': (0, 150)
        }

        angle_loss = 0.0
        for joint_name, (min_angle, max_angle) in angle_limits.items():
            joint_angles = angles.get(joint_name, jt.zeros(1))
            below_min = jt.maximum(min_angle - joint_angles, 0)
            above_max = jt.maximum(joint_angles - max_angle, 0)
            angle_loss += jt.mean(below_min + above_max)

        return angle_loss

    def compute_joint_angles(self, bones: jt.Var) -> Dict[str, jt.Var]:
        """
        计算关节角度
        Args:
            bones: [B, N, 3] 骨骼向量
        Returns:
            angles: 关节角度字典
        """
        # 模拟关节名称到骨骼索引的映射
        joint_to_bone = {
            'shoulder': (1, 4),  # 肩到肘
            'elbow': (4, 7),    # 肘到腕
            'hip': (10, 13),    # 髋到膝
            'knee': (13, 16)    # 膝到踝
        }

        angles = {}
        for joint_name, (bone1_idx, bone2_idx) in joint_to_bone.items():
            bone1 = bones[:, bone1_idx]
            bone2 = bones[:, bone2_idx]

            # 计算夹角（度）
            cos_angle = jt.sum(bone1 * bone2, dim=1) / (jt.norm(bone1, dim=1) * jt.norm(bone2, dim=1) + 1e-8)
            angle = jt.acos(jt.clamp(cos_angle, -1.0, 1.0)) * 180 / jt.pi
            angles[joint_name] = angle

        return angles

class SkinLoss(nn.Module):
    '''
    TODO:可能可以加入关于平滑性的限制
    '''
    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()
        # 默认损失权重
        self.loss_weights = loss_weights or {
            'mse': 1.0,          # 与目标皮肤权重的MSE损失
            'validity': 0.1,     # 权重和为1的约束
            'sparsity': 0.1      # 稀疏性约束
        }
        # 验证权重
        for weight in self.loss_weights.values():
            assert weight >= 0, "Loss weights must be non-negative"
        self.mse = nn.MSELoss()

    def forward(self, pred_skin: jt.Var, target_skin: jt.Var) -> jt.Var:
        """
        计算皮肤权重损失
        Args:
            pred_skin: [B, N, J] 预测的皮肤权重
            target_skin: [B, N, J] 目标皮肤权重
        Returns:
            total_loss: 总损失
        """
        total_loss = 0.0

        # 1. MSE损失
        mse_loss = self.mse(pred_skin, target_skin)
        total_loss += self.loss_weights['mse'] * mse_loss

        # 2. 有效性约束（权重和为1）
        validity_loss = jt.mean(jt.abs(jt.sum(pred_skin, dim=-1) - 1.0))
        total_loss += self.loss_weights['validity'] * validity_loss

        # 3. 稀疏性约束（鼓励每个顶点只受少数几个关节影响）
        sparsity_loss = jt.sum(jt.abs(pred_skin))
        total_loss += self.loss_weights['sparsity'] * sparsity_loss

        return total_loss