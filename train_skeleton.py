import jittor as jt
import numpy as np
import os
import argparse
import time
import random
import copy # Import copy for deepcopy

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from models.skeleton import create_model
from dataset.format import parents

from models.metrics import J2J
import wandb
from scipy.spatial.transform import Rotation

# Set Jittor flags
jt.flags.use_cuda = 1

# 对称关节对列表（左右对称的关节索引）
symmetric_pairs = [
    (9,13), (8,12), (7,11), (6,10),
    (14,18), (15,19), (16,20), (17,21),
]

# --- 辅助函数 (保持不变，或根据需要转换为Jittor Tensor操作) ---
# 注意: reflect_across_plane 和 find_reflaection_plane_colomna 需要处理 Jittor Tensor

def reflect_across_plane(points, plane_point, plane_normal):
    """
    将点 points 绕平面 (plane_point, plane_normal) 做反射。
    points: Jittor tensor (B, N, 3)
    plane_point: Jittor tensor (B, 3)
    plane_normal: Jittor tensor (B, 3)
    """
    vec = points - plane_point.unsqueeze(1) # (B, N, 3) - (B, 1, 3)
    proj = (vec * plane_normal.unsqueeze(1)).sum(dim=-1, keepdims=True)  # (B, N, 1)
    reflected = points - 2 * proj * plane_normal.unsqueeze(1) # (B, N, 3)
    return reflected

def find_reflaection_plane_colomna(target, pairs=None): # pairs is not used inside but kept for compatibility
    """
    通过脊柱节点(0,1,2,3,4,5)计算人体的对称平面
    target: Jittor tensor (B, N, 3) 关节点坐标
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
    # 避免除以零
    spine_dir_norm = jt.norm(spine_dir, dim=-1, keepdims=True)
    spine_dir = spine_dir / (spine_dir_norm + 1e-6)
    
    # 3. 计算垂直于脊柱方向的向量作为平面法向量
    # 使用脊柱节点拟合一个平面
    spine_centered = spine_nodes - c.unsqueeze(1)  # (B, 6, 3)
    
    # 计算协方差矩阵 (Jittor的batch matmul)
    # (B, 3, 6) @ (B, 6, 3) -> (B, 3, 3)
    cov = jt.matmul(spine_centered.permute(0, 2, 1), spine_centered) 
    
    # 计算特征值和特征向量 (Jittor的SVD支持batch)
    U, S, V = jt.linalg.svd(cov)
    
    # 取最小特征值对应的特征向量作为平面法向量
    n = U[:, :, -1]  # (B, 3)
    
    # 确保法向量方向一致性：如果法向量与脊柱方向点积为负，则反转法向量
    dot_product = (n * spine_dir).sum(dim=-1, keepdims=True)
    n = jt.where(dot_product < 0, -n, n) 
    
    # 单位化法向量
    n_norm = jt.norm(n, dim=-1, keepdims=True)
    n = n / (n_norm + 1e-6)
    
    return c, n

def symmetry_loss(pred, target, pairs):
    """
    每个样本独立推断对称平面，根据反射点计算对称损失
    pred: (B, J, 3)
    target: (B, J, 3)
    pairs: 对称点对列表
    """
    if pairs is None or len(pairs) == 0:
        return jt.zeros(1)

    plane_c, plane_n = find_reflaection_plane_colomna(target, pairs)  # (B, 3), (B, 3)
    loss = jt.zeros(1) # Initialize loss as Jittor tensor

    for left_idx, right_idx in pairs:
        left_joint_pred = pred[:, left_idx, :]         # (B, 3)
        right_joint_pred = pred[:, right_idx, :]       # (B, 3)
        
        # 反射预测的右关节
        right_mirrored_pred = reflect_across_plane(right_joint_pred.unsqueeze(1), plane_c, plane_n).squeeze(1) # (B, 3)
        
        # 计算对称损失
        loss += jt.norm(left_joint_pred - right_mirrored_pred, dim=-1).mean()

    return loss / len(pairs)

def topology_loss(pred, target, parents):
    """保持父子节点之间的相对向量结构一致"""
    loss = jt.zeros(1) # Initialize loss as Jittor tensor
    for i, p in enumerate(parents):
        if p is None: continue # Use 'is None' for None check
        pred_vec = pred[:, i, :] - pred[:, p, :]
        target_vec = target[:, i, :] - target[:, p, :]
        loss += jt.norm(pred_vec - target_vec, dim=-1).mean()
    return loss

def chamfer_distance_jittor(set_a, set_b):
    """
    计算两个点集之间的Chamfer Distance。
    set_a: Jittor tensor (B, N, 3) 或 (B, J, 3)
    set_b: Jittor tensor (B, M, 3) 或 (B, J, 3)

    返回: Jittor tensor (scalar) - 批次平均的 Chamfer Distance (平方距离)。
    """
    # 计算 set_a 中每个点到 set_b 中所有点的平方欧氏距离
    dist_ab_sq = jt.sum((set_a.unsqueeze(2) - set_b.unsqueeze(1))**2, dim=-1) # (B, N, M)

    # 找到 set_a 中每个点到 set_b 中最近点的平方距离
    dist_a_to_b_sq = dist_ab_sq.min(dim=-1, keepdims=False) # (B, N)

    # 计算 set_b 中每个点到 set_a 中所有点的平方欧氏距离
    dist_ba_sq = jt.sum((set_b.unsqueeze(2) - set_a.unsqueeze(1))**2, dim=-1) # (B, M, N)
    # 找到 set_b 中每个点到 set_a 中最近点的平方距离
    dist_b_to_a_sq = dist_ba_sq.min(dim=-1, keepdims=False) # (B, M)

    # Chamfer Distance 是两个方向的平均
    cd_sq_per_sample = dist_a_to_b_sq.mean(dim=-1) + dist_b_to_a_sq.mean(dim=-1) # (B,)
    return cd_sq_per_sample.mean() # 返回批次平均的平方Chamfer Distance

def relative_position_loss(pred, target):
    """保持骨骼中心和相对位置一致性"""
    pred_center = pred.mean(dim=1, keepdims=True)
    target_center = target.mean(dim=1, keepdims=True)
    pred_rel = pred - pred_center
    target_rel = target - target_center
    return jt.norm(pred_rel - target_rel, dim=-1).mean()

# --- 数据增强函数 (操作 NumPy 数组) ---

def np_random_rotate_3d(vertices, joints, max_angle=30):
    """
    随机旋转3D模型 (NumPy版本)
    Args:
        vertices: 顶点坐标 (N, 3)
        joints: 骨骼节点位置 (J, 3)
        max_angle: 最大旋转角度（度）
    Returns:
        旋转后的顶点、骨骼
    """
    angles = np.random.uniform(-max_angle, max_angle, 3)
    R = Rotation.from_euler('xyz', angles, degrees=True)
    
    rotated_vertices = R.apply(vertices)
    rotated_joints = R.apply(joints)
    
    return rotated_vertices, rotated_joints

def np_random_scale(vertices, joints, scale_range=(0.9, 1.1)):
    """
    随机缩放3D模型 (NumPy版本)
    Args:
        vertices: 顶点坐标 (N, 3)
        joints: 骨骼节点位置 (J, 3)
        scale_range: 缩放范围
    Returns:
        缩放后的顶点、骨骼
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    
    scaled_vertices = vertices * scale
    scaled_joints = joints * scale
    
    return scaled_vertices, scaled_joints

def np_add_gaussian_noise(vertices, noise_std=0.01):
    """
    添加高斯噪声到顶点 (NumPy版本)
    Args:
        vertices: 顶点坐标 (N, 3)
        noise_std: 噪声标准差
    Returns:
        添加噪声后的顶点
    """
    noise = np.random.normal(0, noise_std, vertices.shape)
    return vertices + noise

def np_random_joint_perturbation(joints, max_offset=0.02):
    """
    随机扰动骨骼节点位置 (NumPy版本)
    Args:
        joints: 骨骼节点位置 (J, 3)
        max_offset: 最大偏移量
    Returns:
        扰动后的骨骼节点
    """
    offset = np.random.uniform(-max_offset, max_offset, joints.shape)
    return joints + offset

# --- 批处理数据增强函数 (在 Jittor 与 NumPy 之间转换) ---

def augment_data_for_batch(vertices_batch_jt, joints_batch_jt, 
                           rotate=True, scale=True, add_noise=True, perturb_joints=True,
                           max_angle=30, scale_range=(0.9, 1.1), noise_std=0.01, max_offset=0.02):
    """
    对一个批次的 Jittor Tensor 数据进行实时增强。
    Args:
        vertices_batch_jt: Jittor Tensor, 顶点数据 (B, N, 3)
        joints_batch_jt: Jittor Tensor, 骨骼节点数据 (B, J, 3)
        ... (增强参数)
    Returns:
        augmented_vertices_batch_jt: Jittor Tensor, 增强后的顶点数据 (B, N, 3)
        augmented_joints_batch_jt: Jittor Tensor, 增强后的骨骼节点数据 (B, J, 3)
    """
    augmented_vertices_list = []
    augmented_joints_list = []

    # 将 Jittor Tensor 转换为 NumPy 数组进行操作
    vertices_batch_np = vertices_batch_jt.numpy()
    joints_batch_np = joints_batch_jt.numpy()

    for i in range(vertices_batch_np.shape[0]): # 遍历批次中的每个样本
        current_vertices = vertices_batch_np[i].copy() # 确保操作的是副本
        current_joints = joints_batch_np[i].copy()

        if rotate:
            current_vertices, current_joints = \
                np_random_rotate_3d(current_vertices, current_joints, max_angle=max_angle)
        
        if scale:
            current_vertices, current_joints = \
                np_random_scale(current_vertices, current_joints, scale_range=scale_range)
        
        if add_noise:
            current_vertices = np_add_gaussian_noise(current_vertices, noise_std=noise_std)
        
        if perturb_joints:
            current_joints = np_random_joint_perturbation(current_joints, max_offset=max_offset)
        
        augmented_vertices_list.append(current_vertices)
        augmented_joints_list.append(current_joints)
    
    # 将增强后的 NumPy 数组列表转换回 Jittor Tensor
    augmented_vertices_batch_jt = jt.array(np.stack(augmented_vertices_list))
    augmented_joints_batch_jt = jt.array(np.stack(augmented_joints_list))

    return augmented_vertices_batch_jt, augmented_joints_batch_jt

def train(args,name):
    patience = 5
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    wandb.login(key="d485e6ef46797558aec48309977203a6795b178f")
    wandb.init(project="jittor", name=name)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        """Helper function to log messages to file and print to console"""
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    # Log training parameters
    log_message(f"Starting training with parameters: {args}")
    
    # Create model
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type
    )
    
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
        transform=transform,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            transform=transform,
        )
    else:
        val_loader = None
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            original_vertices, original_joints = data['vertices'], data['joints']
            
            # --- 实时数据增强 ---
            # 仅对训练数据进行增强
            # augmented_vertices, augmented_joints = augment_data_for_batch(
            #     original_vertices, 
            #     original_joints,
            #     rotate=True,         # 根据 args 控制是否启用
            #     scale=True,          # 根据 args 控制是否启用
            #     add_noise=True,      # 根据 args 控制是否启用
            #     perturb_joints=True, # 根据 args 控制是否启用
            #     max_angle=args.aug_max_angle, # 从命令行参数获取
            #     scale_range=(args.aug_scale_min, args.aug_scale_max), # 从命令行参数获取
            #     noise_std=args.aug_noise_std, # 从命令行参数获取
            #     max_offset=args.aug_max_offset # 从命令行参数获取
            # )

            # 使用增强后的数据作为模型输入和 GT
            # vertices = augmented_vertices.permute(0, 2, 1)  # [B, 3, N] for model input
            # joints_gt = augmented_joints.reshape(-1, 22, 3) # [B, J, 3] for GT

            #不使用每一轮
            vertices = original_vertices.permute(0, 2, 1)
            joints_gt = original_joints.reshape(-1, 22, 3)
            
            outputs = model(vertices)
            
            joints_pred = outputs.reshape(-1, 22, 3)

            # 基础 MSE loss
            loss_pos = criterion(joints_pred, joints_gt)

            # 拓扑先验
            loss_topo = topology_loss(joints_pred, joints_gt, parents)

            # 相对位置先验
            loss_rel = relative_position_loss(joints_pred, joints_gt)

            # 对称性损失
            loss_sym = symmetry_loss(joints_pred, joints_gt, pairs=symmetric_pairs)

            # loss_cd = chamfer_distance_jittor(joints_pred, joints_gt)
            # loss_J2J += J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item() / outputs.shape[0]

            loss_J2J = chamfer_distance_jittor(joints_pred, joints_gt)
            loss = loss_pos + args.lambda_topo * loss_topo + args.lambda_rel * loss_rel + args.lambda_sym * loss_sym + args.lambda_cd * loss_J2J

            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            # Calculate statistics
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Total Loss: {loss.item():.4f} "
                           f"Pos Loss: {loss_pos.item():.4f} "
                           f"Topo Loss: {loss_topo.item():.4f} "
                           f"Rel Loss: {loss_rel.item():.4f} "
                           f"Sym Loss: {loss_sym.item():.4f}"
                           f"J2J Loss: {loss_J2J:.6f}")# 新增 CD Loss
        
        # Calculate epoch statistics
        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss: {train_loss:.4f} "
                   f"Time: {epoch_time:.2f}s "
                   f"LR: {optimizer.lr:.6f}") 
        
        global_step = epoch * len(train_loader) + batch_idx
        wandb.log({
            "skeleton epoch": epoch + 1,
            "skeleton total_loss": train_loss,
            "skeleton pos_loss": loss_pos.item(),
            "skeleton topo_loss": loss_topo.item(),
            "skeleton rel_loss": loss_rel.item(),
            "skeleton sym_loss": loss_sym.item(),
            "skeleton learning_rate": optimizer.lr if hasattr(optimizer, 'lr') else args.learning_rate
        }, step=global_step)

        # Validation phase (No augmentation applied here)
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss = 0.0
            J2J_loss = 0.0
            
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                # Get data and labels (no augmentation for validation)
                vertices, joints = data['vertices'], data['joints']
                joints = joints.reshape(joints.shape[0], -1)
                
                # Reshape input if needed
                if vertices.ndim == 3:  # [B, N, 3]
                    vertices = vertices.permute(0, 2, 1)  # [B, 3, N]
                
                # Forward pass
                outputs = model(vertices)
                loss = criterion(outputs, joints)
                
                # export render results
                if batch_idx == show_id:
                    exporter = Exporter()
                    # export every joint's corresponding skinning
                    exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_ref.png", joints=joints[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_pred.png", joints=outputs[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_pc(path=f"tmp/skeleton/epoch_{epoch}/vertices.png", vertices=vertices[0].permute(1, 0).numpy())

                val_loss += loss.item()
                for i in range(outputs.shape[0]):
                    J2J_loss += J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item() / outputs.shape[0]
            
            # Calculate validation statistics
            val_loss /= len(val_loader)
            J2J_loss /= len(val_loader)
            
            log_message(f"Validation Loss: {val_loss:.4f} J2J Loss: {J2J_loss:.4f}")

            wandb.log({
                "skeleton epoch": epoch + 1,
                "skeleton val_loss": val_loss,
                "skeleton J2J_loss": J2J_loss
            },step=global_step)
            
            # Save best model
            if J2J_loss < best_loss:
                best_loss = J2J_loss
                no_improve_epochs = 0
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    optimizer.lr /= 2
                    no_improve_epochs = 0
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")

    wandb.finish()
    return model, best_loss

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, required=True,
                        help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='',
                        help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='newdata',
                        help='Root directory for the data files')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='pct',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton', 'sym'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')

    # Loss weights
    parser.add_argument('--lambda_topo', type=float, default=0.0,
                        help='Weight for topology loss')
    parser.add_argument('--lambda_rel', type=float, default=0.0,
                        help='Weight for relative position loss')
    parser.add_argument('--lambda_sym', type=float, default=0.0,
                        help='Weight for symmetry loss')
    parser.add_argument('--lambda_cd', type=float, default=0.8,
                        help='Weight for J2J loss')

    # --- Data Augmentation Parameters ---
    parser.add_argument('--aug_max_angle', type=float, default=30.0,
                        help='Max rotation angle in degrees for augmentation')
    parser.add_argument('--aug_scale_min', type=float, default=0.9,
                        help='Min scale factor for augmentation')
    parser.add_argument('--aug_scale_max', type=float, default=1.1,
                        help='Max scale factor for augmentation')
    parser.add_argument('--aug_noise_std', type=float, default=0.01,
                        help='Standard deviation for Gaussian noise augmentation')
    parser.add_argument('--aug_max_offset', type=float, default=0.02,
                        help='Max offset for joint perturbation augmentation')
    
    from datetime import datetime
    import os

    name = "lr+epoch300"
    default_output_dir = os.path.join('output', 'skeleton', name)

    # Output parameters
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    
    args = parser.parse_args()
    
    # Start training
    train(args,name=name)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()