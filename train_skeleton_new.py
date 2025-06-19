from pickletools import optimize
import jittor as jt
import numpy as np
import os
import argparse
import time
import random
import copy # Import copy for deepcopy

from jittor import nn
from jittor import optim
from jittor import lr_scheduler

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

# --- 辅助函数 (保持不变) ---
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

def find_reflaection_plane_colomna(target, pairs=None):
    """
    通过脊柱节点(0,1,2,3,4,5)计算人体的对称平面
    target: Jittor tensor (B, N, 3) 关节点坐标
    返回：c (B, 3), n (B, 3) 表示过点c、法向量为n的对称平面
    """
    spine_nodes = target[:, 0:6, :]
    c = spine_nodes.mean(dim=1)
    spine_start = spine_nodes[:, 0, :]
    spine_end = spine_nodes[:, -1, :]
    spine_dir = spine_end - spine_start
    spine_dir_norm = jt.norm(spine_dir, dim=-1, keepdims=True)
    spine_dir = spine_dir / (spine_dir_norm + 1e-6)
    spine_centered = spine_nodes - c.unsqueeze(1)
    cov = jt.matmul(spine_centered.permute(0, 2, 1), spine_centered)
    U, S, V = jt.linalg.svd(cov)
    n = U[:, :, -1]
    dot_product = (n * spine_dir).sum(dim=-1, keepdims=True)
    n = jt.where(dot_product < 0, -n, n)
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

    plane_c, plane_n = find_reflaection_plane_colomna(target, pairs)
    loss = jt.zeros(1)

    # 从目标关节中提取左右关节以计算实际距离
    left_joints_target = jt.stack([target[:, p[0], :] for p in pairs], dim=1) # (B, num_pairs, 3)
    right_joints_target = jt.stack([target[:, p[1], :] for p in pairs], dim=1) # (B, num_pairs, 3)
    
    # 计算目标骨骼的实际对称关节距离
    target_dist = jt.norm(left_joints_target - right_joints_target, dim=-1) # (B, num_pairs)

    # 从预测关节中提取左右关节
    left_joints_pred = jt.stack([pred[:, p[0], :] for p in pairs], dim=1)
    right_joints_pred = jt.stack([pred[:, p[1], :] for p in pairs], dim=1)

    # 计算预测骨骼的对称关节距离
    pred_dist = jt.norm(left_joints_pred - right_joints_pred, dim=-1)

    # 对称性损失为预测距离与实际距离的MSE
    loss = nn.mse_loss(pred_dist, target_dist)
    
    return loss

def bone_length_loss(pred, target, parents):
    """
    保持骨骼长度一致性 (MSE on bone lengths)
    """
    loss = jt.zeros(1)
    
    # 提取所有骨骼的向量
    pred_bones = []
    target_bones = []
    for i, p in enumerate(parents):
        if p is None: continue
        pred_bones.append(pred[:, i, :] - pred[:, p, :])
        target_bones.append(target[:, i, :] - target[:, p, :])
    
    pred_bones = jt.stack(pred_bones, dim=1) # (B, num_bones, 3)
    target_bones = jt.stack(target_bones, dim=1) # (B, num_bones, 3)

    # 计算骨骼长度
    pred_lengths = jt.norm(pred_bones, dim=-1) # (B, num_bones)
    target_lengths = jt.norm(target_bones, dim=-1) # (B, num_bones)

    # 计算长度的MSE
    return nn.mse_loss(pred_lengths, target_lengths)


# --- 以下函数保持不变 ---
def topology_loss(pred, target, parents):
    """保持父子节点之间的相对向量结构一致"""
    loss = jt.zeros(1)
    for i, p in enumerate(parents):
        if p is None: continue
        pred_vec = pred[:, i, :] - pred[:, p, :]
        target_vec = target[:, i, :] - target[:, p, :]
        loss += jt.norm(pred_vec - target_vec, dim=-1).mean()
    return loss

def chamfer_distance_jittor(set_a, set_b):
    dist_ab_sq = jt.sum((set_a.unsqueeze(2) - set_b.unsqueeze(1))**2, dim=-1)
    dist_a_to_b_sq = dist_ab_sq.min(dim=-1, keepdims=False)
    dist_ba_sq = jt.sum((set_b.unsqueeze(2) - set_a.unsqueeze(1))**2, dim=-1)
    dist_b_to_a_sq = dist_ba_sq.min(dim=-1, keepdims=False)
    cd_sq_per_sample = dist_a_to_b_sq.mean(dim=-1) + dist_b_to_a_sq.mean(dim=-1)
    return cd_sq_per_sample.mean()

def relative_position_loss(pred, target):
    pred_center = pred.mean(dim=1, keepdims=True)
    target_center = target.mean(dim=1, keepdims=True)
    pred_rel = pred - pred_center
    target_rel = target - target_center
    return jt.norm(pred_rel - target_rel, dim=-1).mean()
# --- 数据增强函数 (未使用，保持不变) ---
def np_random_rotate_3d(vertices, joints, max_angle=30):
    angles = np.random.uniform(-max_angle, max_angle, 3)
    R = Rotation.from_euler('xyz', angles, degrees=True)
    rotated_vertices = R.apply(vertices)
    rotated_joints = R.apply(joints)
    return rotated_vertices, rotated_joints
# ... (其他数据增强函数保持不变) ...

# --- 训练函数 (修改以实现图片要求) ---
def train(args,name):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    wandb.login(key="237f5f006b70965c90d08069c63cc560ae78feb4")
    wandb.init(project="jittor", name=name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    log_message(f"Starting training with parameters: {args}")
    
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type
    )
    
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
    
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # 基础损失函数
    criterion = nn.MSELoss()
    
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
    
    best_loss = 99999999
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # 获取数据，根据图片要求【不做数据增强】
            original_vertices, original_joints = data['vertices'], data['joints']
            
            # 直接使用原始数据
            vertices = original_vertices.permute(0, 2, 1)
            joints_gt = original_joints.reshape(-1, 22, 3)
            
            outputs = model(vertices)
            joints_pred = outputs.reshape(-1, 22, 3)

            # --- 根据图片要求构建损失函数 ---
            
            # 基础的关节位置损失 (图片中未明确提及，但通常是基础)
            loss_pos = criterion(joints_pred, joints_gt)

            # 骨骼长度损失 (MSE, 权重是1)
            loss_bone = bone_length_loss(joints_pred, joints_gt, parents)

            # 对称性损失 (权重加一倍)
            # 注意：这里的对称性损失直接比较预测关节和实际关节的对称距离，更符合“对对称关节的距离跟实际一致”
            loss_sym = symmetry_loss(joints_pred, joints_gt, pairs=symmetric_pairs)

            # 总损失: 加起来
            loss = loss_pos + args.lambda_bone * loss_bone + args.lambda_sym * loss_sym
            
            # --- 结束损失函数构建 ---

            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Total Loss: {loss.item():.4f} "
                           f"Pos Loss: {loss_pos.item():.4f} "
                           f"Bone Loss: {loss_bone.item():.4f} "
                           f"Sym Loss: {loss_sym.item():.4f}")
        
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
            "skeleton bone_loss": loss_bone.item(), # Log bone loss
            "skeleton sym_loss": loss_sym.item(),
            "skeleton learning_rate": optimizer.lr if hasattr(optimizer, 'lr') else args.learning_rate
        }, step=global_step)

        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss = 0.0
            J2J_loss = 0.0
            
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                vertices, joints = data['vertices'], data['joints']
                joints = joints.reshape(joints.shape[0], -1)
                
                if vertices.ndim == 3:
                    vertices = vertices.permute(0, 2, 1)
                
                outputs = model(vertices)
                loss = criterion(outputs, joints)
                
                if batch_idx == show_id:
                    exporter = Exporter()
                    os.makedirs(f"tmp/skeleton/epoch_{epoch}", exist_ok=True)
                    exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_ref.png", joints=joints[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_pred.png", joints=outputs[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_pc(path=f"tmp/skeleton/epoch_{epoch}/vertices.png", vertices=vertices[0].permute(1, 0).numpy())

                val_loss += loss.item()
                for i in range(outputs.shape[0]):
                    J2J_loss += J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item() / outputs.shape[0]
            
            val_loss /= len(val_loader)
            J2J_loss /= len(val_loader)
            
            log_message(f"Validation Loss: {val_loss:.4f} J2J Loss: {J2J_loss:.4f}")

            wandb.log({
                "skeleton epoch": epoch + 1,
                "skeleton val_loss": val_loss,
                "skeleton J2J_loss": J2J_loss
            },step=global_step)
            
            if J2J_loss < best_loss:
                best_loss = J2J_loss
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
        
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")
    
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")

    wandb.finish()
    return model, best_loss

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, required=True, help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='', help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='newdata', help='Root directory for the data files')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='pct', choices=['pct', 'pct2', 'custom_pct', 'skeleton', 'sym'], help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'enhanced'], help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='', help='Path to pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs') # 轮数不改
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer to use')
    # --- MODIFIED: learningrate 提高 ---
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate') # 原为 0.0001
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')

    # --- MODIFIED: Loss weights based on image ---
    # “骨骼长度(MSE, 权重是1)” --> 使用新的 bone_length_loss
    parser.add_argument('--lambda_bone', type=float, default=1.0, help='Weight for bone length loss')
    # “对称性 ... 权重加一倍”
    parser.add_argument('--lambda_sym', type=float, default=2.0, help='Weight for symmetry loss') 
    
    # --- DEPRECATED/UNUSED losses (set weight to 0) ---
    # These are kept for argument compatibility but disabled by default
    parser.add_argument('--lambda_topo', type=float, default=0.0, help='Weight for topology loss (vector-based, deprecated by bone length loss)')
    parser.add_argument('--lambda_rel', type=float, default=1.0, help='Weight for relative position loss')
    parser.add_argument('--lambda_cd', type=float, default=0.5, help='Weight for J2J/Chamfer loss in training')

    # --- Data Augmentation Parameters (kept for compatibility but not used) ---
    parser.add_argument('--aug_max_angle', type=float, default=30.0, help='Max rotation angle in degrees for augmentation')
    # ... (other augmentation args)
    
    from datetime import datetime
    import os

    name = "lr0.001_sym2_bone1_cd0.5_rel1" # Changed name to reflect new settings
    default_output_dir = os.path.join('output', 'skeleton', name)

    parser.add_argument('--output_dir', type=str, default=default_output_dir, help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    
    args = parser.parse_args()
    
    train(args,name=name)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()