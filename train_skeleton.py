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
import wandb

# Set Jittor flags
jt.flags.use_cuda = 1

# 对称关节对列表（左右对称的关节索引）
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

def topology_loss(pred, target, parents):
    """保持父子节点之间的相对向量结构一致"""
    # print("parents", parents)
    loss = 0.0
    for i, p in enumerate(parents):
        if p == None: continue
        pred_vec = pred[:, i, :] - pred[:, p, :]
        target_vec = target[:, i, :] - target[:, p, :]
        loss += jt.norm(pred_vec - target_vec, dim=-1).mean()
    return loss

def relative_position_loss(pred, target):
    """保持骨骼中心和相对位置一致性"""
    pred_center = pred.mean(dim=1, keepdims=True)
    target_center = target.mean(dim=1, keepdims=True)
    pred_rel = pred - pred_center
    target_rel = target - target_center
    return jt.norm(pred_rel - target_rel, dim=-1).mean()

def train(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    wandb.login(key="")
    wandb.init(project="jittor", name="loss",mode="offline")

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
            vertices, joints = data['vertices'], data['joints']

            vertices = vertices.permute(0, 2, 1)  # [B, 3, N]

            outputs = model(vertices)
            # joints = joints.reshape(outputs.shape[0], -1)
            # loss = criterion(outputs, joints, parents)

            joints_pred = outputs.reshape(-1, 22, 3)
            joints_gt = joints.reshape(-1, 22, 3)

            # 基础 MSE loss
            loss_pos = criterion(joints_pred, joints_gt)

            # 拓扑先验
            loss_topo = topology_loss(joints_pred, joints_gt, parents)

            # 相对位置先验
            loss_rel = relative_position_loss(joints_pred, joints_gt)

            # 对称性损失
            loss_sym = symmetry_loss(joints_pred, joints_gt, pairs=symmetric_pairs)
            
            loss = loss_pos + 0.2 * loss_topo + 0.2 * loss_rel + 0.5 * loss_sym
            
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
                           f"Sym Loss: {loss_sym.item():.4f}")
        
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
            "skeleton total_loss": loss.item(),
            "skeleton pos_loss": loss_pos.item(),
            "skeleton topo_loss": loss_topo.item(),
            "skeleton rel_loss": loss_rel.item(),
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
                # Get data and labels
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
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
        
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
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    from datetime import datetime
    import os

    now = datetime.now().strftime("%Y%m%d_%H%M%S") 
    default_output_dir = os.path.join('output', now, 'skeleton')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='Directory to save output files')
    os.makedirs(args.output_dir, exist_ok=True)
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    
    args = parser.parse_args()
    
    # Start training
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()