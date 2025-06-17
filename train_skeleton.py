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

def reflect_points(points, axis):
    """
    对points绕axis轴做反射变换。
    points: (B, 3)
    axis: (3,) 单位向量
    返回: 反射后的points (B,3)
    反射公式: p' = p - 2 * n * (n^T p)
    """
    # axis是单位向量 (3,)
    proj = jt.matmul(points, axis.reshape(3,1))  # (B,1)
    reflected = points - 2 * proj * axis  # (B,3)
    return reflected

def find_reflection_axis(target, pairs):
    """
    计算最优反射轴
    target: (B, N, 3) 真实关节
    pairs: 左右对称关节索引对列表
    返回：单位向量axis (3,)
    """
    B = target.shape[0]
    left_pts = []
    right_pts = []
    for left, right in pairs:
        left_pts.append(target[:, left, :])    # (B,3)
        right_pts.append(target[:, right, :])  # (B,3)
    left_pts = jt.stack(left_pts, dim=1)  # (B, num_pairs, 3)
    right_pts = jt.stack(right_pts, dim=1)  # (B, num_pairs, 3)

    # 中心化
    mean_pts = (left_pts + right_pts) / 2  # 对称中心 (B, num_pairs, 3)
    left_centered = left_pts - mean_pts
    right_centered = right_pts - mean_pts

    # 计算左右差向量平均
    diff = left_centered + right_centered  # 反射性质：p_left = R(p_right), R为反射矩阵
    diff = diff.reshape(-1, 3)  # (B*num_pairs, 3)

    # 用SVD求解最优反射轴：
    # 反射轴是使得diff投影最小的方向（最小奇异值对应的方向）
    # 计算协方差矩阵
    cov = jt.matmul(diff.transpose(1, 0), diff)  # (3,3)
    eigvals, eigvecs = jt.linalg.eigh(cov)  # eigvecs: (3,3), eigvals: (3,)
    axis = eigvecs[:, 0]

    axis = axis / jt.norm(axis)
    return axis

def symmetry_loss(pred, target, pairs=None):
    """
    通过target自动推断反射轴，计算pred对称损失
    """
    if pairs is None:
        return jt.zeros(1)
    
    axis = find_reflection_axis(target, pairs)  # 形状 (3,)

    loss = 0.0
    for left, right in pairs:
        left_joint = pred[:, left, :]
        right_joint = pred[:, right, :]
        # 右关节绕axis轴反射
        right_mirrored = reflect_points(right_joint, axis)
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
            
            loss = loss_pos + 0.5 * loss_topo + 0.5 * loss_rel + 0.5 * loss_sym
            
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            # Calculate statistics
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {loss.item():.4f}")
        
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
            "skeleton train_loss": train_loss,
            "skeleton learning_rate": optimizer.lr if hasattr(optimizer, 'lr') else args.learning_rate
        },step=global_step)

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
    parser.add_argument('--data_root', type=str, default='data',
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
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output/skeleton',
                        help='Directory to save output files')
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