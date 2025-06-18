import jittor as jt
import numpy as np
import os
import argparse
import time
import random

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.format import id_to_name
from dataset.sampler import SamplerMix
from models.skin import create_model

from dataset.exporter import Exporter

import wandb

# Set Jittor flags
jt.flags.use_cuda = 1

def train(args, name):
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
    )
    
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
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    # 新增：交叉熵损失
    # 注意：Jittor的CrossEntropyLoss默认包含LogSoftmax，
    # 因此模型输出outputs应该是未经过softmax的原始预测值 (logits)。
    # 如果你的模型outputs已经是经过softmax的，可能需要调整模型输出或损失函数。
    criterion_cross_entropy = nn.CrossEntropyLoss() 
    
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=1024, vertex_samples=512),
        transform=transform,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SamplerMix(num_samples=1024, vertex_samples=512),
            transform=transform,
        )
    else:
        val_loader = None
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss_mse = 0.0
        train_loss_l1 = 0.0
        train_loss_cross_entropy = 0.0 # 新增：交叉熵损失统计
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints, skin = data['vertices'], data['joints'], data['skin']

            vertices: jt.Var
            joints: jt.Var
            skin: jt.Var
            
            outputs = model(vertices, joints) # outputs是预测的蒙皮权重
            
            # 计算各种损失
            loss_mse = criterion_mse(outputs, skin)
            loss_l1 = criterion_l1(outputs, skin)
            
            # 计算交叉熵损失
            # 注意：对于nn.CrossEntropyLoss，target通常是类别索引或概率分布。
            # 如果skin是[batch_size, num_vertices, num_joints]的浮点权重，
            # 需要将其转换为与CrossEntropyLoss兼容的格式。
            # 通常，交叉熵用于单热编码的分类目标。
            # 如果您的skin是连续的概率分布，您可能需要先将skin转换为长整型索引，
            # 或者使用kl_div损失。
            # 这里假设您的'skin'数据虽然是浮点数，但其结构（最后一维是骨骼数量）
            # 可以被看作是某种多分类的概率分布（每个顶点属于哪个骨骼的影响最大）。
            # 为了使用CrossEntropyLoss，我们需要将 skin 转换为长整型类别索引。
            # 这需要假设每个顶点“最主要”受哪个骨骼影响。
            
            # 方案一：将浮点权重转换为离散类别索引（最简单粗暴）
            # 找到每个顶点在哪个骨骼上的权重最大，并将其作为“类别”
            # 注意：这种转换可能丢失很多信息，慎用！
            # 假设skin的形状是 [batch_size, num_vertices, num_joints]
            # cross_entropy_target = jt.argmax(skin, dim=-1) # shape: [batch_size, num_vertices]
            # loss_cross_entropy = criterion_cross_entropy(outputs.permute(0, 2, 1), cross_entropy_target) # outputs需要调整维度以匹配CrossEntropyLoss的期望
            #                                                                                               # CrossEntropyLoss期望 [N, C, ...] 而不是 [N, ..., C]
            
            # 方案二：如果 outputs 和 skin 都是概率分布，且和为1，考虑使用 nn.KLDivLoss
            # KL散度损失 (Kullback-Leibler divergence)
            # 这更适合衡量两个概率分布之间的差异。
            # Jittor的KLDivLoss需要LogSoftmax的输入。
            # 预测值 outputs 经过 log_softmax
            # 真实值 targets 保持原始概率分布
            
            # 假设 outputs 和 skin 的形状都是 [batch_size, num_vertices, num_joints]
            # 并且 num_joints 是分类的“类别”维度。
            # CrossEntropyLoss 期望输入是 [N, C, H, W] 或 [N, C]
            # 在您的例子中，C 是 num_joints。
            # 所以，outputs 的形状需要从 [batch_size, num_vertices, num_joints]
            # 变为 [batch_size * num_vertices, num_joints]
            # 或者 permute 为 [batch_size, num_joints, num_vertices] （如果num_vertices是空间维度）

            # 更稳妥的做法是：将每个顶点的骨骼权重视为一个多分类问题。
            # 将 outputs 和 skin 都 reshape 成 [Batch*Vertices, Num_Joints]
            # 然后 skin 作为一个概率分布，outputs 作为预测的logits。
            
            # 为了确保 CrossEntropyLoss 的正确使用，我们通常需要：
            # 1. 确保 targets 是 long 类型，且代表类别索引（0到C-1）。
            # 2. 确保 predictions（即 outputs）是 logits，且其形状的第二维是类别数。
            
            # 如果您的 'skin' 数据是连续的浮点权重，那么它更像是回归目标，
            # 而不是一个分类问题的one-hot或类别索引。
            # 如果要强制使用交叉熵，通常需要将 'skin' 处理成每个顶点由哪个骨骼“主导”的离散索引。
            # 但是，这会丢失“次要”骨骼的影响信息。

            # 另一种更合理的做法是，如果 outputs 是经过 softmax 的概率，而 skin 也是概率分布，
            # 那么可以使用 Jittor 的 KLDivLoss，并且 outputs 需要先经过 log()。
            # 例如: loss_cross_entropy = nn.KLDivLoss(reduction='batchmean')(jt.log(outputs), skin)

            # 在这里，为了简单地加入交叉熵并假设其适用于您的数据，
            # 我将使用一种常见的处理方式：
            # 假设outputs的最后一个维度是类别维度 (num_joints)，
            # 且skin的最后一个维度也是类别维度。
            # CrossEntropyLoss期望input形状为 (N, C, ...) 且target形状为 (N, ...)
            # 其中C是类别数。
            # 所以我们需要将outputs从 [B, V, J] permute 到 [B, J, V] 才能被识别为 [N, C, ...]
            # 且将 skin 从 [B, V, J] 转换为 [B, V] (每个顶点最影响的骨骼索引)

            # 如果你的 skin 数据是类似 one-hot 的，或者可以转化为索引，
            # 且输出 outputs 是 logits:
            # 假设 skin 是 [B, V, J] 且每个顶点的所有权重和为 1
            # 我们可以找到每个顶点的主导骨骼索引作为目标。
            # 这会将问题视为一个分类问题，每个顶点属于哪个骨骼。
            target_indices = jt.argmax(skin, dim=-1).long() # 找到每个顶点的最大权重所在的骨骼索引
            # outputs_permuted = outputs.permute(0, 2, 1) # 将维度从 [B, V, J] 变为 [B, J, V]
                                                       # 以匹配 CrossEntropyLoss 期望的 [N, C, d1, d2...]
            # 然而，对于 [B, V, J] 这种形状，更自然的理解是：
            # N = B * V (batch size * num_vertices)
            # C = J (num_joints)
            # 那么我们需要将outputs和target_indices reshape
            
            # Reshape outputs to [Batch*Vertices, Num_Joints]
            # Reshape target_indices to [Batch*Vertices]
            num_vertices = outputs.shape[1] # 获取顶点数量
            num_joints = outputs.shape[2] # 获取骨骼数量

            outputs_reshaped = outputs.view(-1, num_joints) # [B*V, J]
            target_indices_reshaped = target_indices.view(-1) # [B*V]
            
            # 计算交叉熵损失
            loss_cross_entropy = criterion_cross_entropy(outputs_reshaped, target_indices_reshaped)

            # 组合所有损失
            # 可以根据需要调整各项损失的权重
            loss = loss_mse + loss_l1 + loss_cross_entropy 
            
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            # Calculate statistics
            train_loss_mse += loss_mse.item()
            train_loss_l1 += loss_l1.item()
            train_loss_cross_entropy += loss_cross_entropy.item() # 统计交叉熵损失
            
            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss mse (L2): {loss_mse.item():.4f} Loss l1: {loss_l1.item():.4f} "
                           f"Loss CE: {loss_cross_entropy.item():.4f}")
        
        # Calculate epoch statistics
        train_loss_mse /= len(train_loader)
        train_loss_l1 /= len(train_loader)
        train_loss_cross_entropy /= len(train_loader) # 平均交叉熵损失
        epoch_time = time.time() - start_time
        
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss mse (L2): {train_loss_mse:.4f} "
                   f"Train Loss l1: {train_loss_l1:.4f} "
                   f"Train Loss CE: {train_loss_cross_entropy:.4f} " # 打印交叉熵损失
                   f"Time: {epoch_time:.2f}s "
                   f"LR: {optimizer.lr:.6f}")
        
        global_step = epoch * len(train_loader) + batch_idx
        wandb.log({
            "skin epoch": epoch + 1,
            "skin train_loss_mse (L2)": train_loss_mse,
            "skin train_loss_l1": train_loss_l1,
            "skin train_loss_cross_entropy": train_loss_cross_entropy, # WandB记录交叉熵损失
            "skin learning_rate": optimizer.lr if hasattr(optimizer, 'lr') else args.learning_rate
        },step=global_step)
        
        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss_mse = 0.0
            val_loss_l1 = 0.0
            val_loss_cross_entropy = 0.0 # 新增：验证集交叉熵损失统计
            
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                # Get data and labels
                vertices, joints, skin = data['vertices'], data['joints'], data['skin']
                
                # Forward pass
                outputs = model(vertices, joints)
                loss_mse = criterion_mse(outputs, skin)
                loss_l1 = criterion_l1(outputs, skin)
                
                # 计算验证集交叉熵损失
                # 同训练集一样处理，确保维度匹配
                target_indices = jt.argmax(skin, dim=-1).long()
                outputs_reshaped = outputs.view(-1, num_joints)
                target_indices_reshaped = target_indices.view(-1)
                loss_cross_entropy = criterion_cross_entropy(outputs_reshaped, target_indices_reshaped)

                val_loss_mse += loss_mse.item()
                val_loss_l1 += loss_l1.item()
                val_loss_cross_entropy += loss_cross_entropy.item() # 统计验证集交叉熵损失
            
            # Calculate validation statistics
            val_loss_mse /= len(val_loader)
            val_loss_l1 /= len(val_loader)
            val_loss_cross_entropy /= len(val_loader) # 平均验证集交叉熵损失
            
            log_message(f"Validation Loss: mse (L2): {val_loss_mse:.4f} l1: {val_loss_l1:.4f} "
                       f"CE: {val_loss_cross_entropy:.4f}") # 打印验证集交叉熵损失
            wandb.log({
                "skin epoch": epoch + 1,
                "skin val_loss_mse (L2)": val_loss_mse,
                "skin val_loss_l1": val_loss_l1,
                "skin val_loss_cross_entropy": val_loss_cross_entropy # WandB记录验证集交叉熵损失
            },step=global_step)
            
            # Save best model - 考虑哪个损失作为最佳模型的评判标准
            # 这里仍然使用 val_loss_l1 作为评判标准，您可以根据需要更改为 val_loss_cross_entropy 或组合损失
            if val_loss_l1 < best_loss: 
                best_loss = val_loss_l1
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with l1 loss {best_loss:.4f} to {model_path}")
        
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
    parser.add_argument('--epochs', type=int, default=1000,
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
    

    name = "Big+data"
    default_output_dir = os.path.join('output', 'skin', name)

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