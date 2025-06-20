import jittor as jt
from jittor import nn  
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    # xyz = xyz.contiguous()
    sampler = FurthestPointSampler(npoint)
    _, fps_idx = sampler(xyz) # [B, npoint]
    # print ('fps size=', fps_idx.size())
    # fps_idx = sampler(xyz).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points



class Point_Transformer2(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer2, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        # self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = Point_Transformer_Last()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(scale=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def execute(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        # add position embedding on each layer
        x = self.pt_last(feature_1, new_xyz)
        x = concat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class MHABlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm = nn.BatchNorm1d(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

    def execute(self, x):
        """
        x: (B, C, N)
        Return: (B, C, N)
        """
        B, C, N = x.shape
        x_in = x
        x = x.permute(0, 2, 1)         # (B, N, C)
        x = self.attn(x, x, x)         # (B, N, C)
        x = x.permute(0, 2, 1)         # (B, C, N)
        x = x + x_in                   # residual
        x = self.norm(x)
        x = self.mlp(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projection
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出映射
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def execute(self, query, key, value):
        """
        query/key/value: (B, N, C)
        return: (B, N, C)
        """
        B, N, C = query.shape
        H = self.num_heads
        D = self.head_dim

        Q = self.q_proj(query).reshape(B, N, H, D).permute(0, 2, 1, 3)  # (B, H, N, D)
        K = self.k_proj(key).reshape(B, N, H, D).permute(0, 2, 1, 3)
        V = self.v_proj(value).reshape(B, N, H, D).permute(0, 2, 1, 3)

        # Attention scores: (B, H, N, N)
        attn_scores = jt.matmul(Q, K.transpose(2, 3)) * self.scale
        attn_weights = nn.softmax(attn_scores, dim=-1)

        # Attention output: (B, H, N, D)
        attn_output = jt.matmul(attn_weights, V)

        # 拼接多头: (B, N, C)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, C)

        return self.out_proj(attn_output)

# class Point_Transformer_Multi(nn.Module):
#     def __init__(self, output_channels=40):
#         super(Point_Transformer_Multi, self).__init__()
        
#         self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(128)

#         self.sa1 = MHABlock(128, num_heads=4)
#         self.sa2 = MHABlock(128, num_heads=4)
#         self.sa3 = MHABlock(128, num_heads=4)
#         self.sa4 = MHABlock(128, num_heads=4)

#         self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
#                                    nn.BatchNorm1d(1024),
#                                    nn.LeakyReLU(scale=0.2))

#         self.linear1 = nn.Linear(1024, 512, bias=False)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.dp1 = nn.Dropout(p=0.5)
#         self.linear2 = nn.Linear(512, 256)
#         self.bn7 = nn.BatchNorm1d(256)
#         self.dp2 = nn.Dropout(p=0.5)
#         self.linear3 = nn.Linear(256, output_channels)

#         self.relu = nn.ReLU()
        
#     def execute(self, x):
#         # x is expected to be [B, 3, N]
#         batch_size, C, N = x.size()
        
#         # Store original input for xyz coordinates
#         x_input = x
        
#         # Apply convolutions
#         x = self.relu(self.bn1(self.conv1(x)))  # B, 128, N
#         x = self.relu(self.bn2(self.conv2(x)))  # B, 128, N

#         # Apply self-attention layers with xyz coordinates
#         x1 = self.sa1(x)
#         x2 = self.sa2(x1)
#         x3 = self.sa3(x2)
#         x4 = self.sa4(x3)
        
#         # Concatenate features from all SA layers
#         x = concat((x1, x2, x3, x4), dim=1)

#         x = self.conv_fuse(x)
#         # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         x = jt.max(x, 2)
#         x = x.view(batch_size, -1)
#         x = self.relu(self.bn6(self.linear1(x)))
#         x = self.dp1(x)
#         x = self.relu(self.bn7(self.linear2(x)))
#         x = self.dp2(x)
#         x = self.linear3(x)
#         return x

class MultiHeadSA_Layer(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(MultiHeadSA_Layer, self).__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Q, K, V for all heads
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)

        # Position encoding projection
        self.xyz_conv = nn.Conv1d(3, channels, 1, bias=False)

        # 输出映射层
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.head_dim ** -0.5

    def execute(self, x, xyz):
        # x: [B, C, N], xyz: [B, 3, N]
        B, C, N = x.shape
        H, D = self.num_heads, self.head_dim

        # 位置编码
        pos_encoding = self.xyz_conv(xyz)  # [B, C, N]
        x = x + pos_encoding

        # Q, K, V 投影
        q = self.q_conv(x).reshape(B, H, D, N)      # [B, H, D, N]
        k = self.k_conv(x).reshape(B, H, D, N)      # [B, H, D, N]
        v = self.v_conv(x).reshape(B, H, D, N)      # [B, H, D, N]

        q = q.permute(0, 1, 3, 2)                   # [B, H, N, D]
        k_t = k                                      # [B, H, D, N]

        # 注意力分数: [B, H, N, N]
        attn = jt.matmul(q, k_t) * self.scale
        attn = self.softmax(attn)
        attn = attn / (1e-9 + attn.sum(dim=2, keepdims=True))  # 归一化

        # 注意力加权值: [B, H, D, N]
        out = jt.matmul(v, attn.permute(0, 1, 3, 2))  # [B, H, D, N]

        # 拼接多头: [B, C, N]
        out = out.reshape(B, C, N)

        # 残差更新
        x_r = self.act(self.after_norm(self.trans_conv(x - out)))
        x = x + x_r

        return x

class Point_Transformer_Multi(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer_Multi, self).__init__()

        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = MultiHeadSA_Layer(128, num_heads=4)
        self.sa2 = MultiHeadSA_Layer(128, num_heads=4)
        self.sa3 = MultiHeadSA_Layer(128, num_heads=4)
        self.sa4 = MultiHeadSA_Layer(128, num_heads=4)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(128 * 4, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(scale=0.2)
        )

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)

        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)

        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()

    def execute(self, x):
        # x: [B, 3, N]
        batch_size, C, N = x.shape

        xyz = x  # 作为坐标信息传入注意力模块

        # Point feature extraction
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 128, N]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 128, N]

        # Apply self-attention with coordinate input
        x1 = self.sa1(x, xyz)  # [B, 128, N]
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)

        # Concatenate outputs from all attention layers
        x = concat([x1, x2, x3, x4], dim=1)  # [B, 128*4, N]

        # Fuse and global max pooling
        x = self.conv_fuse(x)               # [B, 1024, N]
        x = jt.max(x, dim=2)                # [B, 1024]
        x = x.view(batch_size, -1)

        # Fully connected layers
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Point_Transformer(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(scale=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()
        
    def execute(self, x):
        # x is expected to be [B, 3, N]
        batch_size, C, N = x.size()
        
        # Store original input for xyz coordinates
        x_input = x
        
        # Apply convolutions
        x = self.relu(self.bn1(self.conv1(x)))  # B, 128, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, 128, N

        # Apply self-attention layers with xyz coordinates
        x1 = self.sa1(x, x_input)
        x2 = self.sa2(x1, x_input)
        x3 = self.sa3(x2, x_input)
        x4 = self.sa4(x3, x_input)
        
        # Concatenate features from all SA layers
        x = concat((x1, x2, x3, x4), dim=1)

        x = self.conv_fuse(x)
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Point_Transformer_Big(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer_Big, self).__init__()
        
        channels = 128
        self.conv1 = nn.Conv1d(3, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        
        # SA 在点云中学习局部结构之间的依赖关系，通过自注意力机制增强每个点的特征表达能力。
        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)
        self.sa5 = SA_Layer(channels)
        self.sa6 = SA_Layer(channels)
        self.sa7 = SA_Layer(channels)
        self.sa8 = SA_Layer(channels)

        self.conv_fuse = nn.Sequential(nn.Conv1d(8*channels, 16*channels, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(16*channels),
                                   nn.LeakyReLU(scale=0.2))

        self.linear1 = nn.Linear(16*channels, 8*channels, bias=False)
        self.bn6 = nn.BatchNorm1d(8*channels)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(8*channels, 4*channels)
        self.bn7 = nn.BatchNorm1d(4*channels)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(4*channels, 2*channels)
        self.bn8 = nn.BatchNorm1d(2*channels)
        self.dp3 = nn.Dropout(p=0.5)
        self.linear4 = nn.Linear(2*channels, channels)
        self.bn9 = nn.BatchNorm1d(channels)
        self.dp4 = nn.Dropout(p=0.5)
        self.linear5 = nn.Linear(channels, output_channels)

        self.relu = nn.ReLU()
        
    def execute(self, x):
        # x is expected to be [B, 3, N]
        batch_size, C, N = x.size()
        
        # Store original input for xyz coordinates
        x_input = x
        
        # Apply convolutions
        x = self.relu(self.bn1(self.conv1(x)))  # B, 128, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, 128, N

        # Apply self-attention layers with xyz coordinates
        x1 = self.sa1(x, x_input) # 第一个值：每个点的特征语义表示 第二个值：空间位置信息
        x2 = self.sa2(x1, x_input)
        x3 = self.sa3(x2, x_input)
        x4 = self.sa4(x3, x_input)
        x5 = self.sa5(x4, x_input)
        x6 = self.sa6(x5, x_input)
        x7 = self.sa7(x6, x_input)
        x8 = self.sa8(x7, x_input)
        
        # Concatenate features from all SA layers
        x = concat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        x = self.conv_fuse(x)
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.relu(self.bn8(self.linear3(x)))
        x = self.dp3(x)
        x = self.relu(self.bn9(self.linear4(x)))
        x = self.dp4(x)
        x = self.linear5(x)
        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv_pos = nn.Conv1d(3, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)
        self.relu = nn.ReLU()

    def execute(self, x, xyz):
        batch_size, _, N = x.size()
        xyz = xyz.permute(0, 2, 1)
        pos_encoding = self.conv_pos(xyz)
        
        x = self.relu(self.bn1(self.conv1(x)))
        
        x1 = self.sa1(x, pos_encoding)
        x2 = self.sa2(x1, pos_encoding)
        x3 = self.sa3(x2, pos_encoding)
        x4 = self.sa4(x3, pos_encoding)
        x = concat((x1, x2, x3, x4), dim=1)
        return x

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x



class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        # 添加位置编码转换层
        # self.pos_conv = nn.Conv1d(3, channels, 1, bias=False)
        self.xyz_conv = nn.Conv1d(3, channels, 1, bias=False)

    def execute(self, x, xyz):
        # 将3通道的位置信息转换为channels通道
        # pos_encoding = self.pos_conv(xyz)  # [B, channels, N]
        pos_encoding = self.xyz_conv(xyz)
        
        # 现在可以安全地相加
        x = x + pos_encoding
        
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x) # b, c, n        
        x_v = self.v_conv(x)
        energy = nn.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = nn.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

if __name__ == '__main__':
    
    jt.flags.use_cuda=1
    input_points = init.gauss((16, 3, 1024), dtype='float32')  # B, D, N 


    network = Point_Transformer()
    out_logits = network(input_points)
    print (out_logits.shape)

