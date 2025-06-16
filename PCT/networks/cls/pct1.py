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
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128) # in_channels = 3 (xyz_norm) + 64 (new_points) + 64 (new_points repeat)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256) # in_channels = 3 (xyz_norm) + 128 (feature_0) + 128 (feature_0 repeat)
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
        xyz = x.permute(0, 2, 1)  # B, N, 3
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N (3 -> 64)
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N (64 -> 64)
        x = x.permute(0, 2, 1)  # B, N, D
        
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)

        feature_0 = self.gather_local_0(new_feature)  # B, 128, 512
        feature = feature_0.permute(0, 2, 1)  # B, 512, 128
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature)
        
        feature_1 = self.gather_local_1(new_feature)  # B, 256, 256 (输出通道数, 采样点数)

        x = self.pt_last(feature_1, new_xyz) # B, 1024, 256 (4 * 256 = 1024)
        
        x = concat([x, feature_1], dim=1) # B, 1280, 256

        x = self.conv_fuse(x)  # B, 1024, 256
        x = jt.max(x, 2)  # Global Max Pooling: B, 1024
        x = x.view(batch_size, -1)

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

        # SA_Layer 接收特征和 xyz 坐标
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
        x_input_xyz = x # B, 3, N

        # Apply convolutions to extract initial features
        x = self.relu(self.bn1(self.conv1(x)))  # B, 128, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, 128, N

        # Apply self-attention layers with xyz coordinates.
        # SA_Layer expects x (features) and xyz (coordinates).
        # Both should be B, C, N. Here x is B, 128, N and x_input_xyz is B, 3, N.
        # The SA_Layer internally projects xyz to match feature dimensions.
        x1 = self.sa1(x, x_input_xyz)
        x2 = self.sa2(x1, x_input_xyz)
        x3 = self.sa3(x2, x_input_xyz)
        x4 = self.sa4(x3, x_input_xyz)

        # Concatenate features from all SA layers
        # x1, x2, x3, x4 all have shape B, 128, N
        x = concat((x1, x2, x3, x4), dim=1) # B, 512, N

        x = self.conv_fuse(x) # B, 1024, N
        # Global Max Pooling
        x = jt.max(x, 2) # B, 1024
        x = x.view(batch_size, -1)

        # MLP for classification
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        
        # Original: self.conv_pos = nn.Conv1d(3, channels, kernel_size=1, bias=False)
        # Renamed for clarity and to avoid confusion, pos_embed makes more sense for positional embedding.
        self.pos_embed = nn.Conv1d(3, channels, kernel_size=1, bias=False) 

        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()

    def execute(self, x, xyz):
        batch_size, _, N = x.size()
        # x shape: B, C_in, N_sampled (e.g., B, 256, 256)
        # xyz shape: B, N_sampled, 3 (e.g., B, 256, 3)
        
        # Permute xyz to B, 3, N_sampled for convolution
        xyz_permuted = xyz.permute(0, 2, 1) # B, 3, N_sampled
        
        x_with_pos = x + self.pos_embed(xyz_permuted)

        x_processed = self.relu(self.bn1(self.conv1(x_with_pos))) # B, C_out, N_sampled

        # Apply SA layers. Each SA_Layer will further integrate xyz.
        # xyz for SA_Layer should be B, 3, N. So xyz_permuted is correct here.
        x1 = self.sa1(x_processed, xyz_permuted)
        x2 = self.sa2(x1, xyz_permuted)
        x3 = self.sa3(x2, xyz_permuted)
        x4 = self.sa4(x3, xyz_permuted)

        # Concatenate features from all SA layers
        x = concat((x1, x2, x3, x4), dim=1) # B, 4 * channels, N_sampled

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
        b, n, s, d = x.size()  # torch.Size([B, N_group, N_sample, C_in])
        x = x.permute(0, 1, 3, 2)  # [B, N_group, C_in, N_sample]
        # Reshape to treat each group as a batch item for 1D convolutions
        x = x.reshape(-1, d, s)  # [B * N_group, C_in, N_sample]
        
        # Apply 1D convolutions
        x = self.relu(self.bn1(self.conv1(x)))  # [B * N_group, out_channels, N_sample]
        x = self.relu(self.bn2(self.conv2(x)))  # [B * N_group, out_channels, N_sample]
        
        # Max pool over samples (the N_sample dimension)
        x = jt.max(x, 2)  # [B * N_group, out_channels]
        x = x.view(b * n, -1) # Ensure correct view after max pooling
        
        x = x.reshape(b, n, -1).permute(0, 2, 1)  # [B, out_channels, N_group]
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
        # Add a projection for xyz coordinates to embed them into feature space
        # This allows the SA_Layer to learn to combine positional information with features.
        self.xyz_proj = nn.Conv1d(3, channels, 1, bias=False)

    def execute(self, x, xyz):
        # x: B, C, N (features)
        # xyz: B, 3, N (coordinates)

        # Project xyz to the same channel dimension as x
        # This creates a positional embedding from the raw coordinates.
        xyz_feat = self.xyz_proj(xyz)

        x_with_pos = x + xyz_feat

        x_q = self.q_conv(x_with_pos).permute(0, 2, 1)  # B, N, C_query (e.g., B, N, channels // 4)
        x_k = self.k_conv(x_with_pos)  # B, C_key, N (e.g., B, channels // 4, N)
        x_v = self.v_conv(x_with_pos)  # B, C_value, N (e.g., B, channels, N)

        energy = nn.bmm(x_q, x_k)  # B, N, N (attention logits)
        attention = self.softmax(energy)
        # Normalization of attention weights. This ensures sum of attention weights for each query is 1.
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

        x_r = nn.bmm(x_v, attention)  # B, C, N (This assumes attention is applied across the N dimension of V)

        # Residual connection
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r))) # learnable transformation of residual
        x = x + x_r # add the transformed residual
        return x

if __name__ == '__main__':
    jt.flags.use_cuda = 1
    input_points = init.gauss((16, 3, 1024), dtype='float32')  # B, D, N

    print("Testing Point_Transformer:")
    network_pt = Point_Transformer(output_channels=66) # Example output channels
    out_logits_pt = network_pt(input_points)
    print(f"Point_Transformer output shape: {out_logits_pt.shape}") # Expected: [16, 66]

    print("\nTesting Point_Transformer2:")
    network_pt2 = Point_Transformer2(output_channels=66) # Example output channels
    out_logits_pt2 = network_pt2(input_points)
    print(f"Point_Transformer2 output shape: {out_logits_pt2.shape}") # Expected: [16, 66]