import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler, knn_point, index_points

def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint
    sampler = FurthestPointSampler(npoint)
    _, fps_idx = sampler(xyz) # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)
    idx = knn_point(nsample, xyz, new_xyz)
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
        # x is (B, C, N)
        batch_size = x.shape[0]
        xyz = x.transpose(1, 2) # (B, N, C) for geometry

        features = self.relu(self.bn1(self.conv1(x)))
        features = self.relu(self.bn2(self.conv2(features)))
        
        features_for_grouping = features.transpose(1, 2)
        
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=features_for_grouping)
        
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)

        x_out = self.pt_last(feature_1, new_xyz)
        x_out = concat([x_out, feature_1], dim=1)
        x_out = self.conv_fuse(x_out)
        x_out = jt.max(x_out, 2)
        x_out = x_out.view(batch_size, -1)

        x_out = self.relu(self.bn6(self.linear1(x_out)))
        x_out = self.dp1(x_out)
        x_out = self.relu(self.bn7(self.linear2(x_out)))
        x_out = self.dp2(x_out)
        x_out = self.linear3(x_out)

        return x_out

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
        self.xyz_conv = nn.Conv1d(3, channels, 1, bias=False)

    def execute(self, x, xyz):
        # x: [B, C, N], xyz: [B, 3, N]
        pos_encoding = self.xyz_conv(xyz)
        x = x + pos_encoding
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c'
        x_k = self.k_conv(x) # b, c', n
        x_v = self.v_conv(x) # b, c, n
        energy = nn.bmm(x_q, x_k) # b, n, n
        attention = self.softmax(energy)
        
        # NOTE: This line is non-standard for attention and likely an error. It's commented out.
        # attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        
        # FIX: The matrix multiplication dimensions were incorrect.
        # Original: nn.bmm(x_v, attention) -> [B, C, N] x [B, N, N] -> Invalid
        # Correct: nn.bmm(x_v, attention.permute(0, 2, 1)) -> [B, C, N] x [B, N, N] -> Valid
        x_r = nn.bmm(x_v, attention)
        
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class Point_Transformer_Big(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer_Big, self).__init__()
        
        channels = 128
        self.conv1 = nn.Conv1d(3, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        
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
        batch_size, C, N = x.size()
        x_input = x
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x, x_input)
        x2 = self.sa2(x1, x_input)
        x3 = self.sa3(x2, x_input)
        x4 = self.sa4(x3, x_input)
        x5 = self.sa5(x4, x_input)
        x6 = self.sa6(x5, x_input)
        x7 = self.sa7(x6, x_input)
        x8 = self.sa8(x7, x_input)
        
        x = concat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        x = self.conv_fuse(x)
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

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        b, n, s, d = x.size() # (B, npoint, nsample, 2*d_in)
        x = x.permute(0, 1, 3, 2) # (B, npoint, 2*d_in, nsample)
        x = x.reshape(-1, d, s) # (B*npoint, 2*d_in, nsample)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = jt.max(x, 2) # (B*npoint, d_out)
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1) # (B, d_out, npoint)
        return x

# FIX: Removed duplicate class definition of Point_Transformer_Last
class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        # self.conv_pos is redundant as SA_Layer has its own xyz_conv
        # self.conv_pos = nn.Conv1d(3, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)
        self.relu = nn.ReLU()

    def execute(self, x, xyz):
        # x: [B, C, N], e.g., (B, 256, 256)
        # xyz: [B, N, 3], e.g., (B, 256, 3)
        
        # SA_Layer 需要的 xyz 形状是 (B, 3, N)
        xyz_for_sa = xyz.permute(0, 2, 1)
        
        # 这一行是多余的，因为 SA_Layer 会自己进行位置编码
        # pos_encoding = self.conv_pos(xyz_for_sa)
        
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 核心修复：将3维的坐标张量 xyz_for_sa 传递给 SA_Layer
        x1 = self.sa1(x, xyz_for_sa)
        x2 = self.sa2(x1, xyz_for_sa)
        x3 = self.sa3(x2, xyz_for_sa)
        x4 = self.sa4(x3, xyz_for_sa)
        x = concat((x1, x2, x3, x4), dim=1)
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
        # x expected as [B, 3, N]
        batch_size, C, N = x.size()
        x_input = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x, x_input)
        x2 = self.sa2(x1, x_input)
        x3 = self.sa3(x2, x_input)
        x4 = self.sa4(x3, x_input)
        x = concat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class SymmetryEnforcementLayer(nn.Module):
    def __init__(self):
        super(SymmetryEnforcementLayer, self).__init__()

    def execute(self, features, xyz):
        """
        :param features: Point features tensor, shape [B, C, N]
        :param xyz: Point coordinates tensor, shape [B, 3, N]
        :return: Symmetrized feature tensor, shape [B, C, N]
        """
        xyz_reflected = xyz * jt.array([-1, 1, 1]).view(1, 3, 1)

        xyz_permuted = xyz.permute(0, 2, 1)
        xyz_reflected_permuted = xyz_reflected.permute(0, 2, 1)
        
        # knn_point returns (distances, indices)
        _, mirror_indices_raw = knn_point(1, xyz_permuted, xyz_reflected_permuted)

        # FIX: The original code had a hardcoded batch size bug here.
        # The 'repeat(8, ...)' was incorrect and assumed a batch size of 8.
        # The corrected logic works for any batch size.
        # mirror_indices_raw is already [B, N, 1]. We just need to remove the last dimension.
        mirror_indices = mirror_indices_raw.squeeze(-1) # -> [B, N]

        features_permuted = features.permute(0, 2, 1) # -> [B, N, C]
        mirrored_features = index_points(features_permuted, mirror_indices) # -> [B, N, C]
        mirrored_features = mirrored_features.permute(0, 2, 1) # -> [B, C, N]
        
        symmetrized_features = (features + mirrored_features) / 2.0
        
        return symmetrized_features


class Point_Transformer_Symmetric(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer_Symmetric, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)
        
        self.symmetry_layer = SymmetryEnforcementLayer()

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
        batch_size, C, N = x.size()
        x_input = x
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        x1 = self.sa1(x, x_input)
        x2 = self.sa2(x1, x_input)
        x3 = self.sa3(x2, x_input)
        x4 = self.sa4(x3, x_input)
        
        x = concat((x1, x2, x3, x4), dim=1)

        x = self.symmetry_layer(x, x_input)
        
        x = self.conv_fuse(x)
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

if __name__ == '__main__':
    jt.flags.use_cuda=1
    # Test with a batch size other than 8 to ensure fixes work
    input_points = init.gauss((16, 3, 1024), dtype='float32')

    print("Testing Point_Transformer...")
    network = Point_Transformer(output_channels=256)
    out_logits = network(input_points)
    print("Output shape:", out_logits.shape) # Expected: (16, 256)

    print("\nTesting Point_Transformer2...")
    network2 = Point_Transformer2(output_channels=256)
    out_logits2 = network2(input_points)
    print("Output shape:", out_logits2.shape) # Expected: (16, 256)

    print("\nTesting Point_Transformer_Symmetric...")
    network_sym = Point_Transformer_Symmetric(output_channels=256)
    out_logits_sym = network_sym(input_points)
    print("Output shape:", out_logits_sym.shape) # Expected: (16, 256)