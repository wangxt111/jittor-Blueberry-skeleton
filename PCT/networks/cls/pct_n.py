import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat


def _part1by2_10bit(n):
    """
    Spreads the bits of a 10-bit integer by inserting two zeros between each bit.
    This is a key component for calculating 3D Morton codes efficiently.
    Example: ...b2 b1 b0 -> ...00b2 00b1 00b0
    
    Args:
        n (jt.Var): A Jittor Var containing integers to be spread.
    
    Returns:
        jt.Var: The spread-out integers.
    """
    n = n & 0x000003ff # Ensure input is 10-bit
    n = (n ^ (n << 16)) & 0xff0000ff
    n = (n ^ (n << 8))  & 0x0300f00f
    n = (n ^ (n << 4))  & 0x030c30c3
    n = (n ^ (n << 2))  & 0x09249249
    return n

def _compute_morton_code_3d_10bit(coords):
    """
    Computes the 3D Morton code for integer coordinates.
    
    Args:
        coords (jt.Var): Integer coordinates of shape [B, N, 3] and dtype int64.
    
    Returns:
        jt.Var: The Morton codes of shape [B, N].
    """
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]
    return _part1by2_10bit(x) | (_part1by2_10bit(y) << 1) | (_part1by2_10bit(z) << 2)

def serialize_point_cloud_z_order(xyz):
    """
    Serializes a batch of point clouds using the Z-order curve. This is a 
    required preprocessing step for the PointTransformerV3 model.
    
    Args:
        xyz (jt.Var): Input point cloud coordinates, shape [B, 3, N].
    
    Returns:
        jt.Var: Serialized point cloud coordinates, shape [B, 3, N].
    """
    # 1. Normalize coordinates to a [0, 1] unit cube for each cloud in the batch
    xyz_permuted = xyz.permute(0, 2, 1) # to [B, N, 3] for easier processing
    xyz_min, _ = xyz_permuted.min(dim=1, keepdims=True)
    xyz_max, _ = xyz_permuted.max(dim=1, keepdims=True)
    
    xyz_range = xyz_max - xyz_min
    xyz_range[xyz_range < 1e-8] = 1e-8 # Avoid division by zero for flat point clouds

    xyz_normalized = (xyz_permuted - xyz_min) / xyz_range

    # 2. Discretize normalized coordinates onto a 10-bit grid (1024x1024x1024)
    bits = 10
    grid_max_val = (1 << bits) - 1
    int_coords = (xyz_normalized * grid_max_val).int64()

    # 3. Compute Morton codes from the integer coordinates
    morton_codes = _compute_morton_code_3d_10bit(int_coords) # shape [B, N]

    # 4. Get the indices that would sort the Morton codes
    sort_indices = jt.argsort(morton_codes, dim=1) # shape [B, N]

    # 5. Reorder the original point cloud using the sort indices
    # The index for gather must be expanded to match the input tensor's dimensions
    sort_indices_expanded = sort_indices.unsqueeze(1).expand_as(xyz)
    serialized_xyz = jt.gather(xyz, dim=2, index=sort_indices_expanded)

    return serialized_xyz


# ==================================================================
# |                                                                |
# |         Section 2: PointTransformerV3 Model Definition         |
# |                                                                |
# ==================================================================

class SA_Layer_V3(nn.Module):
    """
    Self-Attention Layer for PointTransformerV3.
    It operates on features that have already been positionally encoded
    by the upstream Conditional Positional Encoding (CPE) layer.
    """
    def __init__(self, channels):
        super(SA_Layer_V3, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, x):
        # x shape: [B, C, N]
        x_q = self.q_conv(x).permute(0, 2, 1) # [B, N, C']
        x_k = self.k_conv(x)                 # [B, C', N]
        x_v = self.v_conv(x)                 # [B, C, N]
        
        energy = nn.bmm(x_q, x_k) # [B, N, N]
        
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        
        x_r = nn.bmm(x_v, attention) # [B, C, N]
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class PointTransformerV3(nn.Module):
    """
    PointTransformerV3 Model.
    
    The architecture follows the paper's design philosophy:
    1. It assumes the input point cloud has been pre-processed via serialization.
    2. It uses a 1D convolution with a kernel size > 1 as a Conditional
       Positional Encoding (CPE) layer before the attention blocks.
    3. The Self-Attention layers (SA_Layer_V3) are position-agnostic themselves.
    """
    def __init__(self, output_channels=40):
        super(PointTransformerV3, self).__init__()
        channels = 128
        
        # Initial feature embedding layers
        self.conv1 = nn.Conv1d(3, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        
        # Conditional Positional Encoding (CPE) Layer.
        # This Conv1d with kernel=3 simulates a sparse convolution on the
        # serialized point cloud, implicitly encoding local geometric information.
        self.cpe_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

        # A series of V3 Self-Attention layers
        self.sa1 = SA_Layer_V3(channels)
        self.sa2 = SA_Layer_V3(channels)
        self.sa3 = SA_Layer_V3(channels)
        self.sa4 = SA_Layer_V3(channels)

        # Feature fusion and classification head
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(channels * 4, 1024, kernel_size=1, bias=False),
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
        """
        Args:
            x (jt.Var): A Z-order serialized point cloud of shape [B, 3, N].
        """
        batch_size = x.shape[0]
        
        # 1. Initial feature embedding
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # 2. Apply Conditional Positional Encoding (CPE)
        x = self.cpe_conv(x)

        # 3. Pass through Self-Attention layers
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        
        # 4. Concatenate features and fuse
        x = concat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        
        # 5. Global max pooling and classification
        x = jt.max(x, dim=2) # Global Max Pooling
        x = x.view(batch_size, -1)
        
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x
