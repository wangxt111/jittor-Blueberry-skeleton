import jittor as jt
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_


@jt.no_grad()
def encode(grid_coord, batch, depth, order="z"):
    """
    将点编码为 Hilbert 码或 Z-order 码
    grid_coord: (N, 3) 网格坐标
    batch: (N,) batch 索引
    depth: int, 编码深度
    order: str, 编码顺序
    """
    # 确保输入类型和形状正确
    grid_coord = grid_coord.int32()
    batch = batch.int32()
    
    # 初始化输出
    N = grid_coord.shape[0]
    code = jt.zeros(N, dtype="int64")
    
    # 根据不同的编码顺序生成码
    if order == "z":
        # Z-order 编码
        for i in range(3):  # 3个维度
            # 确保 grid_coord[:, i] 是一维的
            coord_i = grid_coord[:, i].reshape(-1)
            code = code | (coord_i << (i * depth))
    elif order == "hilbert":
        # Hilbert 编码
        for i in range(3):
            # 确保 grid_coord[:, i] 是一维的
            coord_i = grid_coord[:, i].reshape(-1)
            code = code | (coord_i << (i * depth))
        # 添加 Hilbert 曲线变换
        code = hilbert_encode(code, depth)
    else:
        raise ValueError(f"Unknown order: {order}")
    
    # 将 batch 信息编码到高位
    batch_code = batch << (depth * 3)
    code = code | batch_code
    
    return code


@jt.no_grad()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> (depth * 3)
    code = code & ((1 << (depth * 3)) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: jt.Var, depth: int = 16):
    x = grid_coord[:, 0].cast(jt.int64)
    y = grid_coord[:, 1].cast(jt.int64)
    z = grid_coord[:, 2].cast(jt.int64)
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: jt.Var, depth):
    x, y, z = z_order_decode_(code, depth=depth)
    grid_coord = jt.stack([x, y, z], dim=-1)  # (N, 3)
    return grid_coord


def hilbert_encode(grid_coord: jt.Var, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: jt.Var, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)
