import jittor as jt
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_


@jt.no_grad()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.cast(jt.int64)
        code = (batch << (depth * 3)) | code
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
