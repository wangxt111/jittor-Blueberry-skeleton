import jittor as jt
from typing import Optional, Union

class KeyLUT:
    def __init__(self):
        r256 = jt.arange(256).int64()
        r512 = jt.arange(512).int64()
        zero = jt.zeros(256, dtype=jt.int64)
        self._encode = {
            "cpu": (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }
        self._decode = {"cpu": self.key2xyz(r512, 9)}

    def encode_lut(self):
        return self._encode["cpu"]

    def decode_lut(self):
        return self._decode["cpu"]

    def xyz2key(self, x, y, z, depth):
        key = jt.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = key | ((x & mask) << (2 * i + 2)) | ((y & mask) << (2 * i + 1)) | ((z & mask) << (2 * i))
        return key

    def key2xyz(self, key, depth):
        x = jt.zeros_like(key)
        y = jt.zeros_like(key)
        z = jt.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i))) >> (2 * i))
        return x, y, z

# 初始化全局 LUT
_key_lut = KeyLUT()

def xyz2key(
    x: jt.Var,
    y: jt.Var,
    z: jt.Var,
    b: Optional[Union[jt.Var, int]] = None,
    depth: int = 16,
):
    EX, EY, EZ = _key_lut.encode_lut()
    x, y, z = x.cast(jt.int64), y.cast(jt.int64), z.cast(jt.int64)

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[(x & mask)] | EY[(y & mask)] | EZ[(z & mask)]

    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[((x >> 8) & mask)] | EY[((y >> 8) & mask)] | EZ[((z >> 8) & mask)]
        key = (key16 << 24) | key

    if b is not None:
        b = b if isinstance(b, jt.Var) else jt.int64(b)
        b = b.cast(jt.int64)
        key = (b << 48) | key

    return key

def key2xyz(key: jt.Var, depth: int = 16):
    DX, DY, DZ = _key_lut.decode_lut()
    key = key.cast(jt.int64)
    x, y, z = jt.zeros_like(key), jt.zeros_like(key), jt.zeros_like(key)

    b = key >> 48
    key = key & ((1 << 48) - 1)

    n = (depth + 2) // 3
    for i in range(n):
        k = (key >> (i * 9)) & 511
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
        z = z | (DZ[k] << (i * 3))

    return x, y, z, b
