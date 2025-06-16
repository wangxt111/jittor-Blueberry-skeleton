import jittor as jt
import math

def right_shift(binary, k=1, axis=-1):
    if binary.shape[axis] <= k:
        return jt.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = jt.nn.pad(binary[tuple(slicing)], [k, 0], mode="constant", value=0)
    return shifted

def binary2gray(binary, axis=-1):
    shifted = right_shift(binary, axis=axis)
    gray = jt.logical_xor(binary, shifted)
    return gray

def gray2binary(gray, axis=-1):
    out = gray.clone()
    shift = int(2 ** math.floor(math.log2(gray.shape[axis])))
    while shift > 0:
        out = jt.logical_xor(out, right_shift(out, k=shift, axis=axis))
        shift = shift // 2
    return out

def encode(locs, num_dims, num_bits):
    orig_shape = locs.shape
    device = locs.device

    if orig_shape[-1] != num_dims:
        raise ValueError(f"Expected last dimension {num_dims}, got {orig_shape[-1]}.")

    if num_dims * num_bits > 63:
        raise ValueError(f"Too many bits ({num_dims*num_bits}), max 63.")

    locs_uint8 = locs.cast(jt.int64).reshape([-1, num_dims, 8])
    bitpack_mask = (1 << jt.arange(8)).stop_grad()
    bitpack_mask_rev = bitpack_mask[::-1]

    gray = (
        (locs_uint8.unsqueeze(-1).bitwise_and(bitpack_mask_rev))
        .ne(0)
        .cast(jt.uint8)
        .reshape(-1, num_dims, 64)[..., -num_bits:]
    )

    for bit in range(num_bits):
        for dim in range(num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1:] = jt.logical_xor(gray[:, 0, bit + 1:], mask.unsqueeze(1))
            to_flip = jt.logical_and(
                ~mask.unsqueeze(1),
                jt.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:])
            )
            gray[:, dim, bit + 1:] = jt.logical_xor(gray[:, dim, bit + 1:], to_flip)
            gray[:, 0, bit + 1:] = jt.logical_xor(gray[:, 0, bit + 1:], to_flip)

    gray = gray.transpose(1, 2).reshape(-1, num_bits * num_dims)
    hh_bin = gray2binary(gray)
    padded = jt.nn.pad(hh_bin, [64 - num_bits * num_dims, 0], mode="constant", value=0)
    packed = (padded.reshape([-1, 8, 8]) * bitpack_mask).sum(-1).cast(jt.uint8)
    return packed.cast(jt.int64).reshape(orig_shape[:-1])

def decode(hilberts, num_dims, num_bits):
    if num_dims * num_bits > 64:
        raise ValueError(f"Too many bits ({num_dims*num_bits}), max 64.")

    hilberts = hilberts.reshape(-1)
    orig_shape = hilberts.shape
    bitpack_mask = (1 << jt.arange(8)).stop_grad()
    bitpack_mask_rev = bitpack_mask[::-1]

    hh_uint8 = hilberts.cast(jt.int64).reshape(-1, 8)
    hh_bits = (
        hh_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .cast(jt.uint8)
        .reshape(-1, 64)[:, -num_dims * num_bits:]
    )

    gray = binary2gray(hh_bits)
    gray = gray.reshape(-1, num_bits, num_dims).transpose(1, 2)

    for bit in range(num_bits - 1, -1, -1):
        for dim in range(num_dims - 1, -1, -1):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1:] = jt.logical_xor(gray[:, 0, bit + 1:], mask.unsqueeze(1))
            to_flip = jt.logical_and(
                ~mask.unsqueeze(1),
                jt.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:])
            )
            gray[:, dim, bit + 1:] = jt.logical_xor(gray[:, dim, bit + 1:], to_flip)
            gray[:, 0, bit + 1:] = jt.logical_xor(gray[:, 0, bit + 1:], to_flip)

    padded = jt.nn.pad(gray, [64 - num_bits, 0], mode="constant", value=0)
    locs_chopped = padded.flip(-1).reshape(-1, num_dims, 8, 8)
    locs_uint8 = (locs_chopped * bitpack_mask).sum(-1).cast(jt.uint8)
    flat_locs = locs_uint8.cast(jt.int64)
    return flat_locs.reshape(-1, num_dims)
