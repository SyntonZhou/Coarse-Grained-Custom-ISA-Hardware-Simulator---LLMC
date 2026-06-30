#!/usr/bin/env python3
"""
bin_postprocess_fixed.py

Fixed version for BIN -> weights_processed_from_bin.
Key fixes vs previous bin_postprocess.py:
  1) Original BIN FP16 scale files are treated as big-endian (>f2) and rewritten as little-endian fp16.
  2) o_proj.weight_scale.bin and lm_head.weight_scale.bin are converted, not byte-copied.
  3) q/k/v and MLP scales are read with the same explicit endian rule.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Reorder / inverse reorder: hardware 64x8 layout
# ---------------------------------------------------------------------------
def reorder_int8(arr: np.ndarray, block_rows: int = 64, group_cols: int = 8) -> np.ndarray:
    rows, cols = arr.shape
    out = np.empty(rows * cols, dtype=np.int8)
    idx = 0
    for br in range(0, rows, block_rows):
        rb = min(block_rows, rows - br)
        for c in range(0, cols, group_cols):
            cb = min(group_cols, cols - c)
            sub = arr[br:br + rb, c:c + cb].reshape(-1)
            out[idx:idx + sub.size] = sub
            idx += sub.size
    if idx != out.size:
        raise RuntimeError(f"reorder wrote {idx} bytes, expected {out.size}")
    return out.astype(np.int8, copy=False)


def inverse_reorder_int8(data: np.ndarray, rows: int, cols: int,
                         block_rows: int = 64, group_cols: int = 8) -> np.ndarray:
    if data.size != rows * cols:
        raise ValueError(f"int8 size mismatch: got {data.size}, expected {rows * cols} for {rows}x{cols}")
    out = np.zeros((rows, cols), dtype=np.int8)
    idx = 0
    for br in range(0, rows, block_rows):
        rb = min(block_rows, rows - br)
        for c in range(0, cols, group_cols):
            cb = min(group_cols, cols - c)
            n = rb * cb
            sub = data[idx:idx + n].reshape(rb, cb)
            out[br:br + rb, c:c + cb] = sub
            idx += n
    return out


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def read_int8_reordered(path: Path, rows: int, cols: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int8)
    return inverse_reorder_int8(raw, rows, cols)


def read_fp16_be(path: Path, shape=None) -> np.ndarray:
    """Original BIN scale files are big-endian fp16 from convert.py/txt_to_bin_mixed."""
    arr = np.fromfile(path, dtype='>f2').astype(np.float16)
    if shape is not None:
        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(f"{path}: got {arr.size} fp16 values, expected {expected} for shape {shape}")
        arr = arr.reshape(shape)
    return arr


def write_int8_bin(data: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.astype(np.int8, copy=False).tofile(path)


def write_fp16_le(data: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.astype('<f2', copy=False).tofile(path)


def read_txt_floats_stream(path: Path) -> np.ndarray:
    with open(path, 'r', buffering=1024 * 1024) as f:
        n = sum(1 for _ in f)
    arr = np.empty(n, dtype=np.float32)
    with open(path, 'r', buffering=1024 * 1024) as f:
        for i, line in enumerate(f):
            line = line.strip()
            arr[i] = float(line.split()[0]) if line else 0.0
    return arr


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def process_attention(bin_dir: Path, bias_txt_dir: Path, dst_dir: Path, layer: int, max_token: int):
    attn_dir = dst_dir / f"layer{layer}_attn"
    attn_dir.mkdir(parents=True, exist_ok=True)

    for proj in ("q", "k", "v"):
        w_bin = bin_dir / f"model.layers.{layer}.self_attn.{proj}_proj.weight.bin"
        if w_bin.exists():
            w_raw = np.fromfile(w_bin, dtype=np.int8)
            w_orig = inverse_reorder_int8(w_raw, 2048, 2048)
            for h in range(16):
                head = w_orig[:, h * 128:(h + 1) * 128]
                write_int8_bin(reorder_int8(head), attn_dir / f"{proj}_proj.weight.head{h}.bin")

        s_bin = bin_dir / f"model.layers.{layer}.self_attn.{proj}_proj.weight_scale.bin"
        if s_bin.exists():
            s = read_fp16_be(s_bin, (2048, 1))
            for h in range(16):
                head_s = s[h * 128:(h + 1) * 128]
                write_fp16_le(head_s, attn_dir / f"{proj}_proj.weight_scale.head{h}.bin")

        b_txt = bias_txt_dir / f"model.layers.{layer}.self_attn.{proj}_proj.bias.txt"
        if b_txt.exists():
            b = read_txt_floats_stream(b_txt).reshape(1, 2048)
            for h in range(16):
                head_b = b[:, h * 128:(h + 1) * 128]
                expanded = np.repeat(head_b, max_token, axis=0)
                write_fp16_le(expanded, attn_dir / f"{proj}_proj.bias.head{h}.expanded.bin")
        else:
            print(f"  [WARN] missing bias txt: {b_txt}")

    # o_proj: weight is already whole 2048x2048 reordered; copy weight, but convert scale endian.
    w_bin = bin_dir / f"model.layers.{layer}.self_attn.o_proj.weight.bin"
    if w_bin.exists():
        shutil.copy(w_bin, attn_dir / "o_proj.weight.bin")

    s_bin = bin_dir / f"model.layers.{layer}.self_attn.o_proj.weight_scale.bin"
    if s_bin.exists():
        s = read_fp16_be(s_bin, (2048, 1))
        write_fp16_le(s, attn_dir / "o_proj.weight_scale.bin")


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
def process_mlp(bin_dir: Path, dst_dir: Path, layer: int):
    mlp_dir = dst_dir / f"layer{layer}_mlp"
    mlp_dir.mkdir(parents=True, exist_ok=True)
    tiles = [2048, 2048, 1408]

    for proj in ("gate_proj", "up_proj", "down_proj"):
        w_bin = bin_dir / f"model.layers.{layer}.mlp.{proj}.weight.bin"
        if not w_bin.exists():
            continue

        w_raw = np.fromfile(w_bin, dtype=np.int8)
        w_orig = inverse_reorder_int8(w_raw, 5504, 2048)
        row = 0
        for ti, tw in enumerate(tiles):
            tile = w_orig[row:row + tw, :]
            write_int8_bin(reorder_int8(tile), mlp_dir / f"{proj}.weight.tile{ti}.bin")
            row += tw

        s_bin = bin_dir / f"model.layers.{layer}.mlp.{proj}.weight_scale.bin"
        if s_bin.exists():
            if proj in ("gate_proj", "up_proj"):
                s = read_fp16_be(s_bin, (5504, 1))
                row = 0
                for ti, tw in enumerate(tiles):
                    tile_s = s[row:row + tw]
                    write_fp16_le(tile_s, mlp_dir / f"{proj}.weight_scale.tile{ti}.bin")
                    row += tw
            else:
                s = read_fp16_be(s_bin, (2048, 1))
                write_fp16_le(s, mlp_dir / "down_proj.weight_scale.bin")


# ---------------------------------------------------------------------------
# LM_HEAD
# ---------------------------------------------------------------------------
def process_lm_head(bin_dir: Path, dst_dir: Path):
    w_bin = bin_dir / "lm_head.weight.bin"
    if w_bin.exists():
        w_raw = np.fromfile(w_bin, dtype=np.int8)
        vocab = w_raw.size // 2048
        w_orig = inverse_reorder_int8(w_raw, vocab, 2048)
        write_int8_bin(reorder_int8(w_orig), dst_dir / "lm_head.weight.bin")

    s_bin = bin_dir / "lm_head.weight_scale.bin"
    if s_bin.exists():
        # Current file size in your listing is 303872 B = 151936 fp16 values.
        # Keep shape flat but convert endian explicitly.
        s = read_fp16_be(s_bin)
        write_fp16_le(s, dst_dir / "lm_head.weight_scale.bin")


# ---------------------------------------------------------------------------
# RoPE / Mask
# ---------------------------------------------------------------------------
def generate_rope(dst_dir: Path, max_token: int = 1024, head_dim: int = 128):
    half = head_dim // 2
    pos = np.arange(max_token, dtype=np.float32)
    emb_cos = np.zeros((max_token, head_dim), dtype=np.float32)
    emb_sin = np.zeros((max_token, head_dim), dtype=np.float32)
    for m in range(half):
        freq = 10000.0 ** (-(m / half))
        emb_cos[:, m] = np.cos(pos * freq)
        emb_sin[:, m] = -np.sin(pos * freq)
        emb_cos[:, m + half] = np.cos(pos * freq)
        emb_sin[:, m + half] = np.sin(pos * freq)
    write_fp16_le(emb_cos, dst_dir / "W_rope.emb_cos.bin")
    write_fp16_le(emb_sin, dst_dir / "W_rope.emb_sin.bin")
    print(f"  RoPE: {max_token}x{head_dim}")


def generate_mask(dst_dir: Path):
    sizes = list(range(64, 1025, 64))
    all_data = bytearray()
    for size in sizes:
        for row in range(size):
            for col in range(size):
                all_data += b'\x00\xFC' if col > row else b'\x00\x00'
    with open(dst_dir / "W_mask.bin", "wb") as f:
        f.write(all_data)
    print(f"  Mask: {len(sizes)} matrices, {len(all_data)} bytes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="BIN", help="Original BIN directory")
    ap.add_argument("--bias-txt", default="bias_fp16/bias_fp16", help="FP16 bias txt directory")
    ap.add_argument("--dst", default="weights_processed_from_bin_fixed", help="Output directory")
    ap.add_argument("--max-token", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=24)
    args = ap.parse_args()

    bin_dir = Path(args.bin)
    bias_txt_dir = Path(args.bias_txt)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    if not bin_dir.exists():
        sys.exit(f"ERROR: {bin_dir} not found")

    print("Generating RoPE / Mask ...")
    generate_rope(dst, args.max_token)
    generate_mask(dst)

    print("Processing LM_HEAD ...")
    process_lm_head(bin_dir, dst)

    for layer in range(args.n_layers):
        print(f"Processing layer {layer} ...")
        process_attention(bin_dir, bias_txt_dir, dst, layer, args.max_token)
        process_mlp(bin_dir, dst, layer)

    print(f"\nDone. Output: {dst.resolve()}")


if __name__ == "__main__":
    main()
