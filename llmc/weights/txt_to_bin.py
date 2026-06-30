#!/usr/bin/env python3
"""
txt_to_bin.py -- 原始 TXT 权重 → 硬件适配 BIN 的完整预处理流水线。

目录结构（你把原始文件放成这样）：
    weights_raw/
        TXT/              -- 原始 weight / weight_scale / o_proj / lm_head 的 txt
        bias_fp16/
            bias_fp16/    -- 完整 FP16 bias 的 txt（每行一个浮点）

输出：
    weights_processed/
        W_rope.emb_cos.bin
        W_rope.emb_sin.bin
        W_mask.bin
        lm_head.weight.bin
        lm_head.weight_scale.bin
        layer{layer}_attn/
            q_proj.weight.head{h}.bin       (2048,128) INT8 重排
            q_proj.weight_scale.head{h}.bin (128,1)  FP16 LE
            q_proj.bias.head{h}.expanded.bin (1024,128) FP16 LE
            ...
        layer{layer}_mlp/
            gate_proj.weight.tile{t}.bin
            gate_proj.weight_scale.tile{t}.bin
            ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 重排函数（硬件要求的 64x8 块展平）
# ---------------------------------------------------------------------------
def reorder_int8_matrix(arr: np.ndarray, block_rows: int = 64, group_cols: int = 8) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")
    rows, cols = arr.shape
    out = []
    for br in range(0, rows, block_rows):
        for c in range(0, cols, group_cols):
            sub = arr[br:br + block_rows, c:c + group_cols]
            out.append(sub.reshape(-1))
    return np.concatenate(out).astype(np.int8)


# ---------------------------------------------------------------------------
# TXT 读取 & BIN 写入
# ---------------------------------------------------------------------------
def read_txt_floats(path: Path) -> np.ndarray:
    """流式读取，带诊断信息"""
    file_size = path.stat().st_size
    with open(path, 'r', buffering=1024*1024) as f:
        n = sum(1 for _ in f)
    print(f"    [read] {path.name}: {file_size} bytes, {n} lines")
    arr = np.empty(n, dtype=np.float32)
    with open(path, 'r', buffering=1024*1024) as f:
        for i, line in enumerate(f):
            val = line.strip().split()[0] if line.strip() else '0'
            arr[i] = float(val)
    return arr


def write_int8_bin(data: np.ndarray, path: Path):
    data.astype(np.int8).tofile(path)


def write_fp16_le(data: np.ndarray, path: Path):
    data.astype(np.float16).view(np.dtype(np.float16).newbyteorder('<')).tofile(path)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def process_attention(txt_dir: Path, bias_dir: Path, dst_dir: Path, layer: int, max_token: int):
    attn_dir = dst_dir / f"layer{layer}_attn"
    attn_dir.mkdir(parents=True, exist_ok=True)

    for proj in ("q", "k", "v", "o"):
        w_txt = txt_dir / f"model.layers.{layer}.self_attn.{proj}_proj.weight.txt"
        s_txt = txt_dir / f"model.layers.{layer}.self_attn.{proj}_proj.weight_scale.txt"
        if not w_txt.exists():
            print(f"  [SKIP] {w_txt.name} not found")
            continue

        w = read_txt_floats(w_txt)
        expected = 2048 * 2048
        if w.size != expected:
            raise ValueError(
                f"{w_txt.name}: read {w.size} elements, expected {expected}. "
                f"Check if this file is really the weight (should be ~42MB) "
                f"and not weight_scale or bias."
            )
        w = w.reshape(2048, 2048)
        if proj in ("q", "k", "v"):
            for h in range(16):
                head = w[:, h * 128:(h + 1) * 128]
                reordered = reorder_int8_matrix(head)
                write_int8_bin(reordered, attn_dir / f"{proj}_proj.weight.head{h}.bin")
        else:
            reordered = reorder_int8_matrix(w)
            write_int8_bin(reordered, attn_dir / f"{proj}_proj.weight.bin")

        # scale
        if s_txt.exists():
            s = read_txt_floats(s_txt).reshape(2048, 1)
            if proj in ("q", "k", "v"):
                for h in range(16):
                    head_s = s[h * 128:(h + 1) * 128]
                    write_fp16_le(head_s, attn_dir / f"{proj}_proj.weight_scale.head{h}.bin")
            else:
                write_fp16_le(s, attn_dir / f"{proj}_proj.weight_scale.bin")

    # bias: 从 bias_fp16 读取，拆 16 头，每头复制 max_token 份
    for proj in ("q", "k", "v"):
        b_txt = bias_dir / f"model.layers.{layer}.self_attn.{proj}_proj.bias.txt"
        if not b_txt.exists():
            print(f"  [WARN] bias {b_txt.name} not found")
            continue
        b = read_txt_floats(b_txt).reshape(1, 2048)
        for h in range(16):
            head_b = b[:, h * 128:(h + 1) * 128]          # (1, 128)
            expanded = np.repeat(head_b, max_token, axis=0)  # (max_token, 128)
            write_fp16_le(expanded, attn_dir / f"{proj}_proj.bias.head{h}.expanded.bin")


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
def process_mlp(txt_dir: Path, dst_dir: Path, layer: int):
    mlp_dir = dst_dir / f"layer{layer}_mlp"
    mlp_dir.mkdir(parents=True, exist_ok=True)

    for proj in ("gate_proj", "up_proj", "down_proj"):
        w_txt = txt_dir / f"model.layers.{layer}.mlp.{proj}.weight.txt"
        if not w_txt.exists():
            continue
        w = read_txt_floats(w_txt)
        expected = 5504 * 2048
        if w.size != expected:
            raise ValueError(
                f"{w_txt.name}: read {w.size} elements, expected {expected} "
                f"(~11MB). Check file content."
            )
        w = w.reshape(5504, 2048)
        s_txt = txt_dir / f"model.layers.{layer}.mlp.{proj}.weight_scale.txt"
        s = read_txt_floats(s_txt)

        tiles = [2048, 2048, 1408]
        row = 0
        for ti, tw in enumerate(tiles):
            tile = w[row:row + tw, :]
            reordered = reorder_int8_matrix(tile)
            write_int8_bin(reordered, mlp_dir / f"{proj}.weight.tile{ti}.bin")
            row += tw

        if proj in ("gate_proj", "up_proj"):
            row = 0
            for ti, tw in enumerate(tiles):
                tile_s = s[row:row + tw].reshape(-1, 1)
                write_fp16_le(tile_s, mlp_dir / f"{proj}.weight_scale.tile{ti}.bin")
                row += tw
        else:
            write_fp16_le(s.reshape(-1, 1), mlp_dir / f"{proj}.weight_scale.bin")


# ---------------------------------------------------------------------------
# LM_HEAD
# ---------------------------------------------------------------------------
def process_lm_head(txt_dir: Path, dst_dir: Path):
    w_txt = txt_dir / "lm_head.weight.txt"
    s_txt = txt_dir / "lm_head.weight_scale.txt"
    if not w_txt.exists():
        return

    w = read_txt_floats(w_txt)
    vocab = len(w) // 2048
    w = w.reshape(vocab, 2048)
    reordered = reorder_int8_matrix(w)
    write_int8_bin(reordered, dst_dir / "lm_head.weight.bin")

    if s_txt.exists():
        s = read_txt_floats(s_txt).reshape(-1, 1)
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
                if col > row:
                    all_data += b'\x00\xFC'  # FP16 -inf little-endian
                else:
                    all_data += b'\x00\x00'  # FP16 0
    with open(dst_dir / "W_mask.bin", "wb") as f:
        f.write(all_data)
    print(f"  Mask: {len(sizes)} matrices, {len(all_data)} bytes")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="weights_raw", help="原始 TXT 根目录")
    ap.add_argument("--dst", default="weights_processed", help="输出 BIN 目录")
    ap.add_argument("--max-token", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=24)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    txt_dir = src / "TXT"
    bias_dir = src / "bias_fp16" / "bias_fp16"

    if not txt_dir.exists():
        sys.exit(f"ERROR: {txt_dir} not found")
    if not bias_dir.exists():
        print(f"WARN: {bias_dir} not found, bias will be skipped")

    print("Generating RoPE / Mask ...")
    generate_rope(dst, args.max_token)
    generate_mask(dst)

    print("Processing LM_HEAD ...")
    process_lm_head(txt_dir, dst)

    for layer in range(args.n_layers):
        print(f"Processing layer {layer} ...")
        process_attention(txt_dir, bias_dir, dst, layer, args.max_token)
        process_mlp(txt_dir, dst, layer)

    print(f"\nDone. Output: {dst.resolve()}")


if __name__ == "__main__":
    main()