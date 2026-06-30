#!/usr/bin/env python3
"""
generate_golden_fixed.py

Fixed golden generator for weights_processed_from_bin_fixed/.
Key fixes vs previous generate_golden.py:
  1) Uses different matmul conventions for (in,out) and (out,in) weights.
  2) MLP gate/up tile weights are (tile_out, 2048), so use x @ W.T with scale on rows.
  3) MLP down tile weights are (tile_in, 2048), so use h @ W with scale on output columns.
  4) Adds finite-value checks to catch endian/scale problems early.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HIDDEN = 2048
HEAD_DIM = 128
HEADS = 16
MLP_TILES = [2048, 2048, 1408]


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


def load_int8_weight(path: Path, rows: int, cols: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int8)
    return inverse_reorder_int8(raw, rows, cols)


def load_fp16(path: Path, shape=None) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float16)
    if shape is not None:
        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(f"{path}: got {arr.size}, expected {expected} for shape={shape}")
        arr = arr.reshape(shape)
    return arr.astype(np.float32)


def write_fp16(data: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.astype(np.float16).tofile(path)


def check_finite(name: str, arr: np.ndarray, *, strict: bool = True):
    finite = np.isfinite(arr)
    if not finite.all():
        msg = f"[NON-FINITE] {name}: finite={finite.sum()}/{arr.size}, min={np.nanmin(arr)}, max={np.nanmax(arr)}"
        if strict:
            raise FloatingPointError(msg)
        print("WARN", msg)
    return arr


def stats(name: str, arr: np.ndarray):
    a = arr.astype(np.float32)
    print(f"  {name:28s} shape={arr.shape} finite={np.isfinite(a).all()} min={np.nanmin(a):.6g} max={np.nanmax(a):.6g}")


# Weight layout conventions
# q/k/v/o: W is (in_dim, out_dim), scale is per out_dim, result = x @ (W * scale)
def matmul_in_out(a: np.ndarray, w_in_out: np.ndarray, scale_out: np.ndarray, name: str) -> np.ndarray:
    scale = scale_out.reshape(1, -1).astype(np.float32)
    if w_in_out.shape[1] != scale.shape[1]:
        raise ValueError(f"{name}: w shape={w_in_out.shape}, scale={scale.shape}; expected scale len == w.shape[1]")
    w = w_in_out.astype(np.float32) * scale
    check_finite(name + ".w_scaled", w)
    c = a.astype(np.float32) @ w
    check_finite(name + ".out_fp32", c)
    return c.astype(np.float16)


# gate/up: W is (out_dim, in_dim), scale is per out_dim, result = x @ (W*scale[:,None]).T
def matmul_out_in(a: np.ndarray, w_out_in: np.ndarray, scale_out: np.ndarray, name: str) -> np.ndarray:
    scale = scale_out.reshape(-1, 1).astype(np.float32)
    if w_out_in.shape[0] != scale.shape[0]:
        raise ValueError(f"{name}: w shape={w_out_in.shape}, scale={scale.shape}; expected scale len == w.shape[0]")
    w = w_out_in.astype(np.float32) * scale
    check_finite(name + ".w_scaled", w)
    c = a.astype(np.float32) @ w.T
    check_finite(name + ".out_fp32", c)
    return c.astype(np.float16)


def rmsnorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x32 = x.astype(np.float32)
    y = x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return y.astype(np.float16)


def silu(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    return (x32 / (1.0 + np.exp(-x32))).astype(np.float16)


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    x_cos = x32 * cos.astype(np.float32)
    x_sin = x32 * sin.astype(np.float32)
    half = x.shape[1] // 2
    swapped = np.empty_like(x_sin)
    swapped[:, :half] = -x_sin[:, half:]
    swapped[:, half:] = x_sin[:, :half]
    out = x_cos + swapped
    check_finite("rope.out", out)
    return out.astype(np.float16)


def apply_causal_mask(qk: np.ndarray, T: int, R: int) -> np.ndarray:
    qk32 = qk.astype(np.float32)
    for r in range(T):
        qk32[r, r + 1:R] = -np.inf
    return qk32


def softmax_hw(x: np.ndarray, vld_len: int) -> np.ndarray:
    x32 = x.astype(np.float32)
    out = np.zeros_like(x32, dtype=np.float32)
    for i in range(x32.shape[0]):
        row = x32[i, :vld_len]
        mx = np.max(row)
        e = np.exp(row - mx)
        out[i, :vld_len] = e / np.sum(e)
    check_finite("softmax.out", out)
    return out.astype(np.float16)


class Manifest:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.rows = []
        self.seq = 0

    def save(self, name: str, arr: np.ndarray, stage: str):
        self.seq += 1
        fn = f"{self.seq:04d}_{name.replace('.', '_')}.bin"
        write_fp16(arr, self.out_dir / fn)
        self.rows.append({
            "seq": self.seq,
            "name": name,
            "stage": stage,
            "file": fn,
            "shape": "x".join(map(str, arr.shape)),
            "bytes": int(arr.astype(np.float16).nbytes),
        })

    def write(self):
        with open(self.out_dir / "golden_manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["seq", "name", "stage", "file", "shape", "bytes"])
            w.writeheader()
            for r in self.rows:
                w.writerow(r)


def generate_golden(weights_dir: Path, layer: int, T: int, max_token: int,
                    out_dir: Path, bias_txt_dir: Path, seed: int = 42, strict_finite: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out_dir)
    np.random.seed(seed)

    x = (np.random.randn(T, HIDDEN).astype(np.float32) * 0.5).astype(np.float16)
    write_fp16(x, out_dir / "input_activation.bin")
    manifest.rows.append({"seq": 0, "name": "hidden_in", "stage": "input", "file": "input_activation.bin", "shape": f"{T}x{HIDDEN}", "bytes": int(x.nbytes)})
    print(f"[1] input_activation.bin shape={x.shape}")

    x_norm = rmsnorm(x)
    manifest.save(f"L{layer}.norm1", x_norm, "rmsnorm1")
    print("[2] rmsnorm1")

    rope_cos = load_fp16(weights_dir / "W_rope.emb_cos.bin", (max_token, HEAD_DIM))[:T]
    rope_sin = load_fp16(weights_dir / "W_rope.emb_sin.bin", (max_token, HEAD_DIM))[:T]
    R = (T + 63) & ~63
    attn_dir = weights_dir / f"layer{layer}_attn"
    heads_out = []

    for h in range(HEADS):
        wq = load_int8_weight(attn_dir / f"q_proj.weight.head{h}.bin", HIDDEN, HEAD_DIM)
        wk = load_int8_weight(attn_dir / f"k_proj.weight.head{h}.bin", HIDDEN, HEAD_DIM)
        wv = load_int8_weight(attn_dir / f"v_proj.weight.head{h}.bin", HIDDEN, HEAD_DIM)
        sq = load_fp16(attn_dir / f"q_proj.weight_scale.head{h}.bin", (HEAD_DIM, 1)).reshape(-1)
        sk = load_fp16(attn_dir / f"k_proj.weight_scale.head{h}.bin", (HEAD_DIM, 1)).reshape(-1)
        sv = load_fp16(attn_dir / f"v_proj.weight_scale.head{h}.bin", (HEAD_DIM, 1)).reshape(-1)
        check_finite(f"q_scale{h}", sq, strict=strict_finite)
        check_finite(f"k_scale{h}", sk, strict=strict_finite)
        check_finite(f"v_scale{h}", sv, strict=strict_finite)

        q = matmul_in_out(x_norm, wq, sq, f"q{h}")
        k = matmul_in_out(x_norm, wk, sk, f"k{h}")
        v = matmul_in_out(x_norm, wv, sv, f"v{h}")
        manifest.save(f"L{layer}.attn.q{h}", q, "q_linear")
        manifest.save(f"L{layer}.attn.k{h}", k, "k_linear")
        manifest.save(f"L{layer}.attn.v{h}", v, "v_linear")

        bq = load_fp16(attn_dir / f"q_proj.bias.head{h}.expanded.bin", (max_token, HEAD_DIM))[:T]
        bk = load_fp16(attn_dir / f"k_proj.bias.head{h}.expanded.bin", (max_token, HEAD_DIM))[:T]
        bv = load_fp16(attn_dir / f"v_proj.bias.head{h}.expanded.bin", (max_token, HEAD_DIM))[:T]
        q = (q.astype(np.float32) + bq).astype(np.float16)
        k = (k.astype(np.float32) + bk).astype(np.float16)
        v = (v.astype(np.float32) + bv).astype(np.float16)
        manifest.save(f"L{layer}.attn.q_ba{h}", q, "q_bias")
        manifest.save(f"L{layer}.attn.k_ba{h}", k, "k_bias")
        manifest.save(f"L{layer}.attn.v_ba{h}", v, "v_bias")

        q = rope(q, rope_cos, rope_sin)
        k = rope(k, rope_cos, rope_sin)
        manifest.save(f"L{layer}.attn.qr{h}", q, "rope_q")
        manifest.save(f"L{layer}.attn.kr{h}", k, "rope_k")

        qk = np.zeros((T, R), dtype=np.float32)
        qk[:, :T] = q.astype(np.float32) @ k.astype(np.float32).T
        check_finite(f"qk{h}.valid", qk[:, :T])
        manifest.save(f"L{layer}.attn.qk{h}", qk.astype(np.float16), "qk")

        qk_m = apply_causal_mask(qk, T, R)
        sm = softmax_hw(qk_m, T)
        manifest.save(f"L{layer}.attn.sm{h}", sm, "softmax")

        v_pad = np.zeros((R, HEAD_DIM), dtype=np.float32)
        v_pad[:T] = v.astype(np.float32)
        oh = sm.astype(np.float32) @ v_pad
        check_finite(f"attn_o{h}", oh)
        oh = oh.astype(np.float16)
        manifest.save(f"L{layer}.attn.o{h}", oh, "attention_v")
        heads_out.append(oh)

        if h == 0:
            print("[3] attention head0 done")

    concat = np.concatenate(heads_out, axis=1).astype(np.float16)
    manifest.save(f"L{layer}.attn.concat", concat, "concat")
    print("[4] concat")

    wo = load_int8_weight(attn_dir / "o_proj.weight.bin", HIDDEN, HIDDEN)
    so = load_fp16(attn_dir / "o_proj.weight_scale.bin", (HIDDEN, 1)).reshape(-1)
    check_finite("o_proj.scale", so, strict=strict_finite)
    o = matmul_in_out(concat, wo, so, "o_proj")
    manifest.save(f"L{layer}.attn.o_proj", o, "o_proj")

    res1 = (x.astype(np.float32) + o.astype(np.float32)).astype(np.float16)
    manifest.save(f"L{layer}.res1", res1, "residual1")
    print("[5] res1")

    x_norm2 = rmsnorm(res1)
    manifest.save(f"L{layer}.norm2", x_norm2, "rmsnorm2")
    print("[6] rmsnorm2")

    mlp_dir = weights_dir / f"layer{layer}_mlp"
    h_parts = []
    for ti, tw in enumerate(MLP_TILES):
        wg = load_int8_weight(mlp_dir / f"gate_proj.weight.tile{ti}.bin", tw, HIDDEN)
        wu = load_int8_weight(mlp_dir / f"up_proj.weight.tile{ti}.bin", tw, HIDDEN)
        sg = load_fp16(mlp_dir / f"gate_proj.weight_scale.tile{ti}.bin", (tw, 1)).reshape(-1)
        su = load_fp16(mlp_dir / f"up_proj.weight_scale.tile{ti}.bin", (tw, 1)).reshape(-1)
        check_finite(f"gate_s{ti}", sg, strict=strict_finite)
        check_finite(f"up_s{ti}", su, strict=strict_finite)

        gate = matmul_out_in(x_norm2, wg, sg, f"gate{ti}")
        up = matmul_out_in(x_norm2, wu, su, f"up{ti}")
        act = silu(gate)
        hpart = (act.astype(np.float32) * up.astype(np.float32)).astype(np.float16)
        check_finite(f"mlp_mul{ti}", hpart)
        manifest.save(f"L{layer}.mlp.gate{ti}", gate, "mlp_gate")
        manifest.save(f"L{layer}.mlp.up{ti}", up, "mlp_up")
        manifest.save(f"L{layer}.mlp.silu{ti}", act, "mlp_silu")
        manifest.save(f"L{layer}.mlp.mul{ti}", hpart, "mlp_mul")
        h_parts.append(hpart)

    sd = load_fp16(mlp_dir / "down_proj.weight_scale.bin", (HIDDEN, 1)).reshape(-1)
    check_finite("down_s", sd, strict=strict_finite)
    acc = None
    for ti, tw in enumerate(MLP_TILES):
        wd = load_int8_weight(mlp_dir / f"down_proj.weight.tile{ti}.bin", tw, HIDDEN)
        # down weight tile: (tile_in, hidden_out); h_part: (T, tile_in)
        part = matmul_in_out(h_parts[ti], wd, sd, f"down{ti}")
        manifest.save(f"L{layer}.mlp.down_part{ti}", part, "mlp_down_part")
        acc = part.astype(np.float32) if acc is None else acc + part.astype(np.float32)

    mlp_out = acc.astype(np.float16)
    manifest.save(f"L{layer}.mlp.out", mlp_out, "mlp_down_sum")
    res2 = (res1.astype(np.float32) + mlp_out.astype(np.float32)).astype(np.float16)
    manifest.save(f"L{layer}.res2", res2, "final_residual")
    write_fp16(res2, out_dir / "golden_21_final.bin")
    manifest.write()
    print(f"[OK] golden final shape={res2.shape}")
    print(f"     out_dir={out_dir}")
    print(f"     manifest={out_dir / 'golden_manifest.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights_processed_from_bin_fixed")
    ap.add_argument("--out", default="golden_t16_l0_fixed")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--max-token", type=int, default=1024)
    ap.add_argument("--bias-txt", default="bias_fp16/bias_fp16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-strict-finite", action="store_true")
    args = ap.parse_args()

    weights_dir = Path(args.weights)
    if not weights_dir.exists():
        sys.exit(f"ERROR: {weights_dir} not found")
    generate_golden(weights_dir, args.layer, args.tokens, args.max_token,
                    Path(args.out), Path(args.bias_txt), args.seed,
                    strict_finite=not args.no_strict_finite)


if __name__ == "__main__":
    main()
