#!/usr/bin/env python3
"""Generate a pure-MLP board-test package.

The generated graph starts with FP16->INT8 quantization before the first
gate/up matmuls, and quantizes each SwiGLU tile before down_proj.  This keeps
every MLP VEC_MATMUL input consistent with the INT8 hardware path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from . import backend
from .compiler import _make_topology, _schedule_and_allocate
from .ir import DType, Kind
from .models import ModelBuilder, PRESETS
from .schedule import report_schedule
from .weights.generate_golden import inverse_reorder_int8
from .weights import weights_map


HIDDEN = 2048
DEFAULT_LINUX_ROOT = "/home/dcy/桌面/linux-kernel/scripts/qwen"
INSTR_BASE = 0x4000000000


def build_pure_mlp_graph(model: str, tokens: int, layer: int):
    cfg = PRESETS[model]
    mb = ModelBuilder(cfg, tokens)
    x = mb.g.tensor("hidden_in", (tokens, cfg.hidden), DType.FP16, Kind.INPUT)
    out = mb.build_mlp(x, layer, token_num=tokens)
    mb.g.tensors[out].kind = Kind.OUTPUT
    return mb.g


def write_bytes(path: Path, arr: np.ndarray, dtype) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = arr.astype(dtype)
    data.tofile(path)
    return int(data.nbytes)


class GoldenManifest:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.rows: list[dict[str, object]] = []
        self.seq = 0

    def save(self, name: str, arr: np.ndarray, stage: str, dtype=np.float16):
        self.seq += 1
        dtype_name = np.dtype(dtype).name
        fn = f"{self.seq:04d}_{name.replace('.', '_')}.bin"
        nbytes = write_bytes(self.out_dir / fn, arr, dtype)
        self.rows.append({
            "seq": self.seq,
            "name": name,
            "stage": stage,
            "file": fn,
            "shape": "x".join(map(str, arr.shape)),
            "dtype": dtype_name,
            "bytes": nbytes,
        })

    def write(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "golden_manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["seq", "name", "stage", "file", "shape", "dtype", "bytes"],
            )
            w.writeheader()
            for row in self.rows:
                w.writerow(row)


def load_fp16(path: Path, shape=None) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float16)
    if shape is not None:
        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(f"{path}: got {arr.size}, expected {expected} for shape={shape}")
        arr = arr.reshape(shape)
    return arr.astype(np.float32)


def load_int8_weight(path: Path, rows: int, cols: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int8)
    return inverse_reorder_int8(raw, rows, cols)


def fp16_to_int8_rowwise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x32 = x.astype(np.float32)
    max_abs = np.max(np.abs(x32), axis=1)
    scale = max_abs / 127.0
    scale = np.where(scale > 0, scale, 1.0 / 127.0).astype(np.float32)
    q = np.rint(x32 / scale[:, None])
    q = np.clip(q, -128, 127).astype(np.int8)
    deq = (q.astype(np.float32) * scale[:, None]).astype(np.float16)
    return q, scale.astype(np.float16), deq


def int8_matmul_out_in(a_q: np.ndarray, a_scale: np.ndarray,
                       w_out_in: np.ndarray, w_scale: np.ndarray,
                       name: str) -> np.ndarray:
    if w_out_in.shape[0] != w_scale.size:
        raise ValueError(f"{name}: scale length {w_scale.size} != weight rows {w_out_in.shape[0]}")
    acc = a_q.astype(np.float32) @ w_out_in.astype(np.float32).T
    out = acc * a_scale.astype(np.float32).reshape(-1, 1) * w_scale.reshape(1, -1)
    if not np.isfinite(out).all():
        raise FloatingPointError(f"{name}: non-finite output")
    return out.astype(np.float16)


def int8_matmul_in_out(a_q: np.ndarray, a_scale: np.ndarray,
                       w_in_out: np.ndarray, w_scale: np.ndarray,
                       name: str) -> np.ndarray:
    if w_in_out.shape[1] != w_scale.size:
        raise ValueError(f"{name}: scale length {w_scale.size} != weight cols {w_in_out.shape[1]}")
    acc = a_q.astype(np.float32) @ w_in_out.astype(np.float32)
    out = acc * a_scale.astype(np.float32).reshape(-1, 1) * w_scale.reshape(1, -1)
    if not np.isfinite(out).all():
        raise FloatingPointError(f"{name}: non-finite output")
    return out.astype(np.float16)


def silu(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    return (x32 / (1.0 + np.exp(-x32))).astype(np.float16)


def generate_mlp_golden(weights_dir: Path, out_dir: Path, *,
                        model: str, layer: int, tokens: int, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = GoldenManifest(out_dir)
    rng = np.random.default_rng(seed)
    cfg = PRESETS[model]
    hidden = cfg.hidden
    tiles = list(cfg.mlp_tiles)

    x = (rng.standard_normal((tokens, hidden), dtype=np.float32) * 0.5).astype(np.float16)
    write_bytes(out_dir / "input_embedding_fp16.bin", x, np.float16)
    write_bytes(out_dir / "input_activation.bin", x, np.float16)
    manifest.save("hidden_in", x, "input_embedding_fp16", np.float16)

    x_q, x_scale, _ = fp16_to_int8_rowwise(x)
    manifest.save(f"L{layer}.mlp.x_i8", x_q, "quant_in.int8", np.int8)
    manifest.save(f"L{layer}.mlp.x_scale", x_scale, "quant_in.scale", np.float16)

    layer_dir = weights_dir / f"layer{layer}_mlp"
    gate_parts: list[np.ndarray] = []
    up_parts: list[np.ndarray] = []

    for ti, tw in enumerate(tiles):
        wg = load_int8_weight(layer_dir / f"gate_proj.weight.tile{ti}.bin", tw, hidden)
        sg = load_fp16(layer_dir / f"gate_proj.weight_scale.tile{ti}.bin", (tw,))
        gate = int8_matmul_out_in(x_q, x_scale, wg, sg, f"gate{ti}")
        manifest.save(f"L{layer}.mlp.gate{ti}", gate, f"gate_mm{ti}", np.float16)
        gate_parts.append(gate)

        wu = load_int8_weight(layer_dir / f"up_proj.weight.tile{ti}.bin", tw, hidden)
        su = load_fp16(layer_dir / f"up_proj.weight_scale.tile{ti}.bin", (tw,))
        up = int8_matmul_out_in(x_q, x_scale, wu, su, f"up{ti}")
        manifest.save(f"L{layer}.mlp.up{ti}", up, f"up_mm{ti}", np.float16)
        up_parts.append(up)

    h_q_parts: list[tuple[np.ndarray, np.ndarray]] = []
    for ti, tw in enumerate(tiles):
        act = silu(gate_parts[ti])
        manifest.save(f"L{layer}.mlp.act{ti}", act, f"silu{ti}", np.float16)
        h = (act.astype(np.float32) * up_parts[ti].astype(np.float32)).astype(np.float16)
        manifest.save(f"L{layer}.mlp.h{ti}", h, f"mul{ti}", np.float16)
        h_q, h_scale, _ = fp16_to_int8_rowwise(h)
        manifest.save(f"L{layer}.mlp.h_i8_{ti}", h_q, f"quant_h{ti}.int8", np.int8)
        manifest.save(f"L{layer}.mlp.h_scale{ti}", h_scale, f"quant_h{ti}.scale", np.float16)
        h_q_parts.append((h_q, h_scale))

    down_scale = load_fp16(layer_dir / "down_proj.weight_scale.bin", (hidden,))
    partials: list[np.ndarray] = []
    for ti, tw in enumerate(tiles):
        wd = load_int8_weight(layer_dir / f"down_proj.weight.tile{ti}.bin", tw, hidden)
        h_q, h_scale = h_q_parts[ti]
        part = int8_matmul_in_out(h_q, h_scale, wd, down_scale, f"down{ti}")
        manifest.save(f"L{layer}.mlp.down{ti}", part, f"down_mm{ti}", np.float16)
        partials.append(part)

    acc = partials[0]
    for ti in range(1, len(partials)):
        acc = (acc.astype(np.float32) + partials[ti].astype(np.float32)).astype(np.float16)
        manifest.save(f"L{layer}.mlp.acc{ti}", acc, f"add{ti}", np.float16)

    write_bytes(out_dir / "mlp_output_fp16.bin", acc, np.float16)
    manifest.write()
    return out_dir / "input_activation.bin"


def write_artifacts(g, sched, planners, outdir: Path, *, model: str, strategy: str):
    outdir.mkdir(parents=True, exist_ok=True)
    raw, hexl, listing = backend.emit_bin(g, sched)
    c_src = backend.emit_c(g, sched, model=f"{model}_pure_mlp", strategy=strategy)
    report = backend.emit_report(g, sched, planners, {
        "model": model,
        "tokens": g.tensors["hidden_in"].shape[0],
        "pure_mlp_test": True,
        "strategy": strategy,
        "cores": sched.n_cores,
    })
    with open(outdir / "firmware_mlp_unrolled.c", "w", encoding="utf-8", newline="\n") as f:
        f.write(c_src)
    with open(outdir / "program.bin", "wb") as f:
        f.write(raw)
    with open(outdir / "program.hex", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(hexl) + "\n")
    with open(outdir / "program.list", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(listing) + "\n")
    with open(outdir / "hbm_alloc.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(planners["hbm"].table() + "\n")
    write_debug_alloc(g, outdir / "hbm_debug_alloc.txt")
    with open(outdir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    (outdir / "README_EXECUTION.txt").write_text(
        "firmware_mlp_unrolled.c is the TinyRISC-V C source to compile with your normal flow.\n"
        "program.bin is a 16-word-per-op macro-instruction data stream, not executable RV32 code.\n"
        "Only load program.bin to 0x4000000000 when the TinyRISC-V is already running an interpreter firmware that reads this data stream.\n",
        encoding="utf-8",
    )
    return raw


def write_debug_alloc(g, path: Path):
    rows = []
    for t in g.tensors.values():
        if t.addr is None or t.meta.get("alias_base"):
            continue
        rows.append((int(t.addr), t.name, int(t.nbytes), t.kind.value,
                     str(t.meta.get("pc", "UNKNOWN"))))
    rows.sort()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("HBM debug allocation\n")
        f.write("-" * 78 + "\n")
        for addr, name, size, kind, pc in rows:
            f.write(f"  {name:<28} 0x{addr:010X} +0x{size:08X} {kind:<10} {pc}\n")
        f.write("-" * 78 + "\n")


def append_program_load(script_path: Path, program_path: Path, program_size: int,
                        *, linux_root: str, dev: str):
    board_program = weights_map.to_linux_path(program_path, linux_root)
    with open(script_path, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n# program image -> instruction region\n")
        f.write(
            f"dma-to-device -d {dev} -a 0x{INSTR_BASE:010X} "
            f"-f {board_program} -s {program_size}\n"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen", choices=sorted(PRESETS))
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--weights", default="weights_processed_from_bin_fixed")
    ap.add_argument("--out", default="out/mlp_fp16int8_test")
    ap.add_argument("--golden-out", default="golden_mlp_t16_l0_fp16int8")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--strategy", default="single",
                    choices=["single", "round_robin", "critical", "lane_aware_critical"])
    ap.add_argument("--linux-root", default=DEFAULT_LINUX_ROOT)
    ap.add_argument("--dev", default="/dev/qdma01000-MM-0")
    ap.add_argument("--skip-bringup", action="store_true")
    ap.add_argument("--load-macro-program", action="store_true",
                    help="append program.bin DMA load to 0x4000000000; use only with an interpreter firmware")
    ap.add_argument("--recycle-scratch", action="store_true",
                    help="reuse scratch activations; off by default so debug readback preserves every MLP intermediate")
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    golden_dir = Path(args.golden_out)
    weights_dir = Path(args.weights)
    if not weights_dir.exists():
        raise FileNotFoundError(weights_dir)

    g = build_pure_mlp_graph(args.model, args.tokens, args.layer)
    topology = _make_topology(args.cores, shared_dma=False, qos_xlsx=None)
    sched, planners = _schedule_and_allocate(
        g, cores=args.cores, strategy=args.strategy, affinity=None,
        recycle=args.recycle_scratch, topology=topology, pc_aware_schedule=True,
        strict_routes=True,
    )
    raw = write_artifacts(g, sched, planners, outdir,
                          model=args.model, strategy=args.strategy)

    input_bin = generate_mlp_golden(
        weights_dir, golden_dir,
        model=args.model, layer=args.layer, tokens=args.tokens, seed=args.seed,
    )

    load_script = outdir / "dma_mlp_load.sh"
    debug_script = outdir / "dma_mlp_debug_readback.sh"
    weights_map_rc = weights_map.main([
        str(outdir / "hbm_debug_alloc.txt"),
        "--weights", str(weights_dir),
        "--input-bin", str(input_bin),
        "-o", str(load_script),
        "--manifest", str(outdir / "dma_mlp_manifest.csv"),
        "--problem-report", str(outdir / "dma_mlp_problems.csv"),
        "--debug-out", str(debug_script),
        "--golden-dir", str(golden_dir),
        "--golden-manifest", str(golden_dir / "golden_manifest.csv"),
        "--debug-categories", "runtime,output",
        "--tokens", str(args.tokens),
        "--linux-root", args.linux_root,
        "--dev", args.dev,
    ] + (["--skip-bringup"] if args.skip_bringup else []))
    if weights_map_rc:
        return weights_map_rc

    if args.load_macro_program:
        append_program_load(load_script, outdir / "program.bin", len(raw),
                            linux_root=args.linux_root, dev=args.dev)

    print(g.summary())
    print(report_schedule(sched))
    print(f"Golden    : {golden_dir}")
    print(f"Artifacts : {outdir}")
    print(f"Firmware  : {outdir / 'firmware_mlp_unrolled.c'}")
    print(f"Load DMA  : {load_script}")
    print(f"Debug DMA : {debug_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
