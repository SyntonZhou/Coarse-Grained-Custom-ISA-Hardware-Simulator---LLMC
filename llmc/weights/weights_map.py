#!/usr/bin/env python3
"""
weights_map_fixed.py

Generate DMA weight/input loading scripts from llmc hbm_alloc.txt and a
post-processed weights directory, e.g. weights_processed_from_bin/.

Compared with the original weights_map.py, this version is stricter:
  1. Only maps initial resident tensors: weights, weight scales, expanded bias,
     RoPE, mask slice, lm_head, and optional hidden_in input.
  2. Explicitly skips runtime-produced tensors such as L*.attn.k_scale* and
     L*.attn.v_scale*. These must NOT be mapped to weight_scale.head*.bin.
  3. Checks source existence and exact file size before emitting DMA commands.
  4. Generates a CSV manifest plus a missing/mismatch report.
  5. Exits non-zero if --fail-on-problem is enabled and any required file is
     missing or has a size mismatch.

Typical usage:
  python weights_map_fixed.py hbm_alloc.txt \
    --weights weights_processed_from_bin \
    --input-bin golden/input_activation.bin \
    -o dma_weights.sh \
    --manifest dma_weights_manifest.csv \
    -t 16 --max-token 1024 --fail-on-problem
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

try:
    import numpy as np
except Exception:  # allow manifest generation for non-bias paths
    np = None


@dataclass
class HbmEntry:
    name: str
    addr: int
    size: int


@dataclass
class ManifestRow:
    name: str
    addr_hex: str
    size: int
    category: str       # load | input | skip | runtime | output | missing | mismatch
    layout: str
    src: str
    src_size: int
    prepared: str
    prepared_size: int
    dma_file: str
    dma_size: int
    status: str         # OK | SKIP | MISSING | SIZE_MISMATCH | ERROR
    note: str


def q(p: str | Path) -> str:
    return shlex.quote(str(p))


DEFAULT_LINUX_ROOT = "/home/dcy/桌面/linux-kernel/scripts/qwen"

QDMA_BRINGUP = [
    "# Disable VFs and reset qmax",
    "echo 0 > /sys/bus/pci/devices/0000:01:00.0/sriov_numvfs",
    "echo 0 > /sys/bus/pci/devices/0000:01:00.0/qdma/qmax",
    "",
    "# Remove and rescan PCIe device for a full QDMA re-init",
    "echo 1 > /sys/bus/pci/devices/0000:01:00.0/remove",
    "sleep 2",
    "echo 1 > /sys/bus/pci/rescan",
    "sleep 2",
    "",
    "# Re-initialize QDMA from scratch",
    "echo 8 > /sys/bus/pci/devices/0000:01:00.0/qdma/qmax",
    "echo 3 > /sys/bus/pci/devices/0000:01:00.0/sriov_numvfs",
    "sleep 2",
    "",
    "dma-ctl qdma01000 q add idx 0 mode mm dir bi",
    "dma-ctl qdma01000 q start idx 0 dir bi",
]


def to_linux_path(path: str | Path, linux_root: str) -> str:
    """Map generated local paths to the Linux board-side workspace path.

    The board scripts are intentionally emitted without shell quotes because
    the lab commands are normally copied and run in a no-space Linux path.
    """
    s = str(path).replace("\\", "/")
    if not s:
        return ""
    if s.startswith("/"):
        return s

    # If a Windows absolute path leaks in, keep only the project-relative tail
    # when a known artifact directory can be found.
    m = re.match(r"^[A-Za-z]:/(.*)$", s)
    if m:
        tail = m.group(1)
        parts = tail.split("/")
        anchors = [
            "weights_processed_from_bin_fixed",
            "weights_processed_from_bin",
            "weights_processed",
            "golden_t16_l0_fixed",
            "golden_t16_l0",
            "golden",
        ]
        for anchor in anchors:
            if anchor in parts:
                s = "/".join(parts[parts.index(anchor):])
                break
        else:
            s = parts[-1]

    return linux_root.rstrip("/") + "/" + s.lstrip("./")


def script_header(*, include_bringup: bool) -> list[str]:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if include_bringup:
        lines.extend(QDMA_BRINGUP)
        lines.append("")
    return lines


def parse_hbm_alloc(path: str | Path) -> list[HbmEntry]:
    """Parse lines like:  L0.attn.q_w0  0x4200...  +0x40000"""
    pat = re.compile(r"\s*(\S+)\s+0x([0-9A-Fa-f]+)\s+\+0x([0-9A-Fa-f]+)")
    out: list[HbmEntry] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            out.append(HbmEntry(m.group(1), int(m.group(2), 16), int(m.group(3), 16)))
    return out


def stat_size(p: Optional[Path]) -> int:
    if p is None:
        return 0
    try:
        return p.stat().st_size if p.exists() else 0
    except OSError:
        return 0


def read_txt_floats_stream(path: Path):
    if np is None:
        raise RuntimeError("numpy is required to generate bias from txt")
    with open(path, "r", buffering=1024 * 1024) as f:
        n = sum(1 for _ in f)
    arr = np.empty(n, dtype=np.float32)
    with open(path, "r", buffering=1024 * 1024) as f:
        for i, line in enumerate(f):
            line = line.strip()
            arr[i] = float(line.split()[0]) if line else 0.0
    return arr


def write_fp16_le(data, path: Path):
    if np is None:
        raise RuntimeError("numpy is required to write fp16")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.astype(np.float16).tofile(path)


def expand_bias_from_txt(src_txt: Path, dst_bin: Path, head_idx: int, max_token: int):
    b = read_txt_floats_stream(src_txt).reshape(1, 2048)
    head_b = b[:, head_idx * 128:(head_idx + 1) * 128]
    expanded = np.repeat(head_b, max_token, axis=0)
    write_fp16_le(expanded, dst_bin)


def extract_mask(src: Path, dst: Path, token_num: int):
    R = (token_num + 63) & ~63
    sizes = list(range(64, 1025, 64))
    if R not in sizes:
        raise ValueError(f"R={R} not supported by W_mask.bin")
    idx = sizes.index(R)
    offset = sum(s * s * 2 for s in sizes[:idx])
    size = R * R * 2
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f:
        f.seek(offset)
        data = f.read(size)
    if len(data) != size:
        raise IOError(f"short read from {src}: got {len(data)}, expected {size}")
    with open(dst, "wb") as f:
        f.write(data)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def make_row(e: HbmEntry, category: str, layout: str, src: Optional[Path], prepared: Optional[Path],
             dma_file: Optional[Path], dma_size: int, status: str, note: str) -> ManifestRow:
    return ManifestRow(
        name=e.name,
        addr_hex=f"0x{e.addr:010X}",
        size=e.size,
        category=category,
        layout=layout,
        src=str(src) if src else "",
        src_size=stat_size(src),
        prepared=str(prepared) if prepared else "",
        prepared_size=stat_size(prepared),
        dma_file=str(dma_file) if dma_file else "",
        dma_size=dma_size,
        status=status,
        note=note,
    )


def resolve_tensor(e: HbmEntry, weights: Path, tmp_dir: Path, *, token_num: int, max_token: int,
                   bias_txt_dir: Path, input_bin: Optional[Path], allow_generate_bias: bool) -> ManifestRow:
    name = e.name

    # Program/control images are not weight files.
    if "program" in name.lower() or "firmware" in name.lower():
        return make_row(e, "skip", "program_or_firmware", None, None, None, 0, "SKIP",
                        "program image is generated by firmware/codegen, not weights_map")

    # Optional input activation.
    if name == "hidden_in" or name.endswith(".hidden_in"):
        if input_bin is None:
            return make_row(e, "input", "input_activation_fp16", None, None, None, 0, "SKIP",
                            "hidden_in skipped; pass --input-bin to DMA-load input activation")
        return make_row(e, "input", "input_activation_fp16", input_bin, None, input_bin, e.size, "OK", "")

    # Final output is produced by hardware.
    if name.endswith("res2") or name.endswith("output") or name in {"hidden_out", "output"}:
        return make_row(e, "output", "hardware_output", None, None, None, 0, "SKIP", "output tensor")

    # Explicit runtime tensors: must not be initial-loaded.
    if re.match(r"^L\d+\.attn\.[kv]_scale\d+$", name):
        return make_row(e, "runtime", "runtime_attention_scale_fp16", None, None, None, 0, "SKIP",
                        "runtime scale tensor; do not map to weight_scale.head*.bin")

    # RoPE / mask.
    m = re.match(r"^L(\d+)\.attn\.rope_cos$", name)
    if m:
        src = weights / "W_rope.emb_cos.bin"
        return make_row(e, "load", "rope_cos_table_fp16", src, None, src, e.size, "OK", "")

    m = re.match(r"^L(\d+)\.attn\.rope_sin$", name)
    if m:
        src = weights / "W_rope.emb_sin.bin"
        return make_row(e, "load", "rope_sin_table_fp16", src, None, src, e.size, "OK", "")

    m = re.match(r"^L(\d+)\.attn\.mask$", name)
    if m:
        src = weights / "W_mask.bin"
        R = (token_num + 63) & ~63
        prepared = tmp_dir / f"W_mask_R{R}.bin"
        if src.exists() and not prepared.exists():
            extract_mask(src, prepared, token_num)
        return make_row(e, "load", f"mask_R{R}_slice_fp16", src, prepared, prepared, e.size, "OK", "")

    # Attention q/k/v weights, scales, biases.
    m = re.match(r"^L(\d+)\.attn\.([qkv])_w(\d+)$", name)
    if m:
        L, qkv, h = int(m.group(1)), m.group(2), int(m.group(3))
        proj = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}[qkv]
        src = weights / f"layer{L}_attn" / f"{proj}.weight.head{h}.bin"
        return make_row(e, "load", "attn_qkv_head_weight_int8_block64x8", src, None, src, e.size, "OK", "")

    m = re.match(r"^L(\d+)\.attn\.([qkv])_s(\d+)$", name)
    if m:
        L, qkv, h = int(m.group(1)), m.group(2), int(m.group(3))
        proj = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}[qkv]
        src = weights / f"layer{L}_attn" / f"{proj}.weight_scale.head{h}.bin"
        return make_row(e, "load", "attn_qkv_head_weight_scale_fp16", src, None, src, e.size, "OK", "")

    m = re.match(r"^L(\d+)\.attn\.([qkv])_b(\d+)$", name)
    if m:
        L, qkv, h = int(m.group(1)), m.group(2), int(m.group(3))
        proj = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}[qkv]
        src = weights / f"layer{L}_attn" / f"{proj}.bias.head{h}.expanded.bin"
        if not src.exists() and allow_generate_bias:
            b_txt = bias_txt_dir / f"model.layers.{L}.self_attn.{qkv}_proj.bias.txt"
            if b_txt.exists():
                expand_bias_from_txt(b_txt, src, h, max_token)
        return make_row(e, "load", "attn_qkv_head_bias_expanded_fp16", src, None, src, e.size, "OK", "")

    # Attention output projection.
    m = re.match(r"^L(\d+)\.attn\.o_w$", name)
    if m:
        L = int(m.group(1))
        src = weights / f"layer{L}_attn" / "o_proj.weight.bin"
        return make_row(e, "load", "attn_o_weight_int8_block64x8", src, None, src, e.size, "OK", "")

    m = re.match(r"^L(\d+)\.attn\.o_s$", name)
    if m:
        L = int(m.group(1))
        src = weights / f"layer{L}_attn" / "o_proj.weight_scale.bin"
        return make_row(e, "load", "attn_o_weight_scale_fp16", src, None, src, e.size, "OK", "")

    # MLP weights/scales.
    m = re.match(r"^L(\d+)\.mlp\.(gate|up|down)_w(\d+)$", name)
    if m:
        L, proj, tile = int(m.group(1)), m.group(2), int(m.group(3))
        p = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[proj]
        src = weights / f"layer{L}_mlp" / f"{p}.weight.tile{tile}.bin"
        return make_row(e, "load", "mlp_tile_weight_int8_block64x8", src, None, src, e.size, "OK", "")

    m = re.match(r"^L(\d+)\.mlp\.(gate|up)_s(\d+)$", name)
    if m:
        L, proj, tile = int(m.group(1)), m.group(2), int(m.group(3))
        p = {"gate": "gate_proj", "up": "up_proj"}[proj]
        src = weights / f"layer{L}_mlp" / f"{p}.weight_scale.tile{tile}.bin"
        return make_row(e, "load", "mlp_gate_up_tile_scale_fp16", src, None, src, e.size, "OK", "")

    m = re.match(r"^L(\d+)\.mlp\.down_s(\d+)$", name)
    if m:
        L, tile = int(m.group(1)), int(m.group(2))
        src = weights / f"layer{L}_mlp" / "down_proj.weight_scale.bin"
        return make_row(e, "load", "mlp_down_full_weight_scale_fp16", src, None, src, e.size, "OK",
                        f"same down_proj.weight_scale.bin loaded for down_s{tile}")

    # LM head if current graph includes it.
    if re.search(r"lm_head.*scale", name):
        src = weights / "lm_head.weight_scale.bin"
        return make_row(e, "load", "lm_head_weight_scale_fp16", src, None, src, e.size, "OK", "")
    if "lm_head" in name:
        src = weights / "lm_head.weight.bin"
        return make_row(e, "load", "lm_head_weight_int8_block64x8", src, None, src, e.size, "OK", "")

    # Remaining L* tensors are runtime activations/scratch.
    if re.match(r"^L\d+\.", name):
        return make_row(e, "runtime", "runtime_activation_or_scratch", None, None, None, 0, "SKIP",
                        "runtime tensor; initial DMA not required")

    return make_row(e, "skip", "unknown", None, None, None, 0, "SKIP", "unrecognized non-weight tensor")


def validate_row(row: ManifestRow, allow_size_mismatch: bool) -> ManifestRow:
    if row.status == "SKIP":
        return row
    p = Path(row.dma_file) if row.dma_file else None
    if p is None or not p.exists():
        row.status = "MISSING"
        row.note = (row.note + "; " if row.note else "") + "DMA source file missing"
        return row
    actual = p.stat().st_size
    row.dma_size = row.size
    if actual != row.size and not allow_size_mismatch:
        row.status = "SIZE_MISMATCH"
        row.note = (row.note + "; " if row.note else "") + f"source size {actual} != hbm size {row.size}"
    else:
        row.status = "OK"
    return row


def emit_load_script(rows: list[ManifestRow], args) -> str:
    lines = script_header(include_bringup=not args.skip_bringup)
    n_cmd = 0
    for r in rows:
        if r.status == "OK" and r.category in {"load", "input"}:
            src = to_linux_path(r.dma_file, args.linux_root)
            lines.append(f"dma-to-device -d {args.dev} -a {r.addr_hex} -f {src} -s {r.size}")
            n_cmd += 1
        elif r.status == "SKIP":
            lines.append(f"# SKIP {r.name}: {r.note}")
        else:
            lines.append(f"# {r.status} {r.name}: {r.note} src={r.src} prepared={r.prepared}")
    lines.append("")
    lines.append(f"# commands: {n_cmd}")
    return "\n".join(lines) + "\n"


def _debug_categories(text: str) -> set[str]:
    return {x.strip() for x in text.split(",") if x.strip()}


def emit_debug_script(rows: list[ManifestRow], args) -> str:
    cats = _debug_categories(args.debug_categories)
    read_rows = [r for r in rows if r.category in cats and r.size > 0]
    readback_dir = to_linux_path(args.readback_dir, args.linux_root)
    golden_dir = to_linux_path(args.golden_dir, args.linux_root)
    golden_manifest = args.golden_manifest or str(Path(args.golden_dir) / "golden_manifest.csv")
    golden_manifest = to_linux_path(golden_manifest, args.linux_root)

    lines = script_header(include_bringup=False)
    lines.extend([
        "# This script is intended for post-execution debug readback.",
        "# Set BRINGUP=1 only when the QDMA queue is not already initialized.",
        "if [ \"${BRINGUP:-0}\" = \"1\" ]; then",
    ])
    lines.extend([f"  {line}" if line else "" for line in QDMA_BRINGUP])
    lines.extend([
        "fi",
        "",
        f"mkdir -p {readback_dir}",
        "",
    ])

    py_rows: list[tuple[str, str, int, str]] = []
    for r in read_rows:
        fn = f"{sanitize(r.name)}.bin"
        out = f"{readback_dir.rstrip('/')}/{fn}"
        lines.append(f"dma-from-device -d {args.dev} -a {r.addr_hex} -f {out} -s {r.size}")
        py_rows.append((r.name, out, r.size, r.category))

    lines.extend([
        "",
        f"if [ -f {golden_manifest} ]; then",
        f"python3 - <<'PY'",
        "import csv, math, os, sys",
        "",
        f"READBACK_DIR = {readback_dir!r}",
        f"GOLDEN_DIR = {golden_dir!r}",
        f"GOLDEN_MANIFEST = {golden_manifest!r}",
        f"ATOL = {float(args.compare_atol)!r}",
        f"RTOL = {float(args.compare_rtol)!r}",
        "ROWS = [",
    ])
    for name, path, size, category in py_rows:
        lines.append(f"    ({name!r}, {path!r}, {size}, {category!r}),")
    lines.extend([
        "]",
        "",
        "try:",
        "    import numpy as np",
        "except Exception:",
        "    np = None",
        "",
        "golden = {}",
        "with open(GOLDEN_MANIFEST, newline='', encoding='utf-8') as f:",
        "    for row in csv.DictReader(f):",
        "        if row.get('name') and row.get('file'):",
        "            golden[row['name']] = os.path.join(GOLDEN_DIR, row['file'])",
        "",
        "def compare_bytes(a, b):",
        "    if a == b:",
        "        return 'exact'",
        "    if np is None or len(a) % 2 or len(b) % 2:",
        "        return 'raw_mismatch'",
        "    x = np.frombuffer(a, dtype='<f2').astype(np.float32)",
        "    y = np.frombuffer(b, dtype='<f2').astype(np.float32)",
        "    n = min(x.size, y.size)",
        "    if n == 0:",
        "        return 'empty'",
        "    x = x[:n]",
        "    y = y[:n]",
        "    diff = np.abs(x - y)",
        "    denom = np.maximum(np.abs(y), 1e-6)",
        "    rel = diff / denom",
        "    ok = bool(np.all(diff <= (ATOL + RTOL * np.abs(y))))",
        "    return (",
        "        f\"fp16_allclose={ok} max_abs={float(np.nanmax(diff)):.6g} \"",
        "        f\"mean_abs={float(np.nanmean(diff)):.6g} max_rel={float(np.nanmax(rel)):.6g}\"",
        "    )",
        "",
        "matched = 0",
        "missing = 0",
        "print('[COMPARE] readback vs golden manifest')",
        "for name, got_path, size, category in ROWS:",
        "    ref_path = golden.get(name)",
        "    if not ref_path or not os.path.exists(ref_path):",
        "        missing += 1",
        "        continue",
        "    if not os.path.exists(got_path):",
        "        print(f'MISSING_READBACK {name} {got_path}')",
        "        continue",
        "    with open(got_path, 'rb') as f:",
        "        got = f.read(size)",
        "    with open(ref_path, 'rb') as f:",
        "        ref = f.read(size)",
        "    matched += 1",
        "    status = compare_bytes(got, ref)",
        "    print(f'{name}: {status} got={got_path} ref={ref_path}')",
        "print(f'[COMPARE] matched={matched} no_golden={missing}')",
        "PY",
        "else",
        f"  echo '[COMPARE] golden manifest not found: {golden_manifest}'",
        "fi",
        "",
        f"# readback regions: {len(read_rows)}",
    ])
    return "\n".join(lines) + "\n"


def write_script(path: Path, text: str):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def generate(args) -> int:
    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: weights dir not found: {weights}", file=sys.stderr)
        return 2

    entries = parse_hbm_alloc(args.hbm_alloc)
    if not entries:
        print(f"ERROR: no entries parsed from {args.hbm_alloc}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True) if out_path.parent != Path("") else None
    tmp_dir = out_path.with_suffix(".tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_path.with_suffix(".manifest.csv")
    problem_path = Path(args.problem_report) if args.problem_report else out_path.with_suffix(".problems.csv")

    rows: list[ManifestRow] = []
    for e in sorted(entries, key=lambda x: x.addr):
        try:
            row = resolve_tensor(
                e, weights, tmp_dir,
                token_num=args.tokens,
                max_token=args.max_token,
                bias_txt_dir=Path(args.bias_txt),
                input_bin=Path(args.input_bin) if args.input_bin else None,
                allow_generate_bias=args.generate_bias,
            )
            row = validate_row(row, args.allow_size_mismatch)
        except Exception as ex:
            row = make_row(e, "error", "error", None, None, None, 0, "ERROR", str(ex))
        rows.append(row)

    # Write scripts and CSVs.
    write_script(out_path, emit_load_script(rows, args))
    try:
        os.chmod(out_path, 0o755)
    except OSError:
        pass

    debug_out = Path(args.debug_out) if args.debug_out else out_path.with_name(out_path.stem + "_debug_readback.sh")
    debug_out.parent.mkdir(parents=True, exist_ok=True) if debug_out.parent != Path("") else None
    write_script(debug_out, emit_debug_script(rows, args))
    try:
        os.chmod(debug_out, 0o755)
    except OSError:
        pass

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    problems = [r for r in rows if r.status in {"MISSING", "SIZE_MISMATCH", "ERROR"}]
    with open(problem_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in problems:
            w.writerow(asdict(r))

    counts = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    cats = {}
    for r in rows:
        cats[r.category] = cats.get(r.category, 0) + 1

    print(f"DMA script: {out_path}")
    print(f"Debug DMA : {debug_out}")
    print(f"Manifest  : {manifest_path}")
    print(f"Problems  : {problem_path}")
    print(f"Commands  : {sum(1 for r in rows if r.status == 'OK' and r.category in {'load', 'input'})}")
    print(f"Status    : {counts}")
    print(f"Category  : {cats}")

    if problems:
        print("\nProblems found:", file=sys.stderr)
        for r in problems[:20]:
            print(f"  {r.status}: {r.name} {r.note}", file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... {len(problems)-20} more", file=sys.stderr)
        if args.fail_on_problem:
            return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hbm_alloc")
    ap.add_argument("--weights", required=True, help="weights_processed_from_bin/ directory")
    ap.add_argument("--bias-txt", default="bias_fp16/bias_fp16")
    ap.add_argument("--generate-bias", action="store_true", help="generate missing expanded bias from bias txt")
    ap.add_argument("--input-bin", default=None, help="optional input_activation.bin for hidden_in")
    ap.add_argument("-o", "--out", default="dma_weights.sh")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--problem-report", default=None)
    ap.add_argument("-t", "--tokens", type=int, default=16)
    ap.add_argument("--max-token", type=int, default=1024)
    ap.add_argument("--dev", default="/dev/qdma01000-MM-0")
    ap.add_argument("--linux-root", default=DEFAULT_LINUX_ROOT,
                    help="Linux board-side root used to emit absolute script paths")
    ap.add_argument("--skip-bringup", action="store_true",
                    help="do not prepend full QDMA bring-up to the load script")
    ap.add_argument("--debug-out", default=None,
                    help="post-execution readback script path; default: <out>_debug_readback.sh")
    ap.add_argument("--readback-dir", default="debug_readback",
                    help="Linux board-side directory for dma-from-device outputs")
    ap.add_argument("--golden-dir", default="golden_t16_l0_fixed",
                    help="Linux board-side golden directory used by the debug script")
    ap.add_argument("--golden-manifest", default=None,
                    help="golden manifest path; default: <golden-dir>/golden_manifest.csv")
    ap.add_argument("--debug-categories", default="runtime,output",
                    help="comma-separated manifest categories to read back")
    ap.add_argument("--compare-atol", type=float, default=1e-2)
    ap.add_argument("--compare-rtol", type=float, default=1e-2)
    ap.add_argument("--allow-size-mismatch", action="store_true")
    ap.add_argument("--fail-on-problem", action="store_true")
    args = ap.parse_args(argv)
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
