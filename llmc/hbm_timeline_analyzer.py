#!/usr/bin/env python3
"""
hbm_timeline_analyzer.py

Schedule-driven HBM data-state and traffic timeline analyzer for the llmc model.

It complements hbm_analyzer.py:
  - hbm_analyzer.py: static occupancy + aggregate traffic + conflict checks.
  - hbm_timeline_analyzer.py: dynamic timeline of valid/live HBM data as the
    scheduled program executes wave by wave.

Outputs:
  1. JSON report with:
     - initial resident data: firmware/program image, inputs, weights, scales, KV.
     - per-wave HBM reads/writes.
     - per-wave tensors produced/freed.
     - live/valid HBM occupancy after each wave, per PC and per kind.
     - peak valid occupancy and peak traffic waves.
     - final resident outputs.
  2. CSV tables:
     - timeline_by_wave.csv
     - pc_occupancy_timeline.csv
     - pc_traffic_by_wave.csv
     - tensor_lifetimes.csv
  3. Optional PNG plots:
     - occupancy_over_time.png
     - traffic_over_time.png
     - pc_peak_occupancy.png

Example:
  PYTHONPATH=/path/to/llmc_package python hbm_timeline_analyzer.py \
      --model qwen --tokens 16 --layers 1 --cores 4 --strategy critical \
      --firmware-mode compact -o timeline_out
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from llmc.models import build_graph, PRESETS
from llmc.schedule import Scheduler, CostModel
from llmc import backend
from llmc.ir import Kind
from llmc.memory import PC_SIZE
from llmc.topology import default_versal_hbm_topology, topology_from_qos_xlsx

try:
    from llmc.emit_compact import build_compact_program
except Exception:  # older package
    build_compact_program = None


INSTR_BASE = 0x4000000000


def mb(x: float) -> float:
    return x / 1024.0 / 1024.0


def gb(x: float) -> float:
    return x / 1024.0 / 1024.0 / 1024.0


def align_up(x: int, a: int = 64) -> int:
    return (int(x) + a - 1) // a * a


def file_tree_size(path: str | None) -> Optional[Dict[str, object]]:
    if not path or not os.path.exists(path):
        return None
    total = 0
    n_files = 0
    if os.path.isfile(path):
        total = os.path.getsize(path)
        n_files = 1
    else:
        for root, _, files in os.walk(path):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    total += os.path.getsize(fp)
                    n_files += 1
                except OSError:
                    pass
    return {
        "path": path,
        "files": n_files,
        "bytes": int(total),
        "mib": round(mb(total), 6),
        "gib": round(gb(total), 6),
    }


def dma_manifest_summary(path: str | None) -> Optional[Dict[str, object]]:
    if not path or not os.path.exists(path):
        return None
    by_category_status = defaultdict(lambda: {"count": 0, "hbm_size_bytes": 0, "dma_size_bytes": 0})
    total_hbm = 0
    total_dma = 0
    initial_hbm = 0
    initial_dma = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            category = row.get("category", "")
            status = row.get("status", "")
            hbm_size = int(row.get("size") or 0)
            dma_size = int(row.get("dma_size") or 0)
            key = f"{category}:{status}"
            rec = by_category_status[key]
            rec["count"] += 1
            rec["hbm_size_bytes"] += hbm_size
            rec["dma_size_bytes"] += dma_size
            total_hbm += hbm_size
            total_dma += dma_size
            if status == "OK" and category in {"load", "input"}:
                initial_hbm += hbm_size
                initial_dma += dma_size

    return {
        "path": path,
        "total_hbm_regions_bytes": int(total_hbm),
        "total_hbm_regions_mib": round(mb(total_hbm), 6),
        "total_dma_file_bytes": int(total_dma),
        "total_dma_file_mib": round(mb(total_dma), 6),
        "initial_load_hbm_bytes": int(initial_hbm),
        "initial_load_hbm_mib": round(mb(initial_hbm), 6),
        "initial_load_dma_file_bytes": int(initial_dma),
        "initial_load_dma_file_mib": round(mb(initial_dma), 6),
        "by_category_status": dict(by_category_status),
    }


def flat_order(sched) -> List[int]:
    return [op_i for wave in sched.waves for op_i, _ in wave]


def make_topology(cores: int, *, shared_dma: bool = False, qos_xlsx: str | None = None):
    if qos_xlsx:
        return topology_from_qos_xlsx(qos_xlsx, cores, independent_dma=not shared_dma)
    return default_versal_hbm_topology(cores, independent_dma=not shared_dma)


def clear_addresses(g):
    for t in g.tensors.values():
        t.addr = None
        t.meta.pop("pc", None)
        t.meta.pop("alias_resolved", None)
        t.meta.pop("alias_range_warning", None)


def schedule_and_allocate(g, args):
    topology = make_topology(args.cores, shared_dma=args.shared_dma, qos_xlsx=args.qos_xlsx)

    # Match compiler.py behavior: first allocate, optionally reschedule with PC conflicts.
    sched = Scheduler(g, n_cores=args.cores, strategy=args.strategy,
                      avoid_pc_conflicts=False).schedule()
    planners = backend.allocate(g, sched, recycle=not args.no_recycle,
                                topology=topology, strict_topology=not args.no_strict_routes)

    if args.pc_aware_schedule and args.cores > 1:
        sched = Scheduler(g, n_cores=args.cores, strategy=args.strategy,
                          avoid_pc_conflicts=True).schedule()
        clear_addresses(g)
        planners = backend.allocate(g, sched, recycle=not args.no_recycle,
                                    topology=topology, strict_topology=not args.no_strict_routes)
    return sched, planners, topology


def program_image_bytes(g, sched, mode: str) -> Tuple[int, Dict[str, object]]:
    raw, _, _ = backend.emit_bin(g, sched)
    macro_bytes = len(raw)
    n_macro = len(raw) // 64
    if mode == "macro":
        return macro_bytes, {"mode": "macro", "macro_program_bytes": macro_bytes,
                             "macro_instructions": n_macro}
    if mode == "compact":
        if build_compact_program is None:
            return macro_bytes, {"mode": "compact-unavailable-fallback-macro",
                                 "macro_program_bytes": macro_bytes}
        cp = build_compact_program(g, sched)
        compact_bytes = len(cp.raw) if hasattr(cp, "raw") else len(cp.words) * 4
        stats = cp.stats() if hasattr(cp, "stats") else {}
        return compact_bytes, {
            "mode": "compact",
            "macro_program_bytes": macro_bytes,
            "compact_program_bytes": compact_bytes,
            "program_saved_pct": stats.get("program_saved_pct", round((1 - compact_bytes / macro_bytes) * 100, 2) if macro_bytes else 0.0),
            "records": getattr(cp, "n_instr", n_macro),
            "setreg_full_words": stats.get("setreg_full", n_macro * 16),
            "setreg_compact_words": stats.get("setreg_compact", 0),
            "avg_words_per_instr": stats.get("avg_words_per_instr", round((compact_bytes / 4) / max(1, getattr(cp, "n_instr", n_macro)), 3)),
        }
    raise ValueError("--firmware-mode must be compact or macro")


def compute_lifetimes(g, sched) -> Dict[str, Tuple[int, int]]:
    """Return op-position lifetime [birth_pos, last_use_pos] for tensors.

    Persistent initial tensors have birth_pos=-1. Outputs with no use have
    last_use_pos=end_pos and remain final-resident.
    """
    order = flat_order(sched)
    pos = {op_i: k for k, op_i in enumerate(order)}
    n = len(order)

    birth: Dict[str, int] = {}
    last: Dict[str, int] = {}

    # Initial resident kinds.
    for name, t in g.tensors.items():
        if t.kind in (Kind.INPUT, Kind.WEIGHT, Kind.SCALE, Kind.KVCACHE):
            birth[name] = -1

    for k, op_i in enumerate(order):
        op = g.ops[op_i]
        for t in op.outputs:
            birth.setdefault(t, k)
            # Final outputs should survive to the end even if no later op consumes them.
            ten = g.tensors.get(t)
            if ten is not None and ten.kind == Kind.OUTPUT:
                last[t] = n
        for t in op.inputs:
            last[t] = max(last.get(t, -1), k)

    # Persistent initial data survives until end for occupancy accounting.
    for name, t in g.tensors.items():
        if t.kind in (Kind.INPUT, Kind.WEIGHT, Kind.SCALE, Kind.KVCACHE):
            last[name] = n
        elif name in birth and name not in last:
            # Non-output dead temp, still counted for at least its produce wave.
            last[name] = birth[name]
    return {k: (birth[k], last.get(k, birth[k])) for k in birth}


def tensor_pc(hbm, t) -> str:
    if t.addr is None:
        return "UNPLACED"
    return hbm.pc_of(t.addr)


def active_after_wave(life: Dict[str, Tuple[int, int]], wave_end_pos: int) -> List[str]:
    """Tensors valid after executing ops up to wave_end_pos inclusive."""
    out = []
    for name, (b, e) in life.items():
        if b <= wave_end_pos and e > wave_end_pos:
            out.append(name)
    return out


def kind_name(t) -> str:
    return str(t.kind.value if hasattr(t.kind, "value") else t.kind)


def summarize_names(names: List[str], limit: int = 8) -> List[str]:
    if len(names) <= limit:
        return names
    return names[:limit] + [f"...(+{len(names)-limit})"]


def build_report(args) -> Dict[str, object]:
    g = build_graph(args.model, token_num=args.tokens, n_layers=args.layers,
                    mlp_only=args.mlp_only, decode=args.decode,
                    decode_tokens=args.decode_tokens)
    sched, planners, topology = schedule_and_allocate(g, args)
    hbm = planners["hbm"]
    cost = CostModel(g)

    logical_weight_bytes = sum(int(t.nbytes) for t in g.tensors.values() if t.kind == Kind.WEIGHT)
    logical_scale_bytes = sum(int(t.nbytes) for t in g.tensors.values() if t.kind == Kind.SCALE)
    logical_input_bytes = sum(int(t.nbytes) for t in g.tensors.values() if t.kind == Kind.INPUT)
    logical_kvcache_bytes = sum(int(t.nbytes) for t in g.tensors.values() if t.kind == Kind.KVCACHE)
    logical_output_bytes = sum(int(t.nbytes) for t in g.tensors.values() if t.kind == Kind.OUTPUT)

    prog_bytes, prog_meta = program_image_bytes(g, sched, args.firmware_mode)
    instr_pc = hbm.pc_of(INSTR_BASE)

    life = compute_lifetimes(g, sched)
    order = flat_order(sched)
    pos = {op_i: k for k, op_i in enumerate(order)}

    # Per-tensor lifetime table.
    tensor_rows = []
    for name, (b, e) in life.items():
        t = g.tensors.get(name)
        if t is None or t.addr is None:
            continue
        tensor_rows.append({
            "name": name,
            "kind": kind_name(t),
            "bytes": int(t.nbytes),
            "pc": tensor_pc(hbm, t),
            "addr": int(t.addr),
            "birth_pos": int(b),
            "last_use_pos": int(e),
            "initial": bool(b == -1),
            "final_resident": bool(e >= len(order)),
        })

    # Initial resident occupancy includes program image and initial tensors.
    def occ_for_names(names: List[str]) -> Tuple[Counter, Dict[str, Counter]]:
        per_pc = Counter()
        by_kind = defaultdict(Counter)
        for n in names:
            t = g.tensors.get(n)
            if t is None or t.addr is None:
                continue
            pc = tensor_pc(hbm, t)
            sz = int(t.nbytes)
            per_pc[pc] += sz
            by_kind[pc][kind_name(t)] += sz
        return per_pc, by_kind

    initial_names = [n for n, (b, _) in life.items() if b == -1]
    final_names = [n for n, (_, e) in life.items() if e >= len(order)]
    init_pc, init_by_kind = occ_for_names(initial_names)
    init_pc[instr_pc] += prog_bytes
    init_by_kind[instr_pc]["program_image_" + args.firmware_mode] += prog_bytes

    # Wave timeline.
    wave_rows = []
    pc_traffic_rows = []
    pc_occ_rows = []

    current_names = set(initial_names)
    peak_total = 0
    peak_wave = -1
    peak_per_pc = Counter()
    peak_by_kind = {}

    cumulative_read = Counter()
    cumulative_write = Counter()
    cumulative_program_read_bytes = 0

    # If we model compact program reads dynamically, spread stream reads across waves.
    # This is a control-stream estimate; tensor data traffic is computed separately.
    program_bytes_per_wave = []
    if args.count_program_stream_reads:
        # Approximate each macro/compact record consumed at its wave. Compact records
        # are variable length; without full record->wave accounting here, distribute
        # by number of ops per wave. This preserves total bytes and supports occupancy.
        total_ops = max(1, sum(len(w) for w in sched.waves))
        for w in sched.waves:
            program_bytes_per_wave.append(round(prog_bytes * len(w) / total_ops))
        # Fix rounding.
        diff = prog_bytes - sum(program_bytes_per_wave)
        if program_bytes_per_wave:
            program_bytes_per_wave[-1] += diff
    else:
        program_bytes_per_wave = [0 for _ in sched.waves]

    for wi, wave in enumerate(sched.waves):
        wave_start = min((pos[i] for i, _ in wave), default=0)
        wave_end = max((pos[i] for i, _ in wave), default=-1)

        read_b = Counter()
        write_b = Counter()
        read_n = Counter()
        write_n = Counter()
        produced = []
        consumed_last = []

        compute_cycles = max((cost.cost(g.ops[i]) for i, _ in wave), default=0)

        for op_i, core in wave:
            op = g.ops[op_i]
            for tn in op.inputs:
                t = g.tensors.get(tn)
                if t is None or t.addr is None:
                    continue
                pc = tensor_pc(hbm, t)
                read_b[pc] += int(t.nbytes)
                read_n[pc] += 1
                cumulative_read[pc] += int(t.nbytes)
                if tn in life and life[tn][1] == pos[op_i]:
                    consumed_last.append(tn)
            for tn in op.outputs:
                t = g.tensors.get(tn)
                if t is None or t.addr is None:
                    continue
                pc = tensor_pc(hbm, t)
                write_b[pc] += int(t.nbytes)
                write_n[pc] += 1
                cumulative_write[pc] += int(t.nbytes)
                produced.append(tn)

        # Control program stream reads from instruction/program PC.
        pbytes = int(program_bytes_per_wave[wi])
        if pbytes:
            read_b[instr_pc] += pbytes
            read_n[instr_pc] += 1
            cumulative_read[instr_pc] += pbytes
            cumulative_program_read_bytes += pbytes

        # Active occupancy after this wave.
        active = active_after_wave(life, wave_end)
        occ_pc, occ_by_kind = occ_for_names(active)
        occ_pc[instr_pc] += prog_bytes
        occ_by_kind[instr_pc]["program_image_" + args.firmware_mode] += prog_bytes

        total_occ = sum(occ_pc.values())
        if total_occ > peak_total:
            peak_total = total_occ
            peak_wave = wi
            peak_per_pc = Counter(occ_pc)
            peak_by_kind = {pc: dict(kc) for pc, kc in occ_by_kind.items()}

        # Traffic/hotspot summary.
        all_pcs_traffic = set(read_b) | set(write_b)
        top_read_pc, top_read_bytes = (None, 0)
        if read_b:
            top_read_pc, top_read_bytes = read_b.most_common(1)[0]
        top_write_pc, top_write_bytes = (None, 0)
        if write_b:
            top_write_pc, top_write_bytes = write_b.most_common(1)[0]

        # Write/free/change accounting.
        produced_bytes = sum(int(g.tensors[n].nbytes) for n in produced if n in g.tensors)
        freed = [n for n in consumed_last if n in g.tensors and kind_name(g.tensors[n]) == "activation"]
        freed_bytes = sum(int(g.tensors[n].nbytes) for n in freed if n in g.tensors)

        wave_rows.append({
            "wave": wi,
            "n_ops": len(wave),
            "cores_used": len({c for _, c in wave}),
            "compute_cycles_est": round(compute_cycles),
            "read_bytes": int(sum(read_b.values())),
            "write_bytes": int(sum(write_b.values())),
            "program_read_bytes": pbytes,
            "produced_tensors": len(produced),
            "produced_bytes": int(produced_bytes),
            "freed_activation_tensors": len(freed),
            "freed_activation_bytes": int(freed_bytes),
            "live_bytes_after": int(total_occ),
            "top_read_pc": top_read_pc or "",
            "top_read_mb": round(mb(top_read_bytes), 6),
            "top_write_pc": top_write_pc or "",
            "top_write_mb": round(mb(top_write_bytes), 6),
            "sample_produced": summarize_names(produced),
            "sample_freed": summarize_names(freed),
        })

        for pc in sorted(set(occ_pc) | all_pcs_traffic):
            pc_occ_rows.append({
                "wave": wi,
                "pc": pc,
                "live_bytes_after": int(occ_pc.get(pc, 0)),
                "live_mb_after": round(mb(occ_pc.get(pc, 0)), 6),
            })
            if pc in all_pcs_traffic:
                pc_traffic_rows.append({
                    "wave": wi,
                    "pc": pc,
                    "read_bytes": int(read_b.get(pc, 0)),
                    "write_bytes": int(write_b.get(pc, 0)),
                    "read_ops": int(read_n.get(pc, 0)),
                    "write_ops": int(write_n.get(pc, 0)),
                })

    final_pc, final_by_kind = occ_for_names(final_names)
    final_pc[instr_pc] += prog_bytes
    final_by_kind[instr_pc]["program_image_" + args.firmware_mode] += prog_bytes

    # Persistent static reservation.
    persistent_reserved = Counter()
    persistent_by_kind = defaultdict(Counter)
    allocator_by_kind = Counter()
    for rec in planners["hbm"].allocs:
        allocator_by_kind[str(rec.kind)] += int(rec.size)
        if not rec.live:
            continue
        persistent_reserved[rec.pc] += int(rec.size)
        persistent_by_kind[rec.pc][str(rec.kind)] += int(rec.size)
    persistent_reserved[instr_pc] += prog_bytes
    persistent_by_kind[instr_pc]["program_image_" + args.firmware_mode] += prog_bytes

    scratch_peak_pc = hbm.pc_of(planners["scratch"].base)
    scratch_peak_bytes = int(planners["scratch"].peak)

    # Add scratch peak to static high-water reserved accounting.
    static_high_water = Counter(persistent_reserved)
    static_high_water[scratch_peak_pc] += scratch_peak_bytes
    allocator_high_water = Counter({pc: int(ch.used()) for pc, ch in hbm.chan.items()})

    summary = {
        "model": args.model,
        "tokens": args.tokens,
        "layers": args.layers if args.layers is not None else PRESETS[args.model].n_layers,
        "cores": args.cores,
        "strategy": args.strategy,
        "n_ops": len(g.ops),
        "n_waves": len(sched.waves),
        "firmware_mode": args.firmware_mode,
        "program_image": {
            "base": INSTR_BASE,
            "pc": instr_pc,
            "bytes": prog_bytes,
            **prog_meta,
        },
        "logical_model_bytes": {
            "weights": int(logical_weight_bytes),
            "scales": int(logical_scale_bytes),
            "inputs": int(logical_input_bytes),
            "kvcache": int(logical_kvcache_bytes),
            "outputs": int(logical_output_bytes),
            "initial_tensors": int(logical_weight_bytes + logical_scale_bytes + logical_input_bytes + logical_kvcache_bytes),
            "weights_mib": round(mb(logical_weight_bytes), 6),
            "weights_plus_scales_mib": round(mb(logical_weight_bytes + logical_scale_bytes), 6),
            "initial_tensors_mib": round(mb(logical_weight_bytes + logical_scale_bytes + logical_input_bytes + logical_kvcache_bytes), 6),
        },
        "initial_valid_bytes": int(sum(init_pc.values())),
        "final_valid_bytes": int(sum(final_pc.values())),
        "peak_valid_bytes": int(peak_total),
        "peak_valid_mb": round(mb(peak_total), 6),
        "peak_wave": int(peak_wave),
        "scratch_peak_bytes": scratch_peak_bytes,
        "scratch_peak_mb": round(mb(scratch_peak_bytes), 6),
        "scratch_pc": scratch_peak_pc,
        "total_runtime_read_bytes": int(sum(cumulative_read.values())),
        "total_runtime_write_bytes": int(sum(cumulative_write.values())),
        "program_stream_read_bytes_counted": int(cumulative_program_read_bytes),
        "initial_per_pc": dict(init_pc),
        "final_per_pc": dict(final_pc),
        "peak_per_pc": dict(peak_per_pc),
        "allocator_high_water_per_pc": dict(allocator_high_water),
        "allocator_high_water_total_bytes": int(sum(allocator_high_water.values())),
        "allocator_high_water_total_mib": round(mb(sum(allocator_high_water.values())), 6),
        "allocator_allocated_by_kind": dict(allocator_by_kind),
        "allocator_allocated_by_kind_mib": {k: round(mb(v), 6) for k, v in allocator_by_kind.items()},
        "static_high_water_per_pc": dict(static_high_water),
        "static_high_water_total_bytes": int(sum(static_high_water.values())),
        "static_high_water_total_mib": round(mb(sum(static_high_water.values())), 6),
        "top_runtime_read_pcs": cumulative_read.most_common(8),
        "top_runtime_write_pcs": cumulative_write.most_common(8),
        "dma_manifest": dma_manifest_summary(args.dma_manifest),
        "host_artifacts": {
            "processed_weights_dir": file_tree_size(args.weights_dir),
            "raw_weights_dir": file_tree_size(args.raw_weights_dir),
        },
    }

    return {
        "summary": summary,
        "initial_by_kind_per_pc": {pc: dict(c) for pc, c in init_by_kind.items()},
        "final_by_kind_per_pc": {pc: dict(c) for pc, c in final_by_kind.items()},
        "peak_by_kind_per_pc": peak_by_kind,
        "persistent_reserved_by_kind_per_pc": {pc: dict(c) for pc, c in persistent_by_kind.items()},
        "wave_timeline": wave_rows,
        "pc_occupancy_timeline": pc_occ_rows,
        "pc_traffic_by_wave": pc_traffic_rows,
        "tensor_lifetimes": tensor_rows,
        "topology": topology.to_report() if hasattr(topology, "to_report") else {},
    }


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            for k, v in rr.items():
                if isinstance(v, (list, dict, tuple)):
                    rr[k] = json.dumps(v, ensure_ascii=False)
            w.writerow(rr)


def maybe_plot(outdir: str, rep: Dict[str, object]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths = []
    waves = [r["wave"] for r in rep["wave_timeline"]]
    live_mb = [r["live_bytes_after"] / 1024 / 1024 for r in rep["wave_timeline"]]
    read_mb = [r["read_bytes"] / 1024 / 1024 for r in rep["wave_timeline"]]
    write_mb = [r["write_bytes"] / 1024 / 1024 for r in rep["wave_timeline"]]

    plt.figure(figsize=(9, 5))
    plt.plot(waves, live_mb)
    plt.xlabel("Wave")
    plt.ylabel("Live valid HBM data after wave (MB)")
    plt.title("HBM Valid Data Occupancy over Execution")
    plt.grid(True)
    plt.tight_layout()
    p = os.path.join(outdir, "occupancy_over_time.png")
    plt.savefig(p, dpi=150)
    plt.close()
    paths.append(p)

    plt.figure(figsize=(9, 5))
    plt.plot(waves, read_mb, label="read MB/wave")
    plt.plot(waves, write_mb, label="write MB/wave")
    plt.xlabel("Wave")
    plt.ylabel("Traffic per wave (MB)")
    plt.title("HBM Traffic over Execution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p = os.path.join(outdir, "traffic_over_time.png")
    plt.savefig(p, dpi=150)
    plt.close()
    paths.append(p)

    peak_pc = rep["summary"]["peak_per_pc"]
    if peak_pc:
        pcs = list(peak_pc.keys())
        vals = [peak_pc[pc] / 1024 / 1024 for pc in pcs]
        plt.figure(figsize=(10, 5))
        plt.bar(pcs, vals)
        plt.xlabel("HBM PC")
        plt.ylabel("Peak valid occupancy (MB)")
        plt.title("Peak Valid HBM Occupancy by PC")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis="y")
        plt.tight_layout()
        p = os.path.join(outdir, "pc_peak_occupancy.png")
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)

    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PRESETS), default="qwen")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--layers", type=int, default=None,
                    help="number of layers to model; default is the full preset")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--strategy", choices=["single", "round_robin", "critical"], default="critical")
    ap.add_argument("--mlp-only", action="store_true")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--decode-tokens", type=int, default=1)
    ap.add_argument("--no-recycle", action="store_true")
    ap.add_argument("--pc-aware-schedule", action="store_true", default=True)
    ap.add_argument("--no-pc-aware-schedule", dest="pc_aware_schedule", action="store_false")
    ap.add_argument("--shared-dma", action="store_true")
    ap.add_argument("--qos-xlsx", default=None)
    ap.add_argument("--no-strict-routes", action="store_true")
    ap.add_argument("--firmware-mode", choices=["compact", "macro"], default="compact")
    ap.add_argument("--count-program-stream-reads", action="store_true",
                    help="include program image stream reads as control traffic distributed across waves")
    ap.add_argument("--dma-manifest", default="dma_weights_manifest.csv",
                    help="optional weights_map manifest used to report actual initial DMA load size")
    ap.add_argument("--weights-dir", default="weights_processed_from_bin_fixed",
                    help="optional processed binary weight directory for host artifact size reporting")
    ap.add_argument("--raw-weights-dir", default="weights_raw",
                    help="optional raw/source weight directory for host artifact size reporting")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("-o", "--outdir", default="hbm_timeline_out")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    rep = build_report(args)

    with open(os.path.join(args.outdir, "hbm_timeline_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    write_csv(os.path.join(args.outdir, "timeline_by_wave.csv"), rep["wave_timeline"])
    write_csv(os.path.join(args.outdir, "pc_occupancy_timeline.csv"), rep["pc_occupancy_timeline"])
    write_csv(os.path.join(args.outdir, "pc_traffic_by_wave.csv"), rep["pc_traffic_by_wave"])
    write_csv(os.path.join(args.outdir, "tensor_lifetimes.csv"), rep["tensor_lifetimes"])

    plot_paths = maybe_plot(args.outdir, rep) if args.plots else []

    s = rep["summary"]
    print("HBM timeline analysis complete")
    print(f"  model={s['model']} tokens={s['tokens']} layers={s['layers']} cores={s['cores']} strategy={s['strategy']}")
    print(f"  waves={s['n_waves']} ops={s['n_ops']}")
    print(f"  program image: {s['program_image']['bytes']:,} B on {s['program_image']['pc']} ({s['firmware_mode']})")
    lm = s["logical_model_bytes"]
    print(f"  IR logical weights       : {lm['weights_mib']:.3f} MiB")
    print(f"  IR weights + scales      : {lm['weights_plus_scales_mib']:.3f} MiB")
    print(f"  IR initial tensors       : {lm['initial_tensors_mib']:.3f} MiB")
    print(f"  initial valid (+program) : {mb(s['initial_valid_bytes']):.3f} MiB")
    print(f"  final valid  : {mb(s['final_valid_bytes']):.3f} MB")
    print(f"  peak valid   : {s['peak_valid_mb']:.3f} MB at wave {s['peak_wave']}")
    print(f"  allocator high-water     : {s['allocator_high_water_total_mib']:.3f} MiB")
    print(f"  static high-water+scratch: {s['static_high_water_total_mib']:.3f} MiB")
    print(f"  scratch peak : {s['scratch_peak_mb']:.3f} MB on {s['scratch_pc']}")
    if s.get("dma_manifest"):
        dm = s["dma_manifest"]
        print(f"  DMA manifest initial load: {dm['initial_load_hbm_mib']:.3f} MiB "
              f"({dm['initial_load_dma_file_bytes']:,} B)")
    ha = s.get("host_artifacts", {})
    if ha.get("processed_weights_dir"):
        d = ha["processed_weights_dir"]
        print(f"  processed weight dir     : {d['gib']:.3f} GiB ({d['files']} files)")
    if ha.get("raw_weights_dir"):
        d = ha["raw_weights_dir"]
        print(f"  raw/source weight dir    : {d['gib']:.3f} GiB ({d['files']} files)")
    print(f"  runtime reads: {mb(s['total_runtime_read_bytes']):.3f} MB")
    print(f"  runtime writes: {mb(s['total_runtime_write_bytes']):.3f} MB")
    print("  top read PCs :", s["top_runtime_read_pcs"][:4])
    print("  top write PCs:", s["top_runtime_write_pcs"][:4])
    print(f"  wrote -> {args.outdir}")
    for p in plot_paths:
        print(f"  plot -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
