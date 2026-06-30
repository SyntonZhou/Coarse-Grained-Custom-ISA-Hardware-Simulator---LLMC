#!/usr/bin/env python3
"""Roadmap sweep for the LLMCORE architecture study.

The sweep is intentionally lightweight: it schedules one layer accurately and
linearly extrapolates to the full model.  This is fast enough for architecture
roadmap work while preserving the same per-op cost model used by normal reports.
Use the full compiler/analyzer later for final confirmation of a small set of
candidate points.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from typing import Iterable

from . import backend
from .models import PRESETS, build_graph
from .schedule import Scheduler
from .topology import default_versal_hbm_topology


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def clear_addresses(g) -> None:
    for t in g.tensors.values():
        t.addr = None
        t.meta.pop("pc", None)
        t.meta.pop("alias_resolved", None)
        t.meta.pop("alias_range_warning", None)


def apply_width(scheduler: Scheduler, width_bits: int) -> None:
    scheduler.cost.hw = replace(
        scheduler.cost.hw,
        axis_bytes_per_beat=max(1, int(width_bits) // 8),
    )


def schedule_current_graph(g, *, cores: int, strategy: str, width_bits: int):
    if strategy == "lane_aware_critical":
        topology = default_versal_hbm_topology(cores, independent_dma=True)

        first = Scheduler(g, n_cores=cores, strategy="critical",
                          avoid_pc_conflicts=False)
        apply_width(first, width_bits)
        sched = first.schedule()
        backend.allocate(g, sched, recycle=True, topology=topology,
                         strict_topology=True)

        second = Scheduler(g, n_cores=cores, strategy="critical",
                           avoid_pc_conflicts=True)
        apply_width(second, width_bits)
        sched = second.schedule()
        clear_addresses(g)
        backend.allocate(g, sched, recycle=True, topology=topology,
                         strict_topology=True)
        return sched

    sch = Scheduler(g, n_cores=cores, strategy=strategy,
                    avoid_pc_conflicts=False)
    apply_width(sch, width_bits)
    return sch.schedule()


def workload_specs(prefill_tokens: Iterable[int],
                   decode_contexts: Iterable[int]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for t in prefill_tokens:
        specs.append({
            "workload": f"prefill_T{t}",
            "kind": "prefill",
            "tokens": int(t),
            "decode": False,
            "decode_tokens": 1,
            "output_tokens": int(t),
        })
    for ctx in decode_contexts:
        specs.append({
            "workload": f"decode_ctx{ctx}",
            "kind": "decode",
            "tokens": int(ctx),
            "decode": True,
            "decode_tokens": 1,
            "output_tokens": 1,
        })
    return specs


def bottleneck_breakdown(sched, g) -> tuple[str, float, dict[str, float]]:
    by_type: dict[str, float] = {}
    total = 0.0
    for op in g.ops:
        c = sched.cost.cost(op)
        total += c
        by_type[op.type.value] = by_type.get(op.type.value, 0.0) + c
    if not by_type or total <= 0:
        return "", 0.0, by_type
    name, cyc = max(by_type.items(), key=lambda kv: kv[1])
    return name, cyc / total, by_type


def utilization_and_idle(sched, g, cores: int) -> tuple[float, float]:
    makespan = sched.makespan()
    if makespan <= 0 or cores <= 0:
        return 0.0, 0.0
    busy = 0.0
    idle_capacity = 0.0
    for wave in sched.waves:
        costs = [sched.cost.cost(g.ops[i]) for i, _ in wave]
        wave_time = max(costs, default=0.0)
        busy += sum(costs)
        idle_capacity += wave_time * cores - sum(costs)
    return busy / (makespan * cores), idle_capacity


def run_sweep(args) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_layers = PRESETS[args.model].n_layers if args.layers is None else args.layers
    specs = workload_specs(parse_ints(args.prefill_tokens),
                           parse_ints(args.decode_contexts))
    cores_list = parse_ints(args.cores)
    width_list = parse_ints(args.dma_widths)
    strategies = parse_strings(args.strategies)

    for spec in specs:
        for cores in cores_list:
            for strategy in strategies:
                if strategy == "single" and cores != 1:
                    continue
                for width in width_list:
                    g = build_graph(
                        args.model,
                        token_num=int(spec["tokens"]),
                        n_layers=1,
                        decode=bool(spec["decode"]),
                        decode_tokens=int(spec["decode_tokens"]),
                    )
                    try:
                        sched = schedule_current_graph(
                            g, cores=cores, strategy=strategy,
                            width_bits=width)
                        one_layer_cycles = sched.makespan()
                        full_cycles = one_layer_cycles * n_layers
                        out_tokens = int(spec["output_tokens"])
                        toks_100 = out_tokens * 100e6 / full_cycles
                        toks_target_clk = out_tokens * args.clock_mhz * 1e6 / full_cycles
                        avg_util, idle_capacity = utilization_and_idle(sched, g, cores)
                        bottleneck, bottleneck_pct, _ = bottleneck_breakdown(sched, g)
                        status = "OK"
                        error = ""
                    except Exception as exc:  # keep the table rectangular
                        one_layer_cycles = full_cycles = 0.0
                        toks_100 = toks_target_clk = 0.0
                        avg_util = idle_capacity = 0.0
                        bottleneck = ""
                        bottleneck_pct = 0.0
                        status = "ERROR"
                        error = str(exc)

                    rows.append({
                        "model": args.model,
                        "workload": spec["workload"],
                        "kind": spec["kind"],
                        "context_or_tokens": spec["tokens"],
                        "output_tokens": spec["output_tokens"],
                        "layers_extrapolated": n_layers,
                        "cores": cores,
                        "strategy": strategy,
                        "dma_width_bits": width,
                        "cycles_1layer": round(one_layer_cycles),
                        "cycles_full_est": round(full_cycles),
                        "latency_ms_at_clock": round(full_cycles / (args.clock_mhz * 1e6) * 1000, 6) if full_cycles else 0,
                        "tok_s_100mhz": round(toks_100, 6),
                        f"tok_s_{int(args.clock_mhz)}mhz": round(toks_target_clk, 6),
                        "target_tok_s": args.target_tok_s,
                        "target_ratio": round(toks_target_clk / args.target_tok_s, 6) if args.target_tok_s else 0,
                        "meets_target": bool(toks_target_clk >= args.target_tok_s),
                        "avg_core_util": round(avg_util, 6),
                        "barrier_idle_capacity_cycles_1layer": round(idle_capacity),
                        "bottleneck_op_type": bottleneck,
                        "bottleneck_op_pct_serial": round(bottleneck_pct * 100, 4),
                        "status": status,
                        "error": error,
                    })
    return rows


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["status"] != "OK":
            continue
        key = str(row["workload"])
        old = best.get(key)
        metric_key = next(k for k in row if k.startswith("tok_s_") and k.endswith("mhz") and k != "tok_s_100mhz")
        if old is None or float(row[metric_key]) > float(old[metric_key]):
            best[key] = row
    return [best[k] for k in sorted(best)]


def fastest_meeting_target(rows: list[dict[str, object]],
                           clock_mhz: float) -> dict[str, dict[str, object]]:
    metric = f"tok_s_{int(clock_mhz)}mhz"
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["status"] != "OK" or not row["meets_target"]:
            continue
        key = str(row["workload"])
        old = out.get(key)
        rank = (int(row["dma_width_bits"]), int(row["cores"]))
        if old is None or rank < (int(old["dma_width_bits"]), int(old["cores"])):
            out[key] = row
    return out


def write_markdown(path: str, rows: list[dict[str, object]], args) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metric = f"tok_s_{int(args.clock_mhz)}mhz"
    best = best_rows(rows)
    target = fastest_meeting_target(rows, args.clock_mhz)

    lines: list[str] = []
    lines.append("# Roadmap Sweep")
    lines.append("")
    lines.append("All numbers use one accurately scheduled layer and linear extrapolation to the full model.")
    lines.append("")
    lines.append(f"- Clock: {args.clock_mhz:g} MHz")
    lines.append(f"- Target: {args.target_tok_s:g} tok/s")
    lines.append(f"- Strategies: {args.strategies}")
    lines.append(f"- DMA widths: {args.dma_widths} bits")
    lines.append("")

    lines.append("## Best Observed Points")
    lines.append("")
    lines.append("| workload | best tok/s | cores | DMA width | strategy | target ratio | bottleneck |")
    lines.append("|---|---:|---:|---:|---|---:|---|")
    for row in best:
        lines.append(
            f"| {row['workload']} | {float(row[metric]):.3f} | "
            f"{row['cores']} | {row['dma_width_bits']}b | {row['strategy']} | "
            f"{float(row['target_ratio']):.3f} | {row['bottleneck_op_type']} "
            f"({float(row['bottleneck_op_pct_serial']):.1f}%) |"
        )
    lines.append("")

    lines.append("## Smallest Configurations Meeting Target")
    lines.append("")
    lines.append("| workload | tok/s | cores | DMA width | strategy |")
    lines.append("|---|---:|---:|---:|---|")
    for workload in sorted({str(r["workload"]) for r in rows}):
        row = target.get(workload)
        if row is None:
            lines.append(f"| {workload} | not met | - | - | - |")
        else:
            lines.append(
                f"| {workload} | {float(row[metric]):.3f} | "
                f"{row['cores']} | {row['dma_width_bits']}b | {row['strategy']} |"
            )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `lane_aware_critical` uses a first allocation pass to annotate HBM PCs, then reschedules to avoid same-wave PC conflicts.")
    lines.append("- Token/head/channel/request parallel modes are not included here yet because they require graph partitioning changes, not only a scheduler flag.")
    lines.append("- DMA width scaling assumes the same beat/cycle efficiency and no new NoC/HBM timing-closure penalty.")
    lines.append("")

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(PRESETS), default="qwen")
    ap.add_argument("--prefill-tokens", default="16,64,128")
    ap.add_argument("--decode-contexts", default="16,64,128,256")
    ap.add_argument("--cores", default="1,2,4")
    ap.add_argument("--dma-widths", default="256,512,1024,2048,4096")
    ap.add_argument("--strategies", default="round_robin,critical,lane_aware_critical")
    ap.add_argument("--layers", type=int, default=None,
                    help="default: full model layers from preset")
    ap.add_argument("--clock-mhz", type=float, default=225.0)
    ap.add_argument("--target-tok-s", type=float, default=120.0)
    ap.add_argument("-o", "--outdir", default="out/roadmap_sweep")
    args = ap.parse_args(argv)

    rows = run_sweep(args)
    csv_path = os.path.join(args.outdir, "roadmap_sweep.csv")
    md_path = os.path.join(args.outdir, "roadmap_sweep.md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, args)

    ok = sum(1 for r in rows if r["status"] == "OK")
    err = len(rows) - ok
    print(f"roadmap sweep complete: {ok} OK, {err} errors")
    print(f"  csv -> {csv_path}")
    print(f"  md  -> {md_path}")
    for row in best_rows(rows):
        metric = f"tok_s_{int(args.clock_mhz)}mhz"
        print(f"  best {row['workload']}: {float(row[metric]):.3f} tok/s "
              f"({row['cores']} cores, {row['dma_width_bits']}b, {row['strategy']})")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
