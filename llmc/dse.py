"""dse.py -- design-space exploration for the llmc hardware/software model.

Sweeps token count, layer count, core count, scheduling strategy, scale routing,
and DMA topology.  It produces CSV/JSON/Markdown reports with:
  * pure schedule cycles, topology-aware NoC cycles, speedup and efficiency;
  * roofline arithmetic intensity and compute/memory-bound label;
  * NoC bottleneck classification and HBM/DMA/MMIO breakdown;
  * compact-delta firmware image size and MMIO-write reduction;
  * scale-passing impact and scratch/HBM usage.

Example:
  python -m llmc.dse --model qwen --tokens 4,16,64 --layers 1,2 \
      --cores 1,2,4 --strategies critical,round_robin \
      --scale-modes onchip,hbm --dma-modes independent,shared \
      --qos-xlsx /path/to/qos_Table.xlsx -o dse_out
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from typing import Iterable, List

from . import backend
from .compiler import _make_topology, _schedule_and_allocate
from .emit_compact import build_compact_program
from .models import PRESETS, build_graph
from .noc import NoCConfig, NoCSimulator

_TOPO_CACHE = {}


def _parse_list(s: str, cast=str) -> List:
    if isinstance(s, list):
        return s
    return [cast(x.strip()) for x in str(s).split(',') if x.strip()]


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return statistics.mean(xs) if xs else 0.0


def estimate_onchip_scale_saved_bytes(g) -> int:
    """Mirror backend.emit_report's scale-passing estimate before mutation.

    It approximates the HBM bytes avoided by keeping per-row scales on chip:
    one HBM write of the produced scale plus one later HBM read.
    """
    scale_onchip_bytes = 0
    for op in g.ops:
        put = op.attrs.get("put_scale")
        if put in ("scaleA", "scaleAR"):
            elem1 = int(op.attrs.get("elem1", 0) or 0)
            dim = int(op.attrs.get("dim", 1) or 1)
            if elem1 > 0 and dim > 0:
                rows = elem1 // dim
                scale_onchip_bytes += rows * 2
    return scale_onchip_bytes * 2


def force_scale_through_hbm(g) -> None:
    """Disable on-chip scale forwarding in the IR-level attributes for analysis."""
    for op in g.ops:
        if op.attrs.get("put_scale") in ("scaleA", "scaleAR"):
            op.attrs["put_scale"] = "hbm"
        if op.attrs.get("get_scale") == "scaleAR":
            op.attrs["get_scale"] = "scaleA"


def run_one(*, model: str, tokens: int, layers: int, cores: int, strategy: str,
            scale_mode: str, dma_mode: str, mlp_only: bool, decode: bool,
            decode_tokens: int, pc_aware: bool, strict_routes: bool,
            qos_xlsx: str | None, max_resource_cores: int, clk_ghz: float) -> dict:
    force_hbm_scale = (scale_mode == "hbm")
    shared_dma = (dma_mode == "shared")
    old_force = backend.FORCE_HBM_SCALE
    backend.FORCE_HBM_SCALE = force_hbm_scale
    try:
        g = build_graph(model, token_num=tokens, n_layers=layers, mlp_only=mlp_only,
                        decode=decode, decode_tokens=decode_tokens)
        scale_saved_bytes_if_onchip = estimate_onchip_scale_saved_bytes(g)
        if force_hbm_scale:
            force_scale_through_hbm(g)
        topo_key = (cores, shared_dma, qos_xlsx)
        topo = _TOPO_CACHE.get(topo_key)
        if topo is None:
            topo = _make_topology(cores, shared_dma=shared_dma, qos_xlsx=qos_xlsx)
            _TOPO_CACHE[topo_key] = topo
        sched, planners = _schedule_and_allocate(
            g, cores=cores, strategy=strategy, affinity=None, recycle=True,
            topology=topo, pc_aware_schedule=pc_aware, strict_routes=strict_routes)
        report = backend.emit_report(g, sched, planners, {
            "model": model, "tokens": tokens, "cores": cores,
            "strategy": strategy, "mlp_only": mlp_only, "n_layers": layers,
            "decode": decode, "decode_tokens": decode_tokens,
            "force_hbm_scale": force_hbm_scale,
            "shared_dma": shared_dma,
        })
        noc = NoCSimulator(g, sched, NoCConfig(strict_routes=strict_routes), topology=topo).simulate()
        compact = build_compact_program(g, sched).stats()
        rl = report["roofline"]
        util = report["schedule"].get("utilization", [])
        scale_extra_cycles = (scale_saved_bytes_if_onchip * clk_ghz / 16.0) if force_hbm_scale else 0.0
        makespan = float(noc["makespan_noc_cycles"]) + scale_extra_cycles
        pure = float(noc["makespan_pure_cycles"])
        serial = float(report["schedule"].get("serial_cyc", 0.0))
        token_work = decode_tokens if decode else tokens
        tok_s = token_work * clk_ghz * 1e9 / makespan if makespan else 0.0
        return {
            "model": model, "tokens": tokens, "layers": layers,
            "cores": cores, "strategy": strategy, "scale_mode": scale_mode,
            "dma_mode": dma_mode, "mlp_only": mlp_only, "decode": decode,
            "decode_tokens": decode_tokens,
            "resource_viable": cores <= max_resource_cores,
            "ops": len(g.ops), "tensors": len(g.tensors),
            "waves": len(sched.waves),
            "makespan_pure_cycles": int(round(pure)),
            "makespan_noc_cycles": int(round(makespan)),
            "serial_cycles": int(round(serial)),
            "speedup_vs_serial": round(serial / makespan, 4) if makespan else 0.0,
            "parallel_efficiency": round((serial / makespan) / cores, 4) if makespan and cores else 0.0,
            "noc_overhead_pct": noc["noc_overhead_pct"],
            "bottleneck": noc["bottleneck"],
            "token_per_s_at_clk": round(tok_s, 3),
            "avg_core_util": round(_mean(util), 4),
            "min_core_util": round(min(util), 4) if util else 0.0,
            "max_core_util": round(max(util), 4) if util else 0.0,
            "roofline_ai": round(float(rl.get("arithmetic_intensity", 0.0)), 4),
            "roofline_ridge": round(float(rl.get("ridge_point", 0.0)), 4),
            "roofline_bound": rl.get("bound", "unknown"),
            "macs": int(rl.get("macs", 0)),
            "bytes": int(rl.get("bytes", 0)),
            "scratch_peak_mb": report["memory"].get("scratch_peak_mb", 0.0),
            "hbm_used_mb": round(sum(report["memory"].get("hbm_usage_mb", {}).values()), 4),
            "scale_put_onchip_ops": report["scale_passing"].get("put_scale_onchip_ops", 0),
            "scale_get_onchip_ops": report["scale_passing"].get("get_scale_onchip_ops", 0),
            "scale_saved_hbm_kb": round(scale_saved_bytes_if_onchip / 1024, 4) if scale_mode == "onchip" else 0.0,
            "scale_extra_hbm_kb": round(scale_saved_bytes_if_onchip / 1024, 4) if scale_mode == "hbm" else 0.0,
            "scale_extra_cycles_est": round(scale_extra_cycles, 3),
            "compact_program_bytes": compact["compact_program_bytes"],
            "macro_program_bytes": compact["full_program_bytes"],
            "program_saved_pct": compact["program_saved_pct"],
            "setreg_full": compact["setreg_full"],
            "setreg_compact": compact["setreg_compact"],
            "setreg_saved_pct": compact["setreg_saved_pct"],
            "unrolled_text_est_O0_bytes": compact["setreg_full"] * 20 + compact["n_instr"] * 8 + 256,
            "mmio_cycles": noc["breakdown"].get("mmio", 0),
            "sync_cycles": noc["breakdown"].get("sync", 0),
            "hbm_cycles": noc["breakdown"].get("hbm_noc_dma", noc["breakdown"].get("hbm_contention", 0)),
        }
    finally:
        backend.FORCE_HBM_SCALE = old_force


def write_reports(rows: List[dict], outdir: str, *, args) -> None:
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "dse_results.csv")
    json_path = os.path.join(outdir, "dse_results.json")
    md_path = os.path.join(outdir, "dse_summary.md")
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    viable = [r for r in rows if r.get("resource_viable")]
    best = sorted(viable, key=lambda r: (r["makespan_noc_cycles"], -r["setreg_saved_pct"]))[:10]
    by_core = {}
    for r in viable:
        key = (r["tokens"], r["layers"], r["scale_mode"], r["dma_mode"], r["strategy"])
        by_core.setdefault(key, []).append(r)

    lines = []
    lines.append("# llmc DSE Summary")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Rows: {len(rows)}; resource-viable rows: {len(viable)}; max_resource_cores={args.max_resource_cores}")
    lines.append("")
    lines.append("## Best resource-viable configurations by NoC-aware makespan")
    lines.append("")
    lines.append("| tokens | layers | cores | strategy | scale | DMA | cycles | speedup | eff. | bottleneck | compact KB | SetReg saved |")
    lines.append("|---:|---:|---:|---|---|---|---:|---:|---:|---|---:|---:|")
    for r in best:
        lines.append(f"| {r['tokens']} | {r['layers']} | {r['cores']} | {r['strategy']} | {r['scale_mode']} | {r['dma_mode']} | {r['makespan_noc_cycles']:,} | {r['speedup_vs_serial']:.2f} | {r['parallel_efficiency']:.2f} | {r['bottleneck']} | {r['compact_program_bytes']/1024:.1f} | {r['setreg_saved_pct']:.1f}% |")
    lines.append("")
    lines.append("## High-level observations generated from the sweep")
    lines.append("")
    if rows:
        avg_saved = _mean(r["setreg_saved_pct"] for r in rows)
        avg_prog_saved = _mean(r["program_saved_pct"] for r in rows)
        lines.append(f"- Compact-delta firmware reduces runtime MMIO writes by **{avg_saved:.1f}%** on average and HBM program image size by **{avg_prog_saved:.1f}%** on average versus the 16-word macro-instruction stream.")
        hbm_rows = [r for r in rows if r["scale_mode"] == "hbm"]
        on_rows = [r for r in rows if r["scale_mode"] == "onchip"]
        if hbm_rows and on_rows:
            lines.append("- The `scale_mode` sweep is present in the CSV. Compare paired rows with identical tokens/layers/cores/strategy/DMA to quantify the saved HBM traffic from on-chip scale forwarding.")
        shared = [r for r in rows if r["dma_mode"] == "shared"]
        indep = [r for r in rows if r["dma_mode"] == "independent"]
        if shared and indep:
            lines.append("- The DMA mode sweep is present. Shared-DMA rows expose the upper bound loss if all cores time-multiplex one DMA engine; independent-DMA rows match the compiler's intended parallel data-movement assumption.")
        if any(not r["resource_viable"] for r in rows):
            lines.append("- Rows with `resource_viable=false` are hypothetical. They are useful for studying scaling pressure, but should not be used as implementation recommendations when the device cannot route that many cores.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `dse_results.csv`: flat table for plotting and filtering.")
    lines.append("- `dse_results.json`: same rows in JSON.")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PRESETS), default="qwen")
    ap.add_argument("--tokens", default="4,16,64")
    ap.add_argument("--layers", default="1,2")
    ap.add_argument("--cores", default="1,2,4")
    ap.add_argument("--strategies", default="critical,round_robin")
    ap.add_argument("--scale-modes", default="onchip,hbm", choices=None)
    ap.add_argument("--dma-modes", default="independent,shared")
    ap.add_argument("--mlp-only", action="store_true")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--decode-tokens", type=int, default=1)
    ap.add_argument("--no-pc-aware-schedule", action="store_true")
    ap.add_argument("--no-strict-routes", action="store_true")
    ap.add_argument("--qos-xlsx", default=None)
    ap.add_argument("--max-resource-cores", type=int, default=4,
                    help="cores above this are marked hypothetical/resource-nonviable")
    ap.add_argument("--clk-ghz", type=float, default=0.3)
    ap.add_argument("-o", "--out", default="dse_out")
    args = ap.parse_args(argv)

    rows: List[dict] = []
    for tokens in _parse_list(args.tokens, int):
        for layers in _parse_list(args.layers, int):
            for cores in _parse_list(args.cores, int):
                for strategy in _parse_list(args.strategies, str):
                    for scale_mode in _parse_list(args.scale_modes, str):
                        for dma_mode in _parse_list(args.dma_modes, str):
                            rows.append(run_one(
                                model=args.model, tokens=tokens, layers=layers, cores=cores,
                                strategy=strategy, scale_mode=scale_mode, dma_mode=dma_mode,
                                mlp_only=args.mlp_only, decode=args.decode,
                                decode_tokens=args.decode_tokens,
                                pc_aware=not args.no_pc_aware_schedule,
                                strict_routes=not args.no_strict_routes,
                                qos_xlsx=args.qos_xlsx,
                                max_resource_cores=args.max_resource_cores,
                                clk_ghz=args.clk_ghz,
                            ))
                            print(f"done T{tokens} L{layers} C{cores} {strategy} scale={scale_mode} dma={dma_mode}")
    write_reports(rows, args.out, args=args)
    print(f"wrote {args.out}/dse_results.csv, dse_results.json, dse_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
