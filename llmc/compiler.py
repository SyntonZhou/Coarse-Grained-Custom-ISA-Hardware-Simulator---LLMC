"""
compiler.py -- end-to-end driver + CLI.

Pipeline:
  build IR -> preliminary schedule -> topology-aware allocation
           -> optional PC-aware reschedule -> final allocation
           -> emit firmware.c + program.bin/.hex + run_dma.sh + report.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

from . import backend
from .models import build_graph, PRESETS
from .schedule import Scheduler, report_schedule
from .noc import NoCSimulator, NoCConfig
from .topology import default_versal_hbm_topology, topology_from_qos_xlsx
from .emit_firmware import emit_firmware_interp
from .emit_compact import write_compact_artifacts


def _clear_addresses(g):
    for t in g.tensors.values():
        t.addr = None
        t.meta.pop("pc", None)
        t.meta.pop("alias_resolved", None)
        t.meta.pop("alias_range_warning", None)


def _make_topology(cores: int, *, shared_dma: bool, qos_xlsx: str | None):
    if qos_xlsx:
        return topology_from_qos_xlsx(qos_xlsx, cores, independent_dma=not shared_dma)
    return default_versal_hbm_topology(cores, independent_dma=not shared_dma)


def _schedule_and_allocate(g, *, cores: int, strategy: str, affinity,
                           recycle: bool, topology, pc_aware_schedule: bool,
                           strict_routes: bool):
    # First pass: no PC info yet; this establishes op.core for route constraints.
    sched = Scheduler(g, n_cores=cores, strategy=strategy, affinity=affinity,
                      avoid_pc_conflicts=False).schedule()
    planners = backend.allocate(g, sched, recycle=recycle, topology=topology,
                                strict_topology=strict_routes)

    # Second pass: tensors now carry meta["pc"], so the scheduler can avoid
    # same-wave PC hotspots.  Re-allocate after rescheduling because scratch
    # liveness/reuse depends on final wave order.
    if pc_aware_schedule and cores > 1:
        sched = Scheduler(g, n_cores=cores, strategy=strategy, affinity=affinity,
                          avoid_pc_conflicts=True).schedule()
        _clear_addresses(g)
        planners = backend.allocate(g, sched, recycle=recycle, topology=topology,
                                    strict_topology=strict_routes)
    return sched, planners


def compile_model(model: str = "qwen", tokens: int = 16, cores: int = 1,
                  strategy: str = "critical", n_layers: Optional[int] = None,
                  mlp_only: bool = False, decode: bool = False,
                  decode_tokens: int = 1,
                  core_stride: int = backend.isa.DEFAULT_CORE_STRIDE,
                  affinity=None, outdir: str = "out",
                  pc_aware_schedule: bool = True,
                  shared_dma: bool = False,
                  strict_routes: bool = True,
                  qos_xlsx: str | None = None,
                  verbose: bool = True,
                  firmware_mode: str = "compact",
                  emit_unrolled: bool = False,
                  force_hbm_scale: bool = False) -> Dict[str, object]:
    os.makedirs(outdir, exist_ok=True)
    old_force_hbm_scale = backend.FORCE_HBM_SCALE
    backend.FORCE_HBM_SCALE = force_hbm_scale

    g = build_graph(model, token_num=tokens, n_layers=n_layers, mlp_only=mlp_only,
                    decode=decode, decode_tokens=decode_tokens)
    topology = _make_topology(cores, shared_dma=shared_dma, qos_xlsx=qos_xlsx)
    sched, planners = _schedule_and_allocate(
        g, cores=cores, strategy=strategy, affinity=affinity, recycle=True,
        topology=topology, pc_aware_schedule=pc_aware_schedule,
        strict_routes=strict_routes,
    )

    meta = {
        "model": model, "tokens": tokens, "cores": cores,
        "strategy": strategy, "mlp_only": mlp_only,
        "n_layers": n_layers if n_layers is not None else PRESETS[model].n_layers,
        "core_stride": core_stride, "decode": decode, "decode_tokens": decode_tokens,
        "pc_aware_schedule": pc_aware_schedule,
        "shared_dma": shared_dma,
        "strict_routes": strict_routes,
        "qos_xlsx": qos_xlsx,
        "firmware_mode": firmware_mode,
        "emit_unrolled": emit_unrolled,
        "force_hbm_scale": force_hbm_scale,
    }
    raw, hexl, listing = backend.emit_bin(g, sched)

    program_file = "program.bin"
    program_size = len(raw)
    codegen_stats = {"mode": firmware_mode, "macro_program_bytes": len(raw)}

    if firmware_mode == "compact":
        compact_stats = write_compact_artifacts(g, sched, outdir, model=model, mode="hbm",
                                                core_stride=core_stride)
        program_file = "program_compact.bin"
        program_size = int(compact_stats["compact_program_bytes"])
        codegen_stats.update(compact_stats)
    elif firmware_mode == "interp":
        interp_src, interp_stats = emit_firmware_interp(g, sched, model=model, mode="hbm",
                                                        dedup=True, core_stride=core_stride)
        with open(os.path.join(outdir, "firmware_interp.c"), "w") as f:
            f.write(interp_src)
        codegen_stats.update({"interpreter": interp_stats})
    elif firmware_mode == "unrolled":
        emit_unrolled = True
    else:
        raise ValueError("firmware_mode must be one of: compact, interp, unrolled")

    if emit_unrolled:
        c_src = backend.emit_c(g, sched, model=model, strategy=strategy, core_stride=core_stride)
        with open(os.path.join(outdir, "firmware_unrolled.c"), "w") as f:
            f.write(c_src)
        codegen_stats["unrolled_source_bytes"] = len(c_src.encode())

    dma = backend.emit_dma(g, prog_size=program_size, program_file=program_file)
    report = backend.emit_report(g, sched, planners, meta)
    report["codegen"] = codegen_stats
    noc_sim = NoCSimulator(g, sched, NoCConfig(strict_routes=strict_routes), topology=topology)
    noc_result = noc_sim.simulate()
    report["noc_analysis"] = noc_result

    with open(os.path.join(outdir, "program.bin"), "wb") as f:
        f.write(raw)
    with open(os.path.join(outdir, "program.hex"), "w") as f:
        f.write("\n".join(hexl) + "\n")
    with open(os.path.join(outdir, "program.list"), "w") as f:
        f.write("\n".join(listing) + "\n")
    with open(os.path.join(outdir, "run_dma.sh"), "w", newline="\n") as f:
        f.write(dma)
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(outdir, "hbm_alloc.txt"), "w") as f:
        f.write(planners["hbm"].table() + "\n")

    if verbose:
        print(f"  NoC overhead: {noc_result['noc_overhead_pct']}%")
        print(f"  Bottleneck: {noc_result['bottleneck']}")
        print(g.summary())
        print(report_schedule(sched))
        print(f"  scratch peak: {planners['scratch'].peak_mb():.2f} MB")
        print(f"  macro program: {len(raw)} bytes ({len(raw)//64} instructions)")
        if firmware_mode == "compact":
            print(f"  compact program: {program_size} bytes ({codegen_stats.get('setreg_saved_pct', 0)}% fewer MMIO writes)")
        print(f"  wrote -> {outdir}/  "
              "(firmware_*.c, program*.bin/.hex/.list, run_dma.sh, report.json, hbm_alloc.txt)")
    backend.FORCE_HBM_SCALE = old_force_hbm_scale
    return {"graph": g, "schedule": sched, "planners": planners, "report": report}


def analyze_model(model: str = "qwen", tokens: int = 16, cores: int = 1,
                  strategy: str = "critical", n_layers: Optional[int] = None,
                  mlp_only: bool = False,
                  decode: bool = False, decode_tokens: int = 1,
                  pc_aware_schedule: bool = True,
                  shared_dma: bool = False,
                  strict_routes: bool = True,
                  qos_xlsx: str | None = None,
                  verbose: bool = True,
                  force_hbm_scale: bool = False):
    """Build + schedule + allocate only (no C/bin)."""
    old_force_hbm_scale = backend.FORCE_HBM_SCALE
    backend.FORCE_HBM_SCALE = force_hbm_scale
    g = build_graph(model, token_num=tokens, n_layers=n_layers, mlp_only=mlp_only,
                    decode=decode, decode_tokens=decode_tokens)
    topology = _make_topology(cores, shared_dma=shared_dma, qos_xlsx=qos_xlsx)
    sched, planners = _schedule_and_allocate(
        g, cores=cores, strategy=strategy, affinity=None, recycle=True,
        topology=topology, pc_aware_schedule=pc_aware_schedule,
        strict_routes=strict_routes,
    )
    noc_sim = NoCSimulator(g, sched, NoCConfig(strict_routes=strict_routes), topology=topology)
    noc_result = noc_sim.simulate()
    if verbose:
        print(f"  NoC overhead: {noc_result['noc_overhead_pct']}%")
        print(f"  Bottleneck: {noc_result['bottleneck']}")
        print(g.summary())
        print(report_schedule(sched))
    backend.FORCE_HBM_SCALE = old_force_hbm_scale
    return g, sched


def scaling_study(model="qwen", tokens=16, n_layers=1,
                  pc_aware_schedule=True, shared_dma=False, strict_routes=True, qos_xlsx=None,
                  force_hbm_scale: bool = False):
    """Print the makespan/speedup curve across 1..8 cores for one layer."""
    print(f"\nMulti-core scaling -- {model}, {tokens} tokens, {n_layers} layer(s):")
    print(f"  {'cores':>5} {'waves':>6} {'makespan':>12} {'speedup':>8}")
    for c in (1, 2, 4, 8):
        _, s = analyze_model(model, tokens, c, n_layers=n_layers, verbose=False,
                             pc_aware_schedule=pc_aware_schedule,
                             shared_dma=shared_dma,
                             strict_routes=strict_routes, qos_xlsx=qos_xlsx,
                             force_hbm_scale=force_hbm_scale)
        print(f"  {c:>5} {len(s.waves):>6} {s.makespan():>12,.0f} {s.speedup():>7.2f}x")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="llmc -- edge-LLM accelerator compiler")
    ap.add_argument("--model", choices=list(PRESETS), default="qwen")
    ap.add_argument("--tokens", type=int, default=16, help="prefill token count")
    ap.add_argument("--cores", type=int, default=1, help="compute cores (1..8)")
    ap.add_argument("--strategy", choices=["single", "round_robin", "critical"],
                    default="critical")
    ap.add_argument("--layers", type=int, default=None, help="override layer count")
    ap.add_argument("--mlp-only", action="store_true",
                    help="emit a single norm->MLP->residual slice (fully lowered)")
    ap.add_argument("--analyze", action="store_true",
                    help="schedule + allocate the full graph (incl. attention), no codegen")
    ap.add_argument("--scaling", action="store_true", help="print 1..8 core scaling study")
    ap.add_argument("--decode", action="store_true",
                    help="generate decode phase (single-token autoregressive)")
    ap.add_argument("--decode-tokens", type=int, default=1,
                    help="number of decode tokens to generate (default 1)")
    ap.add_argument("--core-stride", type=lambda s: int(s, 0),
                    default=backend.isa.DEFAULT_CORE_STRIDE,
                    help="per-core MMIO register stride (hex ok), default 0x1000")
    ap.add_argument("--no-pc-aware-schedule", action="store_true",
                    help="disable second-pass scheduling that avoids same-PC hotspots")
    ap.add_argument("--shared-dma", action="store_true",
                    help="model all cores sharing one MM2S/S2MM DMA pair")
    ap.add_argument("--no-strict-routes", action="store_true",
                    help="allow analysis even if topology reachability is violated")
    ap.add_argument("--qos-xlsx", default=None,
                    help="optional Vivado NoC QoS XLSX table to derive NMU->PC topology")
    ap.add_argument("--firmware-mode", choices=["compact", "interp", "unrolled"], default="compact",
                    help="control firmware generation mode; compact is the production path")
    ap.add_argument("--emit-unrolled", action="store_true",
                    help="also emit the giant fully-unrolled C reference")
    ap.add_argument("--force-hbm-scale", action="store_true",
                    help="disable on-chip scale passing and force scale traffic through HBM")
    ap.add_argument("-o", "--out", default="out")
    args = ap.parse_args(argv)

    pc_aware = not args.no_pc_aware_schedule
    strict = not args.no_strict_routes
    if args.scaling:
        scaling_study(args.model, args.tokens, args.layers or 1,
                      pc_aware_schedule=pc_aware, shared_dma=args.shared_dma,
                      strict_routes=strict, qos_xlsx=args.qos_xlsx,
                      force_hbm_scale=args.force_hbm_scale)
        return 0
    if args.analyze:
        analyze_model(model=args.model, tokens=args.tokens, cores=args.cores,
                      mlp_only=args.mlp_only, strategy=args.strategy,
                      n_layers=args.layers, decode=args.decode,
                      decode_tokens=args.decode_tokens,
                      pc_aware_schedule=pc_aware, shared_dma=args.shared_dma,
                      strict_routes=strict, qos_xlsx=args.qos_xlsx,
                      force_hbm_scale=args.force_hbm_scale)
        return 0
    compile_model(model=args.model, tokens=args.tokens, cores=args.cores,
                  strategy=args.strategy, n_layers=args.layers,
                  mlp_only=args.mlp_only, decode=args.decode,
                  decode_tokens=args.decode_tokens, core_stride=args.core_stride,
                  outdir=args.out, pc_aware_schedule=pc_aware,
                  shared_dma=args.shared_dma, strict_routes=strict, qos_xlsx=args.qos_xlsx,
                  firmware_mode=args.firmware_mode, emit_unrolled=args.emit_unrolled,
                  force_hbm_scale=args.force_hbm_scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
