"""
compiler.py -- end-to-end driver + CLI (requirement #5: one command, no knobs
required).

    python -m llmc.compiler --model qwen --tokens 16 --cores 4 --mlp-only -o out/

Pipeline:  build IR  ->  schedule (deps + cores)  ->  allocate (HBM + scratch)
           ->  emit firmware.c + program.bin/.hex + run_dma.sh + report.json
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

def compile_model(model: str = "qwen", tokens: int = 16, cores: int = 1,
                  strategy: str = "critical", n_layers: Optional[int] = None,
                  mlp_only: bool = False, decode: bool = False, decode_tokens: int = 1, core_stride: int = 0x1000,
                  affinity=None, outdir: str = "out",
                  verbose: bool = True) -> Dict[str, object]:
    os.makedirs(outdir, exist_ok=True)

    g = build_graph(model, token_num=tokens, n_layers=n_layers, mlp_only=mlp_only,
                    decode=decode, decode_tokens=decode_tokens)
    sched = Scheduler(g, n_cores=cores, strategy=strategy, affinity=affinity).schedule()
    planners = backend.allocate(g, sched, recycle=True)   # 或 False

    meta = {"model": model, "tokens": tokens, "cores": cores,
            "strategy": strategy, "mlp_only": mlp_only,
            "n_layers": n_layers if n_layers is not None else PRESETS[model].n_layers,
            "core_stride": core_stride, "decode": decode, "decode_tokens": decode_tokens}
    # 确保没有调用 emit_c_decode
    c_src = backend.emit_c(g, sched, model=model, strategy=strategy, core_stride=core_stride)
    raw, hexl, listing = backend.emit_bin(g, sched)
    dma = backend.emit_dma(g, prog_size=len(raw))
    report = backend.emit_report(g, sched, planners, meta)
    # 新增 NoC 分析
    noc_sim = NoCSimulator(g, sched, NoCConfig())
    noc_result = noc_sim.simulate()
    report["noc_analysis"] = noc_result

    with open(os.path.join(outdir, "firmware.c"), "w") as f:
        f.write(c_src)
    with open(os.path.join(outdir, "program.bin"), "wb") as f:
        f.write(raw)
    with open(os.path.join(outdir, "program.hex"), "w") as f:
        f.write("\n".join(hexl) + "\n")
    with open(os.path.join(outdir, "program.list"), "w") as f:
        f.write("\n".join(listing) + "\n")
    with open(os.path.join(outdir, "run_dma.sh"), "w") as f:
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
        print(f"  program: {len(raw)} bytes ({len(raw)//64} instructions)")
        print(f"  wrote -> {outdir}/  "
              "(firmware.c, program.bin/.hex/.list, run_dma.sh, report.json, hbm_alloc.txt)")
    return {"graph": g, "schedule": sched, "planners": planners, "report": report}


def analyze_model(model: str = "qwen", tokens: int = 16, cores: int = 1,
                  strategy: str = "critical", n_layers: Optional[int] = None,
                  mlp_only: bool = False,
                  decode: bool = False, decode_tokens: int = 1,
                  verbose: bool = True):
    """Build + schedule + allocate only (no C/bin); works for the full graph
    including attention before its codegen lands."""
    g = build_graph(model, token_num=tokens, n_layers=n_layers, mlp_only=mlp_only,
                    decode=decode, decode_tokens=decode_tokens)
    sched = Scheduler(g, n_cores=cores, strategy=strategy).schedule()
    backend.allocate(g, sched)
    # 新增 NoC 分析
    noc_sim = NoCSimulator(g, sched, NoCConfig())
    noc_result = noc_sim.simulate()
    if verbose:
        print(f"  NoC overhead: {noc_result['noc_overhead_pct']}%")
        print(f"  Bottleneck: {noc_result['bottleneck']}")        
        print(g.summary())
        print(report_schedule(sched))

    return g, sched


def scaling_study(model="qwen", tokens=16, n_layers=1):
    """Print the makespan/speedup curve across 1..8 cores for one layer."""
    print(f"\nMulti-core scaling -- {model}, {tokens} tokens, {n_layers} layer(s):")
    print(f"  {'cores':>5} {'waves':>6} {'makespan':>12} {'speedup':>8}")
    for c in (1, 2, 4, 8):
        _, s = analyze_model(model, tokens, c, n_layers=n_layers, verbose=False)
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
    ap.add_argument("--core-stride", type=lambda s: int(s, 0), default=0x1000,
                    help="per-core MMIO register stride (hex ok), default 0x1000")
    ap.add_argument("-o", "--out", default="out")
    args = ap.parse_args(argv)

    if args.scaling:
        scaling_study(args.model, args.tokens, args.layers or 1)
        return 0
    if args.analyze:
        analyze_model(model=args.model, tokens=args.tokens, cores=args.cores, mlp_only=args.mlp_only,
                      strategy=args.strategy, n_layers=args.layers, decode=args.decode, decode_tokens=args.decode_tokens)
        return 0
    if args.decode:
        compile_model(model=args.model, tokens=args.tokens, cores=args.cores,
                      strategy=args.strategy, n_layers=args.layers, decode=True,
                      decode_tokens=args.decode_tokens, outdir=args.out)
    else: 
        compile_model(model=args.model, tokens=args.tokens, cores=args.cores,
                    strategy=args.strategy, n_layers=args.layers, mlp_only=args.mlp_only,
                    core_stride=args.core_stride, outdir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
