#!/usr/bin/env python3
"""
hbm_analyzer.py -- static + schedule-driven analysis of HBM usage for an llmc
build. Answers, for a chosen model/cores/strategy:

  1. Occupancy per PC channel, broken down by data kind (instructions, weights,
     scales, input tokens, KV-cache, transient/scratch, outputs).
  2. Runtime read/write *traffic* per channel over the wave schedule -> the
     load pressure on each channel.
  3. Per-wave access-conflict simulation: how many cores hit the same PC in the
     same barrier, and which channels are hotspots (same-type read/read or
     write/write concurrency actually serializes on a channel).
  4. Address-collision verification: persistent tensors must not overlap, the
     instruction image region must be clear, and recycled scratch slots must
     have disjoint live ranges.
  5. Data-dependency-chain analysis: acyclicity, the longest chain (by cost and
     by op count), and how many times it bounces between PCs.

Usage:
  python hbm_analyzer.py --model qwen --tokens 64 --cores 4 -o hbm_report.json
  python hbm_analyzer.py --model qwen --tokens 64 --cores 8 --strategy critical

It builds the graph/schedule/allocation with your own package, so the addresses
are exactly what `compile_model` would emit.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from llmc.models import build_graph, PRESETS
from llmc.schedule import Scheduler, CostModel, critical_path
from llmc import backend
from llmc.memory import PC_SIZE

INSTR_BASE = 0x4000000000           # program.bin DMA target (HBM0_PC0)


def _mb(n: float) -> float:
    return n / 1024 / 1024


# --------------------------------------------------------------------------- #
def build(args):
    g = build_graph(args.model, token_num=args.tokens, n_layers=args.layers,
                    mlp_only=args.mlp_only)
    sched = Scheduler(g, n_cores=args.cores, strategy=args.strategy).schedule()
    planners = backend.allocate(g, sched, recycle=not args.no_recycle)
    return g, sched, planners


def flat_order(sched) -> List[int]:
    return [op_i for wave in sched.waves for op_i, _ in wave]


# --------------------------------------------------------------------------- #
# 1. occupancy per PC + by kind
# --------------------------------------------------------------------------- #
def occupancy(g, sched, planners) -> Dict:
    hbm = planners["hbm"]
    scratch = planners["scratch"]
    n_instr = sum(len(w) for w in sched.waves)
    prog_bytes = n_instr * 64

    per_pc: Dict[str, Dict] = defaultdict(lambda: {"bytes": 0, "by_kind": Counter()})
    # persistent tensors tracked by the HBM allocator
    for rec in hbm.allocs:
        if not rec.live:
            continue
        per_pc[rec.pc]["bytes"] += rec.size
        per_pc[rec.pc]["by_kind"][rec.kind] += rec.size
    # instruction image (placed by the DMA script, not the allocator)
    instr_pc = hbm.pc_of(INSTR_BASE)
    per_pc[instr_pc]["bytes"] += prog_bytes
    per_pc[instr_pc]["by_kind"]["instr"] += prog_bytes
    # transient scratch high-water mark
    scratch_pc = hbm.pc_of(scratch.base)
    if scratch.peak:
        per_pc[scratch_pc]["bytes"] += scratch.peak
        per_pc[scratch_pc]["by_kind"]["scratch"] += scratch.peak

    by_kind_total = Counter()
    for d in per_pc.values():
        for k, v in d["by_kind"].items():
            by_kind_total[k] += v
    return {"per_pc": per_pc, "by_kind_total": by_kind_total,
            "n_instr": n_instr, "prog_bytes": prog_bytes,
            "scratch_peak": scratch.peak}


# --------------------------------------------------------------------------- #
# 2. runtime read/write traffic per PC
# --------------------------------------------------------------------------- #
def traffic(g, sched, planners) -> Dict:
    hbm = planners["hbm"]
    rd_b: Counter = Counter(); wr_b: Counter = Counter()
    rd_n: Counter = Counter(); wr_n: Counter = Counter()
    unplaced = 0
    for op_i in flat_order(sched):
        op = g.ops[op_i]
        for t in op.inputs:
            ten = g.tensors.get(t)
            if ten is None or ten.addr is None:
                unplaced += 1; continue
            pc = hbm.pc_of(ten.addr)
            rd_b[pc] += ten.nbytes; rd_n[pc] += 1
        for t in op.outputs:
            ten = g.tensors.get(t)
            if ten is None or ten.addr is None:
                unplaced += 1; continue
            pc = hbm.pc_of(ten.addr)
            wr_b[pc] += ten.nbytes; wr_n[pc] += 1
    return {"rd_bytes": rd_b, "wr_bytes": wr_b, "rd_ops": rd_n, "wr_ops": wr_n,
            "unplaced": unplaced}


# --------------------------------------------------------------------------- #
# 3. per-wave access conflict simulation
# --------------------------------------------------------------------------- #
def conflicts(g, sched, planners) -> Dict:
    hbm = planners["hbm"]
    peak_hist: Counter = Counter()          # busiest-PC core-count per wave
    pc_hot_waves: Counter = Counter()       # #waves where a PC is shared by >=2 cores
    rr_ww_events = 0                        # same-type (r/r or w/w) sharing events
    worst: List[Tuple[int, str, int]] = []  # (wave, pc, n_cores)
    for wi, wave in enumerate(sched.waves):
        if len(wave) < 2:
            peak_hist[len(wave)] += 1
            continue
        pc_read: Dict[str, set] = defaultdict(set)
        pc_write: Dict[str, set] = defaultdict(set)
        for op_i, core in wave:
            op = g.ops[op_i]
            for t in op.inputs:
                ten = g.tensors.get(t)
                if ten and ten.addr is not None:
                    pc_read[hbm.pc_of(ten.addr)].add(core)
            for t in op.outputs:
                ten = g.tensors.get(t)
                if ten and ten.addr is not None:
                    pc_write[hbm.pc_of(ten.addr)].add(core)
        all_pcs = set(pc_read) | set(pc_write)
        busiest = 0
        for pc in all_pcs:
            ncores = len(pc_read.get(pc, set()) | pc_write.get(pc, set()))
            busiest = max(busiest, ncores)
            if ncores >= 2:
                pc_hot_waves[pc] += 1
            if len(pc_read.get(pc, set())) >= 2 or len(pc_write.get(pc, set())) >= 2:
                rr_ww_events += 1
                worst.append((wi, pc, max(len(pc_read.get(pc, set())),
                                          len(pc_write.get(pc, set())))))
        peak_hist[busiest] += 1
    worst.sort(key=lambda x: -x[2])
    return {"peak_hist": peak_hist, "pc_hot_waves": pc_hot_waves,
            "same_type_events": rr_ww_events, "worst": worst[:8]}


# --------------------------------------------------------------------------- #
# 4. address-collision verification
# --------------------------------------------------------------------------- #
def live_intervals(g, sched) -> Dict[str, Tuple[int, int]]:
    flat = flat_order(sched)
    pos = {op_i: k for k, op_i in enumerate(flat)}
    defp: Dict[str, int] = {}
    lastuse: Dict[str, int] = {}
    for k, op_i in enumerate(flat):
        op = g.ops[op_i]
        for t in op.outputs:
            defp.setdefault(t, k)
        for t in op.inputs:
            lastuse[t] = k
    iv = {}
    for t in set(defp) | set(lastuse):
        a = defp.get(t, lastuse.get(t, 0))
        b = lastuse.get(t, a)
        iv[t] = (min(a, b), max(a, b))
    return iv


def collisions(g, sched, planners) -> Dict:
    hbm = planners["hbm"]
    scratch = planners["scratch"]
    iv = live_intervals(g, sched)
    issues: List[str] = []

    # (a) persistent tensors: sweep per PC for overlapping address ranges
    by_pc: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    for rec in hbm.allocs:
        if rec.live:
            by_pc[rec.pc].append((rec.addr, rec.addr + rec.size, rec.tag))
    persistent_overlaps = 0
    for pc, spans in by_pc.items():
        spans.sort()
        for i in range(1, len(spans)):
            if spans[i][0] < spans[i - 1][1]:
                persistent_overlaps += 1
                if len(issues) < 10:
                    issues.append(f"PERSISTENT overlap in {pc}: "
                                  f"{spans[i-1][2]} [{spans[i-1][0]:#x},{spans[i-1][1]:#x}) "
                                  f"vs {spans[i][2]} @{spans[i][0]:#x}")

    # (b) instruction image region must be clear of allocated tensors
    n_instr = sum(len(w) for w in sched.waves)
    prog_lo, prog_hi = INSTR_BASE, INSTR_BASE + n_instr * 64
    instr_overlaps = 0
    for rec in hbm.allocs:
        if rec.live and rec.addr < prog_hi and rec.addr + rec.size > prog_lo:
            instr_overlaps += 1
            if len(issues) < 10:
                issues.append(f"INSTR-IMAGE overlap: {rec.tag} @{rec.addr:#x} "
                              f"hits [{prog_lo:#x},{prog_hi:#x})")

    # (c) recycled scratch slots: tensors sharing an address must be lifetime-disjoint
    scratch_pc = hbm.pc_of(scratch.base)
    slots: Dict[int, List[str]] = defaultdict(list)
    for t in g.tensors.values():
        if t.addr is not None and hbm.pc_of(t.addr) == scratch_pc and t.name in iv:
            slots[t.addr].append(t.name)
    scratch_violations = 0
    for addr, names in slots.items():
        names.sort(key=lambda n: iv[n][0])
        for i in range(1, len(names)):
            prev, cur = names[i - 1], names[i]
            if iv[cur][0] <= iv[prev][1]:   # live ranges overlap on the same slot
                scratch_violations += 1
                if len(issues) < 12:
                    issues.append(f"SCRATCH reuse conflict @{addr:#x}: "
                                  f"{prev}{iv[prev]} overlaps {cur}{iv[cur]}")
    ok = (persistent_overlaps == 0 and instr_overlaps == 0 and scratch_violations == 0)
    return {"ok": ok, "persistent_overlaps": persistent_overlaps,
            "instr_overlaps": instr_overlaps, "scratch_violations": scratch_violations,
            "n_scratch_slots": len(slots), "issues": issues}


# --------------------------------------------------------------------------- #
# 5. data dependency chains
# --------------------------------------------------------------------------- #
def dependency_chains(g, sched, planners) -> Dict:
    hbm = planners["hbm"]
    cost = CostModel(g)
    deps = g.deps()
    n = len(g.ops)
    succ: List[List[int]] = [[] for _ in range(n)]
    for i, preds in enumerate(deps):
        for p in preds:
            succ[p].append(i)
    acyclic = True
    try:
        g.topo()
    except RuntimeError:
        acyclic = False
    cp = critical_path(g, cost)                       # longest remaining cost to a sink
    # reconstruct the longest-cost chain
    start = max(range(n), key=lambda i: cp[i]) if n else 0
    chain = []
    i = start
    seen = set()
    while True:
        chain.append(i); seen.add(i)
        nxt = [s for s in succ[i] if s not in seen]
        if not nxt:
            break
        i = max(nxt, key=lambda s: cp[s])
    # PC hops along the chain (by each op's first input tensor)
    pcs = []
    for oi in chain:
        addr = None
        for t in g.ops[oi].inputs:
            ten = g.tensors.get(t)
            if ten and ten.addr is not None:
                addr = ten.addr; break
        pcs.append(hbm.pc_of(addr) if addr is not None else "?")
    hops = sum(1 for a, b in zip(pcs, pcs[1:]) if a != b)
    n_edges = sum(len(s) for s in succ)
    chain_cost = sum(cost.cost(g.ops[oi]) for oi in chain)
    return {"acyclic": acyclic, "n_ops": n, "n_edges": n_edges,
            "chain_len": len(chain), "chain_cost": round(chain_cost),
            "pc_hops": hops,
            "chain_head": [g.ops[oi].name for oi in chain[:6]],
            "chain_tail": [g.ops[oi].name for oi in chain[-4:]]}


# --------------------------------------------------------------------------- #
def report(g, sched, planners, args) -> Dict:
    occ = occupancy(g, sched, planners)
    tr = traffic(g, sched, planners)
    cf = conflicts(g, sched, planners)
    col = collisions(g, sched, planners)
    dep = dependency_chains(g, sched, planners)
    hbm = planners["hbm"]
    pc_names = [n for _, n in hbm.regions]

    print(f"\n{'='*72}\nHBM ANALYSIS -- {args.model} T{args.tokens} "
          f"{args.cores} core(s) [{args.strategy}]   {len(sched.waves)} waves\n{'='*72}")

    print("\n[1] OCCUPANCY per PC channel (of 1024 MB each):")
    print(f"    {'PC':<11}{'used':>10}{'%':>7}   breakdown")
    for name in pc_names:
        d = occ["per_pc"].get(name)
        if not d or d["bytes"] == 0:
            continue
        bk = ", ".join(f"{k}:{_mb(v):.2f}MB" for k, v in d["by_kind"].most_common())
        print(f"    {name:<11}{_mb(d['bytes']):>8.2f}MB{100*d['bytes']/PC_SIZE:>6.1f}%   {bk}")
    print("    ---- totals by data kind ----")
    for k, v in occ["by_kind_total"].most_common():
        print(f"      {k:<12} {_mb(v):>9.2f} MB")
    print(f"      {'TOTAL':<12} {_mb(sum(occ['by_kind_total'].values())):>9.2f} MB"
          f"   (instr image {occ['prog_bytes']:,} B, scratch peak {_mb(occ['scratch_peak']):.2f} MB)")

    print("\n[2] RUNTIME TRAFFIC per PC (bytes moved across the schedule):")
    print(f"    {'PC':<11}{'read':>11}{'write':>11}{'rd ops':>9}{'wr ops':>9}")
    pcs_seen = set(tr["rd_bytes"]) | set(tr["wr_bytes"])
    for name in pc_names:
        if name not in pcs_seen:
            continue
        print(f"    {name:<11}{_mb(tr['rd_bytes'][name]):>9.1f}MB"
              f"{_mb(tr['wr_bytes'][name]):>9.1f}MB"
              f"{tr['rd_ops'][name]:>9,}{tr['wr_ops'][name]:>9,}")

    print("\n[3] ACCESS-CONFLICT simulation (per barrier/wave):")
    print("    busiest-PC concurrency histogram (cores hitting one PC in a wave):")
    for k in sorted(cf["peak_hist"]):
        print(f"      {k} core(s) on busiest PC : {cf['peak_hist'][k]:>5} waves")
    if cf["pc_hot_waves"]:
        print("    channels shared by >=2 cores in a wave (hotspots):")
        for pc, c in cf["pc_hot_waves"].most_common():
            print(f"      {pc:<11} hot in {c:>5} waves")
        print(f"    same-type (r/r or w/w) contention events: {cf['same_type_events']}")
    else:
        print("    no PC shared by >=2 cores in any wave (single-core or fully spread).")

    print("\n[4] ADDRESS-COLLISION verification:")
    print(f"    persistent overlaps   : {col['persistent_overlaps']}")
    print(f"    instr-image overlaps  : {col['instr_overlaps']}")
    print(f"    scratch reuse conflicts: {col['scratch_violations']}  "
          f"(over {col['n_scratch_slots']} recycled slots)")
    print(f"    => {'PASS -- no address collisions' if col['ok'] else 'FAIL'}")
    for s in col["issues"][:6]:
        print(f"       ! {s}")

    print("\n[5] DATA-DEPENDENCY chains:")
    print(f"    acyclic: {col and dep['acyclic']}   ops: {dep['n_ops']:,}   "
          f"RAW edges: {dep['n_edges']:,}")
    print(f"    longest chain: {dep['chain_len']} ops, cost ~{dep['chain_cost']:,} cyc, "
          f"{dep['pc_hops']} PC hops")
    print(f"      head: {' -> '.join(dep['chain_head'])} ...")
    print(f"      tail: ... {' -> '.join(dep['chain_tail'])}")

    return {"occupancy": {
                "per_pc": {k: {"bytes": v["bytes"], "by_kind": dict(v["by_kind"])}
                           for k, v in occ["per_pc"].items()},
                "by_kind_total": dict(occ["by_kind_total"]),
                "prog_bytes": occ["prog_bytes"], "scratch_peak": occ["scratch_peak"]},
            "traffic": {"rd_bytes": dict(tr["rd_bytes"]), "wr_bytes": dict(tr["wr_bytes"]),
                        "rd_ops": dict(tr["rd_ops"]), "wr_ops": dict(tr["wr_ops"])},
            "conflicts": {"peak_hist": dict(cf["peak_hist"]),
                          "pc_hot_waves": dict(cf["pc_hot_waves"]),
                          "same_type_events": cf["same_type_events"]},
            "collisions": col, "dependency": dep}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PRESETS), default="qwen")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--strategy", choices=["single", "round_robin", "critical"],
                    default="critical")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--mlp-only", action="store_true")
    ap.add_argument("--no-recycle", action="store_true",
                    help="disable scratch recycling (shows the cost of not reusing)")
    ap.add_argument("-o", "--out", default=None, help="dump JSON report here")
    args = ap.parse_args(argv)

    g, sched, planners = build(args)
    rep = report(g, sched, planners, args)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2, default=str)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())