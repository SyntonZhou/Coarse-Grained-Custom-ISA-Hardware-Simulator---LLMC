"""
backend.py -- lowering dispatch, memory allocation, and emitters.

  lower_op()    : ir.Op -> isa.Instruction (raises NotLowered for Step-2 ops)
  allocate()    : assign HBM addresses to every tensor (persistent weights via
                  HBMAllocator; transient activations via a liveness-driven
                  ScratchPad so the footprint is recycled)
  emit_c()      : multi-core RISC-V bare-metal firmware (wavefront barriers)
  emit_dma()    : QDMA load/run/readback script
  emit_bin()    : the program image (.bin + annotated .hex listing)
  emit_report() : JSON manifest + allocation table + timestamp
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

from . import isa, lowering
from .ir import Graph, Op, OpType, Kind
from .memory import HBMAllocator, ScratchPad, DEFAULT_REGIONS, PC_SIZE
from .schedule import Schedule

class NotLowered(Exception):
    pass

# ===========================================================================
# Op -> Instruction
# ===========================================================================
def _addr(g: Graph, name: str) -> int:
    t = g.tensors[name]
    if t.addr is None:
        raise RuntimeError(f"tensor {name!r} has no address (run allocate() first)")
    return t.addr

FORCE_HBM_SCALE = False  # 实验开关

def lower_op(g, op):
    a = dict(op.attrs)  # 复制一份，避免改原始 graph
    if FORCE_HBM_SCALE:
        # 强制所有 scale 写回 HBM，禁用片上传递
        if a.get("put_scale") in ("scaleA", "scaleAR"):
            a["put_scale"] = "hbm"
        if a.get("get_scale") == "scaleAR":
            a["get_scale"] = "scaleA"
    op.attrs = a  # 写回（临时）
    core = op.core or 0
    rd, wr = list(op.inputs), list(op.outputs)

    if op.type in (OpType.VEC_MATMUL, OpType.MATMUL):
        x, w, s = op.inputs
        fn = lowering.vec_matmul if op.type == OpType.VEC_MATMUL else lowering.matmul
        return fn(_addr(g, x), _addr(g, w), _addr(g, wr[0]), _addr(g, s),
                  a["a_row"], a["b_row"], a["b_col"],
                  get_scale=a.get("get_scale", "scaleA"),
                  put_scale=a.get("put_scale", "none"),
                  next_type=a.get("next_type", 1), reads=rd, writes=wr, core=core)

    if op.type == OpType.SILU:
        return lowering.silu(_addr(g, rd[0]), _addr(g, wr[0]),
                             a["elem"], a["dim"],
                             put_scale=a.get("put_scale", "none"),
                             next_type=a.get("next_type", 1), reads=rd, writes=wr, core=core)

    if op.type == OpType.GELU:
        return lowering.gelu(_addr(g, rd[0]), _addr(g, wr[0]),
                             a["elem"], a["dim"],
                             put_scale=a.get("put_scale", "none"),
                             next_type=a.get("next_type", 1), reads=rd, writes=wr, core=core)

    if op.type == OpType.VU_MUL:
        return lowering.vu_mul(_addr(g, rd[0]), _addr(g, rd[1]), _addr(g, wr[0]),
                               a["elem1"], a["dim"], a["elem2"],
                               put_scale=a.get("put_scale", "none"),
                               next_type=a.get("next_type", 1), reads=rd, writes=wr, core=core)

    if op.type == OpType.VU_ADD:
        return lowering.vu_add(_addr(g, rd[0]), _addr(g, rd[1]), _addr(g, wr[0]),
                               a["elem1"], a["dim"], a["elem2"],
                               put_scale=a.get("put_scale", "none"),
                               next_type=a.get("next_type", 1),
                               addr_out_scale=_addr(g, a["addr_out_scale"]) if "addr_out_scale" in a else None,
                               reads=rd, writes=wr, core=core)

    if op.type == OpType.RESIDUAL:
        return lowering.residual(_addr(g, rd[0]), _addr(g, rd[1]), _addr(g, wr[0]),
                                 a["elem1"], a["dim"], a["elem2"],
                                 put_scale=a.get("put_scale", "none"),
                                 next_type=a.get("next_type", 1), reads=rd, writes=wr, core=core)

    if op.type == OpType.VU_MASK:
        return lowering.vu_mask(_addr(g, rd[0]), _addr(g, rd[1]), _addr(g, wr[0]),
                                a["elem1"], a["dim"], a["elem2"],
                                put_scale=a.get("put_scale", "none"),
                                next_type=a.get("next_type", 1), reads=rd, writes=wr, core=core)

    if op.type == OpType.RMSNORM:
        return lowering.rmsnorm(_addr(g, rd[0]), _addr(g, rd[1]), _addr(g, wr[0]),
                                a["elem1"], a["dim"], a["elem2"],
                                e=a.get("e", 0), r=a.get("r", 0),
                                put_scale=a.get("put_scale", "scaleAR"),
                                next_type=a.get("next_type", 0), reads=rd, writes=wr, core=core)

    if op.type == OpType.SOFTMAX:
        return lowering.softmax(_addr(g, rd[0]), _addr(g, rd[1]), _addr(g, wr[0]),
                                a["elem1"], a["dim"], a["elem2"], a["vld_len"],
                                put_scale=a.get("put_scale", "scaleA"),
                                next_type=a.get("next_type", 0), reads=rd, writes=wr, core=core)

    if op.type == OpType.SWAP:
        return lowering.swap(_addr(g, rd[0]), _addr(g, wr[0]),
                             a["elem"], a["token_num"], a["dim"],
                             put_scale=a.get("put_scale", "none"),
                             next_type=a.get("next_type", 1), reads=rd, writes=wr, core=core)

    if op.type == OpType.REARRANGE:
        return lowering.rearrange(_addr(g, rd[0]), _addr(g, wr[0]),
                                  a["elem_in"], a["elem_out"],
                                  a["token_num"], a["dim"], a["heads"],
                                  put_scale=a.get("put_scale", "none"),
                                  next_type=a.get("next_type", 0), reads=rd, writes=wr, core=core)

    if op.type == OpType.TRANSPOSE:
        return lowering.transpose(_addr(g, rd[0]), _addr(g, wr[0]),
                                  _addr(g, a["addr_out_scale"]) if "addr_out_scale" in a else None,
                                  a["elem_in"], a["elem_out"],
                                  a["token_num"], a["dim"], a["heads"],
                                  put_scale=a.get("put_scale", "hbm"),
                                  next_type=a.get("next_type", 0), reads=rd, writes=wr, core=core)

    if op.type == OpType.CONCAT:
        return lowering.concat(_addr(g, rd[0]), _addr(g, wr[0]),
                               elem2=a["token_num"] * a["dim"],
                               out_elem=a["token_num"] * a["dim"],
                               token_num=a["token_num"], dim=a["dim"], heads=a["heads"],
                               reads=rd, writes=wr, core=core)

    raise NotLowered(f"{op.type.value} ({op.name}) -- no builder yet")

# ===========================================================================
# Allocation
# ===========================================================================
PERSISTENT = {Kind.WEIGHT, Kind.SCALE, Kind.INPUT, Kind.OUTPUT, Kind.KVCACHE}


def allocate(g: Graph, sched: Schedule,
             regions=None, scratch_capacity: int = PC_SIZE, recycle=True) -> Dict[str, object]:
    """Assign every tensor an HBM address.  Returns the planner objects."""
    regions = regions or DEFAULT_REGIONS
    names = [n for _, n in regions]
    hbm = HBMAllocator(regions)
    # persistent activations (model in/out) -> HBM1_PC0; scratch -> HBM1_PC1
    hbm.policy["activation"] = [names[2]]
    scratch_base = dict(regions)  # base->name; invert below
    base_of = {n: b for b, n in regions}
    scratch = ScratchPad(base_of[names[3]], capacity=scratch_capacity, recycle=recycle)

    kind_map = {Kind.WEIGHT: "weight", Kind.SCALE: "scale",
                Kind.KVCACHE: "kvcache", Kind.INPUT: "activation",
                Kind.OUTPUT: "activation"}

    # 1) persistent tensors
    for t in g.tensors.values():
        if t.kind in PERSISTENT:
            t.addr = hbm.alloc(t.nbytes, kind=kind_map[t.kind], tag=t.name)
            # 新增：将 PC 信息写回 tensor，供 NoC contention 分析使用
            t.meta["pc"] = hbm.pc_of(t.addr)

    # 2) transient activations via liveness over the executed order
    flat: List[int] = [i for wave in sched.waves for i, _ in wave]
    pos = {op_i: k for k, op_i in enumerate(flat)}
    last_use: Dict[str, int] = {}
    for op_i in flat:
        for t in g.ops[op_i].inputs:
            last_use[t] = max(last_use.get(t, -1), pos[op_i])

    free_at: Dict[int, List[str]] = {}
    for t, k in last_use.items():
        ten = g.tensors.get(t)
        if ten is not None and ten.kind == Kind.ACT:
            free_at.setdefault(k, []).append(t)

    for k, op_i in enumerate(flat):
        op = g.ops[op_i]

        for t in op.outputs:
            ten = g.tensors[t]
            if ten.kind == Kind.ACT and ten.addr is None and not ten.meta.get("alias_base"):
                ten.addr = scratch.alloc(ten.nbytes, tag=t)
        for t in free_at.get(k, []):
            ten = g.tensors[t]
            if ten.kind == Kind.ACT and ten.addr is not None:
                scratch.free(ten.addr)

    # any never-consumed activation (shouldn't happen) gets a slot
    for t in g.tensors.values():
        if t.kind == Kind.ACT and t.addr is None:
            t.addr = scratch.alloc(t.nbytes, tag=t.name)

    return {"hbm": hbm, "scratch": scratch}


# ===========================================================================
# C emission (multi-core, wavefront)
# ===========================================================================
C_HEADER = """\
#include <stdint.h>

/* ---- generated by llmc {ver} on {ts} ----
 * model={model}  cores={cores}  strategy={strat}
 * NOTE: per-core register stride CORE_STRIDE is an assumption (see README);
 *       confirm against the RTL address map for >1 core.
 *       Engineering default: 0x1000 per core.
 */
#define _P2V(addr) (addr)
#define SetReg(_x,_y)  do{{ (*(volatile uint32_t*)(_P2V(_x))) = (uint32_t)(_y); }}while(0)
#define ReadReg(_x,_y) do{{ (_y) = *(volatile uint32_t*)(_P2V(_x)); }}while(0)

#define CORE_STRIDE   0x{stride:X}u
#define CORE_BASE(c)  (0x{base:08X}u + (uint32_t)(c) * CORE_STRIDE)
#define COREREG(c,r)  (CORE_BASE(c) + (uint32_t)(r))
"""

C_REG_IDX = "\n".join(
    f"#define R_{n:<14} 0x{isa.REG_OFFSET[n]-isa.MMIO_BASE:02X}" for n in isa.REG_NAMES
)

C_WAIT = """
static void wait_done(uint32_t core) {
    uint32_t v = 0;
    while (1) {
        ReadReg(COREREG(core, R_PARA_CONFIG), v);
        if ((v & 0x00000007u) == 0x00000004u) break;
    }
}
"""


def emit_c(g: Graph, sched: Schedule, *, model="model", strategy="critical",
           core_stride: int = 0x40) -> str:
    lines: List[str] = []
    lines.append(C_HEADER.format(ver=__import__("llmc").__version__,
                                 ts=time.strftime("%Y-%m-%d %H:%M:%S"),
                                 model=model, cores=sched.n_cores, strat=strategy,
                                 stride=core_stride, base=isa.MMIO_BASE))
    lines.append(C_REG_IDX)
    lines.append(C_WAIT)
    lines.append("int main(void) {")

    for wi, wave in enumerate(sched.waves):
        tag = ", ".join(g.ops[i].name for i, _ in wave)
        lines.append(f"\n    /* ===== wave {wi}  ({len(wave)} core(s)): {tag} ===== */")
        used = []
        for op_i, core in wave:
            inst = lower_op(g, g.ops[op_i])
            lines.append(f"    /* core{core}: {g.ops[op_i].name} -- {inst.comment} */")
            for rn, val in zip(isa.REG_NAMES, inst.words):
                addr = isa.REG_OFFSET[rn] + core * core_stride
                lines.append(f"    SetReg(0x{addr:08X}, 0x{val:08X});  /* {rn} */")
            used.append(core)
        for core in used:
            lines.append(f"    wait_done({core});")
    lines.append("\n    return 0;")
    lines.append("}")
    return "\n".join(lines)

# ===========================================================================
# Instruction binary + listing
# ===========================================================================
def emit_bin(g: Graph, sched: Schedule):
    """Return (raw_bytes, hex_lines, listing_lines) for the program image."""
    raw = bytearray()
    hexl: List[str] = []
    listing: List[str] = []
    idx = 0
    for wi, wave in enumerate(sched.waves):
        for op_i, core in wave:
            inst = lower_op(g, g.ops[op_i])
            raw += inst.to_bin_le()
            listing.append(f"; instr {idx:04d}  wave {wi}  core{core}  "
                           f"{g.ops[op_i].name}  ({inst.comment})")
            for w in inst.words:
                hexl.append(f"{w:08X}")
                listing.append(f"  {w:08X}")
            idx += 1
    return bytes(raw), hexl, listing


# ===========================================================================
# DMA script
# ===========================================================================
DMA_PREAMBLE = """\
# QDMA bring-up (edit the PCIe BDF if not 0000:01:00.0)
echo 0 > /sys/bus/pci/devices/0000:01:00.0/sriov_numvfs
echo 0 > /sys/bus/pci/devices/0000:01:00.0/qdma/qmax
echo 1 > /sys/bus/pci/devices/0000:01:00.0/remove
sleep 2
echo 1 > /sys/bus/pci/rescan
sleep 2
echo 8 > /sys/bus/pci/devices/0000:01:00.0/qdma/qmax
echo 3 > /sys/bus/pci/devices/0000:01:00.0/sriov_numvfs
sleep 2
dma-ctl qdma01000 q add idx 0 mode mm dir bi
dma-ctl qdma01000 q start idx 0 dir bi
"""

DMA_CLEANUP = """\
# teardown
dma-ctl qdma01000 q stop idx 0 dir bi
dma-ctl qdma01000 q del idx 0 dir bi
echo 0 > /sys/bus/pci/devices/0000:01:00.0/sriov_numvfs
echo 0 > /sys/bus/pci/devices/0000:01:00.0/qdma/qmax
"""

INSTR_BASE = 0x4000000000


def emit_dma(g: Graph, prog_size: int, data_dir: str = "data") -> str:
    lines = [DMA_PREAMBLE, "DEV=/dev/qdma01000-MM-0", ""]
    lines.append("# 1) weights + scales -> HBM (one file per tensor)")
    for t in g.tensors.values():
        if t.kind in (Kind.WEIGHT, Kind.SCALE):
            lines.append(f"dma-to-device -d $DEV -a 0x{t.addr:010X} "
                         f"-f {data_dir}/{t.name}.bin -s {t.nbytes}")
    lines.append("")
    lines.append("# 2) input activation -> HBM")
    for t in g.tensors.values():
        if t.kind == Kind.INPUT:
            lines.append(f"dma-to-device -d $DEV -a 0x{t.addr:010X} "
                         f"-f {data_dir}/{t.name}.bin -s {t.nbytes}")
    lines.append("")
    lines.append("# 3) program image -> instruction region")
    lines.append(f"dma-to-device -d $DEV -a 0x{INSTR_BASE:010X} "
                 f"-f program.bin -s {prog_size}")
    lines.append("")
    lines.append("# 4) (run firmware on the control core here)")
    lines.append("")
    lines.append("# 5) read back outputs")
    for t in g.tensors.values():
        if t.kind == Kind.OUTPUT:
            lines.append(f"dma-from-device -d $DEV -a 0x{t.addr:010X} "
                         f"-f {data_dir}/{t.name}.out.bin -s {t.nbytes}")
    lines.append("")
    lines.append(DMA_CLEANUP)
    return "\n".join(lines)


# ===========================================================================
# Report
# ===========================================================================
def emit_report(g: Graph, sched: Schedule, planners: Dict[str, object],
                meta: Dict[str, object]) -> Dict[str, object]:
    hbm: HBMAllocator = planners["hbm"]
    scratch: ScratchPad = planners["scratch"]
    rl = sched.roofline()
    # -----------------------------------------------------------------
    # 量化感知 scale 传递统计
    # -----------------------------------------------------------------
    scale_put_ops = 0
    scale_get_ops = 0
    scale_onchip_bytes = 0
    for op in g.ops:
        put = op.attrs.get("put_scale")
        get = op.attrs.get("get_scale")
        if put in ("scaleA", "scaleAR"):
            scale_put_ops += 1
            # 估算该 op 留在片上的 scale 大小
            # RMSNorm/Softmax: per-row FP16; VU 系列: per-row FP16
            elem1 = op.attrs.get("elem1", 0)
            dim   = op.attrs.get("dim", 1)
            if elem1 > 0 and dim > 0:
                rows = elem1 // dim
                scale_onchip_bytes += rows * 2   # FP16
        if get in ("scaleA", "scaleAR"):
            scale_get_ops += 1
    # 在 emit_report() 的 return 之前加入
    per_token_peak = scratch.peak_mb()
    if meta.get("decode") and meta.get("decode_tokens", 1) > 1:
        # 估算：4-token 的峰值 ≈ 1-token 峰值（因为串行复用）
        # 或者通过简单启发式：取前 1/4 ops 的 scratch 峰值
        per_token_peak = scratch.peak_mb() / meta.get("decode_tokens", 1) * 1.2  # 保守估计
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tool": "llmc " + __import__("llmc").__version__,
        "meta": meta,
        "graph": {"name": g.name, "ops": len(g.ops), "tensors": len(g.tensors)},
        "schedule": {
            "cores": sched.n_cores,
            "waves": len(sched.waves),
            "makespan_cyc": round(sched.makespan()),
            "serial_cyc": round(sched.serial_cost()),
            "speedup": round(sched.speedup(), 3),
            "utilization": [round(u, 3) for u in sched.utilization()],
            "wave_widths": sched.parallelism_histogram(),
        },
        "roofline": rl,
        "memory": {
            "scratch_peak_mb": round(scratch.peak_mb(), 3),
            "hbm_usage_mb": {k: round(v, 3) for k, v in hbm.usage().items() if v > 0},
        },
        "scale_passing": {
            "put_scale_onchip_ops": scale_put_ops,
            "get_scale_onchip_ops": scale_get_ops,
            "onchip_scale_bytes": scale_onchip_bytes,
            "onchip_scale_kb": round(scale_onchip_bytes / 1024, 2),
            "saved_hbm_bytes_if_disabled": scale_onchip_bytes * 2,   # 避免写回+重新读取
            "saved_hbm_kb": round(scale_onchip_bytes * 2 / 1024, 2),
        },
        "per_token_peak_mb": per_token_peak
    }