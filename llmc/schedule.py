"""
schedule.py -- task scheduling (requirements #1, #2) + roofline analysis (#3).

The hardware completes one configured command per core and signals via the
finish register.  We generalise that to N cores with a *wavefront* schedule: a
sequence of barriers, where each wave holds up to N mutually-independent ops
(one per core) whose dependencies all completed in earlier waves.  This maps
directly onto "configure k cores -> kick -> wait-all -> next wave" and is safe
by construction (ready ops share no edges).

Strategies
  single       : everything on core 0 (reproduces the serial reference).
  round_robin  : ready ops handed to cores cyclically.
  critical     : ready ops ordered by critical-path urgency (shortens makespan).

`affinity` optionally pins an OpType to a subset of cores (custom arrangement).
A CostModel attributes MAC counts and HBM byte traffic per op so the report can
say whether a layer is compute- or memory-wall bound on this board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Optional, Tuple

from .ir import Graph, Op, OpType

 

class _LegacyCostModel:
    """基于 pe_compute 模块实际吞吐率的成本模型."""
    # Effective memory path bandwidth used only for rough scheduling/roofline.
    # Do not use the QoS table's 100/200 MB/s as a hard cap here; the detailed
    # NoC model in noc.py applies per-NMU/per-PC limits.  16 GB/s is a
    # conservative single NoC/DMA path placeholder to be calibrated.
    HBM_GBPS = 16.0
    CLK_GHZ = 0.3
    MACS_PER_CYCLE = 2048 * 64
    # ---- 硬件吞吐率（来自 pe_compute 说明）----
    MU_RESULTS_PER_CYCLE = 64      # MU: 每周期 64 个输出结果（FP16/INT8）
    VEC_FP16_PER_CYCLE = 32      # 向量单元: 每周期 32 个 FP16
    VEC_INT8_PER_CYCLE = 64      # 向量单元: 每周期 64 个 INT8
    MU_FILL_LATENCY = 2048  # K 维度填充周期（可配置）
    MU_PIPELINE_DEPTH = 8   # 乘加树流水线深度

    def __init__(self, graph: Graph):
        self.g = graph

    def macs(self, op: Op) -> int:
        a = op.attrs
        if op.type in (OpType.VEC_MATMUL, OpType.MATMUL):
            return int(a["a_row"]) * int(a["b_row"]) * int(a["b_col"])
        return 0

    def bytes(self, op: Op) -> int:
        b = 0
        for t in op.inputs + op.outputs:
            ten = self.g.tensors.get(t)
            if ten is not None:
                b += ten.nbytes
        return b

    def matmul_latency(self, a_row, b_row, b_col, is_decode=False):
        out_elems = a_row * b_col
        steady = out_elems / self.MU_RESULTS_PER_CYCLE
        if is_decode or out_elems < 1024:  # 小矩阵判定
            startup = b_row + self.MU_PIPELINE_DEPTH  # K + D
            return max(steady, startup)
        return steady

    def cost(self, op: Op) -> float:
        """修正后的延迟模型：基于输出元素吞吐率，而非 MAC 总数."""
        # ---- MatMul: 由 MU 输出吞吐率决定 ----
        if op.type in (OpType.VEC_MATMUL, OpType.MATMUL):
            a = op.attrs
            comp_cycles = self.matmul_latency(
                int(a["a_row"]), int(a["b_row"]), int(a["b_col"]),
                is_decode=int(a["a_row"]) == 1,
            )
        else:
            # ---- 非 MatMul: 向量单元吞吐率 ----
            # 默认按 FP16 的 32/cycle 计算（保守估计）
            # 若后续确认某 op 纯 INT8（如 FP16_INT8、REARRANGE 输出），可升级为 64/cycle
            elems = sum(self.g.tensors[t].numel for t in op.outputs if t in self.g.tensors)
            if elems == 0:
                # 无输出的特殊操作（如 SWAP），按输入估算
                elems = sum(self.g.tensors[t].numel for t in op.inputs if t in self.g.tensors)
            comp_cycles = elems / self.VEC_FP16_PER_CYCLE

        # 内存墙估算（保留，但当前 HBM 带宽值可能需按 QoS 表校准）
        mem_seconds = self.bytes(op) / (self.HBM_GBPS * 1e9)
        mem_cycles = mem_seconds * self.CLK_GHZ * 1e9
        return max(comp_cycles, mem_cycles, 1.0)
 

def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


@dataclass(frozen=True)
class LLMCoreTiming:
    """Calibratable cycle constants for the measured LLMCORE path."""
    dispatch_config_cycles: float = 225.0
    state_response_cycles: float = 3.0
    first_mm2s_latency_cycles: float = 77.0
    mm2s_reissue_cycles: float = 32.0
    s2mm_issue_cycles: float = 32.0
    s2mm_intr_clear_cycles: float = 8.0
    mm2s_beats_per_cycle: float = 0.50
    s2mm_beats_per_cycle: float = 0.50
    axis_bytes_per_beat: int = 32
    mu_pipeline_latency: float = 16.0
    vu_pipeline_latency: float = 8.0
    silu_latency: float = 20.0
    quant_latency: float = 16.0
    softmax_pipeline_latency: float = 16.0
    rmsnorm_pipeline_latency: float = 16.0
    transform_latency: float = 16.0
    scale_flush_issue_cycles: float = 32.0
    done_clear_cycles: float = 3.0

    # Softmax external timeline, calibrated from the closed two-VCD trace:
    # config -> MM2S input1 -> gap -> MM2S input2 -> internal wait
    # -> M_AXIS/S2MM output -> READY.  The reference op has
    # elem=4032, dim=64, input_len1=input_len2=8064 B, output_len=4032 B.
    softmax_config_to_first_mm2s_cycles: float = 78.0
    softmax_input_segment_gap_cycles: float = 72.0
    softmax_mm2s_beats_per_cycle: float = 252.0 / 354.0
    softmax_s2mm_beats_per_cycle: float = 126.0 / 176.0
    softmax_output_tlast_to_ready_cycles: float = 36.0
    softmax_ref_elem: int = 4032
    softmax_ref_dim: int = 64
    softmax_ref_internal_visible_cycles: float = 3728.0


@dataclass
class _Traffic:
    input_bytes: int
    input_segments: int
    output_bytes: int
    output_segments: int = 1
    scale_flush_bytes: int = 0


class CostModel:
    """Hardware-aware first-order latency model for one LLMCORE op."""
    HBM_GBPS = 16.0
    CLK_GHZ = 0.3
    MACS_PER_CYCLE = 512
    VEC_FP16_PER_CYCLE = 32

    def __init__(self, graph: Graph):
        self.g = graph
        self.hw = LLMCoreTiming()

    def macs(self, op: Op) -> int:
        a = op.attrs
        if op.type in (OpType.VEC_MATMUL, OpType.MATMUL):
            return int(a["a_row"]) * int(a["b_row"]) * int(a["b_col"])
        return 0

    def bytes(self, op: Op) -> int:
        tr = self._traffic(op)
        return tr.input_bytes + tr.output_bytes + tr.scale_flush_bytes

    def cost(self, op: Op) -> float:
        if op.type == OpType.SOFTMAX:
            return self._softmax_cost(op)

        tr = self._traffic(op)
        config = self.hw.dispatch_config_cycles + self.hw.state_response_cycles
        input_dma = self._mm2s_cycles(tr.input_bytes, tr.input_segments)
        output_dma = self._s2mm_cycles(tr.output_bytes, tr.output_segments)
        scale_dma = self._s2mm_cycles(
            tr.scale_flush_bytes, 1, is_scale_flush=True)
        compute = self._compute_cycles(op)

        if op.type in (OpType.VEC_MATMUL, OpType.MATMUL):
            body = max(input_dma, compute)
        elif op.type in (OpType.SOFTMAX, OpType.RMSNORM, OpType.FP16_INT8):
            body = input_dma + compute
        else:
            body = max(input_dma, compute)
        return max(config + body + output_dma + scale_dma +
                   self.hw.done_clear_cycles, 1.0)

    def _tensor_bytes(self, name: str) -> int:
        ten = self.g.tensors.get(name)
        return int(ten.nbytes) if ten is not None else 0

    def _traffic(self, op: Op) -> _Traffic:
        if op.type in (OpType.VEC_MATMUL, OpType.MATMUL):
            return self._matmul_traffic(op)
        input_bytes = sum(self._tensor_bytes(t) for t in op.inputs)
        output_bytes = sum(self._tensor_bytes(t) for t in op.outputs)
        return _Traffic(input_bytes, len(op.inputs), output_bytes,
                        1 if output_bytes else 0,
                        self._scale_flush_bytes(op))

    def _matmul_traffic(self, op: Op) -> _Traffic:
        a = op.attrs
        m = int(a["a_row"])
        n = int(a["b_col"])
        m_tiles = _ceil_div(m, 8)
        n_tiles = _ceil_div(n, 64)
        act_bytes = self._tensor_bytes(op.inputs[0])
        weight_bytes = self._tensor_bytes(op.inputs[1]) * m_tiles
        scale_bytes = self._tensor_bytes(op.inputs[2]) * m_tiles
        output_bytes = sum(self._tensor_bytes(t) for t in op.outputs)
        input_segments = m_tiles + (m_tiles * n_tiles) + m_tiles
        return _Traffic(act_bytes + weight_bytes + scale_bytes,
                        input_segments, output_bytes,
                        1 if output_bytes else 0,
                        self._scale_flush_bytes(op))

    def _scale_flush_bytes(self, op: Op) -> int:
        name = op.attrs.get("addr_out_scale")
        if name in self.g.tensors:
            return self._tensor_bytes(str(name))
        if op.attrs.get("put_scale") == "hbm":
            elem = int(op.attrs.get("elem1", op.attrs.get("elem_in", 0)) or 0)
            dim = int(op.attrs.get("dim", 1) or 1)
            rows = _ceil_div(elem, dim) if elem else 0
            return rows * 2
        return 0

    def _beats(self, nbytes: int) -> int:
        return _ceil_div(max(0, nbytes), self.hw.axis_bytes_per_beat)

    def _stream_cycles(self, nbytes: int, beats_per_cycle: float) -> float:
        if nbytes <= 0:
            return 0.0
        if beats_per_cycle <= 0:
            return float("inf")
        return math.ceil(self._beats(nbytes) / beats_per_cycle)

    def _mm2s_cycles(self, nbytes: int, segments: int) -> float:
        if nbytes <= 0 or segments <= 0:
            return 0.0
        setup = (self.hw.first_mm2s_latency_cycles +
                 max(0, segments - 1) * self.hw.mm2s_reissue_cycles)
        return setup + self._stream_cycles(
            nbytes, self.hw.mm2s_beats_per_cycle)

    def _s2mm_cycles(self, nbytes: int, segments: int,
                     is_scale_flush: bool = False) -> float:
        if nbytes <= 0 or segments <= 0:
            return 0.0
        issue = (self.hw.scale_flush_issue_cycles if is_scale_flush
                 else self.hw.s2mm_issue_cycles)
        setup = segments * issue + self.hw.s2mm_intr_clear_cycles
        return setup + self._stream_cycles(
            nbytes, self.hw.s2mm_beats_per_cycle)

    def _compute_cycles(self, op: Op) -> float:
        if op.type in (OpType.VEC_MATMUL, OpType.MATMUL):
            return self._matmul_compute(op)
        if op.type == OpType.SOFTMAX:
            return self._softmax_compute(op)
        if op.type == OpType.RMSNORM:
            return self._rmsnorm_compute(op)
        if op.type == OpType.FP16_INT8:
            elems = self._output_or_input_elems(op)
            return 2 * _ceil_div(elems, 32) + self.hw.quant_latency
        if op.type in (OpType.SILU, OpType.GELU):
            elems = self._attr_elems(op)
            return _ceil_div(elems, 32) + self.hw.silu_latency
        if op.type in (OpType.REARRANGE, OpType.TRANSPOSE, OpType.CONCAT,
                       OpType.SWAP):
            elems = self._output_or_input_elems(op)
            return _ceil_div(elems, 32) + self.hw.transform_latency
        elems = self._attr_elems(op)
        return _ceil_div(elems, 32) + self.hw.vu_pipeline_latency

    def _softmax_cost(self, op: Op) -> float:
        """Measured external Softmax path.

        This intentionally does not reuse the generic _mm2s/_s2mm helpers:
        the closed VCD trace gives a stronger decomposition for Softmax than
        the older first-order DMA model.  The returned cost is measured from
        first PE_TEST config write to state READY, matching the scheduler's
        per-op command cost convention.
        """
        input_windows = [
            self._stream_cycles(n, self.hw.softmax_mm2s_beats_per_cycle)
            for n in (self._tensor_bytes(t) for t in op.inputs)
            if n > 0
        ]
        input_gap = max(0, len(input_windows) - 1) * (
            self.hw.softmax_input_segment_gap_cycles)
        output_bytes = sum(self._tensor_bytes(t) for t in op.outputs)
        output_axis = self._stream_cycles(
            output_bytes, self.hw.softmax_s2mm_beats_per_cycle)

        config_valid_to_ready = (
            self.hw.softmax_config_to_first_mm2s_cycles +
            sum(input_windows) +
            input_gap +
            self._softmax_internal_visible(op) +
            output_axis +
            self.hw.softmax_output_tlast_to_ready_cycles
        )
        return max(self.hw.dispatch_config_cycles + config_valid_to_ready, 1.0)

    def _matmul_compute(self, op: Op) -> float:
        a = op.attrs
        m = int(a["a_row"])
        k = int(a["b_row"])
        n = int(a["b_col"])
        n_tiles = _ceil_div(n, 64)
        k_blocks = _ceil_div(k, 8)
        total = 0.0
        for mt in range(_ceil_div(m, 8)):
            rows = min(8, m - mt * 8)
            total += n_tiles * (rows * k_blocks +
                                self.hw.mu_pipeline_latency)
        return total

    def _softmax_compute(self, op: Op) -> float:
        a = op.attrs
        elem = int(a.get("elem1", self._output_or_input_elems(op)) or 0)
        dim = int(a.get("dim", a.get("vld_len", 1)) or 1)
        rows = _ceil_div(elem, dim) if dim else 1
        row_beats = _ceil_div(dim, 32)
        reduce_delay = 6 + row_beats
        return rows * (2 * row_beats + reduce_delay +
                       self.hw.softmax_pipeline_latency)

    def _softmax_internal_visible(self, op: Op) -> float:
        """Scale the measured input2-TLAST -> output-first-valid latency.

        The reference trace measured 3728 cycles for elem=4032, dim=64.
        We retain the existing structural row/dim dependence and calibrate it
        with the reference ratio, so smaller/larger Softmax ops still scale in
        a predictable way until more RTL/ILA points are available.
        """
        ref_op = Op(
            name="softmax_ref",
            type=OpType.SOFTMAX,
            inputs=[],
            outputs=[],
            attrs={"elem1": self.hw.softmax_ref_elem,
                   "dim": self.hw.softmax_ref_dim},
        )
        ref_model = max(self._softmax_compute(ref_op), 1.0)
        return self._softmax_compute(op) * (
            self.hw.softmax_ref_internal_visible_cycles / ref_model)

    def _rmsnorm_compute(self, op: Op) -> float:
        a = op.attrs
        elem = int(a.get("elem1", self._output_or_input_elems(op)) or 0)
        dim = int(a.get("dim", 2048) or 2048)
        rows = _ceil_div(elem, dim) if dim else 1
        row_beats = _ceil_div(dim, 32)
        reduce_delay = 24 + row_beats
        return rows * (2 * row_beats + reduce_delay +
                       self.hw.rmsnorm_pipeline_latency)

    def _attr_elems(self, op: Op) -> int:
        for key in ("elem", "elem1", "elem_out", "out_elem"):
            if key in op.attrs:
                return int(op.attrs[key])
        return self._output_or_input_elems(op)

    def _output_or_input_elems(self, op: Op) -> int:
        elems = sum(self.g.tensors[t].numel for t in op.outputs
                    if t in self.g.tensors)
        if elems:
            return int(elems)
        return int(sum(self.g.tensors[t].numel for t in op.inputs
                       if t in self.g.tensors))


def critical_path(graph: Graph, cost: CostModel) -> List[float]:
    """Longest remaining cost from each op to any sink (higher = more urgent)."""
    deps = graph.deps()
    n = len(graph.ops)
    succ: List[List[int]] = [[] for _ in range(n)]
    for i, preds in enumerate(deps):
        for p in preds:
            succ[p].append(i)
    cp = [0.0] * n
    for i in reversed(graph.topo()):
        cp[i] = cost.cost(graph.ops[i]) + (max((cp[s] for s in succ[i]), default=0.0))
    return cp


# ---------------------------------------------------------------------------
@dataclass
class Schedule:
    waves: List[List[Tuple[int, int]]]      # [(op_index, core), ...] per wave
    n_cores: int
    graph: Graph
    cost: CostModel

    # -- stats -------------------------------------------------------------
    def makespan(self) -> float:
        """Sum over waves of the wave's slowest op (barrier model)."""
        total = 0.0
        for wave in self.waves:
            total += max((self.cost.cost(self.graph.ops[i]) for i, _ in wave), default=0.0)
        return total

    def serial_cost(self) -> float:
        return sum(self.cost.cost(o) for o in self.graph.ops)

    def speedup(self) -> float:
        ms = self.makespan()
        return (self.serial_cost() / ms) if ms else 1.0

    def utilization(self) -> List[float]:
        busy = [0.0] * self.n_cores
        for wave in self.waves:
            for i, c in wave:
                busy[c] += self.cost.cost(self.graph.ops[i])
        ms = self.makespan() or 1.0
        return [b / ms for b in busy]

    def parallelism_histogram(self) -> Dict[int, int]:
        h: Dict[int, int] = {}
        for wave in self.waves:
            h[len(wave)] = h.get(len(wave), 0) + 1
        return h

    def total_macs(self) -> int:
        return sum(self.cost.macs(o) for o in self.graph.ops)

    def total_bytes(self) -> int:
        return sum(self.cost.bytes(o) for o in self.graph.ops)

    def roofline(self) -> Dict[str, float]:
        macs, bts = self.total_macs(), self.total_bytes()
        ai = macs / bts if bts else 0.0
        ridge = CostModel.MACS_PER_CYCLE / (CostModel.HBM_GBPS * 1e9 / CostModel.CLK_GHZ / 1e9)
        return {"arithmetic_intensity": ai, "ridge_point": ridge,
                "bound": "compute" if ai > ridge else "memory",
                "macs": macs, "bytes": bts}


# ---------------------------------------------------------------------------
class Scheduler:
    def __init__(self, graph: Graph, n_cores: int = 1, strategy: str = "critical",
                 affinity: Optional[Dict[OpType, List[int]]] = None,
                 avoid_pc_conflicts: bool = False):
        if not (1 <= n_cores <= 8):
            raise ValueError("n_cores must be in 1..8")
        self.g = graph
        self.n = n_cores
        self.strategy = strategy
        self.affinity = affinity or {}
        self.avoid_pc_conflicts = avoid_pc_conflicts
        self.cost = CostModel(graph)

    def _allowed_cores(self, op: Op) -> List[int]:
        cores = self.affinity.get(op.type)
        return [c for c in cores if c < self.n] if cores else list(range(self.n))

    def _op_pcs(self, op: Op) -> set[str]:
        """PC set touched by an op, when addresses have already been allocated.

        In the first scheduling pass tensors usually do not yet carry PC info;
        this returns an empty set and scheduling degenerates to the original
        critical/round-robin policy.  After allocation, a second scheduling pass
        can use this to avoid putting multiple heavy ops targeting the same PC
        into one barrier wave.
        """
        pcs = set()
        for tname in op.inputs + op.outputs:
            ten = self.g.tensors.get(tname)
            if ten is None:
                continue
            pc = ten.meta.get("pc")
            if pc and pc != "UNKNOWN":
                pcs.add(str(pc))
        return pcs

    def _select_wave_ops(self, order: List[int]) -> List[int]:
        if not self.avoid_pc_conflicts:
            return order[:self.n]
        pick: List[int] = []
        used_pcs: set[str] = set()
        deferred: List[int] = []
        # First pass: choose ready ops whose PC sets do not overlap.
        for i in order:
            pcs = self._op_pcs(self.g.ops[i])
            conflict = bool(pcs & used_pcs)
            if len(pick) < self.n and not conflict:
                pick.append(i)
                used_pcs |= pcs
            else:
                deferred.append(i)
            if len(pick) == self.n:
                break
        # Second pass: if the wave is under-filled, allow conflicts to preserve
        # parallelism rather than serializing too aggressively.
        if len(pick) < self.n:
            for i in order:
                if i in pick:
                    continue
                pick.append(i)
                if len(pick) == self.n:
                    break
        return pick

    def schedule(self) -> Schedule:
        deps = self.g.deps()
        n_ops = len(self.g.ops)
        succ: List[List[int]] = [[] for _ in range(n_ops)]
        indeg = [0] * n_ops
        for i, preds in enumerate(deps):
            for p in preds:
                succ[p].append(i)
                indeg[i] += 1

        urgency = critical_path(self.g, self.cost) if self.strategy == "critical"             else [0.0] * n_ops
        ready = [i for i in range(n_ops) if indeg[i] == 0]
        waves: List[List[Tuple[int, int]]] = []
        rr = 0

        while ready:
            if self.strategy == "single":
                order = sorted(ready)
                pick = order[:1]
            elif self.strategy == "critical":
                order = sorted(ready, key=lambda i: (-urgency[i], i))
                pick = self._select_wave_ops(order)
            else:  # round_robin
                order = sorted(ready)
                pick = self._select_wave_ops(order)

            wave: List[Tuple[int, int]] = []
            used_cores = set()
            # Prefer lower-numbered cores for deterministic codegen, but avoid
            # reusing a core within a wave.
            for i in pick:
                allowed = [c for c in self._allowed_cores(self.g.ops[i]) if c not in used_cores]
                if not allowed:
                    continue
                if self.strategy == "round_robin":
                    core = allowed[rr % len(allowed)]
                    rr += 1
                else:
                    core = allowed[0]
                used_cores.add(core)
                wave.append((i, core))
                self.g.ops[i].core = core

            scheduled = {i for i, _ in wave}
            waves.append(wave)
            ready = [i for i in ready if i not in scheduled]
            for i, _ in wave:
                for s in succ[i]:
                    indeg[s] -= 1
                    if indeg[s] == 0:
                        ready.append(s)

        return Schedule(waves, self.n, self.g, self.cost)

def report_schedule(sched: Schedule) -> str:
    rl = sched.roofline()
    util = ", ".join(f"c{c}:{u*100:4.0f}%" for c, u in enumerate(sched.utilization()))
    hist = ", ".join(f"{w}-wide x{c}" for w, c in sorted(sched.parallelism_histogram().items()))
    return (
        f"Schedule: {len(sched.waves)} waves on {sched.n_cores} core(s)\n"
        f"  est. makespan {sched.makespan():,.0f} cyc  |  serial {sched.serial_cost():,.0f} cyc"
        f"  |  speedup x{sched.speedup():.2f}\n"
        f"  core utilisation: {util}\n"
        f"  wave widths: {hist}\n"
        f"  roofline: {rl['macs']:,} MACs / {rl['bytes']:,} B  "
        f"-> AI {rl['arithmetic_intensity']:.1f}  (ridge {rl['ridge_point']:.1f}) "
        f"=> {rl['bound']}-bound"
    )
