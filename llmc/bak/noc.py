# noc.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
from .schedule import Schedule
from .ir import Graph, Kind


@dataclass
class NoCConfig:
    """基于 Versal QoS 表校准的 NoC 参数"""
    # 控制核 AXI-Lite 写寄存器：本地操作，约 2 cycles per 32-bit reg
    mmio_cycles_per_reg: float = 1.0
    # 计算核 HBM 访问固定延迟（QoS 表 HBM00_AXI -> hbm_chnl: 24-32 cycles）
    hbm_fixed_latency: float = 28.0
    # 每 HBM 端口带宽（GB/s），用于争用排队
    hbm_port_bw_gbps: float = 28.0
    # 时钟频率（GHz）
    clk_ghz: float = 0.3
    # wait_done 轮询间隔（cycles）
    sync_poll_interval: int = 2
    # HBM 内部 bank 级并行度（同一 channel 可并行服务几个请求）
    hbm_bank_parallelism: int = 8

    # Tier-1: 计算核直连 HBM（按 channel 编号查表）
    hbm_latency_map: Dict[str, float] = field(default_factory=lambda: {
        "chnl4": 24.0, "chnl5": 24.0, "chnl0": 28.0,
        "chnl2": 28.0, "chnl3": 28.0, "chnl6": 32.0, "chnl7": 32.0,
    })
    # Tier-2: 控制核/Slave 路径（MMIO、DMA）
    slave_path_latency: float = 48.0   # 44~52 的中值
    # Tier-3: 控制核配置寄存器开销（per 32-bit reg）
    mmio_cycles_per_reg: float = 0.5   # 占位符，待实测
    mmio_cycles_per_core: float = 17.0
    # 带宽：标称 vs 有效
    hbm_nominal_gbps: float = 460.0    # 硬件标称
    hbm_effective_gbps: float = 200.0  # QoS 表标称（占位符）
    mmio_stall_penalty: float = 0.0
    # 争用模型
    hbm_bank_parallelism: int = 8

class NoCSimulator:
    def __init__(self, graph: Graph, sched: Schedule, cfg: NoCConfig):
        self.g = graph
        self.s = sched
        self.cfg = cfg
        # 从 tensor 地址恢复 PC
        self.tensor_pc = {
            t.name: t.meta.get("pc", "UNKNOWN")
            for t in graph.tensors.values()
        }

    def _mmio_time(self, n_cores: int) -> float:
        """控制核串行配置 n 个核的 MMIO 时间（与计算重叠）"""
        regs = 16
        return n_cores * self.cfg.mmio_cycles_per_core

    def _hbm_contention(self, wave) -> float:
        """
        基于 HBM channel 带宽的排队模型。
        同一 wave 内多核访问同一 PC 时，超出 bank 并行度的请求排队。
        """
        pc_cores: Dict[str, List[int]] = {}
        for op_i, core in wave:
            op = self.g.ops[op_i]
            for t in op.inputs + op.outputs:
                pc = self.tensor_pc.get(t)
                if pc and pc != "UNKNOWN":
                    pc_cores.setdefault(pc, []).append(core)

        max_penalty = 0.0
        for pc, cores in pc_cores.items():
            n = len(set(cores))  # 去重：同一核多次访问同一 PC 只算一次
            if n <= self.cfg.hbm_bank_parallelism:
                continue
            # 超出并行度的请求需要排队
            # 惩罚 = 额外请求数 × 固定延迟 / 并行度
            overflow = n - self.cfg.hbm_bank_parallelism
            penalty = overflow * self.cfg.hbm_fixed_latency / self.cfg.hbm_bank_parallelism
            max_penalty = max(max_penalty, penalty)
        return max_penalty

    def simulate(self) -> Dict:
        total_compute = 0.0
        total_mmio = 0.0
        total_sync = 0.0
        total_contention = 0.0
        makespan_noc = 0.0

        for wi, wave in enumerate(self.s.waves):
            n_cores = len(wave)
            # 计算阶段：该 wave 最慢 op 的时间
            comp = max((self.s.cost.cost(self.g.ops[i]) for i, _ in wave), default=0.0)
            # MMIO 阶段：控制核串行配置
            mmio = self._mmio_time(n_cores)
            # 同步：wait_done 轮询
            sync = self.cfg.sync_poll_interval * 2  # 往返
            # HBM 争用
            contention = self._hbm_contention(wave)

            # 关键：MMIO 与计算重叠，取最大值
            wave_time = max(comp, mmio) + sync + contention

            total_compute += comp
            total_mmio += mmio
            total_sync += sync
            total_contention += contention
            makespan_noc += wave_time

        makespan_pure = self.s.makespan()
        overhead = (makespan_noc - makespan_pure) / makespan_pure * 100 if makespan_pure else 0

        # 瓶颈识别：看哪个分量在 wave_time 中占主导
        def _bottleneck():
            # 统计各分量对总时间的贡献
            if total_mmio > total_compute and total_mmio > total_contention:
                return "control-core-mmio"
            if total_contention > total_compute:
                return "hbm-contention"
            return "compute"

        return {
            "makespan_pure_cycles": round(makespan_pure),
            "makespan_noc_cycles": round(makespan_noc),
            "noc_overhead_pct": round(overhead, 1),
            "breakdown": {
                "compute": round(total_compute, 0),
                "mmio": round(total_mmio, 0),
                "sync": round(total_sync, 0),
                "hbm_contention": round(total_contention, 0),
            },
            "bottleneck": _bottleneck(),
            "config": {
                "mmio_per_reg": self.cfg.mmio_cycles_per_reg,
                "hbm_latency": self.cfg.hbm_fixed_latency,
                "hbm_parallelism": self.cfg.hbm_bank_parallelism,
            }
        }