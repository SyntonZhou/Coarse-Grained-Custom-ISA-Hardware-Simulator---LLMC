# noc.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple

from .schedule import Schedule
from .ir import Graph, Kind
from .topology import HardwareTopology, default_versal_hbm_topology


@dataclass
class NoCConfig:
    """NoC/HBM timing model.

    QoS-table bandwidths are treated as compiler-level requirements/weights, not
    hard runtime caps.  Runtime time is bounded by effective DMA/NMU/PC service
    rates and fixed NMU->PC path latency.
    """
    # control-core AXI-Lite configuration cost.  This is serialized per wave.
    mmio_cycles_per_core: float = 17.0
    # wait_done polling round-trip cost in cycles.
    sync_poll_interval: int = 2
    # accelerator clock in GHz.
    clk_ghz: float = 0.3
    # Runtime effective service rates.  Calibrate these with counters/ILA later.
    hbm_pc_bw_GBps: float = 28.0
    noc_path_bw_GBps: float = 16.0
    dma_bw_GBps: float = 16.0
    # Treat invalid NMU->PC routes as warnings by default so old analyses still run.
    strict_routes: bool = False


@dataclass
class WaveNoCDetail:
    wave: int
    hbm_cycles: float
    pc_cycles: Dict[str, float]
    nmu_cycles: Dict[str, float]
    dma_cycles: Dict[str, float]
    invalid_routes: List[Tuple[str, str, str]]


def _bytes_to_cycles(nbytes: int, bw_GBps: float, clk_ghz: float) -> float:
    if nbytes <= 0:
        return 0.0
    if bw_GBps <= 0:
        return float("inf")
    # nbytes / (GB/s * 1e9) seconds * clk_GHz * 1e9 cycles/sec
    return nbytes * clk_ghz / bw_GBps


class NoCSimulator:
    def __init__(self, graph: Graph, sched: Schedule, cfg: NoCConfig,
                 topology: HardwareTopology | None = None):
        self.g = graph
        self.s = sched
        self.cfg = cfg
        self.topology = topology or default_versal_hbm_topology(
            sched.n_cores, independent_dma=True
        )
        self.tensor_pc = {
            t.name: t.meta.get("pc", "UNKNOWN")
            for t in graph.tensors.values()
        }

    def _mmio_time(self, n_cores: int) -> float:
        """Control core serially configures n cores for one wave."""
        return n_cores * self.cfg.mmio_cycles_per_core

    def _tensor_nbytes(self, tname: str) -> int:
        ten = self.g.tensors.get(tname)
        return int(ten.nbytes) if ten is not None else 0

    def _route_latency(self, nmu: str, pc: str) -> Tuple[float, bool]:
        edge = self.topology.edge(nmu, pc)
        if edge is None:
            if self.cfg.strict_routes:
                raise RuntimeError(f"NoC route not reachable: {nmu} -> {pc}")
            return 0.0, False
        return float(edge.latency_cycles), True

    def _wave_hbm_time(self, wave_index: int, wave) -> WaveNoCDetail:
        """Byte-based HBM/NoC/DMA queue approximation for one barrier wave.

        * Each PC is a service endpoint: all bytes targeting the same PC share
          hbm_pc_bw_GBps.
        * Each NMU path has its own aggregate cap.
        * Each DMA id has its own cap; if the topology uses shared dma0, the
          shared-DMA bottleneck appears here automatically.
        * Different PCs can operate in parallel, so the HBM contribution is the
          max over PC/NMU/DMA service times.
        """
        pc_bytes: DefaultDict[str, int] = defaultdict(int)
        pc_lat: DefaultDict[str, float] = defaultdict(float)
        nmu_bytes: DefaultDict[str, int] = defaultdict(int)
        nmu_lat: DefaultDict[str, float] = defaultdict(float)
        dma_bytes: DefaultDict[str, int] = defaultdict(int)
        invalid_routes: List[Tuple[str, str, str]] = []

        for op_i, core in wave:
            op = self.g.ops[op_i]
            port = self.topology.port_for_core(core)

            for t in op.inputs:
                pc = str(self.tensor_pc.get(t, "UNKNOWN"))
                if pc == "UNKNOWN":
                    continue
                nb = self._tensor_nbytes(t)
                nmu = port.read_nmu
                lat, ok = self._route_latency(nmu, pc)
                if not ok:
                    invalid_routes.append((op.name, nmu, pc))
                pc_bytes[pc] += nb
                pc_lat[pc] = max(pc_lat[pc], lat)
                nmu_bytes[nmu] += nb
                nmu_lat[nmu] = max(nmu_lat[nmu], lat)
                dma_bytes[port.read_dma] += nb

            for t in op.outputs:
                pc = str(self.tensor_pc.get(t, "UNKNOWN"))
                if pc == "UNKNOWN":
                    continue
                nb = self._tensor_nbytes(t)
                nmu = port.write_nmu
                lat, ok = self._route_latency(nmu, pc)
                if not ok:
                    invalid_routes.append((op.name, nmu, pc))
                pc_bytes[pc] += nb
                pc_lat[pc] = max(pc_lat[pc], lat)
                nmu_bytes[nmu] += nb
                nmu_lat[nmu] = max(nmu_lat[nmu], lat)
                dma_bytes[port.write_dma] += nb

        pc_cycles = {
            pc: pc_lat[pc] + _bytes_to_cycles(nb, self.cfg.hbm_pc_bw_GBps, self.cfg.clk_ghz)
            for pc, nb in pc_bytes.items()
        }
        nmu_cycles = {
            nmu: nmu_lat[nmu] + _bytes_to_cycles(nb, self.cfg.noc_path_bw_GBps, self.cfg.clk_ghz)
            for nmu, nb in nmu_bytes.items()
        }
        dma_cycles = {
            dma: _bytes_to_cycles(nb, self.cfg.dma_bw_GBps, self.cfg.clk_ghz)
            for dma, nb in dma_bytes.items()
        }
        hbm_cycles = max([0.0] + list(pc_cycles.values())
                         + list(nmu_cycles.values()) + list(dma_cycles.values()))
        return WaveNoCDetail(wave_index, hbm_cycles, pc_cycles, nmu_cycles,
                             dma_cycles, invalid_routes)

    def simulate(self) -> Dict:
        total_compute = 0.0
        total_mmio = 0.0
        total_sync = 0.0
        total_hbm = 0.0
        makespan_noc = 0.0
        invalid_routes: List[Tuple[int, str, str, str]] = []
        worst_waves: List[Dict[str, object]] = []

        for wi, wave in enumerate(self.s.waves):
            n_cores = len(wave)
            comp = max((self.s.cost.cost(self.g.ops[i]) for i, _ in wave), default=0.0)
            mmio = self._mmio_time(n_cores)
            sync = self.cfg.sync_poll_interval * 2
            detail = self._wave_hbm_time(wi, wave)
            hbm = detail.hbm_cycles

            # MMIO, HBM transfer, and compute are overlapped only to the extent
            # the hardware pipeline permits.  For this coarse barrier model, use
            # max() for the wave body and then add explicit sync polling.
            wave_time = max(comp, mmio, hbm) + sync

            total_compute += comp
            total_mmio += mmio
            total_sync += sync
            total_hbm += hbm
            makespan_noc += wave_time

            for op_name, nmu, pc in detail.invalid_routes:
                invalid_routes.append((wi, op_name, nmu, pc))
            worst_waves.append({
                "wave": wi,
                "compute": round(comp, 1),
                "mmio": round(mmio, 1),
                "hbm_noc": round(hbm, 1),
                "width": n_cores,
                "top_pc": max(detail.pc_cycles.items(), key=lambda kv: kv[1])[0]
                          if detail.pc_cycles else None,
            })

        makespan_pure = self.s.makespan()
        overhead = (makespan_noc - makespan_pure) / makespan_pure * 100 if makespan_pure else 0

        def _bottleneck():
            if total_mmio > total_compute and total_mmio > total_hbm:
                return "control-core-mmio"
            if total_hbm > total_compute and total_hbm > total_mmio:
                return "hbm-noc-dma"
            return "compute"

        worst_waves.sort(key=lambda d: max(d["compute"], d["mmio"], d["hbm_noc"]), reverse=True)

        return {
            "makespan_pure_cycles": round(makespan_pure),
            "makespan_noc_cycles": round(makespan_noc),
            "noc_overhead_pct": round(overhead, 1),
            "breakdown": {
                "compute": round(total_compute, 0),
                "mmio": round(total_mmio, 0),
                "sync": round(total_sync, 0),
                "hbm_noc_dma": round(total_hbm, 0),
            },
            "bottleneck": _bottleneck(),
            "invalid_routes": invalid_routes[:16],
            "invalid_route_count": len(invalid_routes),
            "worst_waves": worst_waves[:8],
            "config": {
                "mmio_cycles_per_core": self.cfg.mmio_cycles_per_core,
                "clk_ghz": self.cfg.clk_ghz,
                "hbm_pc_bw_GBps": self.cfg.hbm_pc_bw_GBps,
                "noc_path_bw_GBps": self.cfg.noc_path_bw_GBps,
                "dma_bw_GBps": self.cfg.dma_bw_GBps,
                "topology": self.topology.to_report(),
            }
        }
