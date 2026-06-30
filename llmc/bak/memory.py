"""
memory.py -- address management (requirement #3).

Two cooperating allocators:

  HBMAllocator   : 16 pseudo-channels x 1 GB.  Guarantees no single tensor
                   straddles a PC boundary (a known stall source on this NoC),
                   supports free() so activation buffers can be recycled across
                   the schedule, and offers a placement *policy* so weights are
                   spread across channels to widen effective bandwidth
                   (memory-wall mitigation) while activations / KV-cache get
                   their own channels (reduces read/write contention).

  ScratchPad     : a relative-offset planner for the on-chip / working region.
                   Tracks live ranges and reuses the lowest freed offset
                   (first-fit) so the transient activation footprint stays small.

Addresses are 64-bit HBM byte addresses compatible with the firmware and the
QDMA scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PC_SIZE = 0x4000_0000  # 1 GB per pseudo-channel

# 16 PCs, matching the address-editor export in compilation.md.
DEFAULT_REGIONS: List[Tuple[int, str]] = [
    (0x4000000000, "HBM0_PC0"), (0x4040000000, "HBM0_PC1"),
    (0x4080000000, "HBM1_PC0"), (0x40C0000000, "HBM1_PC1"),
    (0x4100000000, "HBM2_PC0"), (0x4140000000, "HBM2_PC1"),
    (0x4180000000, "HBM3_PC0"), (0x41C0000000, "HBM3_PC1"),
    (0x4200000000, "HBM4_PC0"), (0x4240000000, "HBM4_PC1"),
    (0x4280000000, "HBM5_PC0"), (0x42C0000000, "HBM5_PC1"),
    (0x4300000000, "HBM6_PC0"), (0x4340000000, "HBM6_PC1"),
    (0x4380000000, "HBM7_PC0"), (0x43C0000000, "HBM7_PC1"),
]

# memory.py 中，给每个 PC 标注 NoC 跳数（假设控制核在 Mesh 左下角）
NOC_HOPS = {
    "HBM0_PC0": 2, "HBM0_PC1": 3,
    "HBM1_PC0": 2, "HBM1_PC1": 3,
    "HBM2_PC0": 4, "HBM2_PC1": 5,
    # ... 根据实际 X/Y 拓扑填写
}

def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@dataclass
class Alloc:
    addr: int
    size: int
    tag: str
    pc: str
    kind: str           # weight | activation | scale | kvcache | instr | scratch
    live: bool = True


class _Channel:
    """One PC: bump pointer plus a free-list of recyclable [start,end) spans."""
    def __init__(self, base: int):
        self.base = base
        self.end = base + PC_SIZE
        self.ptr = base
        self.free: List[Tuple[int, int]] = []   # sorted, non-overlapping

    def _take_from_free(self, size: int, align: int) -> Optional[int]:
        for i, (s, e) in enumerate(self.free):
            a = align_up(s, align)
            if a + size <= e:
                # carve [a, a+size); keep the remainders
                new = []
                if a > s:
                    new.append((s, a))
                if a + size < e:
                    new.append((a + size, e))
                self.free = self.free[:i] + new + self.free[i + 1:]
                return a
        return None

    def alloc(self, size: int, align: int) -> Optional[int]:
        a = self._take_from_free(size, align)
        if a is not None:
            return a
        a = align_up(self.ptr, align)
        if a + size <= self.end:
            self.ptr = a + size
            return a
        return None

    def free_span(self, addr: int, size: int):
        s, e = addr, addr + size
        merged = []
        inserted = False
        for cs, ce in sorted(self.free + [(s, e)]):
            if merged and cs <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], ce))
                inserted = True
            else:
                merged.append((cs, ce))
        self.free = merged

    def used(self) -> int:
        return self.ptr - self.base


class HBMAllocator:
    def __init__(self, regions: Optional[List[Tuple[int, str]]] = None):
        regions = regions or DEFAULT_REGIONS
        self.regions = regions
        self.chan: Dict[str, _Channel] = {name: _Channel(base) for base, name in regions}
        self.allocs: List[Alloc] = []
        self._by_addr: Dict[int, Alloc] = {}
        # default channel policy by data kind (overridable)
        names = [n for _, n in regions]
        self.policy: Dict[str, List[str]] = {
            "instr":      [names[0], names[1]],                 # HBM0
            "weight":     names[4:],                          # spread across the middle PCs
            "activation": [names[2], names[3]],                 # HBM1
            "scratch":    [names[2], names[3]],
            "scale":      [names[14]],                          # HBM7_PC0
            "kvcache":    [names[15]],                          # HBM7_PC1
        }
        self._rr: Dict[str, int] = {k: 0 for k in self.policy}

    # -- core api ----------------------------------------------------------
    def alloc(self, size: int, *, kind: str = "weight", align: int = 64,
              tag: str = "", preferred_pc: Optional[str] = None) -> int:
        if size <= 0:
            raise ValueError(f"alloc size must be > 0 (tag={tag})")
        if size > PC_SIZE:
            raise MemoryError(
                f"tensor {tag!r} ({size} B) exceeds one PC ({PC_SIZE} B); "
                f"it must be tiled across channels before allocation")

        order = self._candidate_order(kind, preferred_pc)
        for name in order:
            addr = self.chan[name].alloc(size, align)
            if addr is not None:
                rec = Alloc(addr, align_up(size, align), tag, name, kind)
                self.allocs.append(rec)
                self._by_addr[addr] = rec
                if kind in self._rr:                  # advance round-robin
                    self._rr[kind] = (self._rr[kind] + 1) % max(1, len(self.policy[kind]))
                return addr
        raise MemoryError(f"no PC can fit {size} B for {tag!r} (kind={kind})")

    def free(self, addr: int):
        rec = self._by_addr.get(addr)
        if rec is None or not rec.live:
            return
        rec.live = False
        self.chan[rec.pc].free_span(rec.addr, rec.size)

    def pc_of(self, addr: int) -> str:
        for base, name in self.regions:
            if base <= addr < base + PC_SIZE:
                return name
        return "UNKNOWN"

    def _candidate_order(self, kind: str, preferred_pc: Optional[str]) -> List[str]:
        if preferred_pc:
            pref = [n for n in self.chan if n.startswith(preferred_pc)]
        else:
            pref = []
        pool = self.policy.get(kind, list(self.chan))
        if pool:
            start = self._rr.get(kind, 0) % len(pool)
            pool = pool[start:] + pool[:start]
        rest = [n for n in self.chan if n not in pref and n not in pool]
        # preferred first, then policy pool (load-balanced), then everything else
        seen, order = set(), []
        for n in pref + pool + rest:
            if n not in seen:
                seen.add(n); order.append(n)
        return order

    # -- reporting ---------------------------------------------------------
    def usage(self) -> Dict[str, float]:
        return {name: ch.used() / 1024 / 1024 for name, ch in self.chan.items()}

    def table(self) -> str:
        lines = ["HBM allocation", "-" * 78]
        for r in self.allocs:
            live = "" if r.live else "  (freed)"
            lines.append(f"  {r.tag:26s} 0x{r.addr:010X} +0x{r.size:08X} "
                         f"{r.kind:10s} {r.pc}{live}")
        lines.append("-" * 78)
        for name, mb in self.usage().items():
            lines.append(f"  {name:10s} used {mb:7.2f} MB / 1024 MB")
        return "\n".join(lines)

    # HBMAllocator 类内新增方法
    def get_alloc_base_addr(self, tag: str) -> Optional[int]:
        """根据tensor名称获取已分配的基地址（alias base专用）"""
        for rec in self.allocs:
            if rec.tag == tag and rec.live:
                return rec.addr
        return None

@dataclass
class _Live:
    off: int
    size: int
    tag: str


class ScratchPad:
    """Relative-offset planner for transient activations within a base buffer."""
    def __init__(self, base_addr: int, capacity: int = PC_SIZE, align: int = 64, recycle=True):
        self.base = base_addr
        self.capacity = capacity
        self.align = align
        self.ptr = 0
        self._free: List[Tuple[int, int]] = []
        self.live: Dict[int, _Live] = {}
        self.peak = 0
        self.recycle = recycle  # 新增

    def alloc(self, size, tag=""):
        size = align_up(size, self.align)
        if not self.recycle:
            # 不回收模式：直接 bump ptr，无视 free list
            off = align_up(self.ptr, self.align)
            self.ptr = off + size
            self.peak = max(self.peak, self.ptr)
            self.live[off] = _Live(off, size, tag)
            return self.base + off
        # first-fit in freed spans
        for i, (s, e) in enumerate(self._free):
            if e - s >= size:
                off = s
                rem = [(s + size, e)] if e - s > size else []
                self._free = self._free[:i] + rem + self._free[i + 1:]
                break
        else:
            off = align_up(self.ptr, self.align)
            self.ptr = off + size
            self.peak = max(self.peak, self.ptr)
        self.live[off] = _Live(off, size, tag)
        return self.base + off

    def free(self, addr: int):
        off = addr - self.base
        rec = self.live.pop(off, None)
        if rec is None:
            return
        self._free = self._merge(self._free + [(off, off + rec.size)])

    @staticmethod
    def _merge(spans):
        out = []
        for s, e in sorted(spans):
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        return out

    def peak_mb(self) -> float:
        return self.peak / 1024 / 1024

# 在 memory.py 文件末尾新增
def compute_alias_addr(base_addr: int, elem_offset: int, elem_bytes: int) -> int:
    """
    计算alias张量真实HBM字节地址
    :param base_addr: 基准tensor完整HBM地址
    :param elem_offset: 元素维度偏移量（如row_idx * head_dim）
    :param elem_bytes: 单个元素字节数（FP16=2）
    :return: alias 起始字节地址
    """
    byte_off = elem_offset * elem_bytes
    return base_addr + byte_off


def get_tensor_nbytes(shape: Tuple[int, ...], dtype_byte: int = 2) -> int:
    """计算张量总字节数，适配FP16权重/激活/KV-cache"""
    total_elem = 1
    for dim in shape:
        total_elem *= dim
    return align_up(total_elem * dtype_byte, align=64)