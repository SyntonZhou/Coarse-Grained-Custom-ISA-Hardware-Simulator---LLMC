"""
ir.py -- the intermediate representation.

A Graph is a DAG of Ops over named Tensors.  This is what the model frontend
emits, what the scheduler analyses for parallelism / dependencies, what the
memory planner allocates, and what the backend lowers to instructions.

Tensors carry shape + dtype + a `kind` (weight / activation / scale / kvcache /
input / output).  An Op names its input and output tensors; data dependencies
are *derived* from producer/consumer relationships rather than declared, so the
frontend stays simple and the scheduler stays correct by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class DType(Enum):
    INT8 = ("int8", 1)
    FP16 = ("fp16", 2)
    FP32 = ("fp32", 4)

    @property
    def bytes(self) -> int:
        return self.value[1]

    def __str__(self) -> str:
        return self.value[0]


class Kind(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    WEIGHT = "weight"
    SCALE = "scale"
    ACT = "activation"
    KVCACHE = "kvcache"


# Primitive op set -- mirrors the firmware's Cal_Type families.
class OpType(str, Enum):
    FP16_INT8 = "fp16_int8"
    VEC_MATMUL = "vec_matmul"
    MATMUL = "matmul"
    SILU = "silu"
    GELU = "gelu"
    VU_MUL = "vu_mul"
    VU_ADD = "vu_add"
    RESIDUAL = "residual"
    RMSNORM = "rmsnorm"
    SOFTMAX = "softmax"
    ROPE = "rope"
    SWAP = "swap"
    REARRANGE = "rearrange"
    TRANSPOSE = "transpose"
    CONCAT = "concat"
    VU_MASK = "vu_mask"


@dataclass
class Tensor:
    name: str
    shape: Tuple[int, ...]
    dtype: DType = DType.FP16
    kind: Kind = Kind.ACT
    addr: Optional[int] = None         # filled by the memory planner
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def nbytes(self) -> int:
        return self.numel * self.dtype.bytes

    def __repr__(self) -> str:
        a = f"@0x{self.addr:X}" if self.addr is not None else "@?"
        return f"{self.name}{list(self.shape)}:{self.dtype}:{self.kind.value}{a}"


@dataclass
class Op:
    name: str
    type: OpType
    inputs: List[str]                  # tensor names
    outputs: List[str]                 # tensor names
    attrs: Dict[str, object] = field(default_factory=dict)
    core: Optional[int] = None         # filled by the scheduler

    def __repr__(self) -> str:
        return f"{self.name}({self.type.value}: {self.inputs} -> {self.outputs})"


class Graph:
    def __init__(self, name: str = "model"):
        self.name = name
        self.tensors: Dict[str, Tensor] = {}
        self.ops: List[Op] = []
        self._uid = 0

    # -- builders ----------------------------------------------------------
    def tensor(self, name: str, shape, dtype=DType.FP16, kind=Kind.ACT,
               addr=None, **meta) -> str:
        if name in self.tensors:
            return name
        self.tensors[name] = Tensor(name, tuple(shape), dtype, kind, addr, dict(meta))
        return name

    def fresh(self, prefix: str, shape, dtype=DType.FP16, kind=Kind.ACT) -> str:
        self._uid += 1
        return self.tensor(f"{prefix}_{self._uid}", shape, dtype, kind)

    def op(self, type_: OpType, inputs: List[str], outputs: List[str],
           name: Optional[str] = None, **attrs) -> Op:
        for t in inputs + outputs:
            if t not in self.tensors:
                raise KeyError(f"op references unknown tensor {t!r}")
        if name is None:
            name = f"{type_.value}_{len(self.ops)}"
        o = Op(name, type_, list(inputs), list(outputs), dict(attrs))
        self.ops.append(o)
        return o

    # -- dependency analysis ----------------------------------------------
    def producers(self) -> Dict[str, int]:
        """tensor name -> index of the op that produces it (last writer)."""
        prod: Dict[str, int] = {}
        for i, o in enumerate(self.ops):
            for t in o.outputs:
                prod[t] = i
        return prod

    def _mem_region(self, tensor_name: str) -> Tuple[str, int, int]:
        """Return a conservative memory region for dependency analysis.

        Normal tensors use their own name as the memory object.  Alias tensors
        use alias_base plus a byte interval.  This lets the scheduler preserve
        RAW/WAR/WAW ordering for KV-cache rows and RoPE table slices before
        physical addresses are assigned.
        """
        t = self.tensors[tensor_name]
        base = str(t.meta.get("alias_base", tensor_name))
        off = int(t.meta.get("alias_offset", 0))
        return base, off, off + t.nbytes

    @staticmethod
    def _region_overlap(a: Tuple[str, int, int], b: Tuple[str, int, int]) -> bool:
        return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]

    def deps(self) -> List[List[int]]:
        """For each op, return predecessor op indices.

        Dependencies include:
          * ordinary RAW/WAW dependencies by tensor name;
          * alias-aware memory dependencies by alias_base + byte interval.

        The second part is intentionally conservative and also enforces WAR for
        overlapping alias regions.  This prevents decode KV-cache updates from
        being reordered before later reads of the same cache interval.
        """
        last_writer: Dict[str, int] = {}
        prior_reads: List[Tuple[Tuple[str, int, int], int]] = []
        prior_writes: List[Tuple[Tuple[str, int, int], int]] = []
        edges: List[List[int]] = []

        for i, o in enumerate(self.ops):
            pred = set()

            # Standard SSA-like dependencies by tensor name.
            for t in o.inputs:
                if t in last_writer:
                    pred.add(last_writer[t])
            for t in o.outputs:
                if t in last_writer:
                    pred.add(last_writer[t])

            read_regions = [self._mem_region(t) for t in o.inputs]
            write_regions = [self._mem_region(t) for t in o.outputs]

            # RAW through alias memory: current read waits for prior writes.
            for rr in read_regions:
                for wr, j in prior_writes:
                    if self._region_overlap(rr, wr):
                        pred.add(j)

            # WAW and WAR through alias memory: current write waits for prior
            # reads/writes to overlapping regions.
            for wr in write_regions:
                for pwr, j in prior_writes:
                    if self._region_overlap(wr, pwr):
                        pred.add(j)
                for rr, j in prior_reads:
                    if self._region_overlap(wr, rr):
                        pred.add(j)

            edges.append(sorted(pred))

            for t in o.outputs:
                last_writer[t] = i
            for rr in read_regions:
                prior_reads.append((rr, i))
            for wr in write_regions:
                prior_writes.append((wr, i))

        return edges

    def topo(self) -> List[int]:
        """Kahn topological order over the derived dependency edges."""
        edges = self.deps()
        n = len(self.ops)
        succ: List[List[int]] = [[] for _ in range(n)]
        indeg = [0] * n
        for i, preds in enumerate(edges):
            for p in preds:
                succ[p].append(i)
                indeg[i] += 1
        ready = [i for i in range(n) if indeg[i] == 0]
        order = []
        while ready:
            ready.sort()                # stable: keep program order among ties
            i = ready.pop(0)
            order.append(i)
            for s in succ[i]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    ready.append(s)
        if len(order) != n:
            raise RuntimeError("cycle detected in graph")
        return order

    def summary(self) -> str:
        n_w = sum(1 for t in self.tensors.values() if t.kind == Kind.WEIGHT)
        wb = sum(t.nbytes for t in self.tensors.values() if t.kind == Kind.WEIGHT)
        return (f"Graph {self.name!r}: {len(self.ops)} ops, "
                f"{len(self.tensors)} tensors ({n_w} weights, "
                f"{wb/1024/1024:.1f} MB).")
