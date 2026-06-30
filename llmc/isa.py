"""
isa.py -- the hardware contract.

Single source of truth for:
  * the 16 x 32-bit (= 512-bit) configuration-register layout of the LLMCORE,
  * the bit-exact packing of high-level fields into those 16 words,
  * the block-tiling math (Matrix_configA/B, VU ping-pong),
  * the Cal_Type opcode table.

Everything downstream (lowering, codegen, dma, bin) depends only on this module,
so if the hardware register map ever changes, this is the one place to edit.

The layout below is taken verbatim from gen_script_v2.py / codegen_v1.py and is
validated bit-for-bit against the hand-written reference firmware in
verify_isa.py (the FP16->INT8 -> MatMul smoke test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# MMIO base + register names (single-core).  Multi-core base offsets live in
# codegen; this is the canonical single-core map matching the reference .c.
# ---------------------------------------------------------------------------
MMIO_BASE = 0x80000000
DEFAULT_CORE_STRIDE = 0x1000

REG_NAMES: List[str] = [
    "INPUT_ADDR1H", "INPUT_ADDR1L", "INPUT_LEN1",
    "INPUT_ADDR2H", "INPUT_ADDR2L", "INPUT_LEN2",
    "OUTPUT_ADDRH", "OUTPUT_ADDRL", "OUTPUT_LEN",
    "INPUT_ADDR3H", "INPUT_ADDR3L",
    "OTHER_PARA1", "OTHER_PARA2",
    "MATRIXA_CONFIG", "MATRIXB_CONFIG", "PARA_CONFIG",
]
NUM_REGS = len(REG_NAMES)
assert NUM_REGS == 16

# Per-register MMIO offset (4 bytes each, contiguous from MMIO_BASE).
REG_OFFSET: Dict[str, int] = {name: MMIO_BASE + 4 * i for i, name in enumerate(REG_NAMES)}

# ---------------------------------------------------------------------------
# Field layout.  Each register is a list of (field_name, bit_width), packed
# MSB-first (first field occupies the top bits).  Widths in each register
# must sum to 32.  Names starting with "reserved" / in RESERVED are zero-filled.
# ---------------------------------------------------------------------------
RESERVED = {"Reserve", "Reserved"}

REG_FIELDS: List[List[Tuple[str, int]]] = [
    [("Param1", 16), ("Input_addr1_H", 16)],                                  # Reg1  INPUT_ADDR1H
    [("Input_addr1_L", 32)],                                                  # Reg2  INPUT_ADDR1L
    [("reserved", 8), ("Input_len1", 24)],                                    # Reg3  INPUT_LEN1
    [("Param2", 16), ("Input_addr2_H", 16)],                                  # Reg4  INPUT_ADDR2H
    [("Input_addr2_L", 32)],                                                  # Reg5  INPUT_ADDR2L
    [("reserved", 8), ("Input_len2", 24)],                                    # Reg6  INPUT_LEN2
    [("Param3", 16), ("Output_addr_H", 16)],                                  # Reg7  OUTPUT_ADDRH
    [("Output_addr_L", 32)],                                                  # Reg8  OUTPUT_ADDRL
    [("reserved", 8), ("Output_len", 24)],                                    # Reg9  OUTPUT_LEN
    [("reserved", 16), ("Input_addr3_H", 16)],                                # Reg10 INPUT_ADDR3H
    [("Input_addr3_L", 32)],                                                  # Reg11 INPUT_ADDR3L
    [("Martix_b_row_config", 16), ("Output_scale_addr4_H", 16)],              # Reg12 OTHER_PARA1
    [("Output_scale_addr4_L", 32)],                                           # Reg13 OTHER_PARA2
    [("Matrix_configA_H", 16), ("Matrix_configA_L", 16)],                     # Reg14 MATRIXA_CONFIG
    [("Matrix_configB_H", 16), ("Matrix_configB_L", 16)],                     # Reg15 MATRIXB_CONFIG
    [("Reserve", 8), ("Transpose_mark", 8), ("Cal_Type", 8),                  # Reg16 PARA_CONFIG
     ("next_type", 1), ("get_scale", 2), ("put_scale", 2),
     ("Config_avail", 1), ("Reserved", 2)],
]
assert len(REG_FIELDS) == NUM_REGS
for _r in REG_FIELDS:
    assert sum(w for _, w in _r) == 32, _r

# ---------------------------------------------------------------------------
# Cal_Type opcode table.  Format is XXXX_Y_ZZZ (4_1_3 bits) where the low 3
# bits select the functional unit / instruction class:
#   000 -> MU linear (matmul)        001 -> VU non-linear        010 -> quant/reorder
# NOTE: the reference firmware writes Cal_Type=0x00 for *both* vec-matmul and
# matrix-matmul (it distinguishes them via Matrix_configA's basis, see below).
# The "_1_" naming on setMatrix_Mul is a label only.  We follow the firmware.
# ---------------------------------------------------------------------------
CAL_TYPE: Dict[str, int] = {
    "MATMUL":     0x00,   # 矩阵-矩阵: 0000_0000
    "VU_MATMUL":  0x80,   # 向量-矩阵: 1000_0000 (硬件待支持)
    "VEC_MATMUL": 0x00,   # 如果当前固件实际用 0x00 表示 MatMul，则 VEC_MATMUL 只是别名
    "SILU":       0x01,   # 0000_0_001
    "GELU":       0x11,   # 0001_0_001
    "SOFTMAX":    0x21,   # 0010_0_001
    "ROPE":       0x31,   # 0011_0_001
    "RESIDUAL":   0x41,   # 0100_0_001
    "RMSNORM":    0x51,   # 0101_0_001
    "LAYERNORM":  0x61,   # 0110_0_001
    "VU_MUL":     0x31,   # 0011_0_001 (shares ROPE datapath)
    "VU_ADD":     0x41,   # 0100_0_001 (shares RESIDUAL datapath)
    "VU_MASK":    0x41,   # firmware fn name says 0100_0_001; see note in lowering
    "SWAP":       0x61,   # 0110_0_001
    "REARRANGE":  0x02,   # 0000_0_010
    "TRANSPOSE":  0x12,   # 0001_0_010
    "CONCAT":     0x22,   # 0010_0_010
    "FP16_INT8":  0x32,   # 0011_0_010
}

# put_scale / get_scale enumerations (table 16).
GET_SCALE = {"scaleA": 0b00, "scaleAR": 0b01}
PUT_SCALE = {"none": 0b00, "scaleA": 0b01, "scaleAR": 0b10, "hbm": 0b11}
NEXT_TYPE = {"linear_int8": 0, "nonlinear_fp16": 1}
TRANSPOSE = {"none": 0b00, "A": 0b01, "B": 0b10, "C": 0b11}

# FP16 hex for 1.0 -- the firmware's default "unit scale" placeholder in Param1/2.
FP16_ONE = 0x3C00


# ===========================================================================
# Packing
# ===========================================================================
# isa.py 中，修改 encode_fields 返回字段映射
def encode_fields(fields: Dict[str, int]) -> Tuple[List[int], Dict[int, Dict[str, int]]]:
    """返回 (words, field_map)，field_map[word_idx] = {field_name: value}"""
    words: List[int] = []
    field_map: Dict[int, Dict[str, int]] = {}
    for ri, reg_def in enumerate(REG_FIELDS):
        word = 0
        pos = 32
        fm: Dict[str, int] = {}
        for name, width in reg_def:
            pos -= width
            if name.startswith("reserved") or name in RESERVED:
                continue
            v = int(fields.get(name, 0))
            if v < 0:
                v &= (1 << width) - 1
            word |= (v & ((1 << width) - 1)) << pos
            fm[name] = v
        words.append(word & 0xFFFFFFFF)
        field_map[ri] = fm
    return words, field_map


# ===========================================================================
# Address + tiling helpers
# ===========================================================================
def split_addr(addr: int) -> Tuple[int, int]:
    """64-bit HBM address -> (high16 = bits47:32, low32)."""
    return (addr >> 32) & 0xFFFF, addr & 0xFFFFFFFF


def mat_a_config_by_arow(a_row: int) -> Tuple[int, int]:
    """MatrixA tiling for vec*matrix: driven by A's row count."""
    if a_row > 0 and a_row % 8 == 0:
        return a_row // 8 - 1, 8
    return a_row // 8, a_row % 8


def mat_a_config_by_brow(b_row: int) -> Tuple[int, int]:
    """MatrixA tiling for matrix*matrix: driven by B_row (== A's col count)."""
    if b_row > 0 and b_row % 8 == 0:
        return b_row // 8 - 1, 8
    return b_row // 8, b_row % 8


def mat_b_config(b_col: int) -> Tuple[int, int]:
    """MatrixB tiling: always driven by B_col, granularity 64."""
    if b_col > 0 and b_col % 64 == 0:
        return b_col // 64 - 1, 64
    return b_col // 64, b_col % 64


def vu_pingpong(total_elems: int) -> Tuple[int, int]:
    """VU large-vector ping-pong split, granularity 8192 elements."""
    if total_elems <= 0:
        return 0, 0
    return (total_elems - 1) // 8192, (total_elems - 1) % 8192 + 1

def round_up_64(x: int) -> int:
    """Round up to the next multiple of 64 (firmware: (x+63) & ~63)."""
    return (x + 63) & ~63

# ===========================================================================
# Instruction object
# ===========================================================================
# Instruction 增加 field_map
@dataclass
class Instruction:
    op: str
    words: List[int]
    field_map: Dict[int, Dict[str, int]] = field(default_factory=dict)  # 新增
    comment: str = ""
    reads: List[str] = field(default_factory=list)
    writes: List[str] = field(default_factory=list)
    core: int = 0
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.words) != NUM_REGS:
            raise ValueError(f"instruction needs {NUM_REGS} words, got {len(self.words)}")

    # -- serialisation -----------------------------------------------------
    def hex_words(self) -> List[str]:
        return [f"0x{w:08X}" for w in self.words]

    def to_bin_le(self) -> bytes:
        """64 bytes, little-endian per word (word0 first) -- the .bin payload."""
        out = bytearray()
        for w in self.words:
            out += int(w & 0xFFFFFFFF).to_bytes(4, "little")
        return bytes(out)

    def to_bitstring(self) -> str:
        """512-bit MSB-first binary string (word0 is the top 32 bits)."""
        return "".join(f"{w:032b}" for w in self.words)