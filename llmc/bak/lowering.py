"""
lowering.py -- operator builders.

Each function takes high-level arguments (HBM addresses, shapes, scale routing,
symbolic read/write tensor names) and returns one isa.Instruction with its 16
register words filled in exactly the way the reference firmware does.

Two conventions coexist in the source material:
  (A) the *standalone* per-op test harness (gen_script_v2.py), and
  (B) the *full-model* firmware (the big reference .c).
Where they disagree (e.g. SiLU's `dimension_num`, the MU `Param1/2` placeholder,
the input-3 register in setVecMatrix_Mul), we follow the documented field meaning
and note it.  See README "Known discrepancies".
"""

from __future__ import annotations

from typing import List, Optional

from . import isa
from .isa import (
    Instruction, encode_fields, split_addr,
    mat_a_config_by_arow, mat_a_config_by_brow, mat_b_config, vu_pingpong,
    CAL_TYPE, GET_SCALE, PUT_SCALE, FP16_ONE,
)

FP16 = 2   # bytes
INT8 = 1   # bytes


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------
def fp16_to_int8(addr_in: int, addr_out: int, addr_scale: int,
                 rows: int, cols: int,
                 put_scale: str = "scaleA", next_type: int = 0,
                 reads=None, writes=None, core: int = 0) -> Instruction:
    """FP16[rows,cols] -> INT8[rows,cols] (+ per-row scale).  Cal_Type 0x32."""
    in_h, in_l = split_addr(addr_in)
    out_h, out_l = split_addr(addr_out)
    sc_h, sc_l = split_addr(addr_scale)
    total = rows * cols
    ma_h, ma_l = vu_pingpong(total)
    f = {
        "Param1": rows,                       # K_V == row count (firmware: "row/K_V")
        "Input_addr1_H": in_h, "Input_addr1_L": in_l, "Input_len1": total * FP16,
        "Input_addr2_H": in_h, "Input_addr2_L": in_l, "Input_len2": 0,
        "Output_addr_H": out_h, "Output_addr_L": out_l, "Output_len": total * INT8,
        "Martix_b_row_config": cols,
        "Output_scale_addr4_H": sc_h, "Output_scale_addr4_L": sc_l,
        "Matrix_configA_H": ma_h, "Matrix_configA_L": ma_l,
        "Cal_Type": CAL_TYPE["FP16_INT8"],
        "next_type": next_type, "get_scale": 0, "put_scale": PUT_SCALE[put_scale],
        "Config_avail": 1,
    }
    words, field_map = encode_fields(f)
    return Instruction("FP16_INT8", words, field_map=field_map,
                       comment=f"FP16->INT8 [{rows}x{cols}]",
                       reads=reads or [], writes=writes or [], core=core)


# ---------------------------------------------------------------------------
# Matrix engine
# ---------------------------------------------------------------------------
def _matmul_common(addr_a, addr_b, addr_c, addr_b_scale, addr_out_scale,
                   a_row, b_row, b_col, ma_h, ma_l,
                   get_scale, put_scale, next_type, param1, param2, param3):
    a_h, a_l = split_addr(addr_a)
    b_h, b_l = split_addr(addr_b)
    c_h, c_l = split_addr(addr_c)
    s_h, s_l = split_addr(addr_b_scale)
    os_h, os_l = split_addr(addr_out_scale if addr_out_scale is not None else addr_b_scale)
    mb_h, mb_l = mat_b_config(b_col)
    return {
        "Param1": param1, "Input_addr1_H": a_h, "Input_addr1_L": a_l,
        "Input_len1": a_row * b_row,                  # INT8 operands
        "Param2": param2, "Input_addr2_H": b_h, "Input_addr2_L": b_l,
        "Input_len2": b_row * b_col,
        "Param3": param3, "Output_addr_H": c_h, "Output_addr_L": c_l,
        "Output_len": a_row * b_col * FP16,           # FP16 result
        "Input_addr3_H": s_h, "Input_addr3_L": s_l,
        "Martix_b_row_config": b_row,
        "Output_scale_addr4_H": os_h, "Output_scale_addr4_L": os_l,
        "Matrix_configA_H": ma_h, "Matrix_configA_L": ma_l,
        "Matrix_configB_H": mb_h, "Matrix_configB_L": mb_l,
        "Cal_Type": CAL_TYPE["VEC_MATMUL"],
        "next_type": next_type,
        "get_scale": GET_SCALE[get_scale] if isinstance(get_scale, str) else get_scale,
        "put_scale": PUT_SCALE[put_scale] if isinstance(put_scale, str) else put_scale,
        "Config_avail": 1,
    }


def vec_matmul(addr_a, addr_b, addr_c, addr_b_scale,
               a_row, b_row, b_col,
               addr_out_scale: Optional[int] = None,
               get_scale="scaleA", put_scale="none", next_type=1,
               param1=FP16_ONE, param2=FP16_ONE, param3=0,
               reads=None, writes=None, core=0) -> Instruction:
    """A[a_row,b_row] (INT8) x B[b_row,b_col] (INT8) -> C[a_row,b_col] (FP16).
    MatrixA tiling driven by a_row (vector*matrix path)."""
    ma_h, ma_l = mat_a_config_by_arow(a_row)
    f = _matmul_common(addr_a, addr_b, addr_c, addr_b_scale, addr_out_scale,
                       a_row, b_row, b_col, ma_h, ma_l,
                       get_scale, put_scale, next_type, param1, param2, param3)
    words, field_map = encode_fields(f)   # 改为接收两个返回值
    return Instruction("VEC_MATMUL", words, field_map=field_map,
                       comment=f"VecMatMul [{a_row}x{b_row}]*[{b_row}x{b_col}]",
                       reads=reads or [], writes=writes or [], core=core)


def matmul(addr_a, addr_b, addr_c, addr_b_scale,
           a_row, b_row, b_col,
           addr_out_scale: Optional[int] = None,
           get_scale="scaleA", put_scale="none", next_type=1,
           param1=FP16_ONE, param2=FP16_ONE, param3=0,
           reads=None, writes=None, core=0) -> Instruction:
    """Matrix*matrix path: MatrixA tiling driven by b_row (== A's col count)."""
    ma_h, ma_l = mat_a_config_by_brow(b_row)
    f = _matmul_common(addr_a, addr_b, addr_c, addr_b_scale, addr_out_scale,
                       a_row, b_row, b_col, ma_h, ma_l,
                       get_scale, put_scale, next_type, param1, param2, param3)
    words, field_map = encode_fields(f)   # 改为接收两个返回值
    return Instruction("MATMUL", words, field_map=field_map,
                       comment=f"MatMul [{a_row}x{b_row}]*[{b_row}x{b_col}]",
                       reads=reads or [], writes=writes or [], core=core)


# ---------------------------------------------------------------------------
# Vector engine -- shared scaffold for the elementwise / norm family
# ---------------------------------------------------------------------------
def _vu_two_src(cal_type, addr1, addr2, addr_out,
                elem1, dim, elem2, put_scale, next_type,
                set_out_len=True, addr_out_scale=None,
                param1=0, param2=0, param3=0, out_bytes=FP16):
    a1h, a1l = split_addr(addr1)
    a2h, a2l = split_addr(addr2)
    oh, ol = split_addr(addr_out)
    osh, osl = split_addr(addr_out_scale) if addr_out_scale is not None else (0, 0)
    maa_h, maa_l = vu_pingpong(elem1)
    mab_h, mab_l = vu_pingpong(elem2)
    f = {
        "Param1": param1, "Input_addr1_H": a1h, "Input_addr1_L": a1l,
        "Input_len1": elem1 * FP16,
        "Param2": param2, "Input_addr2_H": a2h, "Input_addr2_L": a2l,
        "Input_len2": elem2 * FP16,
        "Param3": param3, "Output_addr_H": oh, "Output_addr_L": ol,
        "Output_len": (elem1 * FP16) if set_out_len else 0,
        "Martix_b_row_config": dim,                    # OTHER_PARA1 high = dimension_num
        "Output_scale_addr4_H": osh, "Output_scale_addr4_L": osl,
        "Matrix_configA_H": maa_h, "Matrix_configA_L": maa_l,
        "Matrix_configB_H": mab_h, "Matrix_configB_L": mab_l,
        "Cal_Type": cal_type,
        "Output_len": (elem1 * out_bytes) if set_out_len else 0,
        "next_type": next_type, "get_scale": 0,
        "put_scale": PUT_SCALE[put_scale] if isinstance(put_scale, str) else put_scale,
        "Config_avail": 1,
    }
    return f


def silu(addr_in, addr_out, elem, dim, put_scale="none", next_type=1,
         reads=None, writes=None, core=0) -> Instruction:
    """SiLU activation, FP16->FP16.  Cal_Type 0x01.  Single source."""
    a1h, a1l = split_addr(addr_in)
    oh, ol = split_addr(addr_out)
    ma_h, ma_l = vu_pingpong(elem)
    f = {
        "Input_addr1_H": a1h, "Input_addr1_L": a1l, "Input_len1": elem * FP16,
        "Output_addr_H": oh, "Output_addr_L": ol, "Output_len": elem * FP16,
        "Martix_b_row_config": dim,
        "Matrix_configA_H": ma_h, "Matrix_configA_L": ma_l,
        "Cal_Type": CAL_TYPE["SILU"],
        "next_type": next_type, "get_scale": 0,
        "put_scale": PUT_SCALE[put_scale] if isinstance(put_scale, str) else put_scale,
        "Config_avail": 1,
    }
    words, field_map = encode_fields(f)
    return Instruction("SILU", words, field_map=field_map, comment=f"SiLU [{elem} elems]",
                       reads=reads or [], writes=writes or [], core=core)


def gelu(addr_in, addr_out, elem, dim, put_scale="none", next_type=1,
         reads=None, writes=None, core=0) -> Instruction:
    """GeLU activation, FP16->FP16.  Cal_Type 0x11."""
    a1h, a1l = split_addr(addr_in)
    oh, ol = split_addr(addr_out)
    ma_h, ma_l = vu_pingpong(elem)
    f = {
        "Input_addr1_H": a1h, "Input_addr1_L": a1l, "Input_len1": elem * FP16,
        "Output_addr_H": oh, "Output_addr_L": ol, "Output_len": elem * FP16,
        "Martix_b_row_config": dim,
        "Matrix_configA_H": ma_h, "Matrix_configA_L": ma_l,
        "Cal_Type": CAL_TYPE["GELU"],
        "next_type": next_type, "get_scale": 0,
        "put_scale": PUT_SCALE[put_scale] if isinstance(put_scale, str) else put_scale,
        "Config_avail": 1,
    }
    words, field_map = encode_fields(f)
    return Instruction("GELU", words, field_map=field_map, comment=f"GeLU [{elem} elems]",
                       reads=reads or [], writes=writes or [], core=core)


def vu_mul(addr1, addr2, addr_out, elem1, dim, elem2,
           put_scale="scaleA", next_type=0, reads=None, writes=None, core=0) -> Instruction:
    """Element-wise multiply (also the RoPE multiply datapath).  Cal_Type 0x31.
    NOTE: reference firmware leaves Output_len = 0 for VU_MUL; preserved here."""
    f = _vu_two_src(CAL_TYPE["VU_MUL"], addr1, addr2, addr_out,
                    elem1, dim, elem2, put_scale, next_type, set_out_len=False)
    words, field_map = encode_fields(f)
    return Instruction("VU_MUL", words, field_map=field_map, comment=f"VU_MUL [{elem1} elems]",
                       reads=reads or [], writes=writes or [], core=core)


def vu_add(addr1, addr2, addr_out, elem1, dim, elem2,
           put_scale="none", next_type=1, addr_out_scale=None,
           reads=None, writes=None, core=0) -> Instruction:
    """Element-wise add (also the residual datapath).  Cal_Type 0x41."""
    f = _vu_two_src(CAL_TYPE["VU_ADD"], addr1, addr2, addr_out,
                    elem1, dim, elem2, put_scale, next_type, set_out_len=True)
    words, field_map = encode_fields(f)
    return Instruction("VU_ADD", words, field_map=field_map, comment=f"VU_ADD [{elem1} elems]",
                       reads=reads or [], writes=writes or [], core=core)


def residual(addr1, addr2, addr_out, elem1, dim, elem2,
             put_scale="none", next_type=1, reads=None, writes=None, core=0) -> Instruction:
    """Residual add.  Cal_Type 0x41 (same datapath as VU_ADD)."""
    f = _vu_two_src(CAL_TYPE["RESIDUAL"], addr1, addr2, addr_out,
                    elem1, dim, elem2, put_scale, next_type, set_out_len=True)
    words, field_map = encode_fields(f)
    return Instruction("RESIDUAL", words, field_map=field_map, comment=f"Residual [{elem1} elems]",
                       reads=reads or [], writes=writes or [], core=core)


def rmsnorm(addr_in, addr_orig, addr_out, elem1, dim, elem2,
            e: int = 0, r: int = 0, put_scale="scaleAR", next_type=0,
            reads=None, writes=None, core=0, out_bytes=FP16) -> Instruction:
    """RMSNorm.  Cal_Type 0x51.  `e`/`r` are the firmware's two scalar params
    (epsilon / reciprocal-related); routed via Param1/Param2."""
    f = _vu_two_src(CAL_TYPE["RMSNORM"], addr_in, addr_orig, addr_out,
                    elem1, dim, elem2, put_scale, next_type, set_out_len=True,
                    param1=e, param2=r, out_bytes=INT8)
    words, field_map = encode_fields(f)
    return Instruction("RMSNORM", words, field_map=field_map, comment=f"RMSNorm [{elem1} elems]",
                       reads=reads or [], writes=writes or [], core=core)


# --- attention-family builders (structurally present; param maps marked) ----
def softmax(addr1, addr2, addr_out, elem1, dim, elem2, vld_len: int,
            put_scale="scaleA", next_type=0, reads=None, writes=None, core=0, out_bytes=FP16) -> Instruction:
    """Softmax.  Cal_Type 0x21.  `vld_len` = real (unpadded) length so the
    accumulation tree ignores the 64-padding.  Routed via Param1."""
    f = _vu_two_src(CAL_TYPE["SOFTMAX"], addr1, addr2, addr_out,
                    elem1, dim, elem2, put_scale, next_type, set_out_len=True,
                    param1=vld_len, out_bytes=INT8)
    words, field_map = encode_fields(f)
    return Instruction("SOFTMAX", words, field_map=field_map, comment=f"Softmax vld={vld_len}",
                       reads=reads or [], writes=writes or [], core=core)


def rope_add(addr1, addr2, addr_out, addr_out_scale, elem1, dim, elem2,
             put_scale="hbm", next_type=0, reads=None, writes=None, core=0) -> Instruction:
    """RoPE add stage that may write an output scale to HBM.  Cal_Type 0x41 path."""
    f = _vu_two_src(CAL_TYPE["VU_ADD"], addr1, addr2, addr_out,
                    elem1, dim, elem2, put_scale, next_type, set_out_len=True,
                    addr_out_scale=addr_out_scale)
    words, field_map = encode_fields(f)
    return Instruction("ROPE_ADD", words, field_map=field_map, comment="RoPE add",
                       reads=reads or [], writes=writes or [], core=core)


def swap(addr_in, addr_out, elem, token_num, dim,
         put_scale="none", next_type=1, reads=None, writes=None, core=0) -> Instruction:
    """Swap (reverse negative half).  Cal_Type 0x61.
    Uses INPUT_ADDR2 as the single source (firmware convention)."""
    a2h, a2l = split_addr(addr_in)
    oh, ol = split_addr(addr_out)
    mb_h, mb_l = vu_pingpong(elem)
    f = {
        "Param1": token_num,
        "Input_addr2_H": a2h, "Input_addr2_L": a2l,
        "Input_len2": elem * FP16,
        "Param3": 0,
        "Output_addr_H": oh, "Output_addr_L": ol,
        "Output_len": elem * FP16,
        "Martix_b_row_config": dim,
        "Matrix_configB_H": mb_h, "Matrix_configB_L": mb_l,
        "Cal_Type": CAL_TYPE["SWAP"],
        "next_type": next_type,
        "get_scale": 0,
        "put_scale": PUT_SCALE[put_scale] if isinstance(put_scale, str) else put_scale,
        "Config_avail": 1,
    }
    words, field_map = encode_fields(f)
    return Instruction("SWAP", words, field_map=field_map, comment=f"Swap [{elem} elems]",
                       reads=reads or [], writes=writes or [], core=core)


def rearrange(addr_in, addr_out, elem_in, elem_out, token_num, dim, heads,
              put_scale="none", next_type=0, reads=None, writes=None, core=0) -> Instruction:
    """Rearrange (pad to 64).  Cal_Type 0x02.
    Source via INPUT_ADDR2; token/dim/heads via Param1/2/3."""
    a2h, a2l = split_addr(addr_in)
    oh, ol = split_addr(addr_out)
    mb_h, mb_l = vu_pingpong(elem_in)
    f = {
        "Param1": token_num & 0xFFFF,
        "Param2": dim & 0xFFFF,
        "Param3": heads & 0xFFFF,
        "Input_addr2_H": a2h, "Input_addr2_L": a2l,
        "Input_len2": elem_in * FP16,
        "Output_addr_H": oh, "Output_addr_L": ol,
        "Output_len": elem_out * INT8,
        "Martix_b_row_config": dim,
        "Matrix_configB_H": mb_h, "Matrix_configB_L": mb_l,
        "Cal_Type": CAL_TYPE["REARRANGE"],
        "next_type": next_type,
        "get_scale": 0,
        "put_scale": PUT_SCALE[put_scale] if isinstance(put_scale, str) else put_scale,
        "Config_avail": 1,
    }
    words, field_map = encode_fields(f)
    return Instruction("REARRANGE", words, field_map=field_map, comment=f"Rearrange {token_num}x{dim} -> {elem_out}",
                       reads=reads or [], writes=writes or [], core=core)


def transpose(addr_in, addr_out, addr_out_scale, elem_in, elem_out,
              token_num, dim, heads,
              put_scale="hbm", next_type=0, reads=None, writes=None, core=0) -> Instruction:
    """Transpose (pad to 64).  Cal_Type 0x12.
    Source via INPUT_ADDR2; writes optional scale to HBM."""
    a2h, a2l = split_addr(addr_in)
    oh, ol = split_addr(addr_out)
    osh, osl = split_addr(addr_out_scale) if addr_out_scale is not None else (0, 0)
    mb_h, mb_l = vu_pingpong(elem_in)
    f = {
        "Param1": token_num & 0xFFFF,
        "Param2": dim & 0xFFFF,
        "Param3": heads & 0xFFFF,
        "Input_addr2_H": a2h, "Input_addr2_L": a2l,
        "Input_len2": elem_in * FP16,
        "Output_addr_H": oh, "Output_addr_L": ol,
        "Output_len": elem_out * INT8,
        "Martix_b_row_config": dim,
        "Output_scale_addr4_H": osh, "Output_scale_addr4_L": osl,
        "Matrix_configB_H": mb_h, "Matrix_configB_L": mb_l,
        "Cal_Type": CAL_TYPE["TRANSPOSE"],
        "next_type": next_type,
        "get_scale": 0,
        "put_scale": PUT_SCALE[put_scale] if isinstance(put_scale, str) else put_scale,
        "Config_avail": 1,
    }
    words, field_map = encode_fields(f)
    return Instruction("TRANSPOSE", words, field_map=field_map, comment=f"Transpose {token_num}x{dim} -> {elem_out}",
                       reads=reads or [], writes=writes or [], core=core)


def vu_mask(addr1, addr2, addr_out, elem1, dim, elem2,
            put_scale="none", next_type=1, reads=None, writes=None, core=0) -> Instruction:
    """Causal mask.  Cal_Type 0x41 (shared with VU_ADD / RESIDUAL)."""
    f = _vu_two_src(CAL_TYPE["VU_MASK"], addr1, addr2, addr_out,
                    elem1, dim, elem2, put_scale, next_type, set_out_len=True)
    words, field_map = encode_fields(f)
    return Instruction("VU_MASK", words, field_map=field_map, comment=f"VU_MASK [{elem1} elems]",
                       reads=reads or [], writes=writes or [], core=core)


def concat(addr_in, addr_out, elem2, out_elem, token_num, dim, heads,
           put_scale="none", next_type=0, reads=None, writes=None, core=0) -> Instruction:
    """Multi-head concatenation.  Cal_Type 0x22.  token/dim/heads via Param1/2/3."""
    a2h, a2l = split_addr(addr_in)
    oh, ol = split_addr(addr_out)
    f = {
        "Param1": token_num & 0xFFFF, "Input_addr2_H": a2h,
        "Param2": dim & 0xFFFF, "Param3": heads & 0xFFFF,
        "Input_addr2_L": a2l, "Input_len2": elem2 * FP16,
        "Output_addr_H": oh, "Output_addr_L": ol, "Output_len": out_elem * INT8,
        "Cal_Type": CAL_TYPE["CONCAT"],
        "next_type": next_type, "get_scale": 0,
        "put_scale": PUT_SCALE[put_scale] if isinstance(put_scale, str) else put_scale,
        "Config_avail": 1,
    }
    words, field_map = encode_fields(f)
    return Instruction("CONCAT", words, field_map=field_map, comment=f"Concat {heads} heads",
                       reads=reads or [], writes=writes or [], core=core)