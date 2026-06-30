"""
verify_isa.py -- prove the encoder reproduces the hand-written reference
firmware (the FP16->INT8 -> MatMul smoke test in compilation.md) bit-for-bit.

Run:  python -m llmc.verify_isa     (or)   python verify_isa.py
Exit code 0 and "PASS" iff every one of the 32 register words matches.
"""

from __future__ import annotations

from . import lowering
from .isa import REG_NAMES

# Golden words copied verbatim from the reference .c (SetReg literals).
GOLDEN_FP16_INT8 = [
    0x00100041, 0x00000000, 0x00010000,
    0x00000041, 0x00000000, 0x00000000,
    0x00000042, 0x00000000, 0x00008000,
    0x00000000, 0x00000000,
    0x08000043, 0x00000000,
    0x00032000, 0x00000000, 0x0000320C,
]

GOLDEN_MATMUL = [
    0x3C000042, 0x00000000, 0x00000040,
    0x3C000041, 0x00010000, 0x00002000,
    0x00000041, 0x00030000, 0x00000100,
    0x00000041, 0x00020000,
    0x00400041, 0x00000000,
    0x00000001, 0x00010040, 0x00000084,
]


def _diff(name, got, want):
    ok = got == want
    print(f"\n=== {name}: {'PASS' if ok else 'FAIL'} ===")
    if not ok:
        for rn, g, w in zip(REG_NAMES, got, want):
            flag = "" if g == w else "  <-- mismatch"
            print(f"  {rn:14s} got 0x{g:08X}  want 0x{w:08X}{flag}")
    return ok


def main() -> int:
    # Instruction 1: FP16->INT8, 16 x 2048, scale parked at 0x4300000000.
    i1 = lowering.fp16_to_int8(
        addr_in=0x4100000000, addr_out=0x4200000000, addr_scale=0x4300000000,
        rows=16, cols=2048, put_scale="scaleA", next_type=0,
    )
    # Instruction 2: VecMatMul 1x64 * 64x128; out-scale slot parked at 0x4100000000.
    i2 = lowering.vec_matmul(
        addr_a=0x4200000000, addr_b=0x4100010000, addr_c=0x4100030000,
        addr_b_scale=0x4100020000, addr_out_scale=0x4100000000,
        a_row=1, b_row=64, b_col=128,
        get_scale="scaleA", put_scale="none", next_type=1,
    )

    ok = True
    ok &= _diff("FP16_INT8", i1.words, GOLDEN_FP16_INT8)
    ok &= _diff("VEC_MATMUL", i2.words, GOLDEN_MATMUL)

    print("\n" + ("=" * 48))
    print("RESULT:", "PASS -- encoder is bit-exact." if ok else "FAIL")
    print("=" * 48)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
