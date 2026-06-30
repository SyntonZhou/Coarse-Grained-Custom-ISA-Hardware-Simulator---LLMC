#!/usr/bin/env python3
"""Generate tiny TinyRISC-V smoke tests for the accelerator MMIO path.

The emitted *.c files are the things to compile for TinyRISC-V.  The
program_words_*.bin files are macro-instruction data streams for an interpreter;
they are intentionally not executable RV32 code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import isa, lowering


DEFAULT_LINUX_ROOT = "/home/dcy/桌面/linux-kernel/scripts/qwen"
DEV = "/dev/qdma01000-MM-0"

ADDR_INPUT = 0x4100000000
ADDR_WEIGHT = 0x4100010000
ADDR_WEIGHT_SCALE = 0x4100020000
ADDR_MATMUL_OUT = 0x4100030000
ADDR_QUANT_OUT = 0x4200000000
ADDR_QUANT_SCALE = 0x4300000000


def to_linux_path(path: Path, linux_root: str) -> str:
    return linux_root.rstrip("/") + "/" + path.as_posix().lstrip("./")


def write_bin(path: Path, data: np.ndarray, dtype) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = data.astype(dtype)
    arr.tofile(path)
    return int(arr.nbytes)


def macro_bin(words: list[int], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for w in words:
            f.write(int(w & 0xFFFFFFFF).to_bytes(4, "little"))


def try_disasm_as_rv32(path: Path, out: Path):
    data = path.read_bytes()
    lines = [
        f"; {path.name} interpreted as raw RV32 little-endian code",
        "; This file is macro-instruction data, so illegal/odd decoded ops are expected.",
        "",
    ]
    try:
        from capstone import Cs, CS_ARCH_RISCV, CS_MODE_LITTLE_ENDIAN, CS_MODE_RISCV32
        md = Cs(CS_ARCH_RISCV, CS_MODE_RISCV32 | CS_MODE_LITTLE_ENDIAN)
        for off in range(0, len(data), 4):
            chunk = data[off:off + 4]
            word = int.from_bytes(chunk, "little")
            ins = list(md.disasm(chunk, 0x4000000000 + off))
            if ins and ins[0].size == 4:
                asm = f"{ins[0].mnemonic}\t{ins[0].op_str}".rstrip()
            else:
                asm = f".word\t0x{word:08X}    ; illegal/undecoded as RV32"
            lines.append(f"0x{0x4000000000 + off:010X}:  {word:08X}    {asm}")
    except Exception as ex:
        for off in range(0, len(data), 4):
            word = int.from_bytes(data[off:off + 4], "little")
            lines.append(f"0x{0x4000000000 + off:010X}:  .word 0x{word:08X}")
        lines.append(f"; capstone unavailable: {ex!r}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


REG_MACROS = [
    ("INPUT_ADDR1H", 0x80000000),
    ("INPUT_ADDR1L", 0x80000004),
    ("INPUT_LEN1", 0x80000008),
    ("INPUT_ADDR2H", 0x8000000C),
    ("INPUT_ADDR2L", 0x80000010),
    ("INPUT_LEN2", 0x80000014),
    ("OUTPUT_ADDRH", 0x80000018),
    ("OUTPUT_ADDRL", 0x8000001C),
    ("OUTPUT_LEN", 0x80000020),
    ("INPUT_ADDR3H", 0x80000024),
    ("INPUT_ADDR3L", 0x80000028),
    ("OTHER_PARA1", 0x8000002C),
    ("OTHER_PARA2", 0x80000030),
    ("MATRIXA_CONFIG", 0x80000034),
    ("MATRIXB_CONFIG", 0x80000038),
    ("PARA_CONFIG", 0x8000003C),
]


def emit_unrolled_c(path: Path, instrs: list[tuple[str, list[int], str]]):
    lines = [
        "#include <stdint.h>",
        "",
        "#define _P2V(addr) (addr)",
        "",
        "#define SetReg(_x,_y) do{ (*(volatile uint32_t*)(_P2V(_x))) = (uint32_t)(_y); }while(0)",
        "#define ReadReg(_x,_y) do{ (_y) = *(volatile uint32_t*)(_P2V(_x)); }while(0)",
        "",
    ]
    for name, addr in REG_MACROS:
        lines.append(f"#define {name:<15} 0x{addr:08X}")
    lines.extend([
        "",
        "static void wait_done(void)",
        "{",
        "    while (1) {",
        "        uint32_t finish_reg_val = 0;",
        "        ReadReg(PARA_CONFIG, finish_reg_val);",
        "",
        "        if ((finish_reg_val & 0x00000007u) == 0x00000004u) {",
        "            break;",
        "        }",
        "    }",
        "}",
        "",
        "int main(void)",
        "{",
    ])
    for idx, (name, words, comment) in enumerate(instrs):
        lines.append(f"    /* Instruction {idx}: {name} -- {comment} */")
        for (reg_name, _addr), word in zip(REG_MACROS, words):
            lines.append(f"    SetReg({reg_name}, 0x{word:08X});")
        lines.append("    wait_done();")
        lines.append("")
    lines.extend([
        "    return 0;",
        "}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def emit_load_script(path: Path, outdir: Path, linux_root: str):
    data_dir = outdir / "data"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# QDMA bring-up",
        "echo 0 > /sys/bus/pci/devices/0000:01:00.0/sriov_numvfs",
        "echo 0 > /sys/bus/pci/devices/0000:01:00.0/qdma/qmax",
        "echo 1 > /sys/bus/pci/devices/0000:01:00.0/remove",
        "sleep 2",
        "echo 1 > /sys/bus/pci/rescan",
        "sleep 2",
        "echo 8 > /sys/bus/pci/devices/0000:01:00.0/qdma/qmax",
        "echo 3 > /sys/bus/pci/devices/0000:01:00.0/sriov_numvfs",
        "sleep 2",
        "dma-ctl qdma01000 q add idx 0 mode mm dir bi",
        "dma-ctl qdma01000 q start idx 0 dir bi",
        "",
        f"dma-to-device -d {DEV} -a 0x{ADDR_INPUT:010X} -f {to_linux_path(data_dir / 'input_fp16_16x2048.bin', linux_root)} -s 65536",
        f"dma-to-device -d {DEV} -a 0x{ADDR_WEIGHT:010X} -f {to_linux_path(data_dir / 'weight_int8_64x128.bin', linux_root)} -s 8192",
        f"dma-to-device -d {DEV} -a 0x{ADDR_WEIGHT_SCALE:010X} -f {to_linux_path(data_dir / 'weight_scale_fp16_128.bin', linux_root)} -s 256",
        "",
        "# Compile one of firmware_*.c with your normal TinyRISC-V flow and run that executable.",
        "# program_words_*.bin is NOT an executable; it is only for an interpreter firmware.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def emit_readback_script(path: Path, linux_root: str):
    rb = Path("debug_readback/tinyriscv_smoke")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"mkdir -p {to_linux_path(rb, linux_root)}",
        f"dma-from-device -d {DEV} -a 0x{ADDR_QUANT_OUT:010X} -f {to_linux_path(rb / 'quant_out_int8_16x2048.bin', linux_root)} -s 32768",
        f"dma-from-device -d {DEV} -a 0x{ADDR_QUANT_SCALE:010X} -f {to_linux_path(rb / 'quant_scale_fp16_16.bin', linux_root)} -s 32",
        f"dma-from-device -d {DEV} -a 0x{ADDR_MATMUL_OUT:010X} -f {to_linux_path(rb / 'matmul_out_fp16_1x128.bin', linux_root)} -s 256",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/tinyriscv_smoke")
    ap.add_argument("--linux-root", default=DEFAULT_LINUX_ROOT)
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    data_dir = outdir / "data"
    outdir.mkdir(parents=True, exist_ok=True)

    x = (np.arange(16 * 2048, dtype=np.float32).reshape(16, 2048) % 251) / 251.0
    w = ((np.arange(64 * 128, dtype=np.int16).reshape(64, 128) % 17) - 8).astype(np.int8)
    s = np.ones((128,), dtype=np.float16)
    write_bin(data_dir / "input_fp16_16x2048.bin", x, np.float16)
    write_bin(data_dir / "weight_int8_64x128.bin", w, np.int8)
    write_bin(data_dir / "weight_scale_fp16_128.bin", s, np.float16)

    q = lowering.fp16_to_int8(
        addr_in=ADDR_INPUT, addr_out=ADDR_QUANT_OUT, addr_scale=ADDR_QUANT_SCALE,
        rows=16, cols=2048, put_scale="scaleA", next_type=0,
    )
    mm = lowering.vec_matmul(
        addr_a=ADDR_QUANT_OUT, addr_b=ADDR_WEIGHT, addr_c=ADDR_MATMUL_OUT,
        addr_b_scale=ADDR_WEIGHT_SCALE, addr_out_scale=ADDR_INPUT,
        a_row=1, b_row=64, b_col=128,
        get_scale="scaleA", put_scale="none", next_type=1,
    )

    emit_unrolled_c(outdir / "firmware_01_fp16_int8_unrolled.c",
                    [("fp16_int8", q.words, q.comment)])
    emit_unrolled_c(outdir / "firmware_02_fp16_int8_vecmatmul_unrolled.c",
                    [("fp16_int8", q.words, q.comment),
                     ("vecmatmul", mm.words, mm.comment)])

    macro_bin(q.words, outdir / "program_words_01_fp16_int8.bin")
    macro_bin(q.words + mm.words, outdir / "program_words_02_fp16_int8_vecmatmul.bin")
    try_disasm_as_rv32(outdir / "program_words_01_fp16_int8.bin",
                       outdir / "program_words_01_as_rv32_disasm.s")
    try_disasm_as_rv32(outdir / "program_words_02_fp16_int8_vecmatmul.bin",
                       outdir / "program_words_02_as_rv32_disasm.s")

    emit_load_script(outdir / "dma_load_smoke_data.sh", outdir, args.linux_root)
    emit_readback_script(outdir / "dma_readback_smoke.sh", args.linux_root)

    (outdir / "README.txt").write_text(
        "Use firmware_01_fp16_int8_unrolled.c first. Compile it with your known-good "
        "TinyRISC-V flow, then load/run that executable the normal way.\n"
        "program_words_*.bin is macro-instruction data, not RV32 executable code. "
        "The *_as_rv32_disasm.s files demonstrate why direct execution fails.\n",
        encoding="utf-8",
    )

    print(f"wrote {outdir}")
    print(f"  C executable sources: firmware_01_fp16_int8_unrolled.c, firmware_02_fp16_int8_vecmatmul_unrolled.c")
    print(f"  non-executable macro data: program_words_*.bin")
    print(f"  disasm reports: program_words_*_as_rv32_disasm.s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
