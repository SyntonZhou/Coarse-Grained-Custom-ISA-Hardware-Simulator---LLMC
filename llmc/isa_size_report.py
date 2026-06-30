#!/usr/bin/env python3
"""
isa_size_report.py -- compare the control-core *code* and *data* footprint of a
hand-written baseline against llmc's machine-generated output, and explain the
`.text will not fit in region IMEM` failure.

It auto-detects each input by type:

  *.elf  -> compiled RISC-V firmware. Reports per-section sizes and an opcode
            histogram (by mnemonic and by instruction *class*: load/store/alu/
            branch/jump/csr/muldiv/float/other), plus an IMEM-fit check.
            Uses pyelftools+capstone if installed; otherwise shells out to
            `objdump` (any GNU binutils, including riscv*-objdump).

  *.c    -> generated firmware.c. Counts SetReg / ReadReg / wait_done, attributes
            register writes to op types, and *estimates* the resulting .text size
            at -O0 and -O2 so you can predict the IMEM overflow without building.

  *.list -> program.list. Counts macro-instructions and their byte size (the HBM
            image), grouped by op type. This is the data-driven alternative to
            baking everything into .text.

Examples
  python isa_size_report.py out/firmware.c out/program.list --linker link.ld
  python isa_size_report.py baseline.elf generated.elf --imem 0x20000
  python isa_size_report.py firmware.elf --objdump riscv64-unknown-elf-objdump

The whole point: an ELF's `.text` lives in the (small) on-chip IMEM, whereas
program.bin/.list is *data* that lives in HBM. Fully unrolling every MMIO write
into straight-line C moves the entire program from HBM into IMEM, which is why
it overflows. This tool quantifies that gap.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# RISC-V mnemonic -> class table (covers RV32/64 IMAFD + common pseudo/compressed)
# --------------------------------------------------------------------------- #
_CLASS = {
    "load":   {"lb","lh","lw","ld","lbu","lhu","lwu","flw","fld","flh",
               "c.lw","c.ld","c.lwsp","c.ldsp","c.flw","c.fld","c.fldsp","c.flwsp"},
    "store":  {"sb","sh","sw","sd","fsw","fsd","fsh",
               "c.sw","c.sd","c.swsp","c.sdsp","c.fsw","c.fsd","c.fsdsp","c.fswsp"},
    "alu":    {"add","addi","addw","addiw","sub","subw","and","andi","or","ori",
               "xor","xori","sll","slli","slliw","sllw","srl","srli","srliw","srlw",
               "sra","srai","sraiw","sraw","slt","slti","sltu","sltiu","lui","auipc",
               "mv","li","nop","not","neg","negw","sext.w","seqz","snez","sltz","sgtz",
               "c.add","c.addi","c.addiw","c.addw","c.sub","c.subw","c.and","c.andi",
               "c.or","c.xor","c.mv","c.li","c.lui","c.slli","c.srli","c.srai",
               "c.addi16sp","c.addi4spn","c.nop","zext.b","zext.h","zext.w"},
    "branch": {"beq","bne","blt","bge","bltu","bgeu","beqz","bnez","blez","bgez",
               "bltz","bgtz","bgt","ble","bgtu","bleu","c.beqz","c.bnez"},
    "jump":   {"jal","jalr","j","jr","ret","call","tail","c.j","c.jal","c.jalr",
               "c.jr","c.ebreak"},
    "csr":    {"csrr","csrw","csrs","csrc","csrrw","csrrs","csrrc","csrrwi","csrrsi",
               "csrrci","csrwi","csrsi","csrci","ecall","ebreak","fence","fence.i",
               "fence.tso","sfence.vma","mret","sret","wfi","pause"},
    "muldiv": {"mul","mulh","mulhsu","mulhu","mulw","div","divu","divw","divuw",
               "rem","remu","remw","remuw"},
    "float":  {"fadd.s","fsub.s","fmul.s","fdiv.s","fsqrt.s","fmadd.s","fmsub.s",
               "fadd.d","fsub.d","fmul.d","fdiv.d","fmv.x.w","fmv.w.x","fcvt.s.w",
               "fcvt.w.s","fmax.s","fmin.s","fsgnj.s","feq.s","flt.s","fle.s"},
}
_MNEM2CLASS: Dict[str, str] = {}
for _c, _s in _CLASS.items():
    for _m in _s:
        _MNEM2CLASS[_m] = _c

# float catch-all: anything starting with 'f' and containing '.' we treat as float
def classify(mnem: str) -> str:
    mnem = mnem.lower()
    if mnem in _MNEM2CLASS:
        return _MNEM2CLASS[mnem]
    if mnem.startswith("f") and ("." in mnem):
        return "float"
    return "other"


_CAPSTONE_ARCH = {
    "riscv32": ("CS_ARCH_RISCV", "CS_MODE_RISCV32"),
    "riscv64": ("CS_ARCH_RISCV", "CS_MODE_RISCV64"),
    "x86_64":  ("CS_ARCH_X86",   "CS_MODE_64"),
    "arm64":   ("CS_ARCH_ARM64", "CS_MODE_ARM"),
}


def _human(n: int) -> str:
    f = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if f < 1024 or u == "GB":
            return f"{int(n):,} B" if u == "B" else f"{f:.1f} {u}"
        f /= 1024
    return f"{n} B"


# --------------------------------------------------------------------------- #
# size-string parser (for link.ld LENGTH = 0x20000 / 128K / 1M)
# --------------------------------------------------------------------------- #
def parse_size(s: str) -> Optional[int]:
    s = s.strip().rstrip(";").strip()
    m = re.fullmatch(r"(0[xX][0-9a-fA-F]+|\d+)\s*([KMG]?)", s)
    if not m:
        return None
    val = int(m.group(1), 0)
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[m.group(2)]
    return val * mult


def imem_from_linker(path: str, region: str = "IMEM") -> Optional[int]:
    txt = open(path, "r", errors="replace").read()
    # MEMORY { IMEM (rx) : ORIGIN = 0x..., LENGTH = 0x... }
    for m in re.finditer(
        r"(\w+)\s*\([^)]*\)\s*:\s*ORIGIN\s*=\s*[^,]+,\s*LENGTH\s*=\s*([0-9xXa-fA-F]+\s*[KMG]?)",
        txt,
    ):
        if m.group(1).upper() == region.upper():
            return parse_size(m.group(2))
    return None


# --------------------------------------------------------------------------- #
# ELF analysis
# --------------------------------------------------------------------------- #
def analyze_elf(path: str, arch: str, objdump: Optional[str]) -> Dict:
    res: Dict = {"file": path, "kind": "elf", "sections": OrderedDict(),
                 "mnem": Counter(), "klass": Counter(), "ninstr": 0,
                 "text_bytes": 0, "alloc_bytes": 0, "method": None}
    # --- prefer pyelftools (+capstone) ---
    try:
        from elftools.elf.elffile import ELFFile
        from elftools.elf.constants import SH_FLAGS
        have_pyelf = True
    except Exception:
        have_pyelf = False

    if have_pyelf:
        res["method"] = "pyelftools"
        cap = None
        try:
            import capstone
            an, mn = _CAPSTONE_ARCH.get(arch, _CAPSTONE_ARCH["riscv32"])
            cap = capstone.Cs(getattr(capstone, an), getattr(capstone, mn))
            res["method"] = "pyelftools+capstone"
        except Exception:
            cap = None
        with open(path, "rb") as fh:
            elf = ELFFile(fh)
            for sec in elf.iter_sections():
                flags = sec["sh_flags"]
                if not (flags & SH_FLAGS.SHF_ALLOC):
                    continue
                size = sec["sh_size"]
                res["sections"][sec.name] = size
                res["alloc_bytes"] += size
                is_code = bool(flags & SH_FLAGS.SHF_EXECINSTR)
                if is_code:
                    res["text_bytes"] += size
                    if cap is not None and sec.data_size and sec["sh_type"] != "SHT_NOBITS":
                        for ins in cap.disasm(sec.data(), sec["sh_addr"]):
                            res["mnem"][ins.mnemonic] += 1
                            res["klass"][classify(ins.mnemonic)] += 1
                            res["ninstr"] += 1
        return res

    # --- fallback: objdump ---
    od = objdump or _find_objdump()
    if not od:
        raise RuntimeError(
            "Need pyelftools (+capstone) or an objdump on PATH. "
            "Install:  pip install pyelftools capstone   "
            "or pass --objdump riscv64-unknown-elf-objdump")
    res["method"] = f"objdump ({os.path.basename(od)})"
    # sections
    hdr = subprocess.run([od, "-h", path], capture_output=True, text=True).stdout
    for m in re.finditer(r"^\s*\d+\s+(\.\S+)\s+([0-9a-f]+)\s+([0-9a-f]+).*$",
                         hdr, re.M):
        name, size = m.group(1), int(m.group(2), 16)
        res["sections"][name] = size
        res["alloc_bytes"] += size
        if name in (".text",) or name.startswith(".text"):
            res["text_bytes"] += size
    # disasm
    dis = subprocess.run([od, "-d", path], capture_output=True, text=True).stdout
    for line in dis.splitlines():
        m = re.match(r"^\s*[0-9a-f]+:\s+([0-9a-f]{2}(?: [0-9a-f]{2})*|[0-9a-f]+)\s+([a-z][\w.]*)",
                     line)
        if m:
            mnem = m.group(2)
            res["mnem"][mnem] += 1
            res["klass"][classify(mnem)] += 1
            res["ninstr"] += 1
    return res


def _find_objdump() -> Optional[str]:
    for name in ("riscv64-unknown-elf-objdump", "riscv32-unknown-elf-objdump",
                 "riscv64-linux-gnu-objdump", "riscv32-xilinx-elf-objdump",
                 "riscv-none-elf-objdump", "objdump"):
        p = shutil.which(name)
        if p:
            return p
    return None


# --------------------------------------------------------------------------- #
# firmware.c analysis
# --------------------------------------------------------------------------- #
_SETREG = re.compile(r"\bSetReg\s*\(")
_READREG = re.compile(r"\bReadReg\s*\(")
_WAIT = re.compile(r"\bwait_done\s*\(")
_OPMARK = re.compile(r"/\*\s*core(\d+):\s*(\S+)\s*--\s*(.+?)\s*\*/")
_WAVE = re.compile(r"=====\s*wave\s+(\d+)")

# rough static-size model: instructions emitted per SetReg, by -O level.
# At -O0 each SetReg materialises a 32-bit address and a 32-bit value with no
# reuse (lui+addi for each) then a store ~= 5 insns ~= 20 B. -O2 shares the
# upper immediate of the fixed MMIO base and folds constants ~= 2-3 insns.
_BYTES_PER_SETREG = {"O0": 20, "O2": 10}
_BYTES_PER_WAIT_CALL = 8     # call + return slot per call site


def analyze_c(path: str, opt: str = "O0") -> Dict:
    res: Dict = {"file": path, "kind": "c", "setreg": 0, "readreg": 0,
                 "wait": 0, "ops": 0, "waves": 0, "lines": 0,
                 "by_optype": Counter(), "setreg_by_optype": Counter()}
    cur_type = None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.strip():
                res["lines"] += 1
            if _WAVE.search(line):
                res["waves"] += 1
            m = _OPMARK.search(line)
            if m:
                res["ops"] += 1
                comment = m.group(3)
                cur_type = re.split(r"[\s\[]", comment, 1)[0]
                res["by_optype"][cur_type] += 1
            n = len(_SETREG.findall(line))
            if n:
                res["setreg"] += n
                if cur_type:
                    res["setreg_by_optype"][cur_type] += n
            res["readreg"] += len(_READREG.findall(line))
            res["wait"] += len(_WAIT.findall(line))
    res["src_bytes"] = os.path.getsize(path)
    res["est_text_O0"] = res["setreg"] * _BYTES_PER_SETREG["O0"] + res["wait"] * _BYTES_PER_WAIT_CALL + 256
    res["est_text_O2"] = res["setreg"] * _BYTES_PER_SETREG["O2"] + res["wait"] * _BYTES_PER_WAIT_CALL + 256
    res["est_text"] = res[f"est_text_{opt}"]
    return res


# --------------------------------------------------------------------------- #
# program.list analysis
# --------------------------------------------------------------------------- #
_INSTR = re.compile(r"^;\s*instr\s+(\d+)\s+wave\s+(\d+)\s+core(\d+)\s+(\S+)\s+\((.*)\)")


def analyze_list(path: str) -> Dict:
    res: Dict = {"file": path, "kind": "list", "ninstr": 0,
                 "by_optype": Counter()}
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = _INSTR.match(line)
            if m:
                res["ninstr"] += 1
                comment = m.group(5)
                t = re.split(r"[\s\[]", comment, 1)[0] or m.group(4)
                res["by_optype"][t] += 1
    res["words"] = res["ninstr"] * 16
    res["prog_bytes"] = res["ninstr"] * 64       # 16 words * 4 bytes
    return res


# --------------------------------------------------------------------------- #
# detection + reporting
# --------------------------------------------------------------------------- #
def detect_kind(path: str) -> str:
    low = path.lower()
    if low.endswith(".elf"):
        return "elf"
    if low.endswith(".c") or low.endswith(".h"):
        return "c"
    if low.endswith(".list") or low.endswith(".lst"):
        return "list"
    # sniff ELF magic
    try:
        with open(path, "rb") as fh:
            if fh.read(4) == b"\x7fELF":
                return "elf"
    except Exception:
        pass
    return "c"


def report_one(r: Dict, imem: Optional[int]) -> None:
    print(f"\n=== {r['file']}  [{r['kind']}] ===")
    if r["kind"] == "elf":
        print(f"  analysed via: {r['method']}")
        print("  sections (alloc):")
        for name, size in r["sections"].items():
            print(f"    {name:<14} {_human(size):>12}")
        print(f"    {'TOTAL(alloc)':<14} {_human(r['alloc_bytes']):>12}")
        if r["ninstr"]:
            print(f"  .text instructions: {r['ninstr']:,}")
            print("  by class:")
            tot = sum(r["klass"].values()) or 1
            for k in ("load","store","alu","branch","jump","csr","muldiv","float","other"):
                c = r["klass"].get(k, 0)
                if c:
                    print(f"    {k:<8} {c:>9,}  ({100*c/tot:4.1f}%)")
            print("  top mnemonics: " +
                  ", ".join(f"{m}:{c}" for m, c in r["mnem"].most_common(8)))
            used = {k for k, v in r["klass"].items() if v}
            if not (used & {"muldiv", "float"}):
                print("  note: no mul/div or FP instructions -> a minimal integer "
                      "load/store core suffices for this firmware.")
        _imem_check(r["text_bytes"], imem, label=".text")
    elif r["kind"] == "c":
        print(f"  source size      : {_human(r['src_bytes'])}  ({r['lines']:,} non-blank lines)")
        print(f"  ops / waves      : {r['ops']:,} ops, {r['waves']:,} waves")
        print(f"  SetReg / ReadReg : {r['setreg']:,} / {r['readreg']:,}")
        print(f"  wait_done calls  : {r['wait']:,}")
        if r["ops"]:
            print(f"  SetReg per op    : {r['setreg']/r['ops']:.1f}")
        print(f"  est .text @ -O0  : ~{_human(r['est_text_O0'])}   "
              f"@ -O2  : ~{_human(r['est_text_O2'])}   (estimate; build to confirm)")
        top = r["setreg_by_optype"].most_common(6)
        if top:
            print("  SetReg by op type (top): " +
                  ", ".join(f"{t}:{c}" for t, c in top))
        _imem_check(r["est_text"], imem, label="est .text",
                    note="(estimate -- run the ELF analyzer on the built firmware.elf for the real number)")
    elif r["kind"] == "list":
        print(f"  macro-instructions: {r['ninstr']:,}")
        print(f"  program image     : {r['words']:,} words = {_human(r['prog_bytes'])}  (HBM-resident data)")
        top = r["by_optype"].most_common(8)
        if top:
            print("  by op type: " + ", ".join(f"{t}:{c}" for t, c in top))


def _imem_check(nbytes: int, imem: Optional[int], label: str, note: str = "") -> None:
    if imem is None:
        return
    fits = nbytes <= imem
    verb = "FITS" if fits else "OVERFLOWS"
    pct = 100 * nbytes / imem if imem else 0
    print(f"  IMEM fit: {label} = {_human(nbytes)} vs IMEM {_human(imem)} "
          f"-> {verb} ({pct:.0f}% of IMEM){' ' + note if note else ''}")


def metric_for_compare(r: Dict) -> Dict:
    if r["kind"] == "elf":
        return {"label": "elf", ".text": r["text_bytes"], "instrs": r["ninstr"],
                "alloc": r["alloc_bytes"]}
    if r["kind"] == "c":
        return {"label": "c", ".text(est)": r["est_text"], "setreg": r["setreg"],
                "src": r["src_bytes"], "ops": r["ops"]}
    return {"label": "list", "prog.bin": r["prog_bytes"], "instrs": r["ninstr"]}


_BYTE_KEYS = {".text", ".text(est)", "alloc", "src", "prog.bin"}


def print_compare(results: List[Dict]) -> None:
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    base = results[0]
    bm = metric_for_compare(base)
    for r in results[1:]:
        m = metric_for_compare(r)
        print(f"\n  baseline : {base['file']}")
        print(f"  vs       : {r['file']}")
        keys = [k for k in m if k != "label" and k in bm]
        for k in keys:
            b, g = bm[k], m[k]
            ratio = (g / b) if b else float("inf")
            if k in _BYTE_KEYS:
                bs, gs = _human(b), _human(g)
            else:
                bs, gs = f"{b:,}", f"{g:,}"
            print(f"    {k:<14} baseline={bs:<14} generated={gs:<14} ratio={ratio:.1f}x")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="*.elf / *.c / *.list (first = baseline)")
    ap.add_argument("--arch", default="riscv32",
                    choices=list(_CAPSTONE_ARCH), help="ISA for capstone disasm")
    ap.add_argument("--objdump", default=None, help="path to objdump (fallback)")
    ap.add_argument("--imem", type=lambda s: parse_size(s), default=None,
                    help="IMEM size for fit check (e.g. 0x20000, 128K)")
    ap.add_argument("--linker", default=None,
                    help="link.ld to read IMEM LENGTH from (overridden by --imem)")
    ap.add_argument("--region", default="IMEM", help="linker region name")
    ap.add_argument("--opt", default="O0", choices=["O0", "O2"],
                    help="optimisation level assumed for .c .text estimate")
    args = ap.parse_args(argv)

    imem = args.imem
    if imem is None and args.linker:
        imem = imem_from_linker(args.linker, args.region)
        if imem:
            print(f"(IMEM region '{args.region}' from {args.linker}: {_human(imem)})")

    results = []
    for f in args.files:
        if not os.path.exists(f):
            print(f"!! {f}: not found", file=sys.stderr)
            continue
        kind = detect_kind(f)
        try:
            if kind == "elf":
                r = analyze_elf(f, args.arch, args.objdump)
            elif kind == "list":
                r = analyze_list(f)
            else:
                r = analyze_c(f, args.opt)
        except Exception as e:
            print(f"!! {f}: {e}", file=sys.stderr)
            continue
        results.append(r)
        report_one(r, imem)

    if len(results) >= 2:
        same_kind = [r for r in results if r["kind"] == results[0]["kind"]]
        if len(same_kind) >= 2:
            print_compare(same_kind)
        else:
            print("\n(comparison skipped: inputs are different kinds -- they are "
                  "complementary views, not directly comparable. Pass two .elf or "
                  "two .c or two .list files to compare.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())