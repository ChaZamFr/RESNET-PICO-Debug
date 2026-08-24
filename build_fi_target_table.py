#!/usr/bin/env python3
"""
Builds a (Layer, Register, RangeID) target table for fault injection,
where RangeID comes from chunking each function's REAL, OBSERVED offset
array by COUNT — not fixed-byte-width bins.

KEY DESIGN POINT — grouping by SourceFunction, not by file:
Since function_profiler.py can now merge multiple functions into one CSV
(e.g. resblock_1.csv containing resblock's own instructions plus nested
conv2d/batchnorm_relu calls, each tagged via the SourceFunction column),
grouping by the CSV's filename would silently collide different
functions' offsets — conv2d's offset 14 and batchnorm_relu's offset 14
are completely different instructions but would look identical if binned
together. Grouping by SourceFunction instead means each function
INSTANCE (conv2d_2, conv2d_3, resblock_1, ...) gets its own independent
offset array and RangeID space, exactly as if it still had its own file —
correct regardless of whether your CSVs are merged or one-file-per-call.

Falls back to the file's stem as the grouping key for older CSVs captured
before SourceFunction existed.

CALL-PATH DISAMBIGUATION (NEW):
conv2d is one function reached through many call paths (directly, and
inside each resblock). A raw `break *0xADDR` fires on EVERY invocation, so
an address alone can't say WHICH call — injecting "conv2d_2" would silently
hit conv2d_1. The profiler now records, per row, the chain of `bl`
call-site addresses walked into to reach that instance (CallPath) plus the
function's entry address (FuncEntryAddr). This script carries both through
into the target table so the injector can replay the path (break each bl,
si in, verify entry) before arming the injection address, hitting the
correct invocation. CallPath is empty for instances reached directly
(e.g. conv2d_1) — those need no navigation. Both are constant within a
Layer (one instance = one call path), so they're captured per group.
Older CSVs without these columns still work: the columns come out blank
and the injector falls back to its old direct-break behaviour.

Usage:
    # Everything under profiler_output/, by default (all layers, sorted)
    python3 build_fi_target_table.py

    # Explicit files, in whatever order given
    python3 build_fi_target_table.py --range-size 8 file1.csv file2.csv

    # Only certain functions — matches the BASE function name (conv2d
    # matches conv2d_1, conv2d_2, ... regardless of which physical file
    # they came from; resblock matches resblock_1, resblock_2, but NOT
    # resblock_ds — exact base-name match, not a substring match)
    python3 build_fi_target_table.py --layers conv2d,resblock

Output:
    fi_target_table.csv with columns:
    Layer, Register, StaticClass, RangeID, RangeOffsetStart, RangeOffsetEnd,
    RangeOffsets, DataUses, AddressUses, ControlUses, TotalUses,
    DominantUsage, PCAddrMin, PCAddrMax, PCAddresses, FuncEntryAddr,
    CallPath, ExampleInstruction
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ============================================================================
# EDITABLE CONFIG — change these instead of remembering CLI flags every time.
# CLI flags below override these if passed, so either editing this block or
# using the command line works.
# ============================================================================

RANGE_SIZE = 8                    # real offsets per range (not bytes — see module docstring)
CSV_GLOB = "profiler_output/**/*.csv"   # used only if no files are given on the command line
OUTPUT_PATH = "fi_target_table.csv"
LAYERS_FILTER = None               # e.g. "conv2d,resblock" — None = include every function

# ============================================================================


# --- Same classification tables as extract_register_usage.py ---
_DATA_PROCESSING_MNEMONICS = {
    "movs", "mov", "adds", "add", "subs", "sub", "muls", "mul",
    "ands", "orrs", "orr", "eors", "mvns", "mvn",
    "lsls", "lsl", "lsrs", "lsr", "asrs", "asr", "rors", "ror",
    "cmp", "cmn", "tst", "adcs", "adc", "sbcs", "sbc", "rsbs", "rsb",
    "uxtb", "uxth", "sxtb", "sxth", "rev", "rev16", "revsh", "bics", "bic",
}
_LOAD_STORE_MNEMONICS = {
    "ldr", "ldrb", "ldrh", "ldrsb", "ldrsh",
    "str", "strb", "strh",
}
_STACK_MNEMONICS = {"push", "pop"}
_BRANCH_MNEMONICS = {
    "b", "bl", "blx", "bx",
    "beq", "bne", "bcc", "blo", "bcs", "bhs", "bmi", "bpl",
    "bvs", "bvc", "bhi", "bls", "bge", "blt", "bgt", "ble", "bal",
}
_COMPARE_BRANCH_MNEMONICS = {"cbz", "cbnz"}

_REG_TOKEN_RE = re.compile(r"\b(r1[0-2]|r[0-9]|sp|lr|pc|fp|ip)\b", re.IGNORECASE)
_FUNC_OFFSET_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)\+(\d+)>")
_TRAILING_INSTANCE_RE = re.compile(r"_(\d+|untracked)$")

REGISTER_CLASS = {
    "r0": "DATA_ARG", "r1": "DATA_ARG", "r2": "DATA_ARG", "r3": "DATA_ARG",
    "r4": "DATA_LOCAL", "r5": "DATA_LOCAL", "r6": "DATA_LOCAL",
    "r7": "DATA_LOCAL", "r8": "DATA_LOCAL", "r9": "DATA_LOCAL",
    "r10": "DATA_LOCAL", "r11": "DATA_LOCAL",
    "r12": "DATA_SCRATCH", "ip": "DATA_SCRATCH",
    "sp": "CONTROL_SP", "r13": "CONTROL_SP",
    "lr": "CONTROL_LR", "r14": "CONTROL_LR",
    "pc": "CONTROL_PC", "r15": "CONTROL_PC",
}


def classify_register(name: str) -> str:
    return REGISTER_CLASS.get(name.lower(), "UNKNOWN")


def base_function_name(layer_or_source: str) -> str:
    """
    Strips the trailing instance suffix: "conv2d_2" -> "conv2d",
    "resblock_1" -> "resblock", "resnet_infer_full_untracked" ->
    "resnet_infer_full". Used for the --layers filter, which matches
    against the underlying function, not a specific numbered instance.
    """
    return _TRAILING_INSTANCE_RE.sub("", layer_or_source)


def _strip_width_suffix(mnemonic: str) -> str:
    return mnemonic.split(".")[0]


def _extract_mnemonic_and_operands(instr: str) -> Tuple[str, str]:
    text = instr
    m = re.search(r">:\s*(.*)", text)
    if m:
        text = m.group(1)
    text = text.strip()
    if not text:
        return "", ""
    parts = text.split(None, 1)
    mnemonic = _strip_width_suffix(parts[0].lower())
    operands = parts[1] if len(parts) > 1 else ""
    return mnemonic, operands


def classify_instruction_registers(instr: str) -> List[Tuple[str, str]]:
    mnemonic, operands = _extract_mnemonic_and_operands(instr)
    if not mnemonic:
        return []
    regs_found = [r.lower() for r in _REG_TOKEN_RE.findall(operands)]
    if not regs_found:
        return []
    results: List[Tuple[str, str]] = []

    if mnemonic in _LOAD_STORE_MNEMONICS:
        data_part = operands
        addr_part = ""
        bracket = re.search(r"\[([^\]]*)\]", operands)
        if bracket:
            addr_part = bracket.group(1)
            data_part = operands[: bracket.start()]
        data_regs = [r.lower() for r in _REG_TOKEN_RE.findall(data_part)]
        addr_regs = [r.lower() for r in _REG_TOKEN_RE.findall(addr_part)]
        for r in data_regs:
            results.append((r, "DATA"))
        for r in addr_regs:
            results.append((r, "ADDRESS"))
        seen = {r for r, _ in results}
        for r in regs_found:
            if r not in seen:
                results.append((r, "DATA"))

    elif mnemonic in _STACK_MNEMONICS:
        results.append(("sp", "ADDRESS"))
        for r in regs_found:
            results.append((r, "CONTROL" if r in ("lr", "pc") else "DATA"))

    elif mnemonic in _BRANCH_MNEMONICS:
        for r in regs_found:
            results.append((r, "CONTROL"))

    elif mnemonic in _COMPARE_BRANCH_MNEMONICS:
        for r in regs_found:
            results.append((r, "DATA"))

    elif mnemonic in _DATA_PROCESSING_MNEMONICS:
        for r in regs_found:
            results.append((r, "ADDRESS" if r == "sp" else "DATA"))

    elif mnemonic == "mov" and "pc" in regs_found:
        for r in regs_found:
            results.append((r, "CONTROL"))

    else:
        for r in regs_found:
            results.append((r, "DATA"))

    return results


def extract_offset(instr: str) -> int:
    m = _FUNC_OFFSET_RE.search(instr)
    return int(m.group(2)) if m else -1


def find_column(fieldnames: List[str], candidates: List[str]) -> str:
    for c in candidates:
        if c in fieldnames:
            return c
    return ""


def build_range_map(offsets: List[int], range_size: int) -> Dict[int, int]:
    mapping = {}
    for i, off in enumerate(offsets):
        mapping[off] = (i // range_size) + 1
    return mapping


def build_range_bounds(offsets: List[int], range_size: int) -> Dict[int, Tuple[int, int]]:
    bounds = {}
    for i in range(0, len(offsets), range_size):
        chunk = offsets[i:i + range_size]
        range_id = (i // range_size) + 1
        bounds[range_id] = (chunk[0], chunk[-1])
    return bounds


def load_rows_by_layer(csv_paths: List[Path], layers_filter: Optional[Set[str]]) -> Dict[str, List[Dict]]:
    """
    Reads every CSV and buckets rows by their SourceFunction (falling back
    to the file's own stem for older CSVs without that column) — this is
    the "Layer" identity used everywhere downstream, independent of which
    physical file a row happened to be stored in.
    """
    by_layer: Dict[str, List[Dict]] = {}
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            instr_col = find_column(fieldnames, ["Instruction", "instruction"])
            pc_col = find_column(fieldnames, ["PC", "pc"])
            source_col = find_column(fieldnames, ["SourceFunction", "sourcefunction"])
            # NEW: optional call-path columns (older CSVs won't have them).
            callpath_col = find_column(fieldnames, ["CallPath", "callpath"])
            entry_col = find_column(fieldnames, ["FuncEntryAddr", "funcentryaddr"])
            if not instr_col:
                print(f"[!] {path}: no Instruction column found, skipping")
                continue
            if not source_col:
                print(f"[i] {path}: no SourceFunction column (older CSV) — "
                      f"using filename '{path.stem}' as the layer for all its rows")
            if not callpath_col:
                print(f"[i] {path}: no CallPath column (older CSV) — CallPath will be blank; "
                      f"injector will fall back to direct-break for these targets")

            for row in reader:
                layer = row[source_col] if source_col else path.stem
                if layers_filter is not None and base_function_name(layer) not in layers_filter:
                    continue
                row["_instr_col"] = instr_col
                row["_pc_col"] = pc_col
                row["_callpath_col"] = callpath_col
                row["_entry_col"] = entry_col
                by_layer.setdefault(layer, []).append(row)
    return by_layer


def process_layer(layer: str, rows: List[Dict], range_size: int,
                   groups: Dict[Tuple[str, str, int], Dict]):
    instr_col = rows[0]["_instr_col"]
    pc_col = rows[0]["_pc_col"]
    callpath_col = rows[0]["_callpath_col"]
    entry_col = rows[0]["_entry_col"]

    # CallPath + FuncEntryAddr are constant within a Layer (one instance =
    # one call path). Capture them once from the first row that has them;
    # blank if this CSV predates those columns.
    layer_callpath = ""
    layer_entry = ""
    if callpath_col:
        for row in rows:
            v = row.get(callpath_col, "")
            if v:
                layer_callpath = v
                break
    if entry_col:
        for row in rows:
            v = row.get(entry_col, "")
            if v:
                layer_entry = v
                break

    seen_offsets = set()
    for row in rows:
        off = extract_offset(row.get(instr_col, ""))
        if off >= 0:
            seen_offsets.add(off)
    sorted_offsets = sorted(seen_offsets)
    range_map = build_range_map(sorted_offsets, range_size)
    range_bounds = build_range_bounds(sorted_offsets, range_size)
    print(f"[i] {layer}: {len(sorted_offsets)} distinct real offsets -> "
          f"{max(range_map.values()) if range_map else 0} ranges of {range_size} offsets each"
          + (f"  (callpath={layer_callpath or 'DIRECT'})" if callpath_col else ""))

    for row in rows:
        instr = row.get(instr_col, "")
        offset = extract_offset(instr)
        if offset < 0 or offset not in range_map:
            continue
        range_id = range_map[offset]
        chunk_start, chunk_end = range_bounds[range_id]

        pc_val = None
        if pc_col:
            raw = row.get(pc_col, "")
            try:
                pc_val = int(raw, 16) if raw.lower().startswith("0x") else int(raw)
            except ValueError:
                pc_val = None

        for reg, usage in classify_instruction_registers(instr):
            key = (layer, reg, range_id)
            g = groups.setdefault(key, {
                "DATA": 0, "ADDRESS": 0, "CONTROL": 0,
                "static_class": classify_register(reg),
                "range_offset_start": chunk_start, "range_offset_end": chunk_end,
                "pc_min": pc_val, "pc_max": pc_val,
                "pc_addresses": set(),
                "offsets_in_range": set(),
                "example_instruction": instr.strip(),
                # NEW: carried straight through — same for every group in
                # this layer.
                "call_path": layer_callpath,
                "func_entry_addr": layer_entry,
            })
            g[usage] += 1
            g["offsets_in_range"].add(offset)
            if pc_val is not None:
                g["pc_min"] = pc_val if g["pc_min"] is None else min(g["pc_min"], pc_val)
                g["pc_max"] = pc_val if g["pc_max"] is None else max(g["pc_max"], pc_val)
                g["pc_addresses"].add(pc_val)


def write_table(groups: Dict[Tuple[str, str, int], Dict], out_path: Path):
    fields = [
        "Layer", "Register", "StaticClass", "RangeID",
        "RangeOffsetStart", "RangeOffsetEnd", "RangeOffsets",
        "DataUses", "AddressUses", "ControlUses", "TotalUses", "DominantUsage",
        "PCAddrMin", "PCAddrMax", "PCAddresses",
        # NEW columns — the injector reads these to navigate to the correct
        # call instance before arming the injection address.
        "FuncEntryAddr", "CallPath",
        "ExampleInstruction",
    ]
    rows = []
    for (layer, reg, range_id), g in sorted(groups.items()):
        total = g["DATA"] + g["ADDRESS"] + g["CONTROL"]
        dominant = max(("DATA", "ADDRESS", "CONTROL"), key=lambda k: g[k])
        offs = sorted(g["offsets_in_range"])
        addrs = sorted(g["pc_addresses"])
        rows.append({
            "Layer": layer,
            "Register": reg,
            "StaticClass": g["static_class"],
            "RangeID": range_id,
            "RangeOffsetStart": g["range_offset_start"],
            "RangeOffsetEnd": g["range_offset_end"],
            # Semicolon-separated, not comma — a comma-separated numeric-
            # looking list gets silently mangled by Excel/LibreOffice into
            # a single number (commas read as thousands separators).
            "RangeOffsets": ";".join(str(o) for o in offs),
            "DataUses": g["DATA"],
            "AddressUses": g["ADDRESS"],
            "ControlUses": g["CONTROL"],
            "TotalUses": total,
            "DominantUsage": dominant,
            "PCAddrMin": f"0x{g['pc_min']:x}" if g["pc_min"] is not None else "",
            "PCAddrMax": f"0x{g['pc_max']:x}" if g["pc_max"] is not None else "",
            "PCAddresses": ";".join(f"0x{a:x}" for a in addrs),
            # NEW: CallPath is already ';'-joined absolute bl addresses (or
            # blank for direct-reach instances); FuncEntryAddr is a single
            # '0x...'. Both passed through verbatim from the profiler.
            "FuncEntryAddr": g["func_entry_addr"],
            "CallPath": g["call_path"],
            "ExampleInstruction": g["example_instruction"],
        })
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved fault-injection target table to: {out_path} "
          f"({len(rows)} rows, {len({r['Layer'] for r in rows})} layer(s))")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-size", type=int, default=RANGE_SIZE,
                         help=f"Number of REAL observed offsets per range, per layer (default: {RANGE_SIZE})")
    parser.add_argument("csvs", nargs="*",
                         help=f"Profiler CSV files to process (default: glob {CSV_GLOB}, sorted)")
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    parser.add_argument("--layers", type=str, default=LAYERS_FILTER,
                         help="Comma-separated base function names to include, e.g. "
                              "'conv2d,resblock' — matches conv2d_1, conv2d_2, resblock_1, ... "
                              "regardless of which file they came from. Default: include everything.")
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csvs]
    if not csv_paths:
        csv_paths = sorted(Path(".").glob(CSV_GLOB))  # sorted for deterministic, "in order" processing
        if not csv_paths:
            print(f"[!] No files given and nothing matched default glob '{CSV_GLOB}'. "
                  f"Pass files explicitly or edit CSV_GLOB at the top of this script.")
            return
        print(f"[i] No files given — using default glob '{CSV_GLOB}', found {len(csv_paths)} file(s)")

    csv_paths = [p for p in csv_paths if p.exists()]
    if not csv_paths:
        print("[!] None of the given files exist.")
        return

    layers_filter = None
    if args.layers:
        layers_filter = {s.strip() for s in args.layers.split(",") if s.strip()}
        print(f"[i] Filtering to base function(s): {sorted(layers_filter)}")

    print(f"[+] Reading {len(csv_paths)} file(s): {', '.join(p.name for p in csv_paths)}")
    by_layer = load_rows_by_layer(csv_paths, layers_filter)
    if not by_layer:
        print("[!] No matching rows found (check --layers spelling, or that the files have data).")
        return

    groups: Dict[Tuple[str, str, int], Dict] = {}
    for layer in sorted(by_layer.keys()):  # sorted so per-layer log lines are also deterministic
        process_layer(layer, by_layer[layer], args.range_size, groups)

    write_table(groups, args.output)


if __name__ == "__main__":
    main()
