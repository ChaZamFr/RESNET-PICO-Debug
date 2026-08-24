#!/usr/bin/env python3
"""
TRANSIENT FAULT injection tool -- single-event-upset (SEU) style bit-flip
fault model: a bit is toggled ONCE at the injection point, then execution
proceeds normally. If a later instruction happens to overwrite that
register (extremely likely -- this is normal register usage), the fault
naturally disappears, exactly like a real transient particle-strike upset.

This is the RIGHT model for: soft errors from radiation, cosmic rays,
voltage noise -- any fault that corrupts a value once, not the hardware
storage cell itself.

For PERMANENT stuck-at faults (the bit is physically broken and
re-asserts itself every time something tries to write it), use
inject_permanent_fault.py instead -- that requires actively watching and
re-enforcing the value, a meaningfully different (slower, more invasive)
mechanism, not just a different parameter to this same script.

State machine per trial:
  1. Select RangeID (--range-id, default: random) and Layer (--layer,
     default: random -- pool is "every layer" unless one is named)
  2. Look up active registers in that (Layer, RangeID)
  3. Pick a random register (uniform, or --weighted by TotalUses)
  4. NAVIGATE the recorded call path (CallPath column) so the injection
     address fires on the CORRECT invocation of a shared function
     (conv2d_2 vs conv2d_1); empty CallPath = reached directly, no nav.
  5. Break at a REAL address that register was hit at within the range
  6. Flip a random bit -- unconditional toggle, no target value, always
     a real state change (see sfi_injector.SFIInjector.flip_bit)
  7. Continue, racing isr_hardfault vs. the layer's end address
  8. Compare output/logits against golden; no golden data -> Masked

TIMING: every phase of every trial (plus the golden run) is wall-clock
timed via SFIInjector.timed()/snapshot_timings(). Per-trial phase
durations are printed and written into injection_state.csv as Timing_*
columns; the golden-capture duration is printed once up front. The call-
path navigation is its own timed phase ('navigate').
"""

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuses the GDB state machine (sentinel-framed I/O, force_bit,
# run_to_address/run_to_completion, timed(), navigate_call_path()) already
# built and tested in sfi_injector.py rather than duplicating it.
from sfi_injector import SFIInjector


# ============================================================================
# EDITABLE CONFIG — change these instead of remembering CLI flags every time.
# Every CLI flag below overrides its matching constant here if you pass it,
# so you can either edit this block directly or override on the command line
# for a one-off run — whichever's easier.
# ============================================================================

TABLE_PATH = "fi_target_table.csv"       # output of build_fi_target_table.py

GDB_PATH = "arm-none-eabi-gdb"
ELF_PATH = "build/resnet_pico.elf"
TARGET_REMOTE = "localhost:3333"         # OpenOCD's GDB server, must already be running

LAYER = None                             # e.g. "conv2d_1" — None = random across every layer
RANGE_ID = None                          # e.g. 5 — None = random

WEIGHTED = False                         # True = weight register pick by TotalUses instead of uniform
SEED = None                              # e.g. 42 for reproducible picks — None = fresh randomness each run

HARDFAULT_SYMBOL = "isr_hardfault"       # RP2040/Pico-SDK's default hard fault vector name
DUE_TIMEOUT = 15.0                       # seconds to wait before treating a run as hung/DUE

CAMPAIGN_END_SYMBOL = "CAMPAIGN_END"
USE_CONTINUOUS_FLOW = False

PREDICTION_VAR = "FI_PREDICTION"
SUM_VAR = "FI_SUM"
LOGITS_VAR = "g_logits"
LOGITS_LEN = 10
LOGITS_SCALE = 256.0

GOLDEN_BUFFER_PATH = None
OUTPUT_BUFFER_SYMBOL = "buf_b"
OUTPUT_BUFFER_LENGTH = 32 * 32 * 16
OUTPUT_BUFFER_SCALE = 1.0

NUM_RUNS = 1
STATE_FILE = "injection_state.csv"
CLEAR_STATE_ON_START = False
STOP_ON_CRASH = None
MAX_DEDUP_RETRIES = 200

# ============================================================================


# --- Table loading & selection ---------------------------------------

def load_table(path: Path) -> List[Dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get_layer_range_pairs(rows: List[Dict]) -> List[Tuple[str, int]]:
    return sorted({(r["Layer"], int(r["RangeID"])) for r in rows})


def select_layer_and_range(rows: List[Dict], layer: Optional[str], range_id: Optional[int],
                            rng: random.Random) -> Tuple[str, int]:
    pairs = get_layer_range_pairs(rows)
    if layer is not None:
        pairs = [p for p in pairs if p[0] == layer]
        if not pairs:
            raise ValueError(f"Layer '{layer}' not found in table")
    if range_id is not None:
        pairs = [p for p in pairs if p[1] == range_id]
        if not pairs:
            scope = f"layer '{layer}'" if layer else "any layer"
            raise ValueError(f"RangeID {range_id} not found for {scope}")
    if not pairs:
        raise ValueError("No matching (layer, range) pairs")
    return rng.choice(pairs)


def active_registers_in(rows: List[Dict], layer: str, range_id: int) -> List[Dict]:
    return [r for r in rows if r["Layer"] == layer and int(r["RangeID"]) == range_id]


def pick_register(active: List[Dict], rng: random.Random, weighted: bool) -> Dict:
    if weighted:
        weights = [max(1, int(r["TotalUses"])) for r in active]
        return rng.choices(active, weights=weights, k=1)[0]
    return rng.choice(active)


def pick_address(reg_row: Dict, rng: random.Random) -> str:
    addrs = [a for a in reg_row.get("PCAddresses", "").split(";") if a]
    if not addrs:
        raise ValueError(f"No PCAddresses recorded for {reg_row['Register']} "
                          f"in range {reg_row['RangeID']} — table may be stale, rebuild it")
    return rng.choice(addrs)


def get_layer_end_address(rows: List[Dict], layer: str) -> str:
    layer_rows = [r for r in rows if r["Layer"] == layer]
    if not layer_rows:
        raise ValueError(f"No rows for layer '{layer}'")
    return max(layer_rows, key=lambda r: int(r["PCAddrMax"], 16))["PCAddrMax"]


_TRAILING_INSTANCE_RE = re.compile(r"_(\d+|untracked)$")


def base_function_name(layer: str) -> str:
    return _TRAILING_INSTANCE_RE.sub("", layer)


def find_layer_by_base(rows: List[Dict], base_name: str) -> Optional[str]:
    matches = {r["Layer"] for r in rows if base_function_name(r["Layer"]) == base_name}
    if not matches:
        return None

    def instance_num(layer):
        m = re.search(r"_(\d+)$", layer)
        return int(m.group(1)) if m else -1

    return max(matches, key=instance_num)


# --- Dedup state: persistent, crash-safe, cross-invocation --------------
# Timing_* columns log each trial's phase durations; CallPath is logged so
# results are traceable back to which invocation was targeted. Older state
# files without these columns still load fine (DictReader tolerates missing
# keys); new writes include them.

STATE_FIELDS = ["Layer", "RangeID", "Register", "Address", "Bit", "Outcome", "NaturalBit",
                "CallPath",
                "Timing_navigate", "Timing_to_injection", "Timing_inject",
                "Timing_to_end", "Timing_total"]


def fingerprint(layer: str, range_id: int, register: str, address: str, bit: int) -> Tuple:
    return (layer, range_id, register, address, bit)


def load_used_fingerprints(state_path: Path) -> set:
    if not state_path.exists():
        return set()
    used = set()
    with open(state_path, newline="") as f:
        for row in csv.DictReader(f):
            used.add(fingerprint(row["Layer"], int(row["RangeID"]), row["Register"],
                                  row["Address"], int(row["Bit"])))
    return used


def append_state(state_path: Path, layer: str, range_id: int, register: str,
                  address: str, bit: int, outcome: str,
                  natural_bit: Optional[int] = None,
                  timings: Optional[Dict[str, float]] = None,
                  call_path: str = ""):
    """
    natural_bit records what the bit's value was BEFORE the toggle.
    timings is the per-trial phase-duration dict from
    injector.snapshot_timings() (keys: navigate, to_injection, inject,
    to_end, total) — written into the Timing_* columns, blank if not
    provided. call_path records which invocation was navigated to.
    """
    timings = timings or {}
    is_new_file = not state_path.exists()
    with open(state_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATE_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow({
            "Layer": layer, "RangeID": range_id, "Register": register,
            "Address": address, "Bit": bit, "Outcome": outcome,
            "NaturalBit": "" if natural_bit is None else natural_bit,
            "CallPath": call_path,
            "Timing_navigate": _fmt(timings.get("navigate")),
            "Timing_to_injection": _fmt(timings.get("to_injection")),
            "Timing_inject": _fmt(timings.get("inject")),
            "Timing_to_end": _fmt(timings.get("to_end")),
            "Timing_total": _fmt(timings.get("total")),
        })
        f.flush()


def _fmt(v: Optional[float]) -> str:
    """4-decimal seconds, or blank if a phase never ran (e.g. crashed
    before reaching the injection point)."""
    return "" if v is None else f"{v:.4f}"


def pick_fresh_target(rows: List[Dict], layer_arg: Optional[str], range_id_arg: Optional[int],
                       rng: random.Random, weighted: bool, used: set,
                       max_retries: int) -> Optional[Tuple[str, int, Dict, str, int]]:
    for _ in range(max_retries):
        layer, range_id = select_layer_and_range(rows, layer_arg, range_id_arg, rng)
        active = active_registers_in(rows, layer, range_id)
        reg_row = pick_register(active, rng, weighted)
        address = pick_address(reg_row, rng)
        bit = rng.randint(0, 31)
        fp = fingerprint(layer, range_id, reg_row["Register"], address, bit)
        if fp not in used:
            return layer, range_id, reg_row, address, bit
    return None


# --- Output buffer comparison ------------------------------------------

def load_golden_buffer(path: Optional[Path]) -> Optional[List[float]]:
    if path is None:
        return None
    import json
    return json.loads(path.read_text())


def compare_buffers(golden: Optional[List[float]], faulty: Optional[List[float]]) -> Tuple[str, str]:
    if golden is None or faulty is None:
        return "Masked", "no golden buffer available for comparison — defaulting to Masked per spec"
    if len(golden) != len(faulty):
        print(f"[debug] golden ({len(golden)} values) = {golden}")
        print(f"[debug] faulty ({len(faulty)} values) = {faulty}")
        return "SDC", f"length mismatch: golden={len(golden)} faulty={len(faulty)}"
    if golden == faulty:
        return "Masked", "output buffer identical to golden"
    first_diff = next((i for i, (a, b) in enumerate(zip(golden, faulty)) if a != b), None)
    detail = f"buffers differ at index {first_diff} (golden={golden[first_diff]}, faulty={faulty[first_diff]})"
    return "SDC", detail


# --- Helpers to pull CallPath/FuncEntryAddr off a picked target row -----

def _call_path_of(reg_row: Dict) -> str:
    """CallPath column value, or '' for older tables / direct-reach rows."""
    return (reg_row.get("CallPath") or "").strip()


def _func_entry_of(reg_row: Dict) -> str:
    """FuncEntryAddr column value, or '' for older tables."""
    return (reg_row.get("FuncEntryAddr") or "").strip()


class SingleShotInjector(SFIInjector):
    """Adds the hardfault-race and layer-end mechanics on top of
    SFIInjector's existing connection/breakpoint/bit-force/timing/
    navigation primitives."""

    def run_to_address_or_hardfault(self, target: str, hardfault_symbol: str) -> Tuple[str, Optional[str]]:
        self._run(self._break_cmd(target))
        hf_armed = False
        out = self._run(f"break {hardfault_symbol}", quiet=True)
        if "not defined" not in out.lower() and "no symbol" not in out.lower():
            hf_armed = True
        else:
            print(f"[!] Could not resolve hardfault symbol '{hardfault_symbol}' — "
                  f"proceeding without hardfault breakpoint, relying on timeout only")

        out = self._run_until_stop("continue", timeout=self.due_timeout)
        self._run("delete")

        if out is None:
            return "TIMEOUT", None
        if hf_armed and hardfault_symbol in out:
            return "HARDFAULT", out
        if "Breakpoint" in out:
            return "TARGET", out
        return "UNKNOWN", out

    def capture_output_buffer(self, symbol: Optional[str] = None, length: Optional[int] = None,
                               scale: Optional[float] = None) -> Optional[List[float]]:
        sym = symbol if symbol is not None else OUTPUT_BUFFER_SYMBOL
        n = length if length is not None else OUTPUT_BUFFER_LENGTH
        sc = scale if scale is not None else OUTPUT_BUFFER_SCALE
        if sym is None or n is None:
            print("[!] OUTPUT_BUFFER_SYMBOL/OUTPUT_BUFFER_LENGTH not configured — "
                  "edit the config block at the top of this file. Falling back to no-data.")
            return None
        return self.read_array_scaled(sym, n, sc)


# --- Continuous-loop campaign: golden captured inline, no reset between trials ---

def capture_golden_inline(injector: "SingleShotInjector", campaign_end_symbol: str,
                           prediction_var: str, sum_var: str,
                           logits_var: str, logits_len: int, logits_scale: float) -> Dict:
    """
    Runs from a fresh reflash to the FIRST hit of campaign_end_symbol and
    reads the result globals — this IS the golden run, captured inline in
    the same session that goes on to run every injection trial. The whole
    capture is wall-clock timed under the 'golden' label; the duration is
    printed by the caller via injector.snapshot_timings().

    No call-path navigation here — the golden run is a clean, fault-free
    full inference; reading g_logits at CAMPAIGN_END is not tied to any
    specific call instance.
    """
    injector.reset_timings()
    with injector.timed("golden"):
        injector._run(SingleShotInjector._break_cmd(campaign_end_symbol))
        out = injector._run_until_stop("continue", timeout=injector.due_timeout)
        injector._run("delete")
        if out is None or "Breakpoint" not in out:
            raise RuntimeError(f"Golden run failed to reach '{campaign_end_symbol}' — "
                                f"check the symbol name matches your firmware and that it "
                                f"actually loops back into inference.")
        result = {
            "predicted_class": injector.read_variable(prediction_var),
            "sum": injector.read_variable(sum_var),
            "logits": injector.read_array_scaled(logits_var, logits_len, logits_scale),
        }
    print(f"[timing] golden run: {injector.timings['golden']:.4f}s")
    return result


def classify_via_variables(golden: Dict, faulty: Dict) -> Tuple[str, str]:
    pred, fsum = faulty.get("predicted_class"), faulty.get("sum")
    if pred is None or fsum is None:
        return "Critical_SDC", "could not read prediction/sum after injection"
    if pred != golden["predicted_class"] and fsum != golden["sum"]:
        return "Critical_SDC", f"prediction changed ({golden['predicted_class']} -> {pred}), sum also changed"
    if fsum != golden["sum"] and pred == golden["predicted_class"]:
        return "Safe_SDC", f"sum changed ({golden['sum']} -> {fsum}) but prediction held"
    return "Masked", "prediction and sum both match golden"


def classify_via_logits(golden: Dict, faulty: Dict) -> Tuple[str, str]:
    g_logits, f_logits = golden.get("logits"), faulty.get("logits")
    g_pred, f_pred = golden.get("predicted_class"), faulty.get("predicted_class")

    if f_logits is None or f_pred is None:
        return "CDC", "could not read logits/prediction after injection"

    if g_logits == f_logits:
        return "Masked", "logits identical to golden"

    g_sum, f_sum = sum(g_logits), sum(f_logits)
    diffs = [(i, g, f) for i, (g, f) in enumerate(zip(g_logits, f_logits)) if g != f]
    for i, g, f in diffs:
        print(f"  logit[{i}] changed: {g} -> {f}  (delta {f - g:+})")
    diff_indices = [i for i, _, _ in diffs]

    if f_pred == g_pred:
        return "SDC", (f"logit(s) {diff_indices} differ (sum {g_sum} -> {f_sum}) "
                        f"but prediction held at {g_pred}")

    print(f"Golden prediction: {g_pred}   Faulty prediction: {f_pred}")
    return "CDC", (f"logit(s) {diff_indices} differ (sum {g_sum} -> {f_sum}), "
                    f"prediction changed {g_pred} -> {f_pred}")


def recover(injector: "SingleShotInjector", campaign_end_symbol: str) -> bool:
    print("[!] Recovering: reflash + resync to loop...")
    injector.reflash()
    injector._run(SingleShotInjector._break_cmd(campaign_end_symbol))
    out = injector._run_until_stop("continue", timeout=injector.due_timeout)
    injector._run("delete")
    if out is None or "Breakpoint" not in out:
        print("[!] Recovery failed to resync — subsequent trials may be unreliable.")
        return False
    print("Recovery complete — back in the steady-state loop.")
    return True


def _print_trial_timing(snap: Dict[str, float]):
    """One compact line per trial: each phase + total, blanks for phases
    that never ran (e.g. crashed before injection)."""
    parts = []
    for key in ("navigate", "to_injection", "inject", "to_end", "total"):
        v = snap.get(key)
        parts.append(f"{key}={'--' if v is None else f'{v:.4f}s'}")
    print(f"[timing] {'  '.join(parts)}")


def run_one_trial_continuous(injector: "SingleShotInjector", reg_row: Dict, address: str, bit: int,
                              campaign_end_symbol: str, hardfault_symbol: str,
                              golden: Dict) -> Tuple[str, bool, Optional[int], Dict[str, float]]:
    """
    One trial inside a CONTINUOUS firmware session. Returns
    (outcome, needs_recovery, natural_bit, timings_snapshot).

    NOTE: call-path navigation assumes a known execution position (it
    replays bl breakpoints from the start of a run). In the continuous
    no-reset flow the target is mid-loop, so navigation is only applied
    for direct-reach targets (empty CallPath). A non-empty CallPath in
    continuous mode is reported as DUE rather than injected at the wrong
    invocation — use the reset-per-trial flow for shared-function
    instances.
    """
    injector.reset_timings()

    call_path = _call_path_of(reg_row)
    func_entry = _func_entry_of(reg_row)
    if call_path:
        # Can't reliably replay a from-start call path without a reset.
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return ("DUE", False, None, snap)

    with injector.timed("to_injection"):
        status, _ = injector.run_to_address_or_hardfault(address, hardfault_symbol)
    if status == "HARDFAULT":
        print(f"Hit {hardfault_symbol} before reaching injection point")
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "CRASH", True, None, snap
    if status != "TARGET":
        print(f"Timed out/unexpected waiting for injection point ({status})")
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "DUE", True, None, snap

    with injector.timed("inject"):
        natural_bit = injector.flip_bit(reg_row["Register"], bit)
    print(f"Injected: natural_bit={natural_bit} -> flipped to {1 - natural_bit}")

    with injector.timed("to_end"):
        status, _ = injector.run_to_address_or_hardfault(campaign_end_symbol, hardfault_symbol)
    if status == "HARDFAULT":
        print(f"Hit {hardfault_symbol} after injection")
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "CRASH", True, natural_bit, snap
    if status != "TARGET":
        print(f"Timed out/unexpected waiting for {campaign_end_symbol} ({status})")
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "DUE", True, natural_bit, snap

    faulty = {
        "predicted_class": injector.read_variable(PREDICTION_VAR),
        "sum": injector.read_variable(SUM_VAR),
    }
    outcome, detail = classify_via_variables(golden, faulty)
    print(f"  {detail}")
    snap = injector.snapshot_timings()
    _print_trial_timing(snap)
    return outcome, False, natural_bit, snap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path, nargs="?", default=Path(TABLE_PATH),
                         help=f"fi_target_table.csv from build_fi_target_table.py (default: {TABLE_PATH})")
    parser.add_argument("--layer", type=str, default=LAYER)
    parser.add_argument("--range-id", type=int, default=RANGE_ID)
    parser.add_argument("--weighted", action="store_true", default=WEIGHTED)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gdb", default=GDB_PATH)
    parser.add_argument("--elf", default=ELF_PATH)
    parser.add_argument("--target-remote", default=TARGET_REMOTE)
    parser.add_argument("--hardfault-symbol", default=HARDFAULT_SYMBOL)
    parser.add_argument("--campaign-end-symbol", default=CAMPAIGN_END_SYMBOL)
    parser.add_argument("--continuous", action="store_true", default=USE_CONTINUOUS_FLOW)
    parser.add_argument("--golden-buffer", type=Path,
                         default=Path(GOLDEN_BUFFER_PATH) if GOLDEN_BUFFER_PATH else None)
    parser.add_argument("--due-timeout", type=float, default=DUE_TIMEOUT)
    parser.add_argument("--runs", type=int, default=NUM_RUNS)
    parser.add_argument("--state-file", type=Path, default=Path(STATE_FILE))
    parser.add_argument("--max-dedup-retries", type=int, default=MAX_DEDUP_RETRIES)
    parser.add_argument("--stop-on-crash", dest="stop_on_crash", action="store_true", default=None)
    parser.add_argument("--no-stop-on-crash", dest="stop_on_crash", action="store_false")
    parser.add_argument("--fresh-state", action="store_true", default=CLEAR_STATE_ON_START)
    args = parser.parse_args()

    stop_on_crash = args.stop_on_crash
    if stop_on_crash is None:
        stop_on_crash = STOP_ON_CRASH if STOP_ON_CRASH is not None else (args.runs == 1)

    if args.fresh_state and args.state_file.exists():
        print(f"[i] --fresh-state: clearing {args.state_file}")
        args.state_file.unlink()

    rng = random.Random(args.seed)
    rows = load_table(args.table)
    used = load_used_fingerprints(args.state_file)
    if used:
        print(f"[i] Loaded {len(used)} previously-tested target(s) from {args.state_file}")

    injector = SingleShotInjector(gdb_path=args.gdb, elf_path=args.elf,
                                   target_remote=args.target_remote, due_timeout=args.due_timeout)
    injector.start()
    injector.reflash()

    completed = 0
    total_campaign_time = 0.0
    try:
        if args.continuous:
            print(f"Using continuous flow (campaign_end_symbol={args.campaign_end_symbol})")
            golden = capture_golden_inline(injector, args.campaign_end_symbol,
                                            PREDICTION_VAR, SUM_VAR, LOGITS_VAR, LOGITS_LEN, LOGITS_SCALE)
            print(f"Golden: prediction={golden['predicted_class']}  sum={golden['sum']}  "
                  f"logits={golden['logits']}")

            for run_num in range(1, args.runs + 1):
                print(f"\n--- Run {run_num}/{args.runs} ---")
                picked = pick_fresh_target(rows, args.layer, args.range_id, rng, args.weighted,
                                            used, args.max_dedup_retries)
                if picked is None:
                    print(f"[!] Could not find a fresh target within {args.max_dedup_retries} retries. Stopping.")
                    break
                layer, range_id, reg_row, address, bit = picked
                cp = _call_path_of(reg_row)
                print(f"Target: layer={layer}  range_id={range_id}  register={reg_row['Register']}  "
                      f"address={address}  bit={bit}  call_path={cp or 'DIRECT'}")

                outcome, needs_recovery, natural_bit, snap = run_one_trial_continuous(
                    injector, reg_row, address, bit,
                    args.campaign_end_symbol, args.hardfault_symbol, golden)
                print(f"OUTCOME: {outcome}")
                total_campaign_time += snap.get("total", 0.0)

                fp = fingerprint(layer, range_id, reg_row["Register"], address, bit)
                used.add(fp)
                append_state(args.state_file, layer, range_id, reg_row["Register"],
                             address, bit, outcome, natural_bit, snap, call_path=cp)
                completed += 1

                if needs_recovery:
                    if stop_on_crash:
                        print("[!] Stopping batch (stop_on_crash is enabled).")
                        break
                    recover(injector, args.campaign_end_symbol)

        else:
            print("Using reset-per-trial flow")
            if args.campaign_end_symbol:
                print(f"Capturing golden via CAMPAIGN_END_SYMBOL={args.campaign_end_symbol}")
                golden = capture_golden_inline(injector, args.campaign_end_symbol,
                                                PREDICTION_VAR, SUM_VAR, LOGITS_VAR, LOGITS_LEN, LOGITS_SCALE)
                print(f"Golden: prediction={golden['predicted_class']}  logits={golden['logits']}")

                for run_num in range(1, args.runs + 1):
                    print(f"\n--- Run {run_num}/{args.runs} ---")
                    picked = pick_fresh_target(rows, args.layer, args.range_id, rng, args.weighted,
                                                used, args.max_dedup_retries)
                    if picked is None:
                        print(f"[!] Could not find a fresh target within {args.max_dedup_retries} retries. Stopping.")
                        break
                    layer, range_id, reg_row, address, bit = picked
                    cp = _call_path_of(reg_row)
                    print(f"Target: layer={layer}  range_id={range_id}  register={reg_row['Register']}  "
                          f"address={address}  bit={bit}  call_path={cp or 'DIRECT'}")

                    outcome, natural_bit, snap = run_one_trial_logits(
                        injector, reg_row, address, bit,
                        args.campaign_end_symbol, args.hardfault_symbol, golden)
                    print(f"OUTCOME: {outcome}")
                    total_campaign_time += snap.get("total", 0.0)

                    fp = fingerprint(layer, range_id, reg_row["Register"], address, bit)
                    used.add(fp)
                    append_state(args.state_file, layer, range_id, reg_row["Register"],
                                 address, bit, outcome, natural_bit, snap, call_path=cp)
                    completed += 1

                    if outcome in ("CRASH", "DUE") and stop_on_crash:
                        print("[!] Stopping batch (stop_on_crash is enabled).")
                        break

            else:
                golden_bytes = load_golden_buffer(args.golden_buffer)
                if golden_bytes is None:
                    print("[!] No --golden-buffer given — output comparison will default to Masked per spec")

                fc_layer = find_layer_by_base(rows, "fc")
                whole_model = fc_layer is not None and golden_bytes is not None
                if whole_model:
                    print(f"[i] Whole-model mode (fc layer '{fc_layer}' present) — comparing g_logits at end.")
                    comparison_end = get_layer_end_address(rows, fc_layer)
                elif fc_layer is not None and golden_bytes is None:
                    print(f"[i] fc layer '{fc_layer}' present but no --golden-buffer given — using per-layer buf_b.")

                for run_num in range(1, args.runs + 1):
                    print(f"\n--- Run {run_num}/{args.runs} ---")
                    picked = pick_fresh_target(rows, args.layer, args.range_id, rng, args.weighted,
                                                used, args.max_dedup_retries)
                    if picked is None:
                        print(f"[!] Could not find a fresh target within {args.max_dedup_retries} retries. Stopping.")
                        break
                    layer, range_id, reg_row, address, bit = picked
                    cp = _call_path_of(reg_row)
                    print(f"Target: layer={layer}  range_id={range_id}  register={reg_row['Register']}  "
                          f"address={address}  bit={bit}  call_path={cp or 'DIRECT'}")

                    layer_end = comparison_end if whole_model else get_layer_end_address(rows, layer)
                    outcome, natural_bit, snap = run_one_trial(injector, reg_row, address, bit,
                                                          layer_end, args.hardfault_symbol, golden_bytes, whole_model)
                    print(f"OUTCOME: {outcome}")
                    total_campaign_time += snap.get("total", 0.0)

                    fp = fingerprint(layer, range_id, reg_row["Register"], address, bit)
                    used.add(fp)
                    append_state(args.state_file, layer, range_id, reg_row["Register"],
                                 address, bit, outcome, natural_bit, snap, call_path=cp)
                    completed += 1

                    if outcome in ("CRASH", "DUE") and stop_on_crash:
                        print("[!] Stopping batch (stop_on_crash is enabled).")
                        break

    finally:
        injector.close()
        avg = (total_campaign_time / completed) if completed else 0.0
        print(f"\nCompleted {completed}/{args.runs} run(s). Results appended to {args.state_file}")
        print(f"[timing] total trial time: {total_campaign_time:.4f}s  "
              f"avg/trial: {avg:.4f}s  (excludes one-time golden capture & session setup)")


def run_one_trial_logits(injector: "SingleShotInjector", reg_row: Dict, address: str, bit: int,
                          campaign_end_symbol: str, hardfault_symbol: str,
                          golden: Dict) -> Tuple[str, Optional[int], Dict[str, float]]:
    """
    Reset-per-trial version of the CAMPAIGN_END-based trial. Returns
    (outcome, natural_bit, timings_snapshot). Every phase — reset, call-
    path navigation, run-to-injection, the flip itself, and run-to-
    CAMPAIGN_END — is wall-clock timed; 'total' sums them.
    """
    injector.reset_timings()

    with injector.timed("reset"):
        injector.reset_target()

    # Navigate to the correct call instance BEFORE arming the injection
    # address, so run_to_address()'s breakpoint fires on this invocation.
    call_path = _call_path_of(reg_row)
    func_entry = _func_entry_of(reg_row)
    with injector.timed("navigate"):
        ok, reason = injector.navigate_call_path(call_path, func_entry)
    if not ok:
        print(f"[!] call-path navigation failed: {reason}")
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "DUE", None, snap

    with injector.timed("to_injection"):
        hit, timed_out = injector.run_to_address(address)
    if timed_out or not hit:
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "DUE", None, snap

    with injector.timed("inject"):
        natural_bit = injector.flip_bit(reg_row["Register"], bit)
    print(f"Injected: natural_bit={natural_bit} -> flipped to {1 - natural_bit}")

    with injector.timed("to_end"):
        status, _ = injector.run_to_address_or_hardfault(campaign_end_symbol, hardfault_symbol)
    if status == "HARDFAULT":
        print(f"Hit {hardfault_symbol}")
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "CRASH", natural_bit, snap
    if status != "TARGET":
        print(f"Unexpected stop reason ({status}), treating as DUE")
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "DUE", natural_bit, snap

    print("Reached CAMPAIGN_END cleanly — no hardfault.")
    faulty = {
        "logits": injector.read_array_scaled(LOGITS_VAR, LOGITS_LEN, LOGITS_SCALE),
        "predicted_class": injector.read_variable(PREDICTION_VAR),
    }
    outcome, detail = classify_via_logits(golden, faulty)
    print(f"  {detail}")
    snap = injector.snapshot_timings()
    _print_trial_timing(snap)
    return outcome, natural_bit, snap


def run_one_trial(injector: "SingleShotInjector", reg_row: Dict, address: str, bit: int,
                   layer_end: str, hardfault_symbol: str,
                   golden_bytes: Optional[List[float]], whole_model: bool = False
                   ) -> Tuple[str, Optional[int], Dict[str, float]]:
    """Buffer-comparison trial. Returns (outcome, natural_bit,
    timings_snapshot); all phases wall-clock timed, including call-path
    navigation."""
    injector.reset_timings()

    with injector.timed("reset"):
        injector.reset_target()

    call_path = _call_path_of(reg_row)
    func_entry = _func_entry_of(reg_row)
    with injector.timed("navigate"):
        ok, reason = injector.navigate_call_path(call_path, func_entry)
    if not ok:
        print(f"[!] call-path navigation failed: {reason}")
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "DUE", None, snap

    with injector.timed("to_injection"):
        hit, timed_out = injector.run_to_address(address)
    if timed_out or not hit:
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "DUE", None, snap

    with injector.timed("inject"):
        natural_bit = injector.flip_bit(reg_row["Register"], bit)
    print(f"Injected: natural_bit={natural_bit} -> flipped to {1 - natural_bit}")

    with injector.timed("to_end"):
        status, _ = injector.run_to_address_or_hardfault(layer_end, hardfault_symbol)
    if status == "HARDFAULT":
        print(f"Hit {hardfault_symbol}")
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "CRASH", natural_bit, snap
    if status == "TIMEOUT":
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "DUE", natural_bit, snap
    if status != "TARGET":
        print(f"Unexpected stop reason ({status}), treating as DUE")
        snap = injector.snapshot_timings()
        _print_trial_timing(snap)
        return "DUE", natural_bit, snap

    print("Reached layer end cleanly — no hardfault.")
    faulty_bytes = None
    if golden_bytes is not None:
        if whole_model:
            faulty_bytes = injector.read_array_scaled(LOGITS_VAR, LOGITS_LEN, LOGITS_SCALE)
        else:
            faulty_bytes = injector.capture_output_buffer()
    outcome, detail = compare_buffers(golden_bytes, faulty_bytes)
    print(f"  {detail}")
    snap = injector.snapshot_timings()
    _print_trial_timing(snap)
    return outcome, natural_bit, snap


if __name__ == "__main__":
    main()
