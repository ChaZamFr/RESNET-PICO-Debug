#!/usr/bin/env python3
"""
Captures a GOLDEN (fault-free) run's output for use as
inject_transient_fault.py's --golden-buffer, real Masked/SDC
classification instead of the "no golden data -> Masked" fallback.

DUAL MODE — auto-detected from what's in the target table:

  WHOLE-MODEL mode (an 'fc' layer is present, meaning the table covers a
  full inference chain, not just one isolated layer): breaks at
  CAMPAIGN_END and reads g_logits/FI_PREDICTION — the actual 10-class
  Q8.8 fixed-point output of the whole network — via GDB's typed array
  print, matching read_array_scaled()'s exact scaling (divide by 256.0).
  CAMPAIGN_END is the ONLY point where g_logits and FI_PREDICTION are
  finalized; reading them at the fc layer's end address would capture
  them before they're valid. This matches exactly where every injection
  trial reads them (capture_golden_inline / run_one_trial_logits), so the
  golden reference this script writes is directly comparable. This is the
  right golden reference when you're running a full end-to-end campaign,
  since "did this fault change the final classification" is what actually
  matters, not one intermediate layer's raw feature map.

  SINGLE-LAYER mode (no 'fc' layer in the table — you're only profiling
  one isolated layer, e.g. just conv2d_1): falls back to the original
  behavior — breaks at that layer's own end address and reads its output
  buffer (buf_b, via capture_output_buffer()) as the golden reference.

Uses the exact same GDB connection/reset/breakpoint machinery as
inject_transient_fault.py (same SingleShotInjector class) — reuses its
LOGITS_VAR/LOGITS_LEN/LOGITS_SCALE and OUTPUT_BUFFER_SYMBOL/LENGTH config
constants directly, so configuring those once covers both scripts.

NOTE ON CALL PATHS: the golden run needs NO call-path navigation. It's a
clean, fault-free, full inference; reading g_logits at the fc/layer end
address is not tied to any specific dynamic call instance the way an
injection into a shared function (conv2d_2) is. So this script is
unaffected by the CallPath/FuncEntryAddr columns — it just runs to the
end address and reads the result.

State machine:
  1. Connect, reflash
  2. Detect mode (fc layer present? -> whole-model; else -> single-layer)
  3. Break at CAMPAIGN_END (whole-model) or the layer's end address
     (single-layer), continue
  4. Read either g_logits/FI_PREDICTION at CAMPAIGN_END (whole-model) or
     the layer's output buffer at its end address (single-layer), save as
     JSON

Usage:
    # Whole-model table (has an fc layer) -> reads g_logits automatically,
    # --layer is ignored in this mode
    python3 capture_golden_run.py fi_target_table.csv --output golden_logits.json

    # Single-layer table (no fc layer present) -> reads buf_b for --layer
    python3 capture_golden_run.py fi_target_table.csv --layer conv2d_1 --output golden_conv2d_1.json
"""

import argparse
import json
import re
from pathlib import Path

# Repointed from the old 'run_single_injection' name to the current
# transient-injection module, which defines SingleShotInjector and the
# shared config constants.
from inject_transient_fault import (
    SingleShotInjector, load_table, get_layer_end_address,
    PREDICTION_VAR, LOGITS_VAR, LOGITS_LEN, LOGITS_SCALE,
    CAMPAIGN_END_SYMBOL,
)

# ============================================================================
# EDITABLE CONFIG — same pattern as the other scripts.
# ============================================================================

TABLE_PATH = "fi_target_table.csv"
GDB_PATH = "arm-none-eabi-gdb"
ELF_PATH = "build/resnet_pico.elf"
TARGET_REMOTE = "localhost:3333"

LAYER = "conv2d_1"          # only used in SINGLE-LAYER mode (no fc layer in the table);
                             # ignored entirely in WHOLE-MODEL mode
OUTPUT_PATH = "golden_output.json"

DUE_TIMEOUT = 15.0

# ============================================================================

_TRAILING_INSTANCE_RE = re.compile(r"_(\d+|untracked)$")


def base_function_name(layer: str) -> str:
    """'conv2d_2' -> 'conv2d', 'fc_1' -> 'fc' — same convention as
    build_fi_target_table.py's --layers filter."""
    return _TRAILING_INSTANCE_RE.sub("", layer)


def find_layer_by_base(rows, base_name: str):
    """Returns the layer name matching base_name with the HIGHEST instance
    number (the "final"/most-recent call), or None if base_name isn't
    present anywhere in the table at all."""
    matches = {r["Layer"] for r in rows if base_function_name(r["Layer"]) == base_name}
    if not matches:
        return None

    def instance_num(layer):
        m = re.search(r"_(\d+)$", layer)
        return int(m.group(1)) if m else -1

    return max(matches, key=instance_num)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path, nargs="?", default=Path(TABLE_PATH))
    parser.add_argument("--layer", type=str, default=LAYER,
                         help="[single-layer mode only] which layer's buf_b to capture")
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    parser.add_argument("--gdb", default=GDB_PATH)
    parser.add_argument("--elf", default=ELF_PATH)
    parser.add_argument("--target-remote", default=TARGET_REMOTE)
    parser.add_argument("--campaign-end-symbol", default=CAMPAIGN_END_SYMBOL,
                         help="[whole-model mode] breakpoint where g_logits/FI_PREDICTION "
                              "are finalized — same point every injection trial reads them")
    parser.add_argument("--due-timeout", type=float, default=DUE_TIMEOUT)
    args = parser.parse_args()

    rows = load_table(args.table)

    fc_layer = find_layer_by_base(rows, "fc")
    whole_model = fc_layer is not None

    if whole_model:
        print(f"[i] Detected a full-model table (fc layer '{fc_layer}' present) — "
              f"WHOLE-MODEL mode: capturing golden output from {LOGITS_VAR}/"
              f"{PREDICTION_VAR} (Q8.8 logits, /{LOGITS_SCALE}) at {args.campaign_end_symbol}, "
              f"not a single layer's buffer. --layer is ignored in this mode.")
        target_layer = fc_layer
        # g_logits / FI_PREDICTION are only valid at CAMPAIGN_END — NOT at
        # the fc layer's end address. Break there, matching exactly where
        # every injection trial reads them.
        break_target = args.campaign_end_symbol
        print(f"Whole-model golden break point: {break_target}")
    else:
        print(f"[i] No fc layer found in this table — SINGLE-LAYER mode: "
              f"capturing golden output from layer '{args.layer}'s own output buffer.")
        target_layer = args.layer
        # Single-layer mode is unchanged: buf_b is valid at the layer's own
        # end address.
        break_target = get_layer_end_address(rows, target_layer)
        print(f"Layer: {target_layer}  end address: {break_target}")

    injector = SingleShotInjector(gdb_path=args.gdb, elf_path=args.elf,
                                   target_remote=args.target_remote, due_timeout=args.due_timeout)
    injector.start()
    injector.reflash()

    try:
        injector.reset_target()

        # run_to_target accepts a symbol (CAMPAIGN_END, whole-model) or a
        # raw address (layer end, single-layer) — _break_cmd picks the right
        # break syntax for each.
        hit, timed_out = injector.run_to_target(break_target)
        if timed_out or not hit:
            print("[!] Hang/crash reaching the breakpoint during golden run — "
                  "this shouldn't happen on fault-free hardware. Check your ELF/"
                  "symbol/layer/timeout.")
            return

        print(f"Reached {break_target} cleanly.")

        if whole_model:
            buffer_values = injector.read_array_scaled(LOGITS_VAR, LOGITS_LEN, LOGITS_SCALE)
            if buffer_values is not None:
                predicted_class = injector.read_variable(PREDICTION_VAR)
                print(f"Golden logits: {buffer_values}")
                if predicted_class is not None:
                    print(f"Golden predicted class ({PREDICTION_VAR}): {predicted_class}")
                else:
                    print(f"[i] Could not read {PREDICTION_VAR} — if you don't have that global, "
                          f"the argmax of the logits above is the predicted class.")
        else:
            buffer_values = injector.capture_output_buffer()

        if buffer_values is None:
            missing = f"{LOGITS_VAR}/{LOGITS_LEN}" if whole_model else "OUTPUT_BUFFER_SYMBOL/LENGTH"
            print(f"[!] Read returned None — nothing was saved. Check {missing} "
                  f"in inject_transient_fault.py's config block.")
            return

        args.output.write_text(json.dumps(buffer_values))
        print(f"Saved golden {'logits' if whole_model else 'buffer'} "
              f"({len(buffer_values)} values) to: {args.output}")
        if whole_model:
            print(f"Use it with: python3 inject_transient_fault.py --golden-buffer {args.output}")
        else:
            print(f"Use it with: python3 inject_transient_fault.py --layer {target_layer} "
                  f"--golden-buffer {args.output}")

    finally:
        injector.close()


if __name__ == "__main__":
    main()
