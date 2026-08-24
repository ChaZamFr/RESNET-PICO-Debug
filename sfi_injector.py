#!/usr/bin/env python3
"""
SFI campaign injector — the "Injection loop" state machine from the design
diagram, implemented against GDB, extending the same sentinel-framed
command pattern as function_profiler.py.

State machine per trial:
    Select target -> [navigate call path] -> Halt at target PC -> Inject
    fault -> Resume and capture -> Classify vs golden

CALL-PATH NAVIGATION (navigate_call_path): conv2d is one function reached
through many call paths (directly, and inside each resblock). A raw
`break *0xADDR` fires on EVERY invocation and GDB stops at the first, so an
address alone can't target "conv2d_2". The profiler now records, per target,
the chain of `bl` call-site addresses walked into to reach that instance
(CallPath) plus the function's entry address (FuncEntryAddr).
navigate_call_path() replays that chain — break each bl, continue, si in —
then verifies the landing PC equals FuncEntryAddr, so the subsequent
injection-address breakpoint fires on the correct invocation.

Design decisions locked in (see conversation for rationale):
  - Population source: value-stability epochs (value_epoch_table.csv)
  - Fault persistence: PERMANENT, implemented as a single bit-force at the
    epoch's first breakpoint hit. This works because a value epoch is by
    construction a span where nothing rewrites the register — so one flip
    that's never re-touched is equivalent to a stuck-at fault for the
    whole epoch, no per-instruction re-forcing needed.
  - Sampling: exhaustive per epoch by default (32 bits), matching
    sample_size.py's finding that statistical sampling doesn't save
    anything at this population size.

IMPORTANT ADAPTATION FROM THE PAPER: the paper's BER/excitation count is
computed across many parallel GPU threads naturally taking different
register values for the same static instruction. This board is single-
core/single-threaded, so there is no parallel population at one instant —
the equivalent axis here is the TEST DATASET. The campaign loop below
therefore runs each (register, epoch, bit) population once per test
input, not just once. Excitation rate for a population = fraction of test
inputs where the register's natural (golden) bit value differed from the
forced stuck-at value.

HARDWARE-SPECIFIC INTEGRATION POINTS (marked TODO below) — these depend on
your firmware's serial protocol and aren't something I can fill in without
seeing resnet_pico.c's console format:
  - feed_input(): how to select/load a specific test image before
    triggering inference over your existing USB serial console
  - capture_output(): how to read back the predicted class / logits after
    inference completes
  - DUE detection: currently timeout-only; if your firmware has a
    HardFault_Handler symbol, breakpointing it gives a faster/cleaner
    DUE signal than waiting out the timeout
"""

import csv
import re
import subprocess
import threading
import queue
import time
import itertools
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# EDITABLE CONFIG — change these instead of remembering CLI flags every time.
# CLI flags below override these if passed, so either editing this block or
# using the command line works.
# ============================================================================

GDB_PATH = "arm-none-eabi-gdb"
ELF_PATH = "build/resnet_pico.elf"
WORKLIST_PATH = "fault_worklist.csv"      # output of fault_models.py
RESULTS_PATH = "sfi_results.csv"

# ============================================================================



class SFIInjector:
    def __init__(self, gdb_path: str, elf_path: str, target_remote: str = "localhost:3333",
                 due_timeout: float = 15.0):
        self.gdb_path = gdb_path
        self.elf_path = elf_path
        self.target_remote = target_remote
        self.due_timeout = due_timeout

        self.q = queue.Queue()
        self.gdb: Optional[subprocess.Popen] = None
        self._sentinel_counter = itertools.count()

        # --- Timing ---
        # Every phase timed via time.perf_counter() (monotonic, high-res,
        # unaffected by wall-clock adjustments — the right clock for
        # measuring durations). self.timings accumulates labelled phase
        # durations; the timed() context manager writes into it. Callers
        # snapshot/clear it per trial (see reset_timings()/snapshot_timings()).
        self.timings: Dict[str, float] = {}

    # --- Same sentinel-framed GDB I/O as function_profiler.py ---
    def start(self):
        self.gdb = subprocess.Popen(
            [self.gdb_path, "--nx"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._enqueue_output, args=(self.gdb.stdout, self.q), daemon=True).start()
        time.sleep(1)
        # NOTE: deliberately NOT doing "monitor reset halt" / "load" here —
        # every caller already calls reset_target() explicitly before its
        # first real use (run_trial() does it per-trial; run_single_injection
        # main() does it once before the single trial). Doing it here too
        # meant every run reset+reflashed the target TWICE back-to-back,
        # which is wasteful and can destabilize the SWD link right after a
        # reset on some OpenOCD/probe combinations.
        for cmd in (f"file {self.elf_path}", f"target extended-remote {self.target_remote}",
                    "set remotetimeout 60",
                    "set pagination off", "set confirm off",
                    # Critical for read_array_scaled() on large arrays like a
                    # 32*32*16=16384-element feature map buffer: GDB's default
                    # print limit is 200 elements, after which it prints '...'
                    # with NO closing '}' — silently breaking any regex that
                    # expects a matched brace. 'set print elements 0' removes
                    # the limit entirely. 'set print repeats 0' additionally
                    # stops GDB collapsing runs of identical values into
                    # "<repeats N times>", which would also break parsing.
                    "set print elements 0", "set print repeats 0"):
            self._run(cmd)

    def _enqueue_output(self, out, q):
        for line in iter(out.readline, ""):
            q.put(line)
        out.close()

    def _run(self, cmd: str, timeout: float = 20.0, quiet: bool = False) -> str:
        if not quiet:
            print(f"[GDB] {cmd}")
        sentinel = f"__DONE_{next(self._sentinel_counter)}__"
        self.gdb.stdin.write(cmd + "\n")
        self.gdb.stdin.write(f"echo {sentinel}\\n\n")
        self.gdb.stdin.flush()
        buf, start = "", time.time()
        while time.time() - start < timeout:
            while not self.q.empty():
                buf += self.q.get_nowait()
            if sentinel in buf:
                return buf.split(sentinel)[0]
            time.sleep(0.01)
        print(f"[!] Timeout waiting for sentinel after: {cmd}")
        return buf

    def _run_until_stop(self, cmd: str, timeout: float, quiet: bool = False) -> Optional[str]:
        if not quiet:
            print(f"[GDB] {cmd}")
        sentinel = f"__DONE_{next(self._sentinel_counter)}__"
        self.gdb.stdin.write(cmd + "\n")
        self.gdb.stdin.write(f"echo {sentinel}\\n\n")
        self.gdb.stdin.flush()
        buf, start = "", time.time()
        while time.time() - start < timeout:
            while not self.q.empty():
                buf += self.q.get_nowait()
            if sentinel in buf:
                return buf.split(sentinel)[0]
            time.sleep(0.01)
        return None  # timeout -> caller treats as potential DUE

    # --- Timing helpers -------------------------------------------------
    @contextmanager
    def timed(self, label: str):
        """
        Times the enclosed block and records the duration (seconds) under
        `label` in self.timings. Uses perf_counter() — monotonic and
        high-resolution, the correct clock for measuring elapsed time.
        The duration is recorded even if the block raises, so a phase that
        crashes/times out still shows how long it took before failing.

            with injector.timed("run_to_injection"):
                injector.run_to_address(addr)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[label] = time.perf_counter() - t0

    def reset_timings(self):
        """Clear accumulated phase timings — call at the start of each
        trial so self.timings only holds the current trial's phases."""
        self.timings = {}

    def snapshot_timings(self) -> Dict[str, float]:
        """Return a copy of the current phase timings plus a 'total'
        (sum of all recorded phases). Copy so the caller can stash it
        before the next trial overwrites self.timings."""
        snap = dict(self.timings)
        snap["total"] = sum(self.timings.values())
        return snap

    def close(self):
        if self.gdb:
            # 'set confirm off' (in start()) should mean this never blocks
            # on a y/n prompt anymore, but use a short timeout here anyway
            # — if GDB doesn't respond for any other reason (e.g. target
            # already disconnected), don't let shutdown hang the script.
            # terminate()/kill() run regardless of what _run() returned.
            try:
                self._run("quit", timeout=3.0, quiet=True)
            except Exception:
                pass
            self.gdb.terminate()
            try:
                self.gdb.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.gdb.kill()

    # --- Reset target to a clean state before each trial ---
    def reflash(self):
        """
        Reprograms flash and resets. This is the SLOW operation (~2.6s+
        for this ELF) that trips OpenOCD's keep_alive warning — call it
        ONCE per session (right after start(), before any trials), not
        per-trial. The flash image doesn't change between injection
        trials; only register/RAM state does, and that's already wiped
        clean by reset_target()'s plain reset.
        """
        self._run("monitor reset halt")
        self._run("load")

    def reset_target(self):
        """
        Cheap per-trial reset: halts and resets CPU/register state WITHOUT
        rewriting flash. Safe to call before every single trial — flash
        content persists across a reset (it's non-volatile), so nothing
        here needs reprogramming. This is what eliminates the keep_alive
        warning from campaigns: previously this method did a full reflash
        on every trial (N reflashes for an N-trial campaign); now it's
        just the reset, and reflash() runs exactly once per session.
        """
        self._run("monitor reset halt")

    # --- TODO: hardware-specific hooks ---
    def feed_input(self, input_id: str):
        """
        Select/load test input `input_id` before triggering inference.
        Wire this to your USB serial console's existing input-selection
        command (mentioned in your earlier interactive console work).
        """
        raise NotImplementedError("Wire this to your firmware's input-selection protocol")

    def capture_output(self) -> Dict:
        """
        Read back the DNN's output after inference completes (predicted
        class, and ideally raw logits for a finer-grained classification
        than golden-vs-faulty label match alone).
        Returns e.g. {"predicted_class": 7, "logits": [...]} .
        """
        raise NotImplementedError("Wire this to your firmware's output-reporting protocol")

    # --- Memory capture (used for golden/output buffer comparison) ---
    @staticmethod
    def _addr_expr(token: str) -> str:
        """
        Turns either a bare hex/decimal address ("0x20001000") or a named
        C symbol ("conv2d_out") into a GDB expression that evaluates to
        its starting address. "&symbol" gives the correct numeric address
        for both arrays and scalars in C, so this works uniformly for any
        named global — you only need to know the buffer's NAME, not its
        literal address.
        """
        t = token.strip()
        if t.lower().startswith("0x") or t.isdigit():
            return t
        return f"&{t}"

    def dump_memory_to_file(self, symbol_or_addr: str, length: int, out_path: str) -> bool:
        """
        Uses GDB's built-in `dump binary memory` command to write raw
        bytes DIRECTLY to a file on the host machine — no hex-dump output
        to parse, no manual byte-reconstruction, so this is far more
        robust than scraping `x/Nxb` text output.
        Returns True if the dump appears to have succeeded.
        """
        start = self._addr_expr(symbol_or_addr)
        out = self._run(f"dump binary memory {out_path} ({start}) ({start})+{length}")
        lowered = out.lower()
        if "no symbol" in lowered or "error" in lowered or "cannot access memory" in lowered:
            print(f"[!] dump_memory_to_file failed for '{symbol_or_addr}': {out.strip()}")
            return False
        return True

    def read_memory_bytes(self, symbol_or_addr: str, length: int, scratch_path: str = "/tmp/_gdb_dump.bin") -> Optional[bytes]:
        """Convenience wrapper: dump to a scratch file, read it back as bytes, clean up."""
        if not self.dump_memory_to_file(symbol_or_addr, length, scratch_path):
            return None
        try:
            data = Path(scratch_path).read_bytes()
        except OSError as e:
            print(f"[!] Could not read back dumped file {scratch_path}: {e}")
            return None
        finally:
            Path(scratch_path).unlink(missing_ok=True)
        return data

    # --- Fault injection primitive ---
    def read_register(self, reg: str) -> int:
        out = self._run(f"p/x ${reg}", quiet=True)
        m = re.search(r"=\s*(0x[0-9a-fA-F]+)", out)
        return int(m.group(1), 16) if m else 0

    def read_pc(self) -> Optional[int]:
        """
        Reads the program counter as an integer. Used by
        navigate_call_path() to verify, after replaying a call path and
        stepping in, that execution actually landed at the target
        function's entry. Cast through _reg_expr so the pointer-typed pc
        is returned as a plain number.
        """
        out = self._run(f"p/x {self._reg_expr('pc')}", quiet=True)
        m = re.search(r"=\s*(0x[0-9a-fA-F]+)", out)
        return int(m.group(1), 16) if m else None

    def read_variable(self, var: str) -> Optional[int]:
        """
        Reads any C variable by name via GDB's own expression evaluator
        ('p var') — no address/size bookkeeping needed, unlike a raw
        memory dump. Works for any scalar global GDB can see given the
        loaded ELF's debug info.
        """
        out = self._run(f"p {var}", quiet=True)
        m = re.search(r"=\s*(-?\d+)", out)
        return int(m.group(1)) if m else None

    def read_array_scaled(self, var: str, length: int, scale: float = 1.0) -> Optional[List[float]]:
        """
        Reads `length` elements of an array/pointer variable via GDB's
        '@' artificial-array syntax (e.g. 'p g_logits[0]@10'), dividing
        each raw integer by `scale` — e.g. scale=256.0 for Q8.8
        fixed-point values. Returns None if the array print didn't match
        the expected '{a, b, c, ...}' shape.
        """
        out = self._run(f"p {var}[0]@{length}", quiet=True)
        m = re.search(r"\{([^}]*)\}", out)
        if not m:
            return None
        try:
            return [int(x.strip()) / scale for x in m.group(1).split(",")]
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Pointer-register cast helper.
    #
    # GDB types sp/lr/pc (and their numeric aliases r13/r14/r15) as
    # pointer types (void (*)() / void *) from the ELF's debug info. Its
    # C-like expression evaluator refuses bitwise/shift arithmetic on a
    # pointer operand, so a bare `$sp ^ (1 << bit)` or `($pc >> bit) & 1`
    # dies with:
    #     "Argument to arithmetic operation not a number or boolean."
    # General-purpose r0-r7 come back as plain integers and work without
    # a cast. Casting to (unsigned int) makes the operand integer-typed
    # for ALL registers; for r0-r7 the value round-trips identically, so
    # casting unconditionally is safe and keeps every call site branch-free.
    # This is the single choke point every register read/write arithmetic
    # expression below routes through.
    # ------------------------------------------------------------------
    @staticmethod
    def _reg_expr(reg: str) -> str:
        """GDB expression for reading `reg` as an integer, safe for the
        pointer-typed registers (sp/lr/pc/r13/r14/r15) as well as GPRs."""
        return f"(unsigned int)${reg}"

    def read_bit(self, reg: str, bit: int) -> int:
        """
        Reads just bit `bit` of `reg` via GDB's own arithmetic
        ('p ((unsigned int)$reg >> bit) & 1') — safer than read_register()
        for this purpose, since we only ever need to parse a clean 0/1
        result instead of a full 32-bit hex value. The (unsigned int) cast
        is required for pointer-typed registers (sp/lr/pc); see _reg_expr().
        """
        out = self._run(f"p ({self._reg_expr(reg)} >> {bit}) & 1", quiet=True)
        m = re.search(r"=\s*(-?\d+)", out)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _break_cmd(target: str) -> str:
        """'0x...' -> break at raw address (needs '*'); anything else ->
        break at symbol name (no '*'). Shared by transient (single-shot)
        and permanent (watch-and-reinforce) trial logic."""
        return f"break *{target}" if target.lower().startswith("0x") else f"break {target}"

    # ------------------------------------------------------------------
    # Call-path navigation — the fix for injecting into a SPECIFIC dynamic
    # call instance of a shared function (conv2d_2 vs conv2d_1, etc).
    # ------------------------------------------------------------------
    def navigate_call_path(self, call_path: str, func_entry_addr: str = "",
                            timeout: Optional[float] = None) -> Tuple[bool, str]:
        """
        Replays a recorded call path so a subsequent injection-address
        breakpoint fires on the CORRECT invocation of a shared function.

        `call_path` is the ';'-joined list of `bl` call-site addresses from
        the target table's CallPath column (e.g. "0x100005b4", or
        "0x...;0x..." for a deeper instance). For each hop, in order:
            break *<bl addr>   -> continue -> delete it -> stepi (into callee)
        After the last hop, if `func_entry_addr` is given, verifies the PC
        landed exactly there — catching table/ELF drift instead of silently
        injecting in the wrong place.

        An EMPTY call_path is a no-op success: the instance is reached
        directly (e.g. conv2d_1), so no navigation is needed — the caller
        just arms the injection address as before.

        MUST be called right after reset_target() (execution at the reset
        vector), so the hop breakpoints are hit in their natural first-time
        order from the start of the run.

        Returns (ok, reason). On failure ok=False and `reason` names exactly
        what went wrong (which hop wasn't reached, or expected-vs-actual PC),
        so the caller can log it as the DUE detail. On success reason="".
        """
        timeout = timeout if timeout is not None else self.due_timeout
        hops = [h.strip() for h in call_path.split(";") if h.strip()] if call_path else []

        if not hops:
            return True, ""  # direct-reach instance — nothing to navigate

        for i, bl_addr in enumerate(hops):
            # Break at this call site and run to it. Clean up ALL breakpoints
            # after each hop so a leftover bp from an earlier hop can't be the
            # thing that fires on the next continue.
            self._run(self._break_cmd(bl_addr))
            out = self._run_until_stop("continue", timeout=timeout)
            self._run("delete")

            if out is None:
                return False, (f"call-path navigation timed out reaching hop {i+1}/"
                               f"{len(hops)} (bl site {bl_addr}) — target may not have "
                               f"reached this call, or timeout too short")
            if "Breakpoint" not in out:
                # Could be a hardfault or unexpected stop before the bl site.
                return False, (f"call-path navigation did not hit hop {i+1}/{len(hops)} "
                               f"(bl site {bl_addr}); stop output did not contain a "
                               f"breakpoint hit (possible crash before the call)")

            # Step from the bl INTO the callee. One stepi enters the called
            # function because the bl is a single instruction and we're
            # stopped exactly on it.
            self._run("stepi", quiet=True)

        # Verify we landed where the table says this function starts.
        if func_entry_addr:
            want = func_entry_addr.strip().lower()
            pc = self.read_pc()
            if pc is None:
                return False, "call-path navigation: could not read PC after final stepi to verify landing"
            got = f"0x{pc:x}"
            if got.lower() != want:
                return False, (f"call-path navigation landed at {got} but the table's "
                               f"FuncEntryAddr is {func_entry_addr} — table/ELF drift or "
                               f"a wrong CallPath; refusing to inject at the wrong place")

        return True, ""

    def hold_bit_and_run_to(self, reg: str, bit: int, stuck_value: int, target: str,
                             hardfault_symbol: str, max_reinforcements: int = 50
                             ) -> Tuple[str, int]:
        """
        GENUINE permanent stuck-at: forces reg's bit to stuck_value, then
        arms a hardware watchpoint on that bit and RE-ENFORCES the stuck
        value every single time the watchpoint fires (i.e. every time any
        instruction changes it), continuing until `target` (address or
        symbol) is reached, hardfault_symbol fires, a timeout happens, or
        max_reinforcements is exceeded.

        This is what makes the fault ACTUALLY persist for the whole
        observation window — a one-shot force (force_bit/flip_bit) only
        holds until the next instruction that writes the register, which
        for most GPRs is very soon. A permanent stuck-at fault, by
        definition, has to keep winning that race for as long as you're
        observing it.

        Returns (status, reinforcement_count) where status is one of:
        "TARGET" (reached target, still enforced), "HARDFAULT", "TIMEOUT",
        "REINFORCEMENT_LIMIT" (fired more than max_reinforcements times —
        likely means this bit is written every iteration of a tight loop;
        treat as its own outcome category, not a normal DUE/CRASH, since
        it reflects the fault model's cost rather than the target's
        behavior).
        """
        natural = self.read_bit(reg, bit)
        if natural != stuck_value:
            self._run(f"set ${reg} = {self._reg_expr(reg)} ^ (1 << {bit})")

        self._run(self._break_cmd(target))
        hf_armed = False
        out = self._run(f"break {hardfault_symbol}", quiet=True)
        if "not defined" not in out.lower() and "no symbol" not in out.lower():
            hf_armed = True

        # Hardware watchpoint on just this bit (not the whole register) —
        # fires only when THIS bit's value actually changes, not on every
        # write to the register that happens to leave it alone. The
        # (unsigned int) cast is required for pointer-typed registers
        # (sp/lr/pc), same reason as read_bit()/force_bit(); a bare
        # `watch ($sp >> bit) & 1` throws the same arithmetic-type error.
        self._run(f"watch ({self._reg_expr(reg)} >> {bit}) & 1")

        reinforcements = 0
        while True:
            out = self._run_until_stop("continue", timeout=self.due_timeout)
            if out is None:
                return "TIMEOUT", reinforcements

            if hf_armed and hardfault_symbol in out:
                return "HARDFAULT", reinforcements

            if "Breakpoint" in out and "Watchpoint" not in out and "watchpoint" not in out:
                # Reached target (or another breakpoint) with the fault
                # still correctly enforced — watchpoint didn't fire since
                # the last check, so it's still stuck at stuck_value.
                self._run("delete")
                return "TARGET", reinforcements

            if "atchpoint" in out:  # matches both "Watchpoint" and "Hardware watchpoint"
                reinforcements += 1
                if reinforcements > max_reinforcements:
                    self._run("delete")
                    return "REINFORCEMENT_LIMIT", reinforcements
                current = self.read_bit(reg, bit)
                if current != stuck_value:
                    self._run(f"set ${reg} = {self._reg_expr(reg)} ^ (1 << {bit})")
                continue

            # Unknown stop reason — treat conservatively as a timeout
            # rather than silently proceeding with an unenforced fault.
            return "TIMEOUT", reinforcements

    def force_bit(self, reg: str, bit: int, stuck_value: int) -> int:
        """
        Forces `reg`'s bit `bit` to stuck_value (0 or 1). The actual write
        is delegated ENTIRELY to GDB's own expression evaluator —
        'set $reg = (unsigned int)$reg ^ (1 << bit)' — the same pattern
        confirmed working directly at the GDB prompt, rather than
        reconstructing a full replacement value in Python and overwriting
        the whole register with it.

        This matters: the old approach read the current value via
        read_register(), computed (current | (1<<bit)) or
        (current & ~(1<<bit)) in Python, then sent that ENTIRE value as
        the new register contents. If read_register() ever misparsed
        (returning 0, its silent failure fallback), the register would
        get overwritten down to just the bit mask itself — e.g. bit=5,
        stuck_value=1 on a failed read produces exactly $r2 = 32, not a
        real toggle. Letting GDB compute the XOR itself means a read
        failure here can at most cause a wrong TOGGLE DECISION (skip or
        redundantly re-toggle), never a wholesale register overwrite.

        Only toggles if the natural bit doesn't already equal stuck_value
        — this is what preserves stuck-at semantics (force TO a value)
        while still going through a pure toggle at the GDB level.
        Returns the natural (pre-injection) bit value.

        The (unsigned int) cast on the read side of the XOR is required
        for pointer-typed registers (sp/lr/pc/r13/r14/r15), which GDB
        otherwise refuses to do arithmetic on; see _reg_expr().
        """
        natural = self.read_bit(reg, bit)
        if natural != stuck_value:
            self._run(f"set ${reg} = {self._reg_expr(reg)} ^ (1 << {bit})")
        return natural

    def flip_bit(self, reg: str, bit: int) -> int:
        """
        Unconditional bit flip (classic SEU/transient-fault model) —
        exactly 'set $reg = (unsigned int)$reg ^ (1 << bit)', no stuck-at
        target value, always changes that bit regardless of its current
        state. Returns the natural (pre-flip) bit value, for logging
        purposes only — unlike force_bit(), the decision to write never
        depends on it.

        The (unsigned int) cast is required for pointer-typed registers
        (sp/lr/pc/r13/r14/r15): GDB types them as pointers and rejects a
        bare `$sp ^ (1 << bit)` with "Argument to arithmetic operation not
        a number or boolean." GPRs (r0-r7) round-trip identically through
        the cast, so it's applied unconditionally. See _reg_expr().
        """
        natural = self.read_bit(reg, bit)
        self._run(f"set ${reg} = {self._reg_expr(reg)} ^ (1 << {bit})")
        return natural

    def force_bits(self, reg: str, bits_and_values: List[Tuple[int, int]]) -> bool:
        """
        Force several bits of `reg` at once — the MULTI_BIT fault class
        (mirrors SACA-FI's bitflip2(), which flips a list of positions
        together rather than one at a time). Single read, single write.
        Returns whether any bit was actually excited (natural value
        differed from the forced stuck value).

        Note this path builds the full replacement value in Python and
        writes it as a plain integer literal ('set $reg = 12345'), so the
        RHS has no `$reg` operand and needs no cast. The READ side uses
        read_register() ('p/x $reg'), which is a bare pointer print GDB
        accepts for sp/lr/pc without a cast — so this method already works
        for pointer registers as-is.
        """
        current = self.read_register(reg)
        new_val = current
        excited = False
        for bit, stuck_value in bits_and_values:
            natural = (current >> bit) & 1
            if natural != stuck_value:
                excited = True
            if stuck_value:
                new_val |= (1 << bit)
            else:
                new_val &= ~(1 << bit)
        self._run(f"set ${reg} = {new_val}")
        return excited

    # --- Run to a specific PC address, with DUE-on-timeout ---
    def run_to_address(self, address: str) -> Tuple[bool, bool]:
        """Returns (hit_breakpoint, timed_out). Address form only — always
        prefixes '*'. For a target that may be a SYMBOL (e.g. CAMPAIGN_END),
        use run_to_target() instead."""
        self._run(f"break *{address}")
        out = self._run_until_stop("continue", timeout=self.due_timeout)
        self._run("delete")
        if out is None:
            return False, True
        return ("Breakpoint" in out), False

    def run_to_target(self, target: str) -> Tuple[bool, bool]:
        """Like run_to_address but accepts EITHER a raw address ('0x...')
        or a symbol name ('CAMPAIGN_END') — routes through _break_cmd so a
        symbol gets 'break CAMPAIGN_END' (no '*') and an address gets
        'break *0x...'. Returns (hit_breakpoint, timed_out)."""
        self._run(self._break_cmd(target))
        out = self._run_until_stop("continue", timeout=self.due_timeout)
        self._run("delete")
        if out is None:
            return False, True
        return ("Breakpoint" in out), False

    def run_to_completion(self) -> Tuple[bool, bool]:
        """Continue with no further breakpoints until natural exit/timeout."""
        out = self._run_until_stop("continue", timeout=self.due_timeout)
        return (out is not None), (out is None)

    # --- One full trial: select target -> halt -> inject -> resume -> classify ---
    def run_trial(self, reg: str, layer: str, start_pc: str, bit: int, stuck_value: int,
                   input_id: str, golden_output: Dict,
                   call_path: str = "", func_entry_addr: str = "") -> Dict:
        """SINGLE_BIT / HARD_STUCK trial. Now navigates the call path (if
        any) after reset so the injection address fires on the correct
        invocation of a shared function."""
        self.reset_target()
        self.feed_input(input_id)

        ok, reason = self.navigate_call_path(call_path, func_entry_addr)
        if not ok:
            return self._result([{"reg": reg, "layer": layer, "start_pc": start_pc,
                                    "bit": bit, "stuck_value": stuck_value}],
                                 input_id, outcome="DUE", detail=reason)

        hit, timed_out = self.run_to_address(start_pc)
        if timed_out or not hit:
            return self._result([{"reg": reg, "layer": layer, "start_pc": start_pc,
                                    "bit": bit, "stuck_value": stuck_value}],
                                 input_id, outcome="DUE", detail="hang before reaching injection point")

        natural_bit = self.read_bit(reg, bit)
        excited = natural_bit != stuck_value
        self.force_bit(reg, bit, stuck_value)

        ran_ok, timed_out = self.run_to_completion()
        if timed_out:
            return self._result([{"reg": reg, "layer": layer, "start_pc": start_pc,
                                    "bit": bit, "stuck_value": stuck_value}],
                                 input_id, outcome="DUE", detail="hang/crash after injection", excited=excited)

        faulty_output = self.capture_output()
        outcome = self._classify(golden_output, faulty_output)
        return self._result([{"reg": reg, "layer": layer, "start_pc": start_pc,
                                "bit": bit, "stuck_value": stuck_value}],
                             input_id, outcome=outcome, detail="", excited=excited,
                             faulty_output=faulty_output)

    def run_trial_multi_bit(self, reg: str, layer: str, start_pc: str,
                              bits_and_values: List[Tuple[int, int]],
                              input_id: str, golden_output: Dict,
                              call_path: str = "", func_entry_addr: str = "") -> Dict:
        """MULTI_BIT trial — several bits forced together in ONE register
        at ONE epoch/PC, in a single read-modify-write."""
        self.reset_target()
        self.feed_input(input_id)

        fault_desc = [{"reg": reg, "layer": layer, "start_pc": start_pc, "bit": b, "stuck_value": v}
                       for b, v in bits_and_values]

        ok, reason = self.navigate_call_path(call_path, func_entry_addr)
        if not ok:
            return self._result(fault_desc, input_id, outcome="DUE", detail=reason)

        hit, timed_out = self.run_to_address(start_pc)
        if timed_out or not hit:
            return self._result(fault_desc, input_id, outcome="DUE",
                                 detail="hang before reaching injection point")

        excited = self.force_bits(reg, bits_and_values)

        ran_ok, timed_out = self.run_to_completion()
        if timed_out:
            return self._result(fault_desc, input_id, outcome="DUE",
                                 detail="hang/crash after injection", excited=excited)

        faulty_output = self.capture_output()
        outcome = self._classify(golden_output, faulty_output)
        return self._result(fault_desc, input_id, outcome=outcome, detail="",
                             excited=excited, faulty_output=faulty_output)

    def run_trial_multi_err(self, faults: List[Dict], input_id: str, golden_output: Dict) -> Dict:
        """
        MULTI_ERR trial — several INDEPENDENT faults across different
        (register, epoch) populations, injected within one run in PC
        order. `faults` must already be sorted by start_pc (fault_models.py
        does this — mirrors SACA-FI's sort_err()).

        Mechanically: halt at fault 1's PC, inject, resume toward fault
        2's PC (set as the next breakpoint), inject, ... until all faults
        are placed, then run to completion and classify once for the
        whole group.

        NOTE: call-path navigation is not yet wired for MULTI_ERR (each
        fault could have its own path). If you start using MULTI_ERR with
        shared-function instances, this needs per-fault navigation; for now
        each fault's start_pc is assumed to be reachable directly.
        """
        self.reset_target()
        self.feed_input(input_id)

        any_excited = False
        for f in faults:
            hit, timed_out = self.run_to_address(f["start_pc"])
            if timed_out or not hit:
                return self._result(faults, input_id, outcome="DUE",
                                     detail=f"hang before reaching {f['register']}@{f['start_pc']}")
            reg_name = f["reg"] if "reg" in f else f["register"]
            natural = self.read_bit(reg_name, f["bit"])
            if natural != f["stuck_value"]:
                any_excited = True
            self.force_bit(reg_name, f["bit"], f["stuck_value"])

        ran_ok, timed_out = self.run_to_completion()
        if timed_out:
            return self._result(faults, input_id, outcome="DUE",
                                 detail="hang/crash after all injections", excited=any_excited)

        faulty_output = self.capture_output()
        outcome = self._classify(golden_output, faulty_output)
        return self._result(faults, input_id, outcome=outcome, detail="",
                             excited=any_excited, faulty_output=faulty_output)

    def _classify(self, golden: Dict, faulty: Dict) -> str:
        # Masked: identical prediction. Safe-SDC: numbers changed but final
        # class prediction didn't. Critical-SDC: final class changed.
        # Matches the paper's four-way classification (Section III-C).
        if faulty.get("predicted_class") == golden.get("predicted_class"):
            if faulty.get("logits") == golden.get("logits"):
                return "Masked"
            return "Safe_SDC"
        return "Critical_SDC"

    def _result(self, faults: List[Dict], input_id: str, outcome: str, detail: str,
                excited=None, faulty_output=None) -> Dict:
        """faults is a list so SINGLE_BIT (len 1), MULTI_BIT (len 1 desc +
        many bits), and MULTI_ERR (many descs) all share one result shape."""
        fault_class = ("SINGLE_BIT" if len(faults) == 1
                        else "MULTI_ERR" if len({f.get("register", f.get("reg")) for f in faults}) > 1
                        else "MULTI_BIT")
        return {
            "FaultClass": fault_class,
            "Faults": "; ".join(f"{f.get('register', f.get('reg'))}@{f['start_pc']} bit{f['bit']}={f['stuck_value']}"
                                  for f in faults),
            "InputID": input_id, "Outcome": outcome,
            "Excited": excited, "Detail": detail,
            "FaultyPredictedClass": (faulty_output or {}).get("predicted_class", ""),
        }


def load_worklist(worklist_csv: Path) -> Dict[int, List[Dict]]:
    """
    Loads fault_worklist.csv (from fault_models.py) and groups rows by
    FaultGroupID. A SINGLE_BIT/HARD_STUCK group has 1 row; a MULTI_BIT
    group has 2-6 rows sharing one register; a MULTI_ERR group has 2-6
    rows spanning different registers/populations, already PC-sorted by
    fault_models.py.
    """
    groups: Dict[int, List[Dict]] = {}
    with open(worklist_csv, newline="") as f:
        for row in csv.DictReader(f):
            gid = int(row["fault_group_id"])
            groups.setdefault(gid, []).append(row)
    return groups


def run_campaign(injector: SFIInjector, groups: Dict[int, List[Dict]],
                  golden_outputs: Dict[str, Dict], results_path: Path):
    """
    Outer loop over fault groups, dispatching each to the right trial
    method based on fault_class. This replaces the old single-purpose
    nested-loop version — worklist composition (which fault classes, how
    many, with what counts) now lives entirely in fault_models.py.

    CallPath / FuncEntryAddr are read from the worklist row when present
    (fault_models.py should carry them through from the target table) and
    passed to the trial so shared-function instances land correctly.
    """
    fields = ["FaultClass", "Faults", "InputID", "Outcome", "Excited", "Detail", "FaultyPredictedClass"]
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for gid, rows in sorted(groups.items()):
            fault_class = rows[0]["fault_class"]
            input_id = rows[0]["input_id"]
            golden = golden_outputs[input_id]

            if fault_class in ("SINGLE_BIT", "HARD_STUCK"):
                r = rows[0]
                result = injector.run_trial(r["register"], r["layer"], r["start_pc"],
                                             int(r["bit"]), int(r["stuck_value"]), input_id, golden,
                                             call_path=r.get("CallPath", ""),
                                             func_entry_addr=r.get("FuncEntryAddr", ""))

            elif fault_class == "MULTI_BIT":
                r0 = rows[0]
                bits_and_values = [(int(r["bit"]), int(r["stuck_value"])) for r in rows]
                result = injector.run_trial_multi_bit(r0["register"], r0["layer"], r0["start_pc"],
                                                        bits_and_values, input_id, golden,
                                                        call_path=r0.get("CallPath", ""),
                                                        func_entry_addr=r0.get("FuncEntryAddr", ""))

            elif fault_class == "MULTI_ERR":
                faults = [{"register": r["register"], "layer": r["layer"], "start_pc": r["start_pc"],
                           "bit": int(r["bit"]), "stuck_value": int(r["stuck_value"])} for r in rows]
                result = injector.run_trial_multi_err(faults, input_id, golden)

            else:
                print(f"[!] Unknown fault_class '{fault_class}' in group {gid}, skipping")
                continue

            writer.writerow(result)
            f.flush()
            print(f"[group {gid} | {fault_class} | in={input_id}] -> {result['Outcome']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdb", default=GDB_PATH)
    parser.add_argument("--elf", default=ELF_PATH)
    parser.add_argument("--worklist", type=Path, default=Path(WORKLIST_PATH),
                         help="Output of fault_models.py")
    parser.add_argument("--results", type=Path, default=Path(RESULTS_PATH))
    args = parser.parse_args()

    injector = SFIInjector(gdb_path=args.gdb, elf_path=args.elf)
    groups = load_worklist(args.worklist)

    # TODO: replace with your actual test input list and golden capture,
    # e.g. reuse your existing confusion-matrix notebook's test set.
    input_ids = sorted({rows[0]["input_id"] for rows in groups.values()})
    golden_outputs = {iid: {"predicted_class": 0, "logits": []} for iid in input_ids}  # placeholder

    injector.start()
    injector.reflash()  # once per session — run_trial()'s per-trial reset_target() no longer reflashes
    try:
        run_campaign(injector, groups, golden_outputs, args.results)
    finally:
        injector.close()
