import subprocess
import threading
import queue
import time
import re
import csv
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# --- Instruction-level register usage classification (Thumb-1 / ARMv6-M) ---
# These mnemonic sets drive `FunctionProfiler.classify_instruction_registers`,
# which looks at what an instruction actually DOES with each register it
# references, rather than the register's generic AAPCS role. Heuristic and
# tuned for the Thumb-1 subset a Cortex-M0/M0+ compiler emits — extend these
# sets if you hit mnemonics not covered here.
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


def _strip_width_suffix(mnemonic: str) -> str:
    # GDB sometimes prints Thumb condition/width suffixes, e.g. "bne.n"
    return mnemonic.split(".")[0]


def _extract_mnemonic_and_operands(instr: str) -> Tuple[str, str]:
    # `instr` looks like "<conv2d+14>:\tadds\tr3, r3, #1" — strip the
    # "<label>:" prefix, then split mnemonic from its operand string.
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


class FunctionProfiler:
    def __init__(self, gdb_path: str, elf_path: str, target_remote: str = "localhost:3333"):
        self.gdb_path = gdb_path
        self.elf_path = elf_path
        self.target_remote = target_remote

        self.q = queue.Queue()
        self.gdb: Optional[subprocess.Popen] = None

        # Whitelist of functions to recursively profile (Step 5)
        self.profile_whitelist: Set[str] = {
            "conv2d",
            "batchnorm_relu",
            "resblock",
            "resblock_ds",
            "global_avg_pool",
            "fc", 
            "argmax"
        }

        # Track invocation counts for clean file naming (Step 9)
        self.function_counters: Dict[str, int] = {fn: 0 for fn in self.profile_whitelist}

        # Explicit stack to track active breakpoints per frame across GDB recursion (Step 10)
        self.bp_stack: List[List[int]] = []

        # Monotonic counter used to build unique sentinels for command-output framing
        self._sentinel_counter = itertools.count()

        # --- Architecture-mirroring output layout ---
        # Root folder all CSVs land under.
        self.output_root = Path("profiler_output")
        # Stack of "currently active" output directories, mirroring the
        # recursion depth. A function's CSV is written to whatever
        # directory sits on top of this stack *at the moment it finishes*.
        # A function only pushes its own new directory (and thus becomes
        # a "block" containing its children) if it actually intercepts at
        # least one whitelisted sub-call — pure leaves never get their own
        # folder, they just land in their parent's.
        self.dir_stack: List[Path] = []
        # Sequence counter so sibling/nested folders sort in call order
        # (01_resblock_1, 02_resblock_2, ...) regardless of function name.
        self._folder_seq = itertools.count(1)

        # --- CSV-merging for composite blocks (e.g. resblock, resblock_ds) ---
        # Functions listed here get ALL their descendants' rows merged into
        # ONE CSV (named after the block itself, e.g. resblock_1.csv)
        # instead of each child (conv2d_2, batchnorm_relu_2, conv2d_3, ...)
        # writing its own separate file. Functions NOT in this set (like the
        # top-level resnet_infer_full) keep the original one-file-per-call
        # behavior — their direct children still get individual files.
        self.merge_children_functions: Set[str] = {"resblock", "resblock_ds"}
        # Parallel to dir_stack: at each recursion depth, the own_dir that
        # children's rows should be merged INTO, or None if no active merge
        # scope applies (inherits the nearest active merge-block ancestor,
        # or None if none exists — see profile() for exactly how this is
        # set when a new own_dir is created).
        self.merge_stack: List[Optional[Path]] = []
        # Accumulator: own_dir -> all rows (from the block owner itself and
        # every descendant) waiting to be written out as that block's single
        # merged CSV once the owner's own profile() call finishes.
        self.merged_rows: Dict[Path, List[Dict]] = {}

        # --- Call-path tracking (NEW) --------------------------------------
        # The core fix for instance disambiguation. `conv2d` is one function
        # reached through many call paths (directly, and inside each
        # resblock). A raw `break *0xADDR` fires on EVERY invocation and GDB
        # stops at the first, so an address alone cannot say "which call".
        #
        # This stack records the exact chain of `bl` call-site addresses the
        # profiler stepped through to reach the function it's currently
        # inside. When profile() intercepts a `bl` into a whitelisted callee,
        # it pushes that `bl`'s own PC here right before stepping in, and pops
        # it after the recursive profile() returns — mirroring how dir_stack
        # and merge_stack are maintained.
        #
        # Every captured row is stamped with the current call path (the
        # semicolon-joined contents of this stack). Downstream, the injector
        # replays it: break at each `bl` address in order, `si` into the
        # callee, THEN arm the real injection address — so the address fires
        # on the correct invocation. Top-level rows (reached directly, e.g.
        # conv2d_1) get an EMPTY call path: no navigation needed.
        self.call_path_stack: List[str] = []

        # Parallel to the recursion: the entry address (first instruction)
        # of the function currently being profiled, so every row can also
        # record FuncEntryAddr. The injector uses this to VERIFY, after
        # replaying the call path and `si`-ing in, that it actually landed
        # in the intended function rather than silently injecting in the
        # wrong place if the table and ELF ever drift out of sync.
        self.func_entry_stack: List[str] = []

        # --- Dynamic (per-instruction) register usage tracking ---
        # Unlike REGISTER_CLASS (a static "what's this register normally
        # for" table), this tracks what each register was *actually doing*
        # at each captured instruction across the whole session: DATA
        # (value operand), ADDRESS (memory-address calculation), or
        # CONTROL (branch target / return address). Global (not per
        # function-call) so counts and step-span are session-wide.
        self.register_usage: Dict[str, Dict] = {}
        self._global_step_counter = itertools.count()

    def start(self):
        """Initializes the GDB process and connects to the target."""
        print("Starting GDB Engine...")
        self.gdb = subprocess.Popen(
            [self.gdb_path, "--nx"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        threading.Thread(target=self._enqueue_output, args=(self.gdb.stdout, self.q), daemon=True).start()
        time.sleep(1)

        init_cmds = [
            f"file {self.elf_path}",
            f"target remote {self.target_remote}",
            "monitor reset halt",
            "load",
            "set pagination off",
        ]
        for cmd in init_cmds:
            self._run(cmd)

    def _enqueue_output(self, out, q):
        for line in iter(out.readline, ''):
            q.put(line)
        out.close()

    def send_gdb(self, cmd: str, quiet: bool = False):
        if not quiet:
            print(f"[GDB] {cmd}")
        self.gdb.stdin.write(cmd + "\n")
        self.gdb.stdin.flush()

    # ------------------------------------------------------------------
    # Fix #2: unified, sentinel-framed command execution.
    #
    # The original code had two separate "wait" helpers that both drained
    # the same queue and stopped on a substring match ("(gdb)", "Breakpoint",
    # etc). Because GDB's stdout arrives in a stream with no message
    # boundaries, leftover bytes from one command could bleed into the next
    # call's read, corrupting parsing (e.g. `current_instruction()` reading
    # tail bytes from a *previous* `disassemble` dump). Routing every
    # command through `_run` with a unique echoed sentinel guarantees each
    # call only ever sees output that belongs to itself.
    # ------------------------------------------------------------------
    def _run(self, cmd: str, timeout: float = 20.0, quiet: bool = False) -> str:
        if not quiet:
            print(f"[GDB] {cmd}")
        sentinel = f"__DONE_{next(self._sentinel_counter)}__"

        self.gdb.stdin.write(cmd + "\n")
        self.gdb.stdin.write(f"echo {sentinel}\\n\n")
        self.gdb.stdin.flush()

        buf = ""
        start = time.time()
        while time.time() - start < timeout:
            while not self.q.empty():
                buf += self.q.get_nowait()
            if sentinel in buf:
                return buf.split(sentinel)[0]
            time.sleep(0.01)

        print(f"[!] Timeout waiting for sentinel after command: {cmd}")
        return buf

    def _run_until_stop(self, cmd: str, timeout: float = 120.0, quiet: bool = False) -> Optional[str]:
        """
        Like _run, but for commands that resume execution (e.g. `continue`)
        where GDB won't print the sentinel until the target stops again.
        We still queue the sentinel so the *next* command doesn't have to
        guess where this stop-output ends.
        """
        if not quiet:
            print(f"[GDB] {cmd}")
        sentinel = f"__DONE_{next(self._sentinel_counter)}__"

        self.gdb.stdin.write(cmd + "\n")
        self.gdb.stdin.write(f"echo {sentinel}\\n\n")
        self.gdb.stdin.flush()

        buf = ""
        start = time.time()
        while time.time() - start < timeout:
            while not self.q.empty():
                buf += self.q.get_nowait()
            if sentinel in buf:
                return buf.split(sentinel)[0]
            time.sleep(0.01)

        return None

    def get_offsets(self, function_name: str) -> List[int]:
        out = self._run(f"disassemble {function_name}")
        offsets = []
        for line in out.splitlines():
            m = re.search(r"<\+(\d+)>:", line)
            if m:
                offsets.append(int(m.group(1)))

                if "pop" in line and "pc" in line:
                    break
        offsets = sorted(set(offsets))
        print(offsets)
        return offsets

    def get_function_entry_addr(self, function_name: str) -> str:
        """
        Returns the absolute entry address (first instruction) of
        `function_name` as a '0x...' string, parsed from `disassemble`'s
        first listed instruction line. Recorded per row as FuncEntryAddr so
        the injector can confirm it navigated into the right function after
        replaying a call path. Empty string if it can't be parsed.
        """
        out = self._run(f"disassemble {function_name}", quiet=True)
        for line in out.splitlines():
            # e.g. "   0x10000308 <+0>:\tpush\t{r4, r5, ...}"
            m = re.search(r"(0x[0-9a-fA-F]+)\s+<\+0>:", line)
            if m:
                return m.group(1)
        # Fallback: first hex address on any listed line.
        for line in out.splitlines():
            m = re.search(r"(0x[0-9a-fA-F]+)\s+<\+\d+>:", line)
            if m:
                return m.group(1)
        return ""

    def set_breakpoint_at_offset(self, function_name: str, offset: int) -> Optional[int]:
        out = self._run(f"break *({function_name} + {offset})")
        m = re.search(r"Breakpoint (\d+)", out)
        if m:
            bp_num = int(m.group(1))
            print(f"[+] Set Breakpoint {bp_num} at {function_name}+{offset}")
            return bp_num
        return None

    def delete_breakpoint(self, bp_num: Optional[int]):
        if bp_num is not None:
            self._run(f"delete {bp_num}")
            print(f"[-] Deleted Breakpoint {bp_num}")

    def current_instruction(self) -> Tuple[str, str]:
        out = self._run("x/i $pc", quiet=True)
        m = re.search(r"(0x[0-9a-fA-F]+)[^<]*(<.*)", out)
        if m:
            return m.group(1), m.group(2)
        m_fallback = re.search(r"(0x[0-9a-fA-F]+):\s*(.*)", out)
        if m_fallback:
            return m_fallback.group(1), m_fallback.group(2)
        return "", ""

    # AAPCS/Cortex-M register role classification. Used to tag every
    # captured register so downstream analysis (e.g. pandas filtering on
    # RegClass, or fault-injection outcome correlation) doesn't have to
    # re-derive "is this a data reg or a control reg" from bare names.
    REGISTER_CLASS: Dict[str, str] = {
        # Argument / return-value regs — caller-saved, volatile across calls
        "r0": "DATA_ARG", "r1": "DATA_ARG", "r2": "DATA_ARG", "r3": "DATA_ARG",
        # Local-variable regs — callee-saved, changes here reflect real
        # in-function computation state rather than call-boundary churn
        "r4": "DATA_LOCAL", "r5": "DATA_LOCAL", "r6": "DATA_LOCAL",
        "r7": "DATA_LOCAL", "r8": "DATA_LOCAL", "r9": "DATA_LOCAL",
        "r10": "DATA_LOCAL", "r11": "DATA_LOCAL",
        # Intra-procedure-call scratch — volatile, mostly linker/veneer noise
        "r12": "DATA_SCRATCH", "ip": "DATA_SCRATCH",
        # Control-flow / address regs
        "sp": "CONTROL_SP", "r13": "CONTROL_SP",
        "lr": "CONTROL_LR", "r14": "CONTROL_LR",
        "pc": "CONTROL_PC", "r15": "CONTROL_PC",
        # Status / flags (combined xPSR on Cortex-M; split cpsr/apsr on Cortex-A/R)
        "xpsr": "STATUS", "cpsr": "STATUS", "apsr": "STATUS",
        "ipsr": "STATUS", "epsr": "STATUS",
        # Interrupt-mask / mode-control regs — only present if you explicitly
        # dump them via `info registers all` / `p/x $reg`
        "primask": "STATUS_INT", "faultmask": "STATUS_INT", "basepri": "STATUS_INT",
        "control": "STATUS_MODE", "msp": "CONTROL_SP", "psp": "CONTROL_SP",
    }

    @classmethod
    def classify_register(cls, name: str) -> str:
        return cls.REGISTER_CLASS.get(name.lower(), "UNKNOWN")

    def read_registers(self) -> Dict[str, str]:
        out = self._run("info registers", quiet=True)
        regs = {}
        for line in out.splitlines():
            m = re.match(r"(\w+)\s+(0x[0-9a-fA-F]+)\s+(-?\d+)", line)
            if m:
                name = m.group(1)
                regs[name + "_hex"] = m.group(2)
                regs[name + "_dec"] = m.group(3)
                regs[name + "_class"] = self.classify_register(name)
        return regs

    def classify_instruction_registers(self, instr: str) -> List[Tuple[str, str]]:
        """
        Looks at what THIS instruction is actually doing with each register
        it references (as opposed to REGISTER_CLASS, which is the register's
        generic role). Returns a list of (register_name, usage) pairs where
        usage is one of:
            "DATA"    - carries/receives a value being computed on
            "ADDRESS" - participates in a memory-address calculation
            "CONTROL" - branch target / return address / PC
        Unrecognized mnemonics default every register they touch to DATA —
        if usage stats look off for a given register, check here first for
        a missing mnemonic.
        """
        mnemonic, operands = _extract_mnemonic_and_operands(instr)
        if not mnemonic:
            return []

        regs_found = [r.lower() for r in _REG_TOKEN_RE.findall(operands)]
        if not regs_found:
            return []

        results: List[Tuple[str, str]] = []

        if mnemonic in _LOAD_STORE_MNEMONICS:
            # "rt, [rn, #imm]" or "rt, [rn, rm]" — rt (before the bracket)
            # is the data register; rn/rm (inside the bracket) compute the
            # address.
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
            # sp is the implicit address register (not printed in operands);
            # lr/pc inside the {..} list are control-flow (save/restore of
            # return address), everything else is plain data spill/fill.
            results.append(("sp", "ADDRESS"))
            for r in regs_found:
                results.append((r, "CONTROL" if r in ("lr", "pc") else "DATA"))

        elif mnemonic in _BRANCH_MNEMONICS:
            # Register-form branches (bx rN, blx rN) use the register as a
            # control-flow target. Immediate/label branches have no
            # register operand, so regs_found is empty and nothing is added.
            for r in regs_found:
                results.append((r, "CONTROL"))

        elif mnemonic in _COMPARE_BRANCH_MNEMONICS:
            # cbz/cbnz rN, <label> — rN is being tested (data comparison
            # against zero), not itself a branch target.
            for r in regs_found:
                results.append((r, "DATA"))

        elif mnemonic in _DATA_PROCESSING_MNEMONICS:
            for r in regs_found:
                # sp as an operand of add/sub/mov etc. is virtually always
                # stack-frame setup ("sub sp, #60", "add r7, sp, #0"), not
                # generic arithmetic — treat it as ADDRESS regardless of
                # mnemonic family.
                results.append((r, "ADDRESS" if r == "sp" else "DATA"))

        elif mnemonic == "mov" and "pc" in regs_found:
            # e.g. "mov pc, lr" — manual return without pop
            for r in regs_found:
                results.append((r, "CONTROL"))

        else:
            for r in regs_found:
                results.append((r, "DATA"))

        return results

    def _record_register_usage(self, instr: str) -> int:
        """
        Classifies this instruction's register usage and folds it into the
        session-wide self.register_usage tally. Returns the global step
        index assigned to this instruction (monotonic across the whole
        session, including nested recursive calls).
        """
        global_step = next(self._global_step_counter)
        for reg, usage in self.classify_instruction_registers(instr):
            stats = self.register_usage.setdefault(reg, {
                "DATA": 0, "ADDRESS": 0, "CONTROL": 0,
                "static_class": self.classify_register(reg),
                "first_step": global_step,
                "last_step": global_step,
            })
            stats[usage] += 1
            stats["last_step"] = global_step
        return global_step

    def write_register_usage_summary(self, filename: str = "register_usage_summary.csv") -> Path:
        """
        Writes one row per register seen anywhere in the session: how many
        times it was used as DATA vs ADDRESS vs CONTROL, its static AAPCS
        role, and the global-step span over which it was active. Call this
        once after the top-level profile() call returns.
        """
        self.output_root.mkdir(parents=True, exist_ok=True)
        path = self.output_root / filename
        fields = [
            "Register", "StaticClass",
            "DataUses", "AddressUses", "ControlUses", "TotalUses",
            "FirstGlobalStep", "LastGlobalStep", "StepSpan",
        ]
        rows = []
        for reg, stats in sorted(self.register_usage.items()):
            total = stats["DATA"] + stats["ADDRESS"] + stats["CONTROL"]
            rows.append({
                "Register": reg,
                "StaticClass": stats["static_class"],
                "DataUses": stats["DATA"],
                "AddressUses": stats["ADDRESS"],
                "ControlUses": stats["CONTROL"],
                "TotalUses": total,
                "FirstGlobalStep": stats["first_step"],
                "LastGlobalStep": stats["last_step"],
                "StepSpan": stats["last_step"] - stats["first_step"],
            })
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved register usage summary to: {path}")
        return path

    def parse_called_function(self, instr: str) -> Optional[str]:
        # Step 4: Detect branch-with-link calls to functions
        m = re.search(r"bl\s+0x[0-9a-fA-F]+\s+<([^>+]+)(?:\+\d+)?>", instr)
        if m:
            return m.group(1).strip()
        return None

    def profile(self, function_name: str):
        """Recursively profiles execution blocks using a sliding window breakpoint engine."""
        print(f"\n==== Entering Profile Context: {function_name} ====")

        # Step 9: Manage dynamic tracking filenames
        if function_name in self.function_counters:
            self.function_counters[function_name] += 1
            run_id = self.function_counters[function_name]
            output_csv = f"{function_name}_{run_id}.csv"
        else:
            run_id = None
            output_csv = f"{function_name}_untracked.csv"

        # Used both for folder naming (if this call becomes a block owner)
        # and to tag every row with its provenance (SourceFunction column)
        # — essential once rows from multiple distinct functions/invocations
        # can end up merged into one CSV: the Instruction text alone (e.g.
        # "<conv2d+14>") doesn't distinguish invocation 2 from invocation 3,
        # and doesn't distinguish which function an offset number belongs to
        # once conv2d's and batchnorm_relu's offsets sit in the same file.
        label = f"{function_name}_{run_id}" if run_id is not None else function_name

        # NEW: entry address of THIS function, and a snapshot of the call
        # path that led here. Both are stamped onto every row captured in
        # this frame. call_path is snapshotted (joined) once up front: it's
        # constant for the whole duration of this profile() call, since the
        # stack only changes when we descend into a CHILD (which happens in
        # a nested profile() call with its own snapshot).
        func_entry_addr = self.get_function_entry_addr(function_name)
        call_path_str = ";".join(self.call_path_stack)
        self.func_entry_stack.append(func_entry_addr)

        # The directory this call's CSV lands in, and any children start
        # from, is whatever's active on the stack right now. It's created
        # lazily as "own_dir" below, only if this call turns out to
        # intercept a child call.
        parent_dir = self.dir_stack[-1] if self.dir_stack else self.output_root
        parent_dir.mkdir(parents=True, exist_ok=True)
        own_dir: Optional[Path] = None

        offsets = self.get_offsets(function_name)
        if not offsets:
            print(f"[-] No offsets found or function hidden: {function_name}")
            # Balance the func_entry_stack push above before bailing.
            if self.func_entry_stack:
                self.func_entry_stack.pop()
            return

        # Step 2 & 10: Parent state isolated implicitly by Python call-stack scope
        current_idx = 0
        next_idx = 1
        loop_count = 0
        step = 0
        rows = []

        current_bp = self.set_breakpoint_at_offset(function_name, offsets[current_idx])
        next_bp = self.set_breakpoint_at_offset(function_name, offsets[next_idx]) if len(offsets) > 1 else None

        # Keep a log of breakpoints active in this frame to disable them during sub-calls
        active_frame_bps = [bp for bp in [current_bp, next_bp] if bp is not None]
        self.bp_stack.append(active_frame_bps)

        out = self._run_until_stop("continue")

        try:
            while True:
                if out is None:
                    print(f"[!] Timeout or exit encountered inside {function_name}.")
                    break

                # Parse hit breakpoint ID
                m_bp = re.search(r"Breakpoint (\d+)", out)
                hit_bp = int(m_bp.group(1)) if m_bp else None

                if hit_bp is None:
                    print(f"[*] Target broke context inside {function_name} (Signal/Exit).")
                    break

                pc, instr = self.current_instruction()
                regs = self.read_registers()

                # Fix #1: the disassembly listing has no function name inside
                # the <+N> marker (e.g. "<+108>:"), but `x/i $pc` output does
                # (e.g. "<resnet_infer_full+108>:"). The original regex
                # required '+' immediately after '<', so it could never
                # match the latter and offset was always -1. This version
                # allows an optional function-name prefix before the '+'.
                m_off = re.search(r"<[^>]*\+(\d+)>", instr)
                offset = int(m_off.group(1)) if m_off else -1

                # Tally this instruction's register usage (DATA/ADDRESS/
                # CONTROL) into the session-wide summary, and stamp the row
                # with the same monotonic global step so per-row data can be
                # cross-referenced against register_usage_summary.csv later.
                global_step = self._record_register_usage(instr)

                # NEW: CallPath + FuncEntryAddr stamped on every row.
                #  - CallPath: the ';'-joined bl-site addresses walked into to
                #    reach this function instance. Empty for top-level rows
                #    (reached directly — no navigation needed).
                #  - FuncEntryAddr: this function's entry address, used by the
                #    injector to verify it landed in the right function after
                #    replaying the call path.
                row = {"Step": step, "GlobalStep": global_step, "SourceFunction": label,
                       "CallPath": call_path_str, "FuncEntryAddr": func_entry_addr,
                       "PC": pc, "Instruction": instr, "LoopCount": loop_count}
                row.update(regs)
                rows.append(row)

                print(f"[{function_name} | Step {step}] Captured state at offset +{offset} (BP {hit_bp})")

                # Step 4 & 6: Check for branching into whitelisted functions
                called_function = self.parse_called_function(instr)
                if called_function and called_function in self.profile_whitelist:
                    print(f"[➔] Intercepted call to whitelisted function: {called_function}")

                    # This call just proved itself to be a "block" (it has
                    # children), so lazily create its own output folder and
                    # make it the active directory for this child (and any
                    # further siblings/children of this frame).
                    if own_dir is None:
                        seq = next(self._folder_seq)
                        own_dir = parent_dir / f"{seq:02d}_{label}"
                        own_dir.mkdir(parents=True, exist_ok=True)
                        self.dir_stack.append(own_dir)

                        # If this function is a merge-children type (e.g.
                        # resblock), its own_dir BECOMES the active merge
                        # target for everything nested under it — children's
                        # rows accumulate here instead of writing separate
                        # files. Otherwise, inherit whatever merge target
                        # (if any) was already active from an ancestor, so
                        # a non-merging block nested inside a merging one
                        # still merges correctly; with none active, None
                        # propagates and behavior stays exactly as before.
                        if function_name in self.merge_children_functions:
                            self.merge_stack.append(own_dir)
                            self.merged_rows[own_dir] = []
                        else:
                            self.merge_stack.append(self.merge_stack[-1] if self.merge_stack else None)

                    # Fix #3: remove current_bp from the frame's active-bp
                    # list *before* we delete it, so we don't later try to
                    # re-enable an already-deleted breakpoint ID, and so
                    # bookkeeping of "what's live in this frame" stays
                    # accurate across the recursive call.
                    if current_bp in self.bp_stack[-1]:
                        self.bp_stack[-1].remove(current_bp)

                    # Step 10 (Safeguard): Disable parent frame breakpoints before handing control down
                    for bp in self.bp_stack[-1]:
                        self._run(f"disable {bp}", quiet=True)

                    # NEW: push THIS bl instruction's address onto the call
                    # path stack before stepping in. `pc` here is exactly the
                    # address of the `bl` we're about to step through — the
                    # same call-site address you'd look up by hand (e.g.
                    # 0x100005b4 for `<resblock+64>: bl conv2d`). The nested
                    # profile() call snapshots this extended path for every
                    # row it captures. Popped right after it returns, below.
                    self.call_path_stack.append(pc)

                    # Step 7 & 8: Perform Step-Into and transfer control to child frame recursively
                    self._run("stepi")
                    self.delete_breakpoint(current_bp)

                    self.profile(called_function)  # <--- RECURSION ENGINE

                    # NEW: pop the call-site address now that the child (and
                    # all its descendants) have finished — restoring the path
                    # for any remaining siblings captured in this frame.
                    if self.call_path_stack:
                        self.call_path_stack.pop()

                    # Fix #3: explicitly advance the sliding window past the
                    # call site instead of leaving current_bp/current_idx
                    # pointing at the (now-deleted) pre-call breakpoint.
                    current_bp = next_bp
                    current_idx = next_idx
                    if current_bp is not None and next_idx + 1 < len(offsets):
                        next_idx += 1
                        next_bp = self.set_breakpoint_at_offset(function_name, offsets[next_idx])
                        self.bp_stack[-1].append(next_bp)
                    else:
                        next_bp = None

                    # Step 10 (Restore): Re-enable parent frame breakpoints upon return
                    print(f"[⬅] Returned to parent context: {function_name}")
                    for bp in self.bp_stack[-1]:
                        self._run(f"enable {bp}", quiet=True)

                    if current_bp is None:
                        print(f"[*] No downstream blocks available after call. Breaking function frame loop.")
                        break

                    # Re-sync local GDB output state after returning
                    out = self._run_until_stop("continue")
                    step += 1
                    continue

                # --- Sliding Window Breakpoint Logic Execution ---
                if hit_bp == next_bp and next_bp is not None:
                    self.delete_breakpoint(current_bp)
                    if current_bp in self.bp_stack[-1]:
                        self.bp_stack[-1].remove(current_bp)

                    current_bp = next_bp
                    current_idx = next_idx

                    if next_idx + 1 < len(offsets):
                        next_idx += 1
                        next_bp = self.set_breakpoint_at_offset(function_name, offsets[next_idx])
                        self.bp_stack[-1].append(next_bp)
                    else:
                        next_bp = None
                        print(f"[*] Reached last block offset chain for {function_name}.")

                    loop_count = 0

                elif hit_bp == current_bp:
                    loop_count += 1
                    print(f"[!] Loop detected at current block. Count: {loop_count}/3")

                    if loop_count == 3:
                        print("[!] Loop ceiling reached. Forcing window progression.")
                        self.delete_breakpoint(current_bp)
                        if current_bp in self.bp_stack[-1]:
                            self.bp_stack[-1].remove(current_bp)

                        if next_bp is not None:
                            current_bp = next_bp
                            current_idx = next_idx
                            if next_idx + 1 < len(offsets):
                                next_idx += 1
                                next_bp = self.set_breakpoint_at_offset(function_name, offsets[next_idx])
                                self.bp_stack[-1].append(next_bp)
                            else:
                                next_bp = None
                        else:
                            print(f"[*] No downstream blocks available. Breaking function frame loop.")
                            break
                        loop_count = 0
                else:
                    print(f"[?] Foreign Breakpoint hit ({hit_bp}) inside {function_name}. Exiting frame.")
                    break

                if next_bp is None and current_bp == hit_bp:
                    print(f"[*] Finished tracing all execution structures for {function_name}.")
                    break

                step += 1
                out = self._run_until_stop("continue")

        finally:
            # Clean up remaining frame breakpoints from physical GDB context.
            # Save logic now has three cases instead of two:
            #
            # 1. This call IS the owner of the active merge target (own_dir
            #    was just created above AND it equals merge_stack's top) —
            #    fold its own rows into the shared accumulator (same bucket
            #    its children already appended into), then write the WHOLE
            #    accumulated set out as the single merged CSV.
            # 2. A merge target is active but this call ISN'T its owner
            #    (a leaf like conv2d_2, or any non-owning descendant) —
            #    just append rows into that accumulator. No file written
            #    here; the owner (case 1) writes everything at the end.
            # 3. No merge target active — exactly the original behavior:
            #    leaves save into their parent's folder, blocks save into
            #    their own newly-created folder.
            merge_target = self.merge_stack[-1] if self.merge_stack else None

            if own_dir is not None and own_dir == merge_target:
                self.merged_rows[own_dir].extend(rows)
                all_rows = self.merged_rows.pop(own_dir)
                all_rows.sort(key=lambda r: r["GlobalStep"])  # keep true execution order in the file
                if all_rows:
                    fields = list(all_rows[0].keys())
                    csv_path = own_dir / output_csv
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(all_rows)
                    print(f"Saved merged block telemetry ({len(all_rows)} rows, "
                          f"{len({r['SourceFunction'] for r in all_rows})} source function(s)) to: {csv_path}")

            elif merge_target is not None:
                self.merged_rows[merge_target].extend(rows)

            else:
                save_dir = own_dir if own_dir is not None else parent_dir
                if rows:
                    fields = list(rows[0].keys())
                    csv_path = save_dir / output_csv
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
                    print(f"Saved run telemetry to: {csv_path}")

            # Only pop if we pushed — by this point all recursive children
            # of this call have already returned (and popped their own
            # folders/merge scope, if any), so this is always safe/balanced.
            if own_dir is not None and self.dir_stack and self.dir_stack[-1] == own_dir:
                self.dir_stack.pop()
                if self.merge_stack:
                    self.merge_stack.pop()

            # NEW: balance the func_entry_stack push from the top of this
            # call. call_path_stack is balanced inline at each recursion
            # site (push before stepi, pop after profile() returns), so it
            # needs no cleanup here.
            if self.func_entry_stack:
                self.func_entry_stack.pop()

            if self.bp_stack:
                for bp in list(self.bp_stack[-1]):
                    try:
                        self.delete_breakpoint(bp)
                    except Exception:
                        pass
                self.bp_stack.pop()

    def close(self):
        if self.gdb:
            self.send_gdb("quit")
            self.gdb.terminate()


if __name__ == "__main__":
    # Clean, decoupled execution interface
    profiler = FunctionProfiler(gdb_path="arm-none-eabi-gdb", elf_path="build/resnet_pico.elf")
    try:
        profiler.start()
        profiler.profile("resnet_infer_full")
        profiler.write_register_usage_summary()
    except KeyboardInterrupt:
        print("\n[!] Global automation lifecycle interrupted by user.")
    finally:
        profiler.close()

