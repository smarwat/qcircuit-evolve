"""Evolutionary loop: Claude proposes candidate generators, the evaluator scores them.

    python orchestrator.py --generations 10 --max-calls 20        # default: `claude -p` on your subscription
    python orchestrator.py --mock --generations 2                 # local fake model, no network
    python orchestrator.py --resume runs/<dir> --generations 5    # continue a run
    python orchestrator.py --backend api --budget-usd 20 ...      # paid API: only with explicit --backend api

Each generation picks parents (mostly from the Pareto front over T and Q at n=32),
asks the model for an improvement ("refine", ~70%) or a structurally different
algorithm ("redesign", ~30%), applies the reply (full program or strict
SEARCH/REPLACE diff), evaluates children in parallel processes, and runs the hidden
verify tier on any child that would become the new best.

Run directory: attempts.jsonl, archive.json, pareto.json, best.py, state.json,
system_prompt.txt, candidates/*.py, replies/*.md, private/verify.jsonl.
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import sys
import threading
import time

from archive import Archive
from config import ancilla_budget, load_config
from diffs import DiffError, parse_reply
from evaluator import evaluate, verify
from claude_code_client import ClaudeCodeClient, ClaudeCodeError, UsageLimitError
from llm import ClaudeClient, FatalAPIError, MockClient

HERE = os.path.dirname(os.path.abspath(__file__))
EXIT_USAGE_LIMIT = 75  # EX_TEMPFAIL: stopped cleanly on a usage limit; run --resume later

IDEAS = [
    "Binary GCD / Kaliski variants: different case orderings, fewer comparisons per step, "
    "Kaliski's doubling of the coefficients instead of modular halving (then fix the 2^k factor).",
    "divsteps-style constant-time inversion (Bernstein-Yang): a delta counter plus fixed-shape steps "
    "replaces the full n-bit u>v comparison with a sign test.",
    "Reduce or eliminate the stored history bits (2 per iteration in both seeds).",
    "Recompute control decisions from register values during uncompute instead of storing them "
    "(e.g. the parity of the register just halved, or its top bit after a modular halving).",
    "Cheaper adders and comparators: comparators that share work with the following subtraction, "
    "controlled adders without the AND-into-temp register, constant adders specialised to the bits "
    "of p. (There is no measurement here: every AND you compute costs a Toffoli to undo.)",
    "Register reuse to cut Q: share temporaries, let registers shrink as they lose bits, avoid "
    "duplicate swaps, drop ancillas that are only needed briefly.",
    "Avoid Bennett's full 2x: uncompute the history early with a cheaper reverse pass, or "
    "structure the loop so the inverse pass is shorter than the forward pass.",
]

MODES = {
    "refine": (
        "MODE: refine. Make one focused improvement to the parent that lowers T*Q at n=32 "
        "(or lowers T or Q without raising the other). Prefer SEARCH/REPLACE blocks against the "
        "parent; if the change touches most of the program, send a full program instead."),
    "redesign": (
        "MODE: redesign. Propose a structurally different construction, not a tweak of the parent: "
        "for example Bernstein-Yang divsteps, removing or recomputing the history register, a "
        "different uncompute strategy than a full Bennett reversal, or a different register layout. "
        "Send a complete program in one ```python block."),
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def circuit_api():
    return """\
The builder `c` (do not subclass or replace it):
  c.n, c.p                 width and prime
  c.inp                    list of n input qubit indices, little-endian (qubits 0..n-1)
  c.out                    list of n output qubit indices (qubits n..2n-1), start at 0
  c.anc                    list of ancilla indices (2n .. 2n+budget-1), start at 0
  c.x(t)                   NOT
  c.cx(ctrl, t)            CNOT       (alias c.cnot)
  c.ccx(c1, c2, t)         Toffoli    (alias c.toffoli); qubits must be distinct
  c.alloc(k) / c.release(qs)   optional ancilla allocator (lowest indices first)
  c.mark()                 current gate index
  c.append_inverse(s, e=None)  append the inverse of gates [s, e) (Bennett uncompute)
  c.invert_since(s)        replace gates [s, end) by their inverse (e.g. adder -> subtractor)
Gates are stored in c.gates as array('i') triples (target, c1, c2) with -1 for unused controls."""


def build_system_rules(cfg):
    return """\
You are an expert in reversible circuit design and number-theoretic algorithms, working inside an
automated evolutionary search. Each request shows you a parent program, its evaluator feedback and
some strong programs from the archive. You reply with ONE improved candidate.

REPLY FORMAT
First, 2-6 sentences: the idea, and the T and Q you expect at n=32. Then EITHER
  (a) a complete program in ONE ```python block containing `def build_circuit(c, n, p):`, OR
  (b) one or more edit blocks against the parent, applied in order:
<<<<<<< SEARCH
lines copied exactly from the parent (including indentation)
=======
replacement lines
>>>>>>> REPLACE
Each SEARCH text must occur exactly once in the parent (after earlier blocks are applied). Edits are
applied strictly; one that does not match is rejected and wasted. Do not mix (a) and (b).

Correctness first: a cheaper circuit that fails any test scores nothing. Programs that would become the
new best are re-verified on more inputs and widths you do not see; width-generic, input-independent
constructions pass, anything tuned to the visible tests fails."""


def build_task_spec(cfg):
    w_lines = []
    for w in cfg["widths"]:
        n, p = w["n"], w["p"]
        if w["mode"] == "exhaustive":
            tests = f"every input 1..{p - 1}"
        else:
            tests = (f"edge cases (1, 2, 3, p-1, p-2, (p+1)/2, powers of two and 2^k-1 below p), "
                     f"{cfg['random_samples']} random inputs from a fixed public seed, and "
                     f"{cfg['hidden_samples']} hidden random inputs from a different, secret seed")
        w_lines.append(f"  - n={n}, p={p}: {tests}. Ancilla budget {ancilla_budget(cfg, n)}.")
    ideas = "\n".join(f"  - {s}" for s in IDEAS)
    return f"""\
TASK: reversible circuits for modular inversion, a small-scale version of the prime-field inversion
inside elliptic-curve point addition (cf. Roetteler, Naehrig, Svore & Lauter 2017). Gates: X, CNOT,
Toffoli only.

CONTRACT
Write a Python function `build_circuit(c, n, p)` that emits gates into `c` and works for ANY width n
and odd prime p < 2^n. Register layout:
  - qubits 0..n-1: input a. Must be unchanged at the end.
  - qubits n..2n-1: start at 0, must hold a^-1 mod p at the end.
  - qubits 2n and up: ancillas, at most {cfg['ancilla_per_bit']}n + {cfg['ancilla_const']}. Start at 0, must end at 0.
  - Inputs 0 and p..2^n-1 are don't-cares.
You may do any classical precomputation from n and p (constants, bit patterns, loops, helper
functions). You may NOT import anything, define classes, or use names starting with '__'.
Available builtins: range, len, min, max, abs, pow, divmod, sum, enumerate, zip, list, tuple, dict,
set, sorted, reversed, int, bool, bin, isinstance, map, filter, any, all, iter, next.

{circuit_api()}

EVALUATION (cheapest first, stops at the first failure)
{chr(10).join(w_lines)}
Gate cap {cfg['gate_cap']:,} gates per width. Whole evaluation must finish in {cfg['eval_timeout']:g}s.
The evaluator ignores everything on `c` except the stored gate array: it re-validates every gate,
recounts T and Q, and simulates the circuit with its own code.

SCORE
  T = Toffoli count, Q = number of distinct qubits used (input and output registers always count).
  PRIMARY SCORE = T*Q at n={cfg['score_width']}. Lower is better. Invalid at any width = no score.
  You also see T and Q at every width and the scaling exponent log2(T32/T16).
The archive keeps the Pareto front over (T, Q) at n={cfg['score_width']}, so a program that trades T for Q can
survive even if its T*Q is not the lowest.
Lookup tables, input-specific tricks and anything exponential in n cannot work: they blow the gate
cap or the time limit at n=32, and hidden inputs catch circuits that only handle the visible ones.
Only genuinely algorithmic, polynomial-size constructions score.

IDEAS WORTH TRYING
{ideas}"""


def build_system_prompt(cfg):
    return build_system_rules(cfg) + "\n\n" + build_task_spec(cfg)


def _summary(e):
    return f"T={e['T']:,} Q={e['Q']} T*Q={e['score']:,}" + (" verified" if e.get("verified") else "")


def build_user_prompt(mode, parent, inspirations, archive):
    front = sorted(archive.front(), key=lambda e: e["score"])
    best = archive.best()
    lines = [MODES[mode], "", "Pareto front at n=32:"]
    lines += [f"  {e['id']}: {_summary(e)}" for e in front[:8]]
    if best:
        lines.append(f"Current verified best: {best['id']} with T*Q {best['score']:,}.")
    for e in inspirations:
        lines += ["", f"Inspiration {e['id']} ({_summary(e)}):", "<inspiration>", "```python",
                  e["source"].rstrip("\n"), "```", "</inspiration>"]
    lines += ["", f"Parent program {parent['id']} ({_summary(parent)}):", "<parent>", "```python",
              parent["source"].rstrip("\n"), "```", "</parent>", "",
              "Evaluator feedback for the parent:", parent["feedback"], "",
              f"Propose a candidate that beats T*Q {parent['score']:,} while staying correct at every width."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Worker functions (run in separate processes)
# ---------------------------------------------------------------------------
def _eval_job(source, cfg):
    return evaluate(source, cfg)


def _verify_job(source, cfg, expected):
    return verify(source, cfg, expected)


def _score_row(r, cfg):
    return next((w for w in r["widths"] if w["n"] == cfg["score_width"] and "T" in w), None)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
class Run:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config()
        self.out = None
        self.state = None
        self.lock = threading.Lock()
        self.tries = {}  # gen -> times started in this process (limit waits re-run a generation)

    # -- setup -------------------------------------------------------------------
    def setup(self):
        a, cfg = self.args, self.cfg
        if a.resume:
            self.out = a.resume
            with open(os.path.join(self.out, "state.json")) as f:
                self.state = json.load(f)
            cfg["llm"].update(self.state["llm"])         # keep model/effort stable across resume
            cfg["claude_code"].update(self.state.get("claude_code", {}))
            cfg["evolution"].update(self.state["evolution"])
            self.state["sessions"] = self.state.get("sessions", 0) + 1
            self.archive = Archive.load(self.out)
        else:
            self.out = a.out or os.path.join(HERE, "runs", time.strftime("%Y%m%d-%H%M%S"))
            self.state = {"gen": 0, "spent_usd": 0.0, "calls": 0, "sessions": 0,
                          "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}}
            self.archive = Archive()
        for k, v in (("model", a.model), ("effort", a.effort), ("max_tokens", a.max_tokens)):
            if v:
                cfg["llm"][k] = v
        if a.backend == "claude-code":
            cfg["evolution"]["per_gen"] = cfg["claude_code"]["per_gen"]
            if a.claude_bin:
                cfg["claude_code"]["binary"] = a.claude_bin
            if a.model:
                cfg["claude_code"]["model"] = cfg["claude_code"]["expected_model"] = a.model
            if a.effort:
                cfg["claude_code"]["effort"] = a.effort
        if a.per_gen:
            cfg["evolution"]["per_gen"] = a.per_gen
        if a.workers:
            cfg["evolution"]["workers"] = a.workers
        if a.eval_timeout:
            cfg["eval_timeout"] = a.eval_timeout
        if a.resume and self.state.get("backend") not in (None, a.backend):
            print(f"[resume] note: run was started with backend {self.state['backend']!r}; "
                  f"continuing with {a.backend!r}")
        self.state["llm"] = cfg["llm"]
        self.state["claude_code"] = cfg["claude_code"]
        self.state["evolution"] = cfg["evolution"]
        self.state["backend"] = a.backend
        self.state.setdefault("notional_usd", 0.0)

        for d in ("candidates", "replies", "private"):
            os.makedirs(os.path.join(self.out, d), exist_ok=True)
        with open(os.path.join(self.out, "system_prompt.txt"), "w") as f:
            f.write(build_system_prompt(cfg))

        if a.backend == "mock":
            divsteps = os.path.join(HERE, "seeds", "divsteps.py")
            redesign = open(divsteps).read() if os.path.exists(divsteps) else None
            self.client = MockClient(cfg["llm"], redesign)
        elif a.backend == "claude-code":
            self.client = ClaudeCodeClient(cfg["claude_code"], build_system_prompt(cfg))
            info = self.client.model_info()
            print(f"[backend] claude-code: {info}")
            if not info["binary"]:
                raise SystemExit(f"`{cfg['claude_code']['binary']}` not found on PATH")
        elif a.backend == "api":  # only ever reached with an explicit --backend api
            self.client = ClaudeClient(cfg["llm"], build_system_rules(cfg), build_task_spec(cfg))
            info = self.client.model_info()
            self.state["model_info"] = info
            print(f"[model] {cfg['llm']['model']}: {info}")
            if info.get("max_tokens") and cfg["llm"]["max_tokens"] > info["max_tokens"]:
                print(f"[model] clamping max_tokens {cfg['llm']['max_tokens']} -> {info['max_tokens']}")
                cfg["llm"]["max_tokens"] = info["max_tokens"]
            if info.get("effort_levels") and cfg["llm"]["effort"] not in info["effort_levels"]:
                raise SystemExit(f"effort {cfg['llm']['effort']!r} not supported: {info['effort_levels']}")
        else:
            raise SystemExit(f"unknown backend {a.backend!r}")
        self.pool = cf.ProcessPoolExecutor(max_workers=cfg["evolution"]["workers"])

    # -- logging -------------------------------------------------------------------
    def log_attempt(self, rec):
        with self.lock, open(os.path.join(self.out, "attempts.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def log_private(self, rec):
        with self.lock, open(os.path.join(self.out, "private", "verify.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def save_state(self):
        self.archive.save(self.out)
        tmp = os.path.join(self.out, "state.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, os.path.join(self.out, "state.json"))

    def budget_left(self, inflight=0):
        """--max-calls caps completed + in-flight calls (any backend); --budget-usd caps
        dollar spend and is only accepted with --backend api."""
        mc = self.args.max_calls
        if mc is not None and self.state["calls"] + inflight >= mc:
            return False
        b = self.args.budget_usd
        if b is None:
            return True
        calls = self.state["calls"]
        est = self.state["spent_usd"] / calls if calls else self.cfg["llm"]["est_call_usd"]
        return self.state["spent_usd"] + inflight * est < b

    # -- seeds ---------------------------------------------------------------------
    def seed_archive(self):
        paths = sorted(glob.glob(self.args.seeds))
        srcs = {os.path.splitext(os.path.basename(p))[0]: open(p).read() for p in paths}
        evals = {k: self.pool.submit(_eval_job, s, self.cfg) for k, s in srcs.items()}
        for name, fut in evals.items():
            r = fut.result()
            entry = self.make_entry(f"seed_{name}", 0, None, "seed", srcs[name], r)
            rec = {"id": entry["id"], "gen": 0, "parent": None, "mode": "seed", "valid": r["valid"],
                   "score": r["score"], "feedback": r["feedback"], "eval_s": r["elapsed"]}
            if r["valid"]:
                v = verify(srcs[name], self.cfg, {"T": entry["T"], "Q": entry["Q"]})
                entry["verified"] = v["passed"]
                rec["verify"] = v["feedback"]
                if not v["passed"]:
                    self.log_private({"id": entry["id"], **v})
                else:
                    self.archive.add(entry)
            self.write_candidate(entry["id"], srcs[name])
            self.log_attempt(rec)
            print(f"[seed] {entry['id']}: " + (f"T*Q {r['score']:,} ({rec.get('verify')})" if r["valid"]
                                               else "INVALID " + r["failure"]))
        if not self.archive.entries:
            raise SystemExit("No valid, verified seed; nothing to evolve.")

    def make_entry(self, eid, gen, parent, mode, src, r):
        row = _score_row(r, self.cfg) or {}
        return {"id": eid, "gen": gen, "parent": parent, "mode": mode, "source": src,
                "score": r["score"], "T": row.get("T"), "Q": row.get("Q"),
                "widths": [{k: w.get(k) for k in ("n", "T", "Q", "gates")} for w in r["widths"]],
                "scaling_exponent": r["scaling_exponent"], "feedback": r["feedback"], "verified": None}

    def write_candidate(self, eid, src):
        with open(os.path.join(self.out, "candidates", f"{eid}.py"), "w") as f:
            f.write(src)

    # -- one generation ----------------------------------------------------------
    def generation(self, gen):
        cfg, evo = self.cfg, self.cfg["evolution"]
        tries = self.tries.get(gen, 0)
        self.tries[gen] = tries + 1
        rng = random.Random(f"{self.args.rng_seed}-{gen}-{self.state['sessions']}-{tries}")
        suffix = (f"_r{self.state['sessions']}" if self.state["sessions"] else "") + (f"_t{tries}" if tries else "")
        jobs = []
        for k in range(evo["per_gen"]):
            mode = "redesign" if rng.random() < evo["redesign_prob"] else "refine"
            parent = self.archive.pick_parent(rng, evo["front_parent_prob"])
            insp = self.archive.inspirations(parent, evo["inspirations"], rng)
            jobs.append({"id": f"g{gen:03d}_{k}{suffix}", "mode": mode, "parent": parent,
                         "inspirations": [e["id"] for e in insp],
                         "prompt": build_user_prompt(mode, parent, insp, self.archive)})

        first_event = threading.Event()
        stats = {"cost": 0.0, "notional": 0.0, "inflight": 0, "fatal": None, "limit": None, "skipped": 0}

        def call(job, wait_for_cache):
            if wait_for_cache:  # let the first request create the prompt cache before the rest start
                first_event.wait(timeout=120)
            with self.lock:
                if stats["fatal"] or stats["limit"] or not self.budget_left(inflight=stats["inflight"]):
                    stats["skipped"] += 1
                    first_event.set()
                    return job, None
                stats["inflight"] += 1
            try:
                res = self.client.call(job["prompt"], on_first_event=first_event.set)
            except UsageLimitError as e:
                with self.lock:
                    stats["limit"] = e
                    stats["inflight"] -= 1
                first_event.set()
                return job, {"error": f"usage limit: {e}", "limit": True}
            except FatalAPIError as e:
                with self.lock:
                    stats["fatal"] = str(e)
                    stats["inflight"] -= 1
                first_event.set()
                return job, {"error": str(e)}
            except ClaudeCodeError as e:  # counted against --max-calls, logged as a failed attempt
                with self.lock:
                    self.state["calls"] += 1
                    stats["inflight"] -= 1
                first_event.set()
                return job, {"error": f"claude -p failed: {e}"}
            with self.lock:
                self.state["calls"] += 1
                self.state["spent_usd"] += res["cost_usd"]
                stats["cost"] += res["cost_usd"]
                notional = res.get("notional_cost_usd") or 0.0
                self.state["notional_usd"] += notional
                stats["notional"] += notional
                for t in self.state["tokens"]:
                    self.state["tokens"][t] += res["usage"].get(t, 0)
                stats["inflight"] -= 1
            return job, res

        evals, attempts = [], []
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(max_workers=len(jobs)) as tp:
            futs = [tp.submit(call, job, i > 0) for i, job in enumerate(jobs)]
            for fut in cf.as_completed(futs):
                job, res = fut.result()
                if res is None:
                    continue  # skipped: budget reached or a fatal error elsewhere
                rec = {"id": job["id"], "gen": gen, "parent": job["parent"]["id"], "mode": job["mode"],
                       "inspirations": job["inspirations"]}
                rec["backend"] = self.args.backend
                if "error" in res:
                    rec.update(valid=False, feedback=f"API error: {res['error']}",
                               usage_limit=bool(res.get("limit")))
                    self.log_attempt(rec)
                    continue
                rec.update(tokens=res["usage"], cost_usd=round(res["cost_usd"], 6), api_s=res["api_s"],
                           stop_reason=res["stop_reason"], api_attempts=res["attempts"],
                           request_id=res["request_id"], model=res.get("model"))
                if self.args.backend == "claude-code":
                    rec.update(cost_billed=False, notional_cost_usd=res.get("notional_cost_usd"),
                               models_used=res.get("models_used"), model_mismatch=res.get("model_mismatch"),
                               num_turns=res.get("num_turns"), duration_ms=res.get("duration_ms"),
                               duration_api_ms=res.get("duration_api_ms"))
                with open(os.path.join(self.out, "replies", f"{job['id']}.md"), "w") as f:
                    f.write(res["text"])
                    if res.get("thinking_summary"):
                        f.write("\n\n<!-- thinking summary -->\n" + res["thinking_summary"])
                if res["stop_reason"] == "max_tokens":
                    rec.update(valid=False, edit_format=None,
                               feedback="reply truncated: hit max_tokens before finishing")
                    self.log_attempt(rec)
                    continue
                try:
                    src, fmt = parse_reply(job["parent"]["source"], res["text"])
                except DiffError as e:
                    rec.update(valid=False, edit_format="diff" if "<<<<<<<" in res["text"] else None,
                               feedback=f"edit rejected: {e}")
                    self.log_attempt(rec)
                    continue
                rec["edit_format"] = fmt
                self.write_candidate(job["id"], src)
                dup = self.archive.duplicate_of(src)
                if dup:
                    rec.update(valid=False, feedback=f"duplicate of {dup}; not re-evaluated")
                    self.log_attempt(rec)
                    continue
                evals.append((job, rec, src, self.pool.submit(_eval_job, src, cfg)))

            # Evaluate (in parallel processes), then verify would-be new bests.
            results = []
            for job, rec, src, fut in evals:
                r = fut.result()
                rec.update(valid=r["valid"], score=r["score"], eval_s=r["elapsed"],
                           failure_kind=r["failure_kind"], feedback=r["feedback"])
                entry = self.make_entry(job["id"], gen, job["parent"]["id"], job["mode"], src, r)
                rec.update(T=entry["T"], Q=entry["Q"])
                results.append((rec, entry, r))
            to_verify = [(rec, entry) for rec, entry, r in results
                         if r["valid"] and self.archive.would_be_new_best(r["score"])]
            vfuts = [(rec, entry, self.pool.submit(_verify_job, entry["source"], cfg,
                                                   {"T": entry["T"], "Q": entry["Q"]}))
                     for rec, entry in to_verify]
            for rec, entry, vf in vfuts:
                v = vf.result()
                rec["verify"] = v["feedback"]
                rec["verify_s"] = v["elapsed"]
                entry["verified"] = v["passed"]
                if not v["passed"]:
                    rec.update(valid=False, feedback="failed verification")
                    self.log_private({"id": entry["id"], **v})
            for rec, entry, r in results:
                if rec["valid"]:
                    self.archive.add(entry)
                self.log_attempt(rec)
                attempts.append(rec)

        # A generation cut short by a fatal error or usage limit before any candidate came
        # back is not counted as done, so --resume (or the wait) re-runs it.
        stats["completed"] = not ((stats["fatal"] or stats["limit"]) and not attempts)
        if stats["completed"]:
            self.state["gen"] = gen
        self.save_state()
        best = self.archive.best()
        valid = sum(1 for r in attempts if r.get("valid"))
        if self.args.backend == "claude-code":
            money = f"notional ${stats['notional']:.3f} (not billed), {self.state['calls']} calls total"
        else:
            money = f"gen ${stats['cost']:.3f}, total ${self.state['spent_usd']:.3f}"
        print(f"[gen {gen}] {len(jobs)} proposed, {valid} valid, front {len(self.archive.front())}, "
              f"best {best['id']} T*Q {best['score']:,} | {money}, {time.monotonic() - t0:.0f}s", flush=True)
        return stats

    def stop_message(self):
        if self.args.max_calls is not None and self.state["calls"] >= self.args.max_calls:
            return f"[stop] --max-calls {self.args.max_calls} reached."
        return (f"[stop] budget ${self.args.budget_usd:.2f} reached "
                f"(spent ${self.state['spent_usd']:.3f}).")

    def handle_limit(self, err):
        """Usage limit: wait for the reported reset, or exit cleanly for --resume.
        Returns True to continue, False to stop."""
        cc = self.cfg["claude_code"]
        self.save_state()
        reset, now = err.reset_at, time.time()
        resume = f"python orchestrator.py --resume {self.out} --generations <N>"
        if self.args.on_limit == "wait" and reset and 0 < reset - now <= cc["max_limit_wait_s"]:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(reset))
            print(f"[limit] usage limit hit: {err}. State saved. Waiting until {when} "
                  f"({(reset - now) / 60:.0f} min), then continuing.", flush=True)
            while time.time() < reset + 60:
                time.sleep(min(60, reset + 60 - time.time() + 1))
            return True
        why = "no reset time reported" if not reset else (
            "--on-limit exit" if self.args.on_limit == "exit" else "reset is too far away")
        print(f"[stop] usage limit hit: {err}. State saved ({why}). Resume later with:\n  {resume}")
        return False

    # -- main loop -----------------------------------------------------------------
    def run(self):
        self.setup()
        try:
            if not self.archive.entries:
                self.seed_archive()
                self.save_state()
            done = 0
            while done < self.args.generations:
                if not self.budget_left():
                    print(self.stop_message())
                    break
                stats = self.generation(self.state["gen"] + 1)
                done += stats["completed"]
                if stats["fatal"]:
                    print(f"[stop] fatal error: {stats['fatal']}", file=sys.stderr)
                    return 1
                if stats["limit"] and not self.handle_limit(stats["limit"]):
                    return EXIT_USAGE_LIMIT
        except KeyboardInterrupt:
            print("[stop] interrupted; saving state.", file=sys.stderr)
            self.save_state()
            return 130
        finally:
            self.pool.shutdown(cancel_futures=True)
        best = self.archive.best()
        spent = (f"notional ${self.state['notional_usd']:.3f} (subscription, not billed)"
                 if self.args.backend == "claude-code" else f"spent ${self.state['spent_usd']:.3f}")
        print(f"Done. Best: {best['id']} T*Q {best['score']:,}. {spent} over {self.state['calls']} calls. "
              f"Run dir: {self.out}")
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="claude-code", choices=["claude-code", "api", "mock"],
                    help="claude-code (default): `claude -p` on your subscription; api: paid Anthropic API, "
                         "only ever used when passed explicitly; mock: local fake model")
    ap.add_argument("--generations", type=int, default=10, help="generations to run (more, when resuming)")
    ap.add_argument("--per-gen", type=int, default=None,
                    help="candidates per generation (default 1 for claude-code, 4 otherwise)")
    ap.add_argument("--workers", type=int, default=None, help="parallel evaluation processes (config: 4)")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="stop after this many model calls in total (cumulative across --resume)")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="--backend api only: stop when cumulative API spend reaches this")
    ap.add_argument("--on-limit", default=None, choices=["wait", "exit"],
                    help="claude-code usage limit: wait for the reported reset (default) or exit for --resume")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--eval-timeout", type=float, default=None)
    ap.add_argument("--seeds", default=os.path.join(HERE, "seeds", "*.py"))
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--claude-bin", default=None, help="path to the claude CLI (default: `claude` on PATH)")
    ap.add_argument("--mock", action="store_true", help="same as --backend mock")
    ap.add_argument("--print-system-prompt", action="store_true")
    args = ap.parse_args(argv)
    if args.print_system_prompt:
        print(build_system_prompt(load_config()))
        return 0
    if args.mock:
        args.backend = "mock"
    if args.budget_usd is not None and args.backend != "api":
        print("--budget-usd only applies to --backend api; use --max-calls to limit "
              f"the {args.backend} backend.", file=sys.stderr)
        return 2
    if args.on_limit is None:
        args.on_limit = load_config()["claude_code"]["on_limit"]
    if args.backend == "api" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (--backend api needs it).", file=sys.stderr)
        return 2
    return Run(args).run()


if __name__ == "__main__":
    sys.exit(main())
