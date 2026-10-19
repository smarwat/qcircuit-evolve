"""Evolutionary loop: Claude proposes candidate generators, the evaluator scores them.

    python orchestrator.py --mock --generations 2          # no API calls
    python orchestrator.py --generations 30 --children 4   # needs ANTHROPIC_API_KEY

Each generation picks parents from the archive (best valid candidates), asks the
model for an improved `build_circuit(c, n, p)`, evaluates every child, and keeps
the best. Everything is logged under --out (default runs/<timestamp>/).
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import time

from config import ancilla_budget, load_config
from evaluator import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "claude-opus-5-5"

IDEAS = [
    "Binary GCD / Kaliski variants: different case orderings, fewer comparisons per step, "
    "Kaliski's doubling of the coefficients instead of modular halving (then fix the 2^k factor).",
    "divsteps-style constant-time inversion (Bernstein-Yang): a delta counter plus fixed-shape steps "
    "can replace the full n-bit u>v comparison with a sign test.",
    "Reduce or eliminate the stored history bits (the seed stores 2 per iteration, 4n in total).",
    "Recompute control decisions from register values during uncompute instead of storing them "
    "(e.g. the parity of the register just halved, or its top bit after a modular halving).",
    "Cheaper adders and comparators: comparators that share work with the following subtraction, "
    "controlled adders without the AND-into-temp register, constant adders specialised to the bits "
    "of p. (There is no measurement here: every AND you compute costs a Toffoli to undo.)",
    "Register reuse to cut Q: share temporaries, let u/v shrink as they lose bits, avoid "
    "separate x1/x2 swaps, drop ancillas that are only needed briefly.",
    "Avoid Bennett's full 2x: uncompute the history early with a cheaper reverse pass, or "
    "structure the loop so the inverse pass is shorter than the forward pass.",
]


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


def build_system_prompt(cfg):
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
You are designing reversible circuits for modular inversion: a small-scale version of the
prime-field inversion inside elliptic-curve point addition (cf. Roetteler, Naehrig, Svore &
Lauter 2017). Gates: X, CNOT, Toffoli only.

CONTRACT
Write a Python function `build_circuit(c, n, p)` that emits gates into `c` and works for ANY
width n and odd prime p < 2^n. Register layout:
  - qubits 0..n-1: input a. Must be unchanged at the end.
  - qubits n..2n-1: start at 0, must hold a^-1 mod p at the end.
  - qubits 2n and up: ancillas, at most {cfg['ancilla_per_bit']}n + {cfg['ancilla_const']}. Start at 0, must end at 0.
  - Inputs 0 and p..2^n-1 are don't-cares.
You may do any classical precomputation from n and p (constants, bit patterns, loops, helper
functions). You may NOT import anything, define classes, or use names starting with '__'.
Available builtins: range, len, min, max, abs, pow, divmod, sum, enumerate, zip, list, tuple,
dict, set, sorted, reversed, int, bool, bin, isinstance, map, filter, any, all, iter, next.

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
Lookup tables, input-specific tricks and anything exponential in n cannot work: they blow the gate
cap or the time limit at n=32, and the hidden inputs catch circuits that only handle the public
ones. Only genuinely algorithmic, polynomial-size constructions score.

IDEAS WORTH TRYING
{ideas}

OUTPUT FORMAT
Briefly state the idea you are trying and the T and Q you expect, then give the complete
candidate as ONE ```python code block containing `def build_circuit(c, n, p):` (helpers may be
nested inside it or defined at top level). Correctness first: a cheaper circuit that fails any
test scores nothing."""


def build_user_prompt(parent, archive, cfg):
    board = []
    for i, e in enumerate(sorted(archive, key=lambda e: e["score"])[:6]):
        widths = ", ".join(f"n={w['n']}: T={w['T']:,} Q={w['Q']}" for w in e["eval"]["widths"] if "T" in w)
        board.append(f"  #{i + 1} {e['id']}: score {e['score']:,} ({widths})")
    return f"""\
Current leaderboard (lower is better):
{chr(10).join(board)}

Parent candidate {parent['id']}:
```python
{parent['source'].rstrip()}
```

Evaluator feedback for the parent:
{parent['eval']['feedback']}

Propose a candidate that beats the parent's score of {parent['score']:,} while staying correct at every width."""


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    blocks = [b for b in blocks if "def build_circuit" in b]
    if blocks:
        return max(blocks, key=len)
    return text if "def build_circuit" in text else None


# ---------------------------------------------------------------------------
# Proposers
# ---------------------------------------------------------------------------
class ClaudeProposer:
    def __init__(self, model, max_tokens, system):
        import anthropic  # imported lazily so --mock never needs the SDK

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system = system

    def propose(self, user_prompt):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class MockProposer:
    """Deterministic stand-in for the model: returns small edits of the parent so the
    loop, evaluation, logging and failure feedback all get exercised without the API."""

    def __init__(self, rng):
        self.rng = rng
        self.calls = 0

    def propose(self, user_prompt):
        m = re.search(r"```python\n(.*?)```", user_prompt, re.S)
        src = m.group(1)
        self.calls += 1
        kind = ["same", "wasteful", "broken"][self.calls % 3]
        if kind == "wasteful":  # correct but adds a cancelling Toffoli pair
            src = src.rstrip() + "\n    c.ccx(c.inp[0], c.inp[1], c.anc[-1])\n    c.ccx(c.inp[0], c.inp[1], c.anc[-1])\n"
        elif kind == "broken":  # forgets to uncompute
            src = src.replace("    c.append_inverse(start, end)", "    pass  # mock: skipped uncompute")
        else:
            src = f"# mock edit {self.calls}\n" + src
        return f"Mock proposal ({kind}).\n```python\n{src}```\n"


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--children", type=int, default=4, help="candidates proposed per generation")
    ap.add_argument("--archive-size", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--seeds", default=os.path.join(HERE, "seeds", "*.py"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-timeout", type=float, default=None)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="use a local mock proposer (no API calls)")
    ap.add_argument("--print-system-prompt", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.eval_timeout:
        cfg["eval_timeout"] = args.eval_timeout
    system = build_system_prompt(cfg)
    if args.print_system_prompt:
        print(system)
        return 0

    out = args.out or os.path.join(HERE, "runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.join(out, "candidates"), exist_ok=True)
    with open(os.path.join(out, "system_prompt.txt"), "w") as f:
        f.write(system)
    log = open(os.path.join(out, "log.jsonl"), "a")
    rng = random.Random(args.rng_seed)
    proposer = MockProposer(rng) if args.mock else ClaudeProposer(args.model, args.max_tokens, system)

    def record(entry):
        path = os.path.join(out, "candidates", f"{entry['id']}.py")
        with open(path, "w") as f:
            f.write(entry["source"])
        log.write(json.dumps({k: v for k, v in entry.items() if k != "source"}) + "\n")
        log.flush()

    archive = []
    for path in sorted(glob.glob(args.seeds)):
        with open(path) as f:
            src = f.read()
        r = evaluate(src, cfg)
        entry = {"id": "seed_" + os.path.splitext(os.path.basename(path))[0], "gen": 0,
                 "parent": None, "source": src, "eval": r, "score": r["score"]}
        record(entry)
        print(f"[seed] {entry['id']}: {'score ' + format(r['score'], ',') if r['valid'] else 'INVALID'}"
              f"  ({r['elapsed']:.1f}s)")
        if r["valid"]:
            archive.append(entry)
    if not archive:
        print("No valid seed; nothing to evolve.", file=sys.stderr)
        return 1

    for gen in range(1, args.generations + 1):
        archive.sort(key=lambda e: e["score"])
        children = []
        for k in range(args.children):
            # Bias towards the best, but keep some diversity.
            parent = archive[min(int(rng.expovariate(1.0)), len(archive) - 1)]
            cid = f"g{gen:03d}_{k}"
            try:
                reply = proposer.propose(build_user_prompt(parent, archive, cfg))
            except Exception as e:  # noqa: BLE001 - keep the loop alive on API errors
                print(f"[gen {gen}] {cid}: proposer error: {e}", file=sys.stderr)
                continue
            src = extract_code(reply)
            if src is None:
                r = {"valid": False, "score": None, "widths": [], "feedback": "no code block found",
                     "failure": "no code block found", "elapsed": 0.0}
                src = reply
            else:
                r = evaluate(src, cfg)
            entry = {"id": cid, "gen": gen, "parent": parent["id"], "source": src, "eval": r,
                     "score": r["score"], "reply_head": reply[:400]}
            record(entry)
            status = f"score {r['score']:,}" if r["valid"] else "INVALID: " + (r["failure"] or "")[:100]
            print(f"[gen {gen}] {cid} <- {parent['id']}: {status}")
            if r["valid"]:
                children.append(entry)
        archive = sorted(archive + children, key=lambda e: e["score"])[: args.archive_size]
        best = archive[0]
        print(f"[gen {gen}] best so far: {best['id']} score {best['score']:,}")
        with open(os.path.join(out, "best.py"), "w") as f:
            f.write(best["source"])

    log.close()
    print(f"Done. Best: {archive[0]['id']} score {archive[0]['score']:,}. Logs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
