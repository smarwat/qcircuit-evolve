"""Strict, deterministic evaluator for width-generic reversible modular inversion.

For each width (cheapest first, stop at the first failure):
  1. Run the candidate's build_circuit(c, n, p) in a sandboxed subprocess and
     receive its raw gate array.
  2. Re-validate every gate, recount T (Toffolis) and Q (qubits) from the array.
  3. Simulate bit-sliced (one Python int per qubit, bit k = test input k) with
     this module's own code, and check input preserved / output = a^-1 mod p /
     ancillas clean.
Score = T*Q at the score width (n=32), lower is better.

A separate verify tier (`verify()`, run by the orchestrator only on would-be new
bests) re-checks n=32 on 20,000 fresh + adversarial inputs and runs the same
generator at hidden widths/primes. Its feedback is only "failed verification".

Usage: python evaluator.py path/to/candidate.py [--verify]
"""

import json
import math
import os
import random
import subprocess
import sys
import time
from array import array

from adversarial import adversarial_inputs, is_prime
from config import ancilla_budget, load_config
from sandbox import SourceRejected, check_source

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "sandbox_runner.py")


class CandidateFailure(Exception):
    """A check failed. `public` is False when the message must not name inputs."""

    def __init__(self, msg, kind="failed"):
        super().__init__(msg)
        self.kind = kind


# --------------------------------------------------------------------------
# Test inputs
# --------------------------------------------------------------------------
def edge_inputs(n, p):
    xs = [1, 2, 3, p - 1, p - 2, (p + 1) // 2]
    xs += [1 << k for k in range(n) if (1 << k) < p]
    xs += [(1 << k) - 1 for k in range(1, n + 1) if (1 << k) - 1 < p]
    seen, out = set(), []
    for x in xs:
        if 1 <= x < p and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def random_inputs(seed, n, p, count):
    rng = random.Random(seed * 1_000_003 + n)
    return [rng.randrange(1, p) for _ in range(count)]


def inputs_for_width(cfg, width):
    """Return (inputs, n_public). Inputs [0, n_public) are public; the rest hidden."""
    n, p = width["n"], width["p"]
    if width["mode"] == "exhaustive":
        xs = list(range(1, p))  # 0 and p..2^n-1 are don't-cares
        return xs, len(xs)
    public = edge_inputs(n, p) + random_inputs(cfg["public_seed"], n, p, cfg["random_samples"])
    hidden = random_inputs(cfg["hidden_seed"], n, p, cfg["hidden_samples"])
    return public + hidden, len(public)


# --------------------------------------------------------------------------
# Running the candidate
# --------------------------------------------------------------------------
def generate_gates(source, n, p, cfg, timeout):
    job = {
        "source": source,
        "n": n,
        "p": p,
        "num_ancillas": ancilla_budget(cfg, n),
        "gate_cap": cfg["gate_cap"],
        "mem_limit": cfg.get("child_memory_limit"),
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", RUNNER],
            input=json.dumps(job).encode(),
            capture_output=True,
            timeout=max(timeout, 0.01),
            cwd=HERE,
        )
    except subprocess.TimeoutExpired:
        raise CandidateFailure(f"timed out while generating the circuit at n={n} "
                               f"(evaluation budget {cfg['eval_timeout']:g}s)", "timeout") from None
    raw = proc.stdout
    nl = raw.find(b"\n")
    try:
        header = json.loads(raw[:nl]) if nl >= 0 else None
    except ValueError:
        header = None
    if not isinstance(header, dict):
        tail = proc.stderr.decode(errors="replace")[-300:]
        raise CandidateFailure(f"candidate process crashed at n={n} (exit code {proc.returncode}). {tail}",
                               "crash")
    if not header.get("ok"):
        kind = str(header.get("kind", "exception"))
        err = str(header.get("error", ""))[:2000]
        if kind == "gate_cap":
            raise CandidateFailure(f"exceeded the gate cap at n={n}: {err}", "gate_cap")
        if kind == "rejected":
            raise CandidateFailure(f"source rejected: {err}", "rejected")
        raise CandidateFailure(f"build_circuit raised at n={n}:\n{err}", "exception")
    payload = raw[nl + 1 :]
    gates = array("i")
    if len(payload) % gates.itemsize:
        raise CandidateFailure(f"malformed gate payload at n={n}", "crash")
    gates.frombytes(payload)
    return gates


# --------------------------------------------------------------------------
# Validation and counting (never trusts anything but the raw array)
# --------------------------------------------------------------------------
def validate(gates, num_qubits, n, gate_cap):
    """Return (T, Q, num_gates). Raise CandidateFailure on any illegal gate."""
    if len(gates) % 3:
        raise CandidateFailure(f"gate array length {len(gates)} is not a multiple of 3 at n={n}", "invalid")
    G = len(gates) // 3
    if G > gate_cap:
        raise CandidateFailure(f"exceeded the gate cap at n={n}: {G:,} gates > {gate_cap:,}", "gate_cap")
    ts, c1s, c2s = gates[0::3], gates[1::3], gates[2::3]
    T = 0
    N = num_qubits
    # Bulk range check at C speed; if it fails, the loop below finds the gate.
    in_range = G == 0 or (min(ts) >= 0 and max(ts) < N and min(c1s) >= -1 and max(c1s) < N
                          and min(c2s) >= -1 and max(c2s) < N)
    for i, (t, a, b) in enumerate(zip(ts, c1s, c2s)):
        if not in_range and not (0 <= t < N and -1 <= a < N and -1 <= b < N):
            raise CandidateFailure(f"gate #{i} at n={n} uses a qubit outside [0, {num_qubits}): "
                                   f"(target={t}, c1={a}, c2={b})", "invalid")
        if b >= 0:
            if a < 0 or a == b or a == t or b == t:
                raise CandidateFailure(f"gate #{i} at n={n} is not a legal Toffoli: "
                                       f"(target={t}, c1={a}, c2={b})", "invalid")
            T += 1
        elif a >= 0:
            if a == t:
                raise CandidateFailure(f"gate #{i} at n={n} is a CNOT with control == target ({t})", "invalid")
        elif a != -1 or b != -1:
            raise CandidateFailure(f"gate #{i} at n={n} has a bad encoding ({t}, {a}, {b})", "invalid")
    touched = set(ts)
    touched.update(c1s)
    touched.update(c2s)
    touched.discard(-1)
    touched.update(range(2 * n))  # the input and output registers always count
    return T, len(touched), G


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def simulate_bitsliced(gates, state, mask):
    """Apply gates in place. state[q] is an int whose bit k is qubit q on input k."""
    for t, a, b in zip(gates[0::3], gates[1::3], gates[2::3]):
        if b >= 0:
            state[t] ^= state[a] & state[b]
        elif a >= 0:
            state[t] ^= state[a]
        else:
            state[t] ^= mask
    return state


def reference_simulate(gates, bits):
    """Plain per-input simulator: `bits` is a list of 0/1 per qubit. Returns a new list."""
    s = list(bits)
    for i in range(0, len(gates), 3):
        t, a, b = gates[i], gates[i + 1], gates[i + 2]
        if b >= 0:
            s[t] ^= s[a] & s[b]
        elif a >= 0:
            s[t] ^= s[a]
        else:
            s[t] ^= 1
    return s


def _slice(values, nbits):
    """Transpose: returns ints r[j] whose bit k is bit j of values[k]."""
    out = []
    for j in range(nbits):
        v = 0
        for k, x in enumerate(values):
            if (x >> j) & 1:
                v |= 1 << k
        out.append(v)
    return out


def _unslice(state, qubits, k):
    return sum(((state[q] >> k) & 1) << j for j, q in enumerate(qubits))


def check_width(gates, cfg, width, num_qubits):
    xs, n_public = inputs_for_width(cfg, width)
    return check_inputs(gates, width["n"], width["p"], xs, n_public, num_qubits)


def check_inputs(gates, n, p, xs, n_public, num_qubits):
    """Simulate all of xs at once. Inputs [0, n_public) are reported in detail;
    failures beyond that only say 'failed hidden tests'."""
    K = len(xs)
    mask = (1 << K) - 1
    a_bits = _slice(xs, n)
    want = _slice([pow(x, -1, p) for x in xs], n)
    state = [0] * num_qubits
    state[:n] = a_bits
    simulate_bitsliced(gates, state, mask)

    bad_in = 0
    for j in range(n):
        bad_in |= state[j] ^ a_bits[j]
    bad_out = 0
    for j in range(n):
        bad_out |= state[n + j] ^ want[j]
    bad_anc = 0
    for q in range(2 * n, num_qubits):
        bad_anc |= state[q]
    bad = bad_in | bad_out | bad_anc
    public_bad = bad & ((1 << n_public) - 1)
    if public_bad:
        k = (public_bad & -public_bad).bit_length() - 1
        x = xs[k]
        nfail = bin(public_bad).count("1")
        probs = []
        if (bad_in >> k) & 1:
            probs.append(f"input register was modified (holds {_unslice(state, range(n), k)})")
        if (bad_out >> k) & 1:
            got = _unslice(state, range(n, 2 * n), k)
            probs.append(f"output register holds {got}, expected a^-1 mod p = {pow(x, -1, p)}")
        if (bad_anc >> k) & 1:
            dirty = [q for q in range(2 * n, num_qubits) if (state[q] >> k) & 1]
            more = f" (+{len(dirty) - 8} more)" if len(dirty) > 8 else ""
            probs.append(f"ancilla qubits not returned to 0: {dirty[:8]}{more}")
        raise CandidateFailure(f"wrong at n={n}, p={p}, input a={x}: " + "; ".join(probs)
                               + f". ({nfail} of {n_public} public inputs fail at this width.)", "wrong")
    if bad:
        raise CandidateFailure(f"failed hidden tests at n={n}", "hidden")
    return K


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------
def evaluate(source, cfg=None):
    cfg = cfg or load_config()
    t0 = time.monotonic()
    deadline = t0 + cfg["eval_timeout"]
    rows, failure, failure_kind = [], None, None

    try:
        check_source(source)
    except SourceRejected as e:
        failure, failure_kind = f"source rejected: {e}", "rejected"

    for width in cfg["widths"] if failure is None else []:
        n, p = width["n"], width["p"]
        num_qubits = 2 * n + ancilla_budget(cfg, n)
        row = {"n": n, "p": p}
        try:
            g0 = time.monotonic()
            gates = generate_gates(source, n, p, cfg, deadline - g0)
            row["gen_s"] = round(time.monotonic() - g0, 3)
            T, Q, G = validate(gates, num_qubits, n, cfg["gate_cap"])
            row.update(T=T, Q=Q, TQ=T * Q, gates=G)
            s0 = time.monotonic()
            row["tests"] = check_width(gates, cfg, width, num_qubits)
            row["sim_s"] = round(time.monotonic() - s0, 3)
            if time.monotonic() > deadline:
                raise CandidateFailure(f"evaluation exceeded {cfg['eval_timeout']:g}s at n={n}", "timeout")
            row["passed"] = True
            rows.append(row)
        except CandidateFailure as e:
            row["passed"] = False
            rows.append(row)
            failure, failure_kind = str(e), e.kind
            break

    by_n = {r["n"]: r for r in rows if r.get("passed")}
    valid = failure is None
    score = by_n[cfg["score_width"]]["TQ"] if valid else None
    lo, hi = cfg["scaling_widths"]
    scaling = None
    if lo in by_n and hi in by_n and by_n[lo]["T"] > 0 and by_n[hi]["T"] > 0:
        scaling = math.log2(by_n[hi]["T"] / by_n[lo]["T"])
    result = {
        "valid": valid,
        "score": score,
        "widths": rows,
        "scaling_exponent": scaling,
        "failure": failure,
        "failure_kind": failure_kind,
        "elapsed": round(time.monotonic() - t0, 3),
    }
    result["feedback"] = format_feedback(result, cfg)
    return result


# --------------------------------------------------------------------------
# Verify tier (only for candidates that would become the new best)
# --------------------------------------------------------------------------
VERIFY_FAILED = "failed verification"


def verify_targets(cfg):
    """[(n, p, inputs)] for the verify tier. Never shown to the model."""
    v = cfg["verify"]
    sw = next(w for w in cfg["widths"] if w["n"] == cfg["score_width"])
    targets = [(sw["n"], sw["p"], v["random_score_width"])]
    targets += [(w["n"], w["p"], v["random_extra"]) for w in v["extra_widths"]]
    out = []
    for n, p, count in targets:
        if not (is_prime(p) and p % 2 == 1 and p < (1 << n)):
            raise ValueError(f"verify config: {p} is not an odd prime below 2^{n}")
        xs = edge_inputs(n, p)
        xs += list(adversarial_inputs(n, p, v["seed"], v["adversarial_per_metric"]))
        xs += random_inputs(v["seed"], n, p, count)
        out.append((n, p, xs))
    return out


def verify(source, cfg=None, expected=None):
    """Harder, hidden checks. `expected` = {"T": .., "Q": ..} at the score width
    from the normal evaluation; the verified circuit must be the scored one.

    Returns {"passed", "feedback", "private", "elapsed"}. Only `feedback` may be
    shown to the model, and it never names inputs or widths. `private` is for
    the human-readable run log."""
    cfg = cfg or load_config()
    t0 = time.monotonic()
    deadline = t0 + cfg["eval_timeout"]
    checked = []
    try:
        check_source(source)
        for n, p, xs in verify_targets(cfg):
            num_qubits = 2 * n + ancilla_budget(cfg, n)
            gates = generate_gates(source, n, p, cfg, deadline - time.monotonic())
            T, Q, _ = validate(gates, num_qubits, n, cfg["gate_cap"])
            if expected and n == cfg["score_width"] and p == next(
                    w["p"] for w in cfg["widths"] if w["n"] == n):
                if (T, Q) != (expected["T"], expected["Q"]):
                    raise CandidateFailure(f"circuit at n={n} differs from the scored one: "
                                           f"T,Q = {T},{Q} vs {expected['T']},{expected['Q']}", "mismatch")
            check_inputs(gates, n, p, xs, len(xs), num_qubits)
            if time.monotonic() > deadline:
                raise CandidateFailure(f"verification exceeded {cfg['eval_timeout']:g}s", "timeout")
            checked.append({"n": n, "p": p, "inputs": len(xs), "T": T, "Q": Q})
    except (CandidateFailure, SourceRejected) as e:
        return {"passed": False, "feedback": VERIFY_FAILED,
                "private": {"detail": str(e), "kind": getattr(e, "kind", "rejected"), "checked": checked},
                "elapsed": round(time.monotonic() - t0, 3)}
    return {"passed": True, "feedback": "passed verification", "private": {"checked": checked},
            "elapsed": round(time.monotonic() - t0, 3)}


def format_feedback(r, cfg):
    lines = []
    if r["valid"]:
        lines.append("VALID at every width.")
    else:
        lines.append("INVALID: " + r["failure"])
    if r["widths"]:
        lines.append("Per-width counts (T = Toffolis, Q = qubits used incl. the 2n register qubits):")
    for w in r["widths"]:
        if "T" in w:
            status = "pass" if w.get("passed") else "FAIL"
            lines.append(f"  n={w['n']:<2} p={w['p']:<10} T={w['T']:>10,}  Q={w['Q']:>5}  "
                         f"T*Q={w['TQ']:>14,}  gates={w['gates']:>9,}  [{status}]")
        else:
            lines.append(f"  n={w['n']:<2} p={w['p']:<10} no valid circuit  [FAIL]")
    lo, hi = cfg["scaling_widths"]
    if r["scaling_exponent"] is not None:
        e = r["scaling_exponent"]
        lines.append(f"Scaling exponent log2(T{hi}/T{lo}) = {e:.3f}  "
                     f"(T ~ n^{e:.2f}; 1-2 = polynomial algorithm, ~17 = exponential (lookup-style))")
    if r["valid"]:
        lines.append(f"PRIMARY SCORE (T*Q at n={cfg['score_width']}): {r['score']:,}  (lower is better)")
    else:
        lines.append("PRIMARY SCORE: none (candidate must pass every width)")
    lines.append(f"Evaluation time: {r['elapsed']:.1f}s")
    return "\n".join(lines)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__)
        return 2
    with open(args[0]) as f:
        src = f.read()
    r = evaluate(src)
    print(r["feedback"])
    if r["valid"] and "--verify" in argv:
        w = next(w for w in r["widths"] if w["n"] == load_config()["score_width"])
        v = verify(src, expected={"T": w["T"], "Q": w["Q"]})
        print(f"Verify: {v['feedback']} ({v['elapsed']:.1f}s)")
        if not v["passed"]:
            print(f"  [private] {v['private']['detail']}")
        return 0 if v["passed"] else 1
    return 0 if r["valid"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
