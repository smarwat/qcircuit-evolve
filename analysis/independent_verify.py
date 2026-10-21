"""Independent verification of candidate circuits with Qiskit + Qiskit Aer.

Shares no code with evaluator.py, circuit.py or sandbox_runner.py: it builds each
candidate with its own minimal builder, exports the gate list to OpenQASM 2 through
Qiskit, reloads the QASM, and simulates the reloaded circuit with Aer's
matrix_product_state method.

Checks
  basis states  n=4 and n=8: every input 1..p-1; n=16 and n=32: 200 random inputs plus
                edge cases and adversarial (hard step-count) inputs.
  quantum       n=4 and n=8, preparations of the input register:
                  (a) Hadamards on all n input qubits (includes don't-care inputs), n=4 and 8
                  (b) uniform superposition over the valid inputs 1..p-1, compiled to u/cx
                      gates, n=4 only (at n=8 it would repeat a ~1 h simulation)
                For each: exact marginal distribution of the (input, output) qubits, exact
                <sum of Z> over all ancillas (equals their number iff every ancilla is |0>
                with probability 1, i.e. the state is a product with the ancillas), and
                4096 measurement shots.
  selftest      Before the quantum checks, every Aer MPS primitive used is compared with
                the exact statevector method on a random 14-qubit reversible circuit. Two
                primitives that failed this comparison in Aer 0.18.0 are not used:
                save_amplitudes and initialize.

Run with the verification venv:
    .venv-verify/bin/python analysis/independent_verify.py results/ best=results/best.py
    .venv-verify/bin/python analysis/independent_verify.py results/ best=results/best.py --quantum-only
"""

import gzip
import json
import os
import random
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adversarial import adversarial_inputs  # step-count search only; no evaluator code  # noqa: E402

WIDTHS = [(4, 13), (8, 251), (16, 65521), (32, 4294967291)]
ANCILLA_BUDGET = lambda n: 16 * n + 64  # noqa: E731 - the task's published budget
SIM = AerSimulator(method="matrix_product_state")


# ---------------------------------------------------------------------------
# Minimal independent builder (same public API the candidates use)
# ---------------------------------------------------------------------------
class Builder:
    def __init__(self, n, p):
        self.n, self.p = n, p
        self.num_ancillas = ANCILLA_BUDGET(n)
        self.num_qubits = 2 * n + self.num_ancillas
        self.inp = list(range(n))
        self.out = list(range(n, 2 * n))
        self.anc = list(range(2 * n, self.num_qubits))
        self.ops = []
        self._free = self.anc[::-1]

    def _q(self, *qs):
        if len(set(qs)) != len(qs) or not all(0 <= q < self.num_qubits for q in qs):
            raise ValueError(f"bad qubits {qs}")

    def x(self, t):
        self._q(t)
        self.ops.append(("x", t))

    def cx(self, c, t):
        self._q(c, t)
        self.ops.append(("cx", c, t))

    def ccx(self, a, b, t):
        self._q(a, b, t)
        self.ops.append(("ccx", a, b, t))

    cnot, toffoli = cx, ccx

    def mark(self):
        return len(self.ops)

    def num_gates(self):
        return len(self.ops)

    def append_inverse(self, s, e=None):
        self.ops.extend(reversed(self.ops[s:len(self.ops) if e is None else e]))

    def invert_since(self, s):
        self.ops[s:] = list(reversed(self.ops[s:]))

    def alloc(self, k=1):
        return [self._free.pop() for _ in range(k)]

    def release(self, qs):
        self._free.extend(sorted(qs, reverse=True))


def build(source, n, p):
    ns = {}
    exec(compile(source, "<candidate>", "exec"), ns)
    b = Builder(n, p)
    r = ns["build_circuit"](b, n, p)
    if r is not None and hasattr(r, "__next__"):
        for _ in r:
            pass
    return b.ops


def to_qiskit(ops, n):
    """Compact qubit numbering: inputs 0..n-1, outputs n..2n-1, then touched ancillas."""
    touched = sorted({q for op in ops for q in op[1:]} - set(range(2 * n)))
    remap = {q: i for i, q in enumerate(list(range(2 * n)) + touched)}
    qc = QuantumCircuit(len(remap))
    for op in ops:
        if op[0] == "x":
            qc.x(remap[op[1]])
        elif op[0] == "cx":
            qc.cx(remap[op[1]], remap[op[2]])
        else:
            qc.ccx(remap[op[1]], remap[op[2]], remap[op[3]])
    return qc


def toffoli_depth(qc):
    """Toffoli layers, with CNOT/X free but respecting qubit dependencies."""
    t = [0] * qc.num_qubits
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        d = max(t[q] for q in qs) + (1 if inst.operation.name == "ccx" else 0)
        for q in qs:
            t[q] = d
    return max(t) if t else 0


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
def run_basis(base, n, p, inputs, batch=64):
    """Return list of (a, input_reg, output_reg, ancillas_all_zero) per input."""
    Q = base.num_qubits
    out = []
    for i in range(0, len(inputs), batch):
        circs = []
        for a in inputs[i:i + batch]:
            qc = QuantumCircuit(Q, Q)
            for j in range(n):
                if (a >> j) & 1:
                    qc.x(j)
            qc.compose(base, inplace=True)
            qc.measure(range(Q), range(Q))
            circs.append(qc)
        res = SIM.run(circs, shots=1).result()
        for k, a in enumerate(inputs[i:i + batch]):
            bits = next(iter(res.get_counts(k)))[::-1]  # bits[q] = qubit q
            reg_in = int(bits[:n][::-1], 2)
            reg_out = int(bits[n:2 * n][::-1], 2)
            out.append((a, reg_in, reg_out, "1" not in bits[2 * n:]))
    return out


def valid_prep_circuit(n, p):
    """Uniform superposition over 1..p-1 as ordinary u/cx gates. (Aer MPS mishandles the
    `initialize` instruction in long circuits; see selftest().)"""
    amp = np.zeros(2 ** n)
    amp[1:p] = 1 / np.sqrt(p - 1)
    prep = QuantumCircuit(n)
    prep.append(StatePreparation(amp), range(n))
    return transpile(prep, basis_gates=["u", "cx"], optimization_level=1)


def _prob(d, i):
    return float(d.get(i, d.get(hex(i), 0.0)))  # Aer returns int or hex keys depending on version


def selftest(seed=0, nq=14, gates=400):
    """Check every Aer MPS primitive used below against the exact statevector method on a
    random reversible circuit, with both input preparations. Returns a report; the caller
    refuses to run the quantum checks if any primitive disagrees."""
    rng = random.Random(seed)
    circ = QuantumCircuit(nq)
    for _ in range(gates):
        a, b, c = rng.sample(range(nq), 3)
        k = rng.randrange(3)
        circ.x(a) if k == 0 else circ.cx(a, b) if k == 1 else circ.ccx(a, b, c)
    n, p = 4, 13
    zsum = SparsePauliOp.from_sparse_list([("Z", [q], 1.0) for q in range(2 * n, nq)], nq)
    report = {"qubits": nq, "gates": gates, "checks": {}}
    for label, prep in (("hadamard", lambda q: q.h(range(n))),
                        ("valid_compiled", lambda q: q.compose(valid_prep_circuit(n, p), range(n), inplace=True))):
        vals = {}
        for method in ("statevector", "matrix_product_state"):
            qc = QuantumCircuit(nq)
            prep(qc)
            qc.compose(circ, inplace=True)
            qc.save_probabilities_dict(list(range(2 * n)), label="io")
            qc.save_expectation_value(zsum, list(range(nq)), label="z")
            d = AerSimulator(method=method).run(qc).result().data(0)
            vals[method] = ({i: _prob(d["io"], i) for i in range(2 ** (2 * n))}, float(np.real(d["z"])))
        sv, mps = vals["statevector"], vals["matrix_product_state"]
        ok_p = all(abs(sv[0][i] - mps[0][i]) < 1e-9 for i in sv[0])
        ok_z = abs(sv[1] - mps[1]) < 1e-9
        report["checks"][label] = {"probabilities_match": ok_p, "expectation_matches": ok_z}
    report["passed"] = all(v["probabilities_match"] and v["expectation_matches"] for v in report["checks"].values())
    report["known_bad_primitives_not_used"] = [
        "save_amplitudes (returned zeros after long circuits under MPS, statevector disagreed)",
        "initialize (gave wrong probabilities and expectation values under MPS in this setting)"]
    return report


def quantum_check(base, n, p, prep):
    """prep: 'hadamard' (H on every input qubit) or 'valid' (compiled uniform superposition
    over 1..p-1). Exact marginal probabilities over (input, output), exact <sum Z> over all
    ancillas, and 4096 measurement shots."""
    Q = base.num_qubits
    qc = QuantumCircuit(Q, Q)
    if prep == "hadamard":
        qc.h(range(n))
        weights = {a: 1 / 2 ** n for a in range(2 ** n)}
    else:
        qc.compose(valid_prep_circuit(n, p), range(n), inplace=True)
        weights = {a: 1 / (p - 1) for a in range(1, p)}
    qc.compose(base, inplace=True)
    valid = list(range(1, p))
    idx = [a | (pow(a, -1, p) << n) for a in valid]  # (a, a^-1) on the input/output qubits
    n_anc = Q - 2 * n
    qc.save_probabilities_dict(list(range(2 * n)), label="io_probs")
    if n_anc:
        zsum = SparsePauliOp.from_sparse_list([("Z", [q], 1.0) for q in range(2 * n, Q)], Q)
        qc.save_expectation_value(zsum, list(range(Q)), label="anc_z")
    # Saves come first, measurements last: Aer simulates once and samples the shots.
    qc.measure(range(Q), range(Q))
    t0 = time.time()
    result = SIM.run(qc, shots=4096, seed_simulator=1).result()
    sim_seconds = time.time() - t0
    res = result.data(0)
    counts = result.get_counts(0)
    io = res["io_probs"]
    io_valid = sum(_prob(io, i) for i in idx)
    expected_valid = sum(weights[a] for a in valid)
    anc_z = float(np.real(res["anc_z"])) if n_anc else 0.0
    # Support of the (input, output) distribution: one output per input means no branch
    # spread over several outputs.
    support = {}
    for key, pr in io.items():
        k = int(key, 16) if isinstance(key, str) else int(key)
        if pr > 1e-12:
            support.setdefault(k & ((1 << n) - 1), []).append((k >> n, float(pr)))
    multi = sum(1 for v in support.values() if len(v) > 1)

    bad_pairs = dirty = dontcare = dontcare_dirty = 0
    for key, c in counts.items():
        bits = key[::-1]
        a = int(bits[:n][::-1], 2)
        o = int(bits[n:2 * n][::-1], 2)
        anc_zero = "1" not in bits[2 * n:]
        if not 1 <= a < p:
            dontcare += c
            dontcare_dirty += 0 if anc_zero else c
            continue
        if o != pow(a, -1, p):
            bad_pairs += c
        if not anc_zero:
            dirty += c
    return {
        "prep": prep,
        "io_mass_on_correct_valid_pairs": io_valid,
        "expected_mass": expected_valid,
        "ancillas": n_anc,
        "ancilla_z_sum": anc_z,  # equals the number of ancillas iff every ancilla is |0> w.p. 1
        "inputs_in_support": len(support),
        "inputs_with_more_than_one_output": multi,
        "dont_care_outputs": {str(a): [o for o, _ in v] for a, v in sorted(support.items()) if not 1 <= a < p},
        "shots": 4096,
        "shots_on_valid_inputs_with_wrong_output": bad_pairs,
        "shots_on_valid_inputs_with_nonzero_ancilla": dirty,
        "shots_on_dont_care_inputs": dontcare,
        "dont_care_shots_with_nonzero_ancilla": dontcare_dirty,
        "sim_seconds": round(sim_seconds, 1),
    }


def edge_inputs(n, p):
    xs = [1, 2, 3, p - 1, p - 2, (p + 1) // 2]
    xs += [1 << k for k in range(n) if (1 << k) < p] + [(1 << k) - 1 for k in range(1, n + 1) if (1 << k) - 1 < p]
    return sorted({x for x in xs if 1 <= x < p})


# ---------------------------------------------------------------------------
QUANTUM_PREPS = {4: ("hadamard", "valid"), 8: ("hadamard",)}  # n=8 "valid" would double a ~1h simulation


def save(outdir, reports, extra=None):
    path = os.path.join(outdir, "verification.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    meta = {"qiskit": __import__("qiskit").__version__, "qiskit_aer": __import__("qiskit_aer").__version__,
            "method": "matrix_product_state"}
    data = {"meta": meta, "reports": reports, **{k: v for k, v in old.items() if k not in ("meta", "reports")}}
    data.update(extra or {})
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def load_base(source, name, n, p, outdir):
    ops = build(source, n, p)
    qc = to_qiskit(ops, n)
    text = qasm2.dumps(qc)
    qpath = os.path.join(outdir, "qasm", f"{name}_n{n}.qasm.gz")
    os.makedirs(os.path.dirname(qpath), exist_ok=True)
    with gzip.open(qpath, "wt") as f:
        f.write(text)
    with gzip.open(qpath, "rt") as f:  # reload from the file we wrote
        return qasm2.loads(f.read()), qpath, len(text)


def basis_stage(name, source, outdir, reports):
    report = {"name": name, "widths": []}
    reports.append(report)
    for n, p in WIDTHS:
        t0 = time.time()
        base, qpath, nbytes = load_base(source, name, n, p, outdir)
        ops_count = dict(base.count_ops())
        row = {"n": n, "p": p, "qasm": os.path.relpath(qpath, outdir), "qasm_bytes": nbytes,
               "qubits": base.num_qubits, "toffolis": ops_count.get("ccx", 0), "cnots": ops_count.get("cx", 0),
               "nots": ops_count.get("x", 0), "gates": sum(ops_count.values()),
               "toffoli_depth": toffoli_depth(base), "depth_all_gates": base.depth()}
        if n in (4, 8):
            inputs = list(range(1, p))
        else:
            rng = random.Random(20261020 + n)
            inputs = sorted(set(rng.randrange(1, p) for _ in range(200)) | set(edge_inputs(n, p))
                            | set(adversarial_inputs(n, p, 4242)))
        res = run_basis(base, n, p, inputs)
        fails = [r for r in res if not (r[1] == r[0] and r[2] == pow(r[0], -1, p) and r[3])]
        row["basis"] = {"inputs": len(inputs), "failures": len(fails),
                        "first_failures": [{"a": a, "input_reg": i, "output_reg": o, "ancillas_zero": z}
                                           for a, i, o, z in fails[:5]]}
        row["seconds"] = round(time.time() - t0, 1)
        print(f"[{name}] n={n}: Q={row['qubits']} T={row['toffolis']} Tdepth={row['toffoli_depth']} "
              f"basis {len(inputs) - len(fails)}/{len(inputs)} ok, {row['seconds']}s", flush=True)
        report["widths"].append(row)
        save(outdir, reports)
    return report


def quantum_stage(name, source, outdir, reports, report):
    st = selftest()
    save(outdir, reports, {"mps_selftest": st})
    print(f"[selftest] Aer MPS vs statevector on a random {st['qubits']}-qubit reversible circuit: "
          f"{'PASS' if st['passed'] else 'FAIL'} {st['checks']}", flush=True)
    if not st["passed"]:
        raise SystemExit("Aer MPS disagrees with statevector on the primitives used; not running quantum checks")
    rows = {r["n"]: r for r in report["widths"]}
    # The n=8 superposition is the slow part (MPS bond dimension up to 2^8).
    for n, preps in QUANTUM_PREPS.items():
        p = dict(WIDTHS)[n]
        base, _, _ = load_base(source, name, n, p, outdir)
        row = rows[n]
        row["quantum"] = []
        for prep in preps:
            q = quantum_check(base, n, p, prep)
            row["quantum"].append(q)
            print(f"[{name}] n={n} quantum ({prep}): io mass on correct pairs {q['io_mass_on_correct_valid_pairs']:.12f} "
                  f"(expected {q['expected_mass']:.12f}), <sum Z_anc> {q['ancilla_z_sum']:.9f} of {q['ancillas']}, "
                  f"bad shots {q['shots_on_valid_inputs_with_wrong_output']}/{q['shots_on_valid_inputs_with_nonzero_ancilla']}, "
                  f"{q['sim_seconds']}s", flush=True)
            save(outdir, reports)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    outdir = args[0]
    quantum_only = "--quantum-only" in argv
    reports = []
    if quantum_only:  # keep the finished basis-state results, (re)run only the quantum stage
        reports = json.load(open(os.path.join(outdir, "verification.json")))["reports"]
    for spec in args[1:]:
        name, path = spec.split("=", 1)
        with open(path) as f:
            source = f.read()
        report = next((r for r in reports if r["name"] == name), None) if quantum_only else None
        if report is None:
            report = basis_stage(name, source, outdir, reports)
        quantum_stage(name, source, outdir, reports, report)
    save(outdir, reports)


if __name__ == "__main__":
    main(sys.argv)
