"""Resource report for candidate circuits, computed without evaluator.py.

For each candidate and width: Toffoli count, qubit count (distinct qubits touched,
with the 2n input/output qubits always counted), Toffoli depth (Toffoli layers; X and
CNOT are free but respect qubit dependencies), and T-count under two conventions:
7 T per Toffoli, and 4 T per Toffoli (measurement-based uncomputation, as requested).

Also fits T(n) and Q(n) over every width n = 4..32 (largest prime below 2^n; the four
task widths use the task primes) and reports the growth order.

    .venv-verify/bin/python analysis/resources.py results/ seed=seeds/binary_euclid.py best=<path>
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from adversarial import is_prime  # noqa: E402  (number theory helper, not evaluator code)
from independent_verify import WIDTHS, build  # noqa: E402

TASK_PRIMES = dict(WIDTHS)


def prime_for(n):
    if n in TASK_PRIMES:
        return TASK_PRIMES[n]
    p = (1 << n) - 1
    while not is_prime(p):
        p -= 2
    return p


def measure(ops, n):
    toffolis = cnots = nots = 0
    touched = set(range(2 * n))
    t = {}
    depth = 0
    for op in ops:
        qs = op[1:]
        touched.update(qs)
        d = max(t.get(q, 0) for q in qs)
        if op[0] == "ccx":
            toffolis += 1
            d += 1
            depth = max(depth, d)
        elif op[0] == "cx":
            cnots += 1
        else:
            nots += 1
        for q in qs:
            t[q] = d
    return {"toffolis": toffolis, "qubits": len(touched), "toffoli_depth": depth, "cnots": cnots, "nots": nots,
            "gates": len(ops), "t_count_7": 7 * toffolis, "t_count_4": 4 * toffolis,
            "tq": toffolis * len(touched)}


def fit(ns, ys):
    ns, ys = np.asarray(ns, float), np.asarray(ys, float)
    quad = np.polyfit(ns, ys, 2)
    lin = np.polyfit(ns, ys, 1)
    big = ns >= 16
    slope = float(np.polyfit(np.log2(ns[big]), np.log2(ys[big]), 1)[0])
    resid = lambda c: float(np.max(np.abs(np.polyval(c, ns) - ys) / ys))  # noqa: E731
    return {"quadratic": [float(c) for c in quad], "quadratic_max_rel_err": resid(quad),
            "linear": [float(c) for c in lin], "linear_max_rel_err": resid(lin),
            "loglog_slope_n16_32": slope}


def report(name, source):
    rows = []
    for n in range(4, 33):
        p = prime_for(n)
        r = measure(build(source, n, p), n)
        r.update(n=n, p=p, task_width=n in TASK_PRIMES)
        rows.append(r)
    ns = [r["n"] for r in rows]
    return {"name": name, "rows": rows,
            "fit_T": fit(ns, [r["toffolis"] for r in rows]),
            "fit_Q": fit(ns, [r["qubits"] for r in rows]),
            "fit_Tdepth": fit(ns, [r["toffoli_depth"] for r in rows])}


def main(argv):
    out = argv[1]
    reps = []
    for spec in argv[2:]:
        name, path = spec.split("=", 1)
        with open(path) as f:
            reps.append(report(name, f.read()))
        print(f"{name}: n=32 {reps[-1]['rows'][-1]}", flush=True)
    with open(os.path.join(out, "resources.json"), "w") as f:
        json.dump(reps, f, indent=1)


if __name__ == "__main__":
    main(sys.argv)
