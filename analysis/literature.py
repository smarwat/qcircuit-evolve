"""Published reversible modular-inversion costs, as formulas with citations.

Formulas were transcribed from the papers' text (PDFs from arXiv, read 2026-10-20):

RNSL 2017  M. Roetteler, M. Naehrig, K. M. Svore, K. Lauter, "Quantum resource estimates
           for computing elliptic curve discrete logarithms", ASIACRYPT 2017,
           arXiv:1706.06752. Table 1 (p. 19), row "inv modp".
HJNRS 2020 T. Häner, S. Jaques, M. Naehrig, M. Roetteler, M. Soeken, "Improved quantum
           circuits for elliptic curve discrete logarithms", PQCrypto 2020,
           arXiv:2001.09580. Table 4 (p. 21), "Modular Inversion", three variants.
"""

import math

LG = math.log2


def _clog(n):
    return math.ceil(LG(n))


def _flog(n):
    return math.floor(LG(n))


SOURCES = [
    {
        "key": "RNSL2017",
        "variant": "inv modp",
        "cite": "Roetteler, Naehrig, Svore & Lauter 2017 (arXiv:1706.06752), Table 1, row 'inv modp'",
        "qubits_formula": "7n + 2⌈log2 n⌉ + 9",
        "qubits": lambda n: 7 * n + 2 * _clog(n) + 9,
        "metric": "Toffoli",
        "cost_formula": "32 n² log2 n  (leading order only; no lower-order term given for inversion)",
        "cost": lambda n: 32 * n * n * LG(n),
        "notes": [
            "Computes the Montgomery inverse x^-1·2^n mod p of an input in Montgomery form (Sec. 3.4).",
            "Input preserved; result XOR-ed into a fresh n-qubit register; full Bennett uncompute (Sec. 3.4, Table 1 caption).",
            "Qubit total includes the n input and n output qubits (Table 1 'total' column).",
            "Fixed 2n rounds with a counter: constant-time, input-independent gate sequence (Sec. 3.4).",
            "Designed to minimise qubits first, then Toffolis (Sec. 5); constant adders use dirty-ancilla incrementers, hence the log n factor.",
        ],
    },
    {
        "key": "HJNRS2020-LowW",
        "variant": "Low W (low width)",
        "cite": "Häner, Jaques, Naehrig, Roetteler & Soeken 2020 (arXiv:2001.09580), Table 4, Modular Inversion, Low W",
        "qubits_formula": "7n + ⌊lg n⌋ + 11",
        "qubits": lambda n: 7 * n + _flog(n) + 11,
        "metric": "T-count",
        "cost_formula": "236 n² lg n + 1.30·2^18",
        "cost": lambda n: 236 * n * n * LG(n) + 1.30 * 2 ** 18,
        "notes": [],
    },
    {
        "key": "HJNRS2020-LowT",
        "variant": "Low T (low T-count)",
        "cite": "Häner, Jaques, Naehrig, Roetteler & Soeken 2020 (arXiv:2001.09580), Table 4, Modular Inversion, Low T",
        "qubits_formula": "9n + ⌊lg n⌋ + 11",
        "qubits": lambda n: 9 * n + _flog(n) + 11,
        "metric": "T-count",
        "cost_formula": "496 n² + 1.70·2^14",
        "cost": lambda n: 496 * n * n + 1.70 * 2 ** 14,
        "notes": [],
    },
    {
        "key": "HJNRS2020-LowD",
        "variant": "Low D (low depth)",
        "cite": "Häner, Jaques, Naehrig, Roetteler & Soeken 2020 (arXiv:2001.09580), Table 4, Modular Inversion, Low D",
        "qubits_formula": "10n + 12.8",
        "qubits": lambda n: 10 * n + 12.8,
        "metric": "T-count",
        "cost_formula": "1062 n² − 1.28·2^16",
        "cost": lambda n: 1062 * n * n - 1.28 * 2 ** 16,
        "notes": [],
    },
]

HJNRS_NOTES = [
    "Same function as RNSL: Montgomery inverse x^-1·2^n mod p; input preserved (2n in/out qubits in Table 4); Bennett uncompute; 2n fixed rounds (Sec. 4.2).",
    "Costs are T-counts, not Toffoli counts: AND = 4 T, AND† = 0 T (measurement-based), Toffoli = 7 T (Sec. 2).",
    "Formulas are linear-regression fits to Q# resource counts for every n from 4 to 64 plus 110-521 bits; only the leading term was fitted (Appendix D), so n=32 lies inside the data but the constant terms are approximate.",
    "Improves RNSL by performing each round's operations once with controlled swaps and correcting the pseudo-inverse in parallel during uncomputation (Sec. 4.2).",
]


def evaluate(n):
    rows = []
    for s in SOURCES:
        rows.append({"key": s["key"], "variant": s["variant"], "cite": s["cite"], "metric": s["metric"],
                     "qubits_formula": s["qubits_formula"], "qubits": s["qubits"](n),
                     "cost_formula": s["cost_formula"], "cost": s["cost"](n), "n": n,
                     "notes": s["notes"] or HJNRS_NOTES})
    return rows
