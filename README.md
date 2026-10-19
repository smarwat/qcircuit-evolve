# qcircuit-evolve

An evolutionary loop where Claude proposes reversible circuits (X / CNOT / Toffoli)
and a strict, deterministic evaluator scores them.

**Task:** width-generic modular inversion, `a -> a^-1 mod p`. This is a small-scale version
of the prime-field inversion inside elliptic-curve point addition (Roetteler, Naehrig,
Svore & Lauter 2017). Candidates are *generators*, not fixed circuits: they must work for
any width `n` and odd prime `p < 2^n`. Scoring is at n=32, where lookup tables cannot fit,
so only algorithmic constructions score.

## Files

| file | role |
|---|---|
| `config.py` | widths, primes, ancilla budget, gate cap, timeout, seeds. `load_config(**overrides)` |
| `circuit.py` | the `Circuit` builder handed to candidates (runs only inside the sandbox) |
| `sandbox.py` | AST checks (no imports, classes or `__` names) and the restricted builtins |
| `sandbox_runner.py` | child process: runs one candidate at one width, returns raw gate bytes |
| `evaluator.py` | validation, T/Q counting, bit-sliced simulation, checks, feedback. CLI: `python evaluator.py cand.py` |
| `orchestrator.py` | evolutionary loop, system prompt generated from config, `--mock` mode |
| `seeds/binary_euclid.py` | correct width-generic baseline |
| `tests/` | pytest suite |

## Candidate contract

```python
def build_circuit(c, n, p):
    ...  # emit gates with c.x(t), c.cx(ctrl, t), c.ccx(c1, c2, t)
```

| qubits | role |
|---|---|
| `0 .. n-1` (`c.inp`) | input `a`, little-endian. Must be unchanged at the end. |
| `n .. 2n-1` (`c.out`) | start at 0, must hold `a^-1 mod p` at the end. |
| `2n ..` (`c.anc`) | ancillas, budget `16n + 64` (`ancilla_per_bit`, `ancilla_const` in config). Start and end at 0. |

Inputs 0 and `p .. 2^n-1` are don't-cares (not tested). The generator may do any classical
precomputation from `n` and `p` but may not import anything. Helpers on `c`: `alloc/release`
(ancilla allocator), `mark()`, `append_inverse(start, end)` (Bennett uncompute) and
`invert_since(start)` (turn the gates just emitted into their inverse, e.g. adder to subtractor).
A function that returns a Python generator also works; the runner exhausts it.

## Evaluation

Cheapest width first, stopping at the first failure:

| n | p | inputs |
|---|---|---|
| 4 | 13 | all 1..12 |
| 8 | 251 | all 1..250 |
| 16 | 65521 | edge cases + 256 public random + 256 hidden random |
| 32 | 4294967291 | edge cases + 256 public random + 256 hidden random |

Edge cases are 1, 2, 3, p-1, p-2, (p+1)/2, every power of two below p and every 2^k-1 below p.
Public randoms use `public_seed`. Hidden randoms use `hidden_seed`, which the model never sees
(override it with `QCE_HIDDEN_SEED`).

For each width:

1. The candidate runs in a subprocess (`python -I -S sandbox_runner.py`) with restricted builtins.
   The source is AST-checked first: no `import`, no `class`, no identifier starting with `__`.
   The child's stdout is moved to a private fd before candidate code runs.
2. The child returns the raw `array('i')` of gates `(target, c1, c2)`, with -1 for unused controls.
   The parent re-validates every gate (range, distinct qubits, encoding), enforces the gate cap
   (5,000,000 per width), and recounts T and Q. Nothing else on the builder object is used.
3. Bit-sliced simulation in the parent: one Python int per qubit, where bit k is that qubit's
   value on test input k. One pass over the gates covers every input.
4. Checks per input: input register unchanged, output == `pow(a, -1, p)`, all ancillas 0.

Public failures name the width, the input and the failed check, e.g.
`wrong at n=4, p=13, input a=2: output register holds 2, expected a^-1 mod p = 7`.
Hidden failures only say `failed hidden tests at n=16`.

The timeout (180s, `eval_timeout`) covers the whole evaluation: generation and simulation at
every width.

## Score

* **T** = Toffoli count, **Q** = distinct qubits referenced by any gate, plus the 2n
  input/output qubits (always counted).
* **Primary score = T·Q at n=32**, lower is better. Invalid at any width means no score.
* Feedback also reports T, Q, T·Q and gate count at every width, plus the scaling exponent
  `log2(T32/T16)`: about 2 for the O(n²) seed, about 17 for a lookup table.

## Seed: reversible binary extended Euclid

`seeds/binary_euclid.py` is in the Kaliski family. It keeps `u = x1·a`, `v = x2·a (mod p)`,
starting from `u, v, x1, x2 = a, p, 1, 0`. Each of 2n iterations halves one register: u even;
u odd and v even; both odd with u>v; both odd with u≤v. Subtract first when needed. The
coefficient is halved mod p. Halving mod p instead of Kaliski's doubling means there is no
input-dependent `2^k` Montgomery factor to correct at the end. After v reaches 0, case B
repeats harmlessly, so no "done" flag is needed.

Each iteration stores 2 history bits (4n total), conditionally swaps so the operated register
is always `u`, and shifts right by relabelling qubits (no gates). The building blocks are
Cuccaro ripple adders and comparators. A final Bennett pass copies `x1` out and reverses
everything.

| n | p | T | Q | T·Q | gates |
|---|---|---|---|---|---|
| 4 | 13 | 1,792 | 55 | 98,560 | 5,876 |
| 8 | 251 | 6,912 | 103 | 711,936 | 22,504 |
| 16 | 65521 | 27,136 | 199 | 5,400,064 | 87,500 |
| 32 | 4294967291 | 107,520 | 391 | **42,040,320** | 348,064 |

T = 4n·(26n+8) exactly, Q = 12n+7. Scaling exponent 1.986. It fits the default ancilla budget
(10n+7 ancillas against 16n+64), so the budget is unchanged.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python evaluator.py seeds/binary_euclid.py
.venv/bin/python orchestrator.py --mock --generations 2          # no API calls
.venv/bin/python orchestrator.py --print-system-prompt
ANTHROPIC_API_KEY=... .venv/bin/python orchestrator.py --generations 30 --children 4
.venv/bin/python -m pytest tests -q
```

Each orchestrator run writes `runs/<timestamp>/` with `system_prompt.txt`, `log.jsonl`
(per-candidate eval results), `candidates/*.py` and `best.py`. `--mock` uses a deterministic
local proposer. It cycles through the parent unchanged, the parent plus a wasteful cancelling
Toffoli pair, and the parent with its uncompute removed, so success and failure paths both run.

## Timing (Apple Silicon, Python 3.14)

* Seed, full evaluation at all four widths: **~0.25 s**.
* Worst case, a valid circuit padded to ~5M gates at every width: **~8.6 s**. That is about
  1.3 s generating and transferring each width, plus 0.2–0.4 s simulating it.
* Lookup table: passes n=4, 8 and 16 (2.6M gates at n=16), then hits the gate cap at n=32
  after ~1.9 s total.

## Notes and caveats

* **Rare-path bugs.** Sampled widths can't prove correctness on every input. For the seed's
  algorithm, the edge cases already hit the worst-case 2n-1 step count at every width; 200k
  random inputs at n=32 only reach 60 of 64. An exhaustive n=16 check is affordable: set
  `"mode": "exhaustive"` on the n=16 width. That costs ~0.8 s for the seed and ~15 s at the gate cap.
* **Sandbox limits.** The sandbox is defence in depth, not a security boundary: memory limits
  (`RLIMIT_AS`) only apply on Linux. Scoring cannot be gamed through the sandbox, because the
  parent re-validates and re-simulates everything. Someone with repo access can read `hidden_seed`.
