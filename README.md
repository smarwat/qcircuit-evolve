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
| `config.py` | task (widths, primes, budget, caps), verify tier, LLM settings and prices, evolution settings |
| `circuit.py` | the `Circuit` builder handed to candidates (runs only inside the sandbox) |
| `sandbox.py`, `sandbox_runner.py` | AST checks + restricted builtins; child process that runs one candidate at one width |
| `evaluator.py` | validation, T/Q counting, bit-sliced simulation, checks, feedback, verify tier. CLI: `python evaluator.py cand.py [--verify]` |
| `adversarial.py` | step counters (binary Euclid, Kaliski, divsteps, binary GCD), hill-climbing search for hard inputs, `is_prime` |
| `orchestrator.py` | evolutionary loop (prompts, modes, parallel eval, verify, budget, resume, logging) |
| `llm.py` | `ClaudeClient` (streaming, prompt caching, usage/cost, retries) and `MockClient` |
| `diffs.py` | strict SEARCH/REPLACE diff parsing and application |
| `archive.py` | all valid programs, Pareto front over (T, Q) at n=32, verified best, persistence |
| `seeds/binary_euclid.py`, `seeds/divsteps.py` | two correct, structurally different baselines |
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
| `2n ..` (`c.anc`) | ancillas, budget `16n + 64`. Start and end at 0. |

Inputs 0 and `p .. 2^n-1` are don't-cares. The generator may do any classical
precomputation from `n` and `p` but may not import anything, define classes or use
`__` names. Helpers on `c`: `alloc/release`, `mark()`, `append_inverse(start, end)`
(Bennett uncompute), `invert_since(start)` (emit f, turn it into f⁻¹).

## Evaluation (what the model sees)

Cheapest width first, stopping at the first failure:

| n | p | inputs |
|---|---|---|
| 4 | 13 | all 1..12 |
| 8 | 251 | all 1..250 |
| 16 | 65521 | **all 1..65520** |
| 32 | 4294967291 | edge cases + 256 public random + 256 hidden random |

Edge cases are 1, 2, 3, p-1, p-2, (p+1)/2, every power of two below p and every 2^k-1 below p.

1. The candidate runs in a subprocess (`python -I -S sandbox_runner.py`) with restricted builtins,
   after an AST check that rejects `import`, `class` and `__` names.
2. The parent receives the raw gate array, re-validates every gate, enforces the cap
   (5,000,000 gates per width), and recounts T and Q. Nothing on the builder object is trusted.
3. Bit-sliced simulation in the parent: one Python int per qubit, where bit k is test input k.
4. Checks: input unchanged, output == `pow(a, -1, p)`, all ancillas 0.

Public failures name the width, the input and the failed check. Hidden failures say only
`failed hidden tests at n=32`. The timeout (180s) covers the whole evaluation.

**Score:** T = Toffolis, Q = distinct qubits referenced by any gate (plus the 2n
input/output qubits, which always count). The primary score is **T·Q at n=32**, lower is better.
Feedback also gives per-width T/Q and `log2(T32/T16)`.

## Verify tier (hidden)

A candidate becomes the new best only if `evaluator.verify()` also passes. It is run only
on candidates whose T·Q would beat the current verified best.

* **n=32, p=4294967291:** 20,000 fresh random inputs (secret `verify.seed`), the edge cases, and
  adversarial inputs. The adversarial inputs are the 16 hardest per step metric, found by random
  restarts + hill climbing (`adversarial.py`) under four step counters: the seed's binary
  Euclid, Kaliski's almost-inverse, Bernstein-Yang divsteps, and Stein's binary GCD. The
  regenerated circuit must have the same T and Q as the scored one.
* **The same generator at hidden widths and primes** (values redacted; since rotated). Each is tested on
  edge + 2,000 random + adversarial inputs. Primality is checked in code (deterministic
  Miller-Rabin).
* **Feedback:** the model only ever sees `failed verification`. The detail (width, input,
  check) goes to `private/verify.jsonl` in the run directory.

Why adversarial inputs matter: for divsteps at n=32, everything the normal tier tests needs at
most 75 iterations, while the search finds inputs needing 78. A divsteps circuit cut to 75
iterations at n=32 only passes the normal tier and fails verify. That case is tested in
`tests/test_verify.py`. For binary Euclid, the edge case 2^(n-1) already needs the worst-case
2n-1 steps, so the normal tier catches iteration cuts there.

## Seeds

| seed | n=4 T/Q | n=8 T/Q | n=16 T/Q | n=32 T/Q | T·Q @32 | log2(T32/T16) |
|---|---|---|---|---|---|---|
| `binary_euclid` | 1,792 / 55 | 6,912 / 103 | 27,136 / 199 | 107,520 / 391 | **42,040,320** | 1.99 |
| `divsteps` | 7,528 / 85 | 22,642 / 138 | 78,932 / 249 | 293,248 / 470 | 137,826,560 | 1.89 |

* **`binary_euclid`:** u, v halved mod p with 2 history bits per step and 2n steps, then a full
  Bennett reversal. T = 4n(26n+8), Q = 12n+7.
* **`divsteps`:** ⌊(49n+80)/17⌋ steps (Bernstein-Yang's bound), no comparator, no swaps. The
  swap case is `g -= f; f += g`, and delta is kept as `e = delta - 1` so `delta > 0` is a sign bit.
  It is dominated today, but it gives redesign mode a structurally different starting point.

Both pass verify. (Known free improvement: binary Euclid needs only 2n-1 iterations.)

## Orchestrator

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python orchestrator.py --generations 10 --max-calls 20          # default: claude -p, your subscription
.venv/bin/python orchestrator.py --resume runs/<dir> --generations 5      # continue a run
.venv/bin/python orchestrator.py --mock --generations 2                   # local fake model, no network
.venv/bin/python orchestrator.py --backend api --budget-usd 20 ...        # paid API, only when asked for explicitly
.venv/bin/python orchestrator.py --print-system-prompt
```

### Backends

| `--backend` | what runs | budget control | default per-gen |
|---|---|---|---|
| `claude-code` (default) | one headless `claude -p` subprocess per candidate, on your Claude Code subscription | `--max-calls` | 1 |
| `api` | Anthropic Messages API (`llm.ClaudeClient`); constructed only when `--backend api` is passed | `--budget-usd` (also `--max-calls`) | 4 |
| `mock` (`--mock`) | deterministic local fake | `--max-calls` | 4 |

`--budget-usd` is rejected for any backend but `api`.

**claude-code call shape** (flags checked against `claude --help`, Claude Code 2.1.157):

```
claude -p --output-format json --model claude-opus-5-5 --effort high --tools "" \
       --max-turns 1 --no-session-persistence --system-prompt "<rules + task spec>"   < user prompt on stdin
```

* `--tools ""` disables every tool, so the model can only answer in text. `--max-turns` is accepted
  by the CLI but not listed in `--help`; it is a backstop.
* Each call runs in a fresh, empty temp directory (deleted afterwards), so the model cannot see
  the repo or `config.py`.
* The subprocess environment is an allowlist (`PATH`, `HOME`, `USER`, locale, `TMPDIR`, ...). No
  `ANTHROPIC_*` variables (API keys, base URLs) and no `CLAUDE_CODE_*` variables inherited from a
  parent Claude session (e.g. `CLAUDE_CODE_SIMPLE`, which forces API-key auth). The CLI can only
  use your subscription login.
* The JSON result is parsed for `result`, `is_error` and `api_error_status`. A failed call still
  reports `subtype: "success"`, so `is_error` is what counts. It also yields `num_turns`,
  `duration_ms`, `usage` and `modelUsage`. The answering model is the `modelUsage` entry with
  the most output tokens. If it isn't `claude-opus-5-5`, a loud warning is printed and the
  attempt is marked `model_mismatch: true`. `total_cost_usd` is logged as
  `notional_cost_usd` with `cost_billed: false`.
* Errors:
  * **Usage/rate limits** (status 429 or limit wording, in the JSON or on stderr): state is
    saved. If a reset time is reported, the run waits until then (at most 6h) and continues.
    Otherwise it exits with code 75 and prints the `--resume` command. Use `--on-limit exit`
    to never wait.
  * **Auth failures (401/403):** the run stops and asks you to run `claude auth login`.
  * **Transient errors:** retried with backoff.
  * **Malformed output:** logged as a failed attempt.
* A generation cut short by a limit or auth error is not counted, so `--resume` re-runs it.

* **Modes:** `refine` (70%) asks for a focused improvement, preferably as SEARCH/REPLACE diffs.
  `redesign` (30%) asks for a structurally different algorithm as a full program.
* **Replies:** either a full program or OpenEvolve-style diffs against the parent:
  `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE`. Diffs apply strictly: whole-line,
  exact-whitespace matches, each SEARCH matching exactly once, in order. Anything else is
  logged as `edit rejected: block k: ...`.
* **Archive:** every valid program, plus the Pareto front over (T, Q) at n=32. Parents come
  from the front 80% of the time and from any valid program otherwise. Each prompt shows
  2 inspirations from the front and the parent's evaluator feedback.
* **Parallelism:** API calls run in threads. The first call of each generation starts alone,
  and the rest start once its stream begins, so they hit the prompt cache. Evaluation and
  verify run in a process pool (`workers`, default 4).
* **Prompt caching:** the system prompt is two static blocks (rules, task spec). One
  `cache_control: {"type": "ephemeral", "ttl": "1h"}` breakpoint on the last block caches both.
* **Cost control:** each call logs input, output, cache-read and cache-write tokens and
  `cost_usd`, computed from `config.LLM["prices_per_mtok"]`. `--budget-usd` caps the run's
  cumulative spend, including spend before a resume. No new call starts once spent plus
  estimated in-flight cost reaches the cap, and the run stops cleanly. In-flight calls can
  overshoot by at most one generation's calls.
* **Resume:** `--resume RUN_DIR` reloads `archive.json` and `state.json` and runs N more
  generations with the same model and effort settings.
* **Retries:** the SDK's own retries, then our backoff loop (up to 6 retries, honouring
  `retry-after`) for 408/409/429/5xx, connection errors, and mid-stream `overloaded_error` /
  `api_error`. Other 4xx errors stop the run.

Run directory: `attempts.jsonl` (every attempt: parent, mode, edit format, tokens, cost, API
and eval time, score, T/Q, feedback, verify result), `pareto.json`, `archive.json`, `best.py`,
`state.json`, `system_prompt.txt`, `candidates/`, `replies/` and `private/verify.jsonl`.
Each generation prints one progress line.

### Claude API settings (`config.LLM`)

Checked against anthropic-sdk-python 1.15.0 types and the platform.claude.com docs on 2026-10-18:

| setting | value | source |
|---|---|---|
| model | `claude-opus-5-5` | Models overview |
| max output | 128K (default `max_tokens` 64K; clamped to the Models API value at startup) | Models overview, `models.retrieve()` |
| thinking | adaptive, always on; `enabled`/`disabled` return 400; display defaults to `omitted` | Thinking docs |
| effort | `output_config.effort`, default `high` here (the API default for Opus 5.5 is `medium`); fixed per run because changing it invalidates the cache | Effort docs |
| caching | `cache_control` on system blocks, TTL `5m`/`1h`, min 512 tokens on Opus 5.5 | Prompt caching docs, SDK types |
| usage | `input_tokens`, `output_tokens` (incl. thinking), `cache_read_input_tokens`, `cache_creation_input_tokens`, `cache_creation.ephemeral_{5m,1h}_input_tokens` | SDK `Usage` |
| prices / MTok | input $4, output $20, cache write $5 (5m) / $8 (1h), cache read $0.20 | Pricing page |
| sampling params | not sent (`temperature`/`top_p` return 400 on Opus 5.5) | Thinking docs |

## Tests

`.venv/bin/python -m pytest tests -q` runs 69 tests in about 70s. None of them call the real
`claude` CLI or the API. They cover:
- the claude-code backend against fake `claude` executables: success, usage limit (JSON and
  stderr), model mismatch, malformed output, auth error, the environment/cwd/argv check,
  limit → resume, `--max-calls`, and the limit wait/exit logic
- both seeds are valid and pass verify; seed counts match the design formulas
- the bit-sliced simulator matches a reference simulator
- wrong output, modified input, dirty ancilla, counter tampering, illegal gates, imports and
  escapes, infinite loops and the gate cap are all rejected
- a lookup table fails at n=32; hidden-set failures don't leak inputs
- a divsteps circuit cut to fewer iterations only at n=32 passes normal and fails verify;
  verify feedback reveals nothing
- primality of every prime; hidden widths and primes are absent from the prompt
- diffs, the Pareto archive, cost maths, and the Claude request shape plus retry on mid-stream
  overload (offline, real SDK types)
- mock runs: 2 generations, every diff and verify path, resume, and the budget stop

## Timing

The full normal evaluation takes about 0.8s for the seed (n=16 exhaustive adds about 0.5s).
Verify takes about 2s. The worst case at the gate cap is about 15s for normal evaluation.
The adversarial search costs about 0.9s per width, once per process (cached).
