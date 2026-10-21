# Shor's Edge: architecture

## Data flow

```
                    ┌──────────────────────────── orchestrator.py ────────────────────────────┐
                    │                                                                          │
 archive.json ──► pick parent (Pareto front) + 2 inspirations ──► prompt (refine / redesign; default 70/30, final run 60/40)
                    │                                                     │
                    │                                    backend: claude_code_client.py (claude -p, default)
                    │                                             llm.py ClaudeClient (--backend api)
                    │                                             llm.py MockClient (--mock)
                    │                                                     │ reply
                    │                                       diffs.py: full program or strict SEARCH/REPLACE
                    │                                                     │ candidate source
                    │                         ┌───────────── evaluator.py (process pool) ─────────────┐
                    │                         │ sandbox_runner.py (subprocess, restricted builtins,     │
                    │                         │   AST check via sandbox.py, circuit.py builder)         │
                    │                         │   └─► raw gate array (bytes) ──► parent re-validates,  │
                    │                         │       recounts T and Q, bit-sliced simulation, checks   │
                    │                         │ normal tier: n=4,8,16 exhaustive; n=32 sampled          │
                    │                         └─────────────────────────────────────────────────────────┘
                    │                                                     │ valid and would beat the best?
                    │                         evaluator.verify(): hidden widths/primes, 20k fresh +
                    │                         adversarial inputs (adversarial.py), T/Q must match
                    │                                                     │
                    └──── archive.py: add valid programs, Pareto front over (T, Q) at n=32, verified best
                                       │
                         runs/<dir>/attempts.jsonl, archive.json, pareto.json, best.py, replies/, candidates/
```

## Files

| path | role |
|---|---|
| `config.py` | task definition (widths, primes, budgets, caps), verify-tier sizes, model and backend settings, evolution settings; loads the secret hidden values |
| `hidden_values.example.json` | documented format with **public** example values; the real `hidden_values.json` is git-ignored |
| `make_hidden_values.py` | writes a fresh secret `hidden_values.json` (random widths and primes, checked in code) |
| `circuit.py` | the builder object candidates receive inside the sandbox (convenience only; never trusted) |
| `sandbox.py` | AST checks (no imports, classes or `__` names) and the restricted builtins |
| `sandbox_runner.py` | child process: runs one candidate at one width, returns the raw gate array |
| `evaluator.py` | validation, T/Q counting, bit-sliced simulation, feedback text, verify tier |
| `adversarial.py` | step counters for four GCD-type algorithms, hill-climbing search for worst-case inputs, `is_prime` |
| `orchestrator.py` | the loop: prompts, modes, backends, parallel evaluation, verify, budget / call limits, usage-limit handling, resume, logging |
| `claude_code_client.py` | default backend: one isolated `claude -p` subprocess per candidate |
| `llm.py` | paid API backend (`--backend api` only) and the mock backend |
| `diffs.py` | strict SEARCH/REPLACE parsing and application |
| `archive.py` | all valid programs, Pareto front, verified best, persistence |
| `seeds/` | two correct baselines: binary extended Euclid, Bernstein-Yang divsteps |
| `tests/` | offline test suite (never calls the real CLI or API) |
| `analysis/independent_verify.py` | Qiskit/Aer cross-check sharing no code with the evaluator |
| `analysis/resources.py` | Toffoli, qubit, Toffoli-depth, T-count at every n = 4..32, with fits |
| `analysis/literature.py` | published inversion cost formulas with citations |
| `analysis/reverify.py` | re-runs the evaluator's tiers with the current hidden values (counts only in the output) |
| `analysis/run_analysis.py` | plots, call statistics, lineage |
| `analysis/make_results.py` | writes `results/RESULTS.md` from the JSON outputs |
| `results/` | the final best, QASM exports, JSON data, plots, the final run's logs and lineage |

## Trust boundaries

1. **Candidate code is untrusted.** It runs in a separate process with restricted builtins after an
   AST check. Only the raw gate array crosses back. The parent re-validates every gate, recounts
   T and Q, and simulates with its own code. Nothing on the builder object is believed.
2. **The model never sees hidden information.** The system prompt is built from the public task
   config. Hidden-set failures say only `failed hidden tests at n=…`; verify failures say only
   `failed verification`. Tests check both.
3. **The `claude -p` subprocess is isolated**: empty temp directory, allowlisted environment (no
   `ANTHROPIC_*` or parent-session variables), `--tools ""`, and `--restricted --strict-mcp-config
   --mcp-config '{"mcpServers":{}}' --disable-slash-commands` so no MCP servers, plugins or skills
   load.
4. **The independent verifier** (`analysis/`) shares no code with `evaluator.py`, `circuit.py` or
   `sandbox_runner.py`. It self-tests the simulator primitives it relies on.
