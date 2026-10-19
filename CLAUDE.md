# qcircuit-evolve: notes for Claude

Evolutionary search where Claude proposes reversible circuits (X/CNOT/Toffoli) for
width-generic modular inversion and a strict evaluator scores them. README.md has the details.

## Candidate contract

- A candidate is Python source defining `build_circuit(c, n, p)`. It emits gates via `c.x`,
  `c.cx`, `c.ccx` and must work for any width n and odd prime p < 2^n.
- Qubits `0..n-1` hold input a and must be unchanged. Qubits `n..2n-1` start at 0 and must end
  as a^-1 mod p. Ancillas from `2n` up (budget 16n+64) must start and end at 0.
  Inputs 0 and >= p are don't-cares.
- Candidates may not import anything, define classes or use `__` names. They run in a
  subprocess with restricted builtins.

## Scoring

- **Normal tier:** n=4, 8 and 16 exhaustive; n=32 on edges + 256 public + 256 hidden random
  inputs. Stops at the first failure.
- **Score:** T·Q at n=32 (Toffolis × qubits used), lower is better.
- **Verify tier** (`evaluator.verify`): runs only on would-be new bests. It checks n=32 on 20k
  fresh + adversarial inputs and the hidden widths/primes in `config.TASK["verify"]`.
  Feedback to the model is only "failed verification".

## Commands

```bash
.venv/bin/python -m pytest tests -q                          # ~70s, must stay green
.venv/bin/python evaluator.py seeds/binary_euclid.py --verify
.venv/bin/python orchestrator.py --mock --generations 2      # local fake model, no network
.venv/bin/python orchestrator.py --generations N --max-calls M    # real run: `claude -p` on the user's subscription
.venv/bin/python orchestrator.py --resume runs/<dir> --generations N
.venv/bin/python adversarial.py 32 4294967291                # inspect hardest inputs
```

The default backend is `claude-code` (headless `claude -p`, subscription, no per-call billing).
The paid API backend runs ONLY with an explicit `--backend api`. Never pass it, and never
add code paths that fall back to it, unless the user asks. Tests must never invoke the real
`claude` CLI or the API: use fake executables (`--claude-bin`) or `--mock`.

## Invariants (do not break)

- **Never trust the circuit object.** The stored gate array returned by the sandbox is the
  circuit. Re-validate every gate, recount T and Q from the array, and simulate with the
  evaluator's own code. Never read counts, methods or attributes from `c`.
- **Never loosen correctness checks to make a candidate pass.** No tolerances, no skipped
  widths, no sampling that replaces exhaustive checks, no relaxed ancilla or input checks,
  no fuzzy diff matching. If a candidate fails, it fails.
- **Never leak hidden information to the model.** Hidden-set failures say only
  `failed hidden tests at n=…`; verify failures say only `failed verification`. Hidden seeds,
  verify widths and verify primes must never appear in prompts or model-visible feedback
  (tests check this).
- Keep `--mock` working without network access or the `anthropic` package.
- The `claude -p` subprocess must keep running in an empty temp dir with the allowlisted
  environment (no ANTHROPIC_* / CLAUDE_CODE_* variables) and `--tools ""`.
- Do not change effort/thinking mid-run: it invalidates the prompt cache.
- Pricing lives in `config.LLM["prices_per_mtok"]`. Re-check it against the pricing page when the
  model changes.
