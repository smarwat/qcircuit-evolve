"""Task configuration for width-generic reversible modular inversion.

Everything the evaluator and the orchestrator's system prompt need lives here.
`load_config()` returns a fresh dict so callers (tests, orchestrator flags) can
override fields without touching the module-level defaults.
"""

import copy
import os

TASK = {
    "name": "modinv",
    # Widths are evaluated in this order, cheapest first; evaluation stops at
    # the first failing width.
    "widths": [
        {"n": 4, "p": 13, "mode": "exhaustive"},
        {"n": 8, "p": 251, "mode": "exhaustive"},
        {"n": 16, "p": 65521, "mode": "exhaustive"},
        {"n": 32, "p": 4294967291, "mode": "sampled"},
    ],
    # The primary score is T*Q at this width.
    "score_width": 32,
    # Scaling exponent is log2(T[hi] / T[lo]).
    "scaling_widths": (16, 32),
    # Ancilla budget = ancilla_per_bit * n + ancilla_const.
    "ancilla_per_bit": 16,
    "ancilla_const": 64,
    # Hard cap on stored gates per width (X + CNOT + Toffoli).
    "gate_cap": 5_000_000,
    # Wall-clock budget for one full evaluation (all widths, generation and
    # simulation), in seconds.
    "eval_timeout": 180.0,
    # Random inputs for sampled widths.
    "random_samples": 256,
    "public_seed": 20240613,
    "hidden_samples": 256,
    # Hidden seed: never shown to the model. Override with QCE_HIDDEN_SEED.
    "hidden_seed": 0,  # redacted: rotated secret (published earlier, now burned)
    # Memory limit for the candidate subprocess (bytes). Only enforced where
    # the OS supports RLIMIT_AS (Linux); ignored on macOS.
    "child_memory_limit": 4 * 1024**3,
    # Verify tier: run only on candidates that would become the new best.
    # Never shown to the model (not in the system prompt, not in feedback).
    "verify": {
        # Secret seed for verify's random inputs and adversarial search.
        # Override with QCE_VERIFY_SEED.
        "seed": 0,  # redacted: rotated secret (published earlier, now burned)
        "random_score_width": 20_000,   # fresh random inputs at the score width
        "random_extra": 2_000,          # per hidden extra width
        "adversarial_per_metric": 16,   # hardest inputs per step-count metric
        # Hidden extra widths/primes; primality is checked in code (adversarial.is_prime).
        "extra_widths": [
            {"n": 0, "p": 0},  # redacted: rotated secret (published earlier, now burned)
            {"n": 0, "p": 0},  # redacted: rotated secret (published earlier, now burned)
            {"n": 0, "p": 0},  # redacted: rotated secret (published earlier, now burned)
        ],
    },
}


LLM = {
    "model": "claude-opus-5-5",
    # Opus 5.5 max output is 128K tokens (Models overview, checked 2026-10-18).
    # Thinking counts toward max_tokens; requests are streamed so large values work.
    "max_tokens": 64_000,
    # output_config.effort: low | medium | high | xhigh | max. Opus 5.5 defaults to
    # "medium"; keep it fixed for a run (changing effort invalidates the prompt cache).
    "effort": "high",
    # Opus 5.5: adaptive thinking is always on; "enabled"/"disabled" return 400.
    # display "omitted" (the model default) or "summarized" to log thinking summaries.
    "thinking": {"type": "adaptive", "display": "omitted"},
    # Prompt-cache TTL for the system prompt + task spec: "5m" or "1h".
    "cache_ttl": "1h",
    # Our own retry loop (on top of the SDK's) for 429 / 5xx / overloaded / connection errors.
    "max_retries": 6,
    "retry_base_delay": 2.0,
    "retry_max_delay": 90.0,
    # USD per million tokens for claude-opus-5-5, from
    # https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-18).
    # Standard (global) routing, non-batch, non-fast-mode.
    "prices_per_mtok": {
        "input": 4.00,
        "output": 20.00,          # includes thinking tokens
        "cache_write_5m": 5.00,
        "cache_write_1h": 8.00,
        "cache_read": 0.20,
    },
    # Used only to decide whether another call fits under --budget-usd before any
    # real call has completed; afterwards the running average is used.
    "est_call_usd": 0.50,
}

# Default backend: the user's Claude Code subscription via headless `claude -p`.
# Flags checked against `claude --help` (Claude Code 2.1.157 and 2.1.295) on 2026-10-19;
# --max-turns is accepted by the CLI but not listed in --help. claude-opus-5-5 needs
# Claude Code >= 2.1.280, and the isolation flags (--restricted etc.) need a recent CLI.
CLAUDE_CODE = {
    "binary": "claude",
    "model": "claude-opus-5-5",
    "expected_model": "claude-opus-5-5",   # warn loudly if another model answers
    "effort": "high",                       # --effort low|medium|high|xhigh|max
    "max_turns": 1,
    "timeout_s": 3600,                      # per call (a redesign call took 18.5 min)
    "per_gen": 1,                           # default candidates per generation for this backend
    "max_retries": 3,                       # transient CLI/API errors (5xx, overloaded)
    "retry_base_delay": 15.0,
    "retry_max_delay": 300.0,
    "on_limit": "wait",                     # usage limit: "wait" for the reported reset, or "exit"
    "max_limit_wait_s": 6 * 3600,           # never sleep longer than this; exit instead
    # The subprocess gets ONLY these variables: no ANTHROPIC_* (API keys, base URLs),
    # no CLAUDE_CODE_* / CLAUDECODE session variables inherited from a parent session.
    "env_allowlist": ["PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE",
                      "TMPDIR", "TERM", "__CF_USER_TEXT_ENCODING"],
}

EVOLUTION = {
    "per_gen": 4,                # candidates proposed per generation
    "redesign_prob": 0.30,       # else "refine"
    "front_parent_prob": 0.80,   # pick parent from the Pareto front, else any valid program
    "inspirations": 2,           # programs from the front shown alongside the parent
    "workers": 4,                # parallel evaluation processes
}


def load_config(**overrides):
    cfg = copy.deepcopy(TASK)
    cfg["llm"] = copy.deepcopy(LLM)
    cfg["claude_code"] = copy.deepcopy(CLAUDE_CODE)
    cfg["evolution"] = copy.deepcopy(EVOLUTION)
    env_seed = os.environ.get("QCE_HIDDEN_SEED")
    if env_seed:
        cfg["hidden_seed"] = int(env_seed)
    env_vseed = os.environ.get("QCE_VERIFY_SEED")
    if env_vseed:
        cfg["verify"]["seed"] = int(env_vseed)
    cfg.update(overrides)
    return cfg


def ancilla_budget(cfg, n):
    return cfg["ancilla_per_bit"] * n + cfg["ancilla_const"]
