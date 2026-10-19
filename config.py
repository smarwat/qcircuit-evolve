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
        {"n": 16, "p": 65521, "mode": "sampled"},
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
}


def load_config(**overrides):
    cfg = copy.deepcopy(TASK)
    env_seed = os.environ.get("QCE_HIDDEN_SEED")
    if env_seed:
        cfg["hidden_seed"] = int(env_seed)
    cfg.update(overrides)
    return cfg


def ancilla_budget(cfg, n):
    return cfg["ancilla_per_bit"] * n + cfg["ancilla_const"]
