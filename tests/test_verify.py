"""The verify tier: hidden widths, fresh randoms and adversarial inputs."""

import os
import re

import pytest

from adversarial import METRICS, adversarial_inputs, is_prime, steps_binary_euclid, steps_divsteps
from config import load_config
from conftest import ROOT
from evaluator import VERIFY_FAILED, evaluate, inputs_for_width, verify, verify_targets
from orchestrator import build_system_prompt


def read_seed(name):
    with open(os.path.join(ROOT, "seeds", f"{name}.py")) as f:
        return f.read()


def expected_at_32(result):
    w = next(w for w in result["widths"] if w["n"] == 32)
    return {"T": w["T"], "Q": w["Q"]}


@pytest.mark.parametrize("name", ["binary_euclid", "divsteps"])
def test_seeds_pass_verify(name):
    src = read_seed(name)
    r = evaluate(src)
    assert r["valid"], r["feedback"]
    v = verify(src, expected=expected_at_32(r))
    assert v["passed"], v["private"]
    assert v["feedback"] == "passed verification"
    assert len(v["private"]["checked"]) == 1 + len(load_config()["verify"]["extra_widths"])


def test_fewer_iterations_only_at_32_passes_normal_but_fails_verify():
    cfg = load_config()
    w32 = next(w for w in cfg["widths"] if w["n"] == 32)
    p = w32["p"]
    xs, _ = inputs_for_width(cfg, w32)  # public AND hidden inputs of the normal tier
    cut = max(steps_divsteps(x, p) for x in xs)
    verify_xs = next(vx for n, q, vx in verify_targets(cfg) if (n, q) == (32, p))
    assert max(steps_divsteps(x, p) for x in verify_xs) > cut  # adversarial search found harder inputs
    assert cut < (49 * 32 + 80) // 17

    src = read_seed("divsteps").replace(
        "iters = (49 * n + 80) // 17",
        f"iters = (49 * n + 80) // 17 if n != 32 else {cut}")
    r = evaluate(src, cfg)
    assert r["valid"], r["feedback"]  # the normal tier can't tell
    v = verify(src, cfg, expected_at_32(r))
    assert not v["passed"]
    assert v["feedback"] == VERIFY_FAILED
    assert "n=32" in v["private"]["detail"]  # the human log knows where


def test_verify_feedback_reveals_nothing():
    # Correct at the public widths, wrong only at the hidden width.
    src = read_seed("binary_euclid").replace(
        "    c.append_inverse(start, end)\n",
        "    c.append_inverse(start, end)\n    if n == -1:\n        c.x(c.anc[-1])\n")
    r = evaluate(src)
    assert r["valid"]
    v = verify(src, expected=expected_at_32(r))
    assert not v["passed"]
    assert v["feedback"] == "failed verification"
    assert not re.search(r"\d", v["feedback"])
    assert "n=" in v["private"]["detail"]


def test_verify_rejects_a_different_circuit_than_the_scored_one(seed_src, seed_result):
    v = verify(seed_src, expected={"T": 1, "Q": 1})
    assert not v["passed"] and v["private"]["kind"] == "mismatch"


def test_hidden_primes_are_prime_and_never_shown():
    cfg = load_config()
    system = build_system_prompt(cfg)
    for w in cfg["widths"] + cfg["verify"]["extra_widths"]:
        assert is_prime(w["p"]) and w["p"] < 2 ** w["n"] and w["p"] % 2 == 1
    public_ns = {w["n"] for w in cfg["widths"]}
    for w in cfg["verify"]["extra_widths"]:
        assert str(w["p"]) not in system and f"{w['p']:,}" not in system
        if w["n"] not in public_ns:
            assert f"n={w['n']}" not in system
    assert str(cfg["verify"]["seed"]) not in system
    assert all(is_prime(q) for q in (2, 3, 13, 251, 65521)) and not any(is_prime(q) for q in (1, 91, 561, 65535))


def test_adversarial_search_is_deterministic_and_hard():
    p = 4294967291
    seed = load_config()["verify"]["seed"]
    a = adversarial_inputs(32, p, seed)
    assert a == adversarial_inputs.__wrapped__(32, p, seed)
    assert all(1 <= x < p for x in a)
    # Includes the binary-Euclid worst case region and beats random sampling for divsteps.
    import random
    rng = random.Random(0)
    rand_max = max(steps_divsteps(rng.randrange(1, p), p) for _ in range(5000))
    assert max(steps_divsteps(x, p) for x in a) > rand_max
    assert set(METRICS) >= {"binary_euclid", "kaliski", "divsteps"}
    assert steps_binary_euclid(1 << 31, p) == 63
