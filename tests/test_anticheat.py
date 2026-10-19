"""Every cheat or mistake must be rejected, or scored honestly from the raw gate list."""

import re
import time

import pytest

from config import load_config
from evaluator import evaluate, inputs_for_width

SEED_TAIL = "    c.append_inverse(start, end)\n"


def seed_plus(seed_src, *lines):
    """The seed with extra statements appended at the end of build_circuit."""
    assert seed_src.endswith(SEED_TAIL)
    return seed_src + "".join(f"    {ln}\n" for ln in lines)


def small_cfg(**kw):
    """Only the cheap exhaustive widths, for fast tests."""
    cfg = load_config(**kw)
    cfg["widths"] = cfg["widths"][:2]
    cfg["score_width"] = 8
    cfg["scaling_widths"] = (4, 8)
    return cfg


# --------------------------------------------------------------------------
# Wrong answers
# --------------------------------------------------------------------------
def test_wrong_output_rejected():
    src = "def build_circuit(c, n, p):\n    for i in range(n):\n        c.cx(c.inp[i], c.out[i])\n"
    r = evaluate(src, small_cfg())
    assert not r["valid"] and r["failure_kind"] == "wrong"
    assert "n=4" in r["failure"] and "input a=2" in r["failure"]
    assert "output register holds 2, expected a^-1 mod p = 7" in r["failure"]
    assert r["score"] is None


def test_modified_input_rejected(seed_src):
    r = evaluate(seed_plus(seed_src, "c.x(c.inp[0])"), small_cfg())
    assert not r["valid"] and r["failure_kind"] == "wrong"
    assert "input register was modified" in r["failure"]


def test_dirty_ancilla_rejected(seed_src):
    r = evaluate(seed_plus(seed_src, "c.cx(c.inp[0], c.anc[-1])"), small_cfg())
    assert not r["valid"] and r["failure_kind"] == "wrong"
    assert "ancilla qubits not returned to 0" in r["failure"]
    assert "input a=1:" in r["failure"]


# --------------------------------------------------------------------------
# Tampering with the builder object
# --------------------------------------------------------------------------
def test_overwriting_toffoli_count_is_ignored(seed_src, seed_result):
    r = evaluate(seed_plus(seed_src, "c.toffoli_count = 0"))
    assert r["valid"]
    assert [w["T"] for w in r["widths"]] == [w["T"] for w in seed_result["widths"]]
    assert r["score"] == seed_result["score"]


def test_rebinding_gates_does_not_change_the_circuit(seed_src, seed_result):
    r = evaluate(seed_plus(seed_src, "c.gates = c.gates[:0]"), small_cfg())
    assert r["valid"]
    assert r["widths"][0]["T"] == seed_result["widths"][0]["T"]


@pytest.mark.parametrize("gate,msg", [
    ("(c.out[0], c.out[0], -1)", "control == target"),
    ("(c.out[0], c.inp[0], c.inp[0])", "not a legal Toffoli"),
    ("(c.out[0], -1, c.inp[0])", "not a legal Toffoli"),
    ("(c.num_qubits, -1, -1)", "outside"),
    ("(-5, -1, -1)", "outside"),
    ("(c.out[0], -7, -1)", "outside"),
])
def test_illegal_gate_appended_to_gates_rejected(seed_src, gate, msg):
    r = evaluate(seed_plus(seed_src, f"c.gates.extend({gate})"), small_cfg())
    assert not r["valid"] and r["failure_kind"] == "invalid", r["failure"]
    assert msg in r["failure"]


def test_widening_the_builder_does_not_widen_the_check(seed_src):
    r = evaluate(seed_plus(seed_src, "c.num_qubits = 10 ** 9", "c.x(10 ** 8)", "c.x(10 ** 8)"), small_cfg())
    assert not r["valid"] and r["failure_kind"] == "invalid"


# --------------------------------------------------------------------------
# Sandbox
# --------------------------------------------------------------------------
@pytest.mark.parametrize("src", [
    "import os\ndef build_circuit(c, n, p):\n    pass\n",
    "def build_circuit(c, n, p):\n    from os import path\n",
    "def build_circuit(c, n, p):\n    __import__('os')\n",
    "def build_circuit(c, n, p):\n    ().__class__.__base__.__subclasses__()\n",
])
def test_imports_and_escapes_rejected(src):
    r = evaluate(src, small_cfg())
    assert not r["valid"] and r["failure_kind"] == "rejected"
    assert r["widths"] == []


def test_dangerous_builtins_unavailable():
    src = "def build_circuit(c, n, p):\n    open('/etc/passwd').read()\n"
    r = evaluate(src, small_cfg())
    assert not r["valid"] and r["failure_kind"] == "exception"
    assert "NameError" in r["failure"] and "open" in r["failure"]


def test_infinite_loop_times_out():
    src = "def build_circuit(c, n, p):\n    while True:\n        pass\n"
    t0 = time.monotonic()
    r = evaluate(src, load_config(eval_timeout=3.0))
    assert time.monotonic() - t0 < 15
    assert not r["valid"] and r["failure_kind"] == "timeout"
    assert "timed out" in r["failure"] and "n=4" in r["failure"]


def test_gate_cap_via_builder():
    src = "def build_circuit(c, n, p):\n    for i in range(2000):\n        c.x(c.anc[0])\n"
    r = evaluate(src, small_cfg(gate_cap=1000))
    assert not r["valid"] and r["failure_kind"] == "gate_cap"


def test_gate_cap_via_direct_append():
    src = "def build_circuit(c, n, p):\n    for i in range(2000):\n        c.gates.extend((c.anc[0], -1, -1))\n"
    r = evaluate(src, small_cfg(gate_cap=1000))
    assert not r["valid"] and r["failure_kind"] == "gate_cap"
    assert "2,000 gates > 1,000" in r["failure"]


# --------------------------------------------------------------------------
# Lookup tables and overfitting
# --------------------------------------------------------------------------
TABLE = '''
def build_circuit(c, n, p):
    chain = c.anc[: n - 1]           # chain[i] = AND of (flipped) input bits 0..i+1

    def entry(x, prev):
        flip = ~x & ((1 << n) - 1)   # flip zero bits so input == x becomes all ones
        d = flip ^ prev
        for i in range(n):
            if (d >> i) & 1:
                c.x(c.inp[i])
        m0 = c.mark()
        c.ccx(c.inp[0], c.inp[1], chain[0])
        for i in range(2, n):
            c.ccx(chain[i - 2], c.inp[i], chain[i - 1])
        m1 = c.mark()
        inv = pow(x, -1, p)
        for i in range(n):
            if (inv >> i) & 1:
                c.cx(chain[n - 2], c.out[i])
        c.append_inverse(m0, m1)
        return flip

    prev = 0
    for x in TABLE_INPUTS(n, p):
        prev = entry(x, prev)
    for i in range(n):
        if (prev >> i) & 1:
            c.x(c.inp[i])
'''


def test_lookup_table_passes_small_widths_then_fails_cleanly_at_32():
    src = "def TABLE_INPUTS(n, p):\n    return range(1, p)\n" + TABLE
    t0 = time.monotonic()
    r = evaluate(src)
    elapsed = time.monotonic() - t0
    assert not r["valid"] and r["score"] is None
    rows = {w["n"]: w for w in r["widths"]}
    assert rows[4]["passed"] and rows[8]["passed"]
    assert rows[32]["passed"] is False
    assert r["failure_kind"] in ("gate_cap", "timeout")
    assert "n=32" in r["failure"]
    assert elapsed < load_config()["eval_timeout"] + 10
    # The table is exponential: its n=16 circuit dwarfs the polynomial seed's.
    if rows.get(16, {}).get("passed"):
        assert rows[16]["T"] > 1_000_000


def test_hidden_failures_do_not_leak_inputs():
    cfg = load_config()
    public = {}
    hidden = {}
    for w in cfg["widths"]:
        if w["mode"] == "sampled":
            xs, k = inputs_for_width(cfg, w)
            public[w["n"]], hidden[w["n"]] = xs[:k], xs[k:]
    src = (f"PUBLIC = {public!r}\n"
           "def TABLE_INPUTS(n, p):\n    return range(1, p) if n <= 8 else PUBLIC[n]\n" + TABLE)
    r = evaluate(src, cfg)
    assert not r["valid"] and r["failure_kind"] == "hidden"
    assert r["failure"] == "failed hidden tests at n=16"
    text = r["feedback"] + "\n" + r["failure"]
    leaked = [x for x in set(hidden[16]) - set(public[16]) if re.search(rf"(?<![\d,]){x}(?![\d,])", text)]
    assert leaked == []
    assert "input a=" not in text
