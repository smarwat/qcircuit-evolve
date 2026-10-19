import json
import os
import random
import subprocess
import sys

import pytest

from archive import Archive, dominates
from config import load_config
from conftest import ROOT
from diffs import DiffError, apply_diff, parse_reply
from llm import ClaudeClient, cost_usd, usage_to_dict
from orchestrator import build_system_prompt, build_system_rules, build_task_spec

ENV = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}


def run_orch(*args, timeout=600):
    return subprocess.run([sys.executable, os.path.join(ROOT, "orchestrator.py"), *args],
                          capture_output=True, text=True, timeout=timeout, cwd=ROOT, env=ENV)


def attempts(out):
    return [json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines()]


# --------------------------------------------------------------------------
# End to end (mock)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run") / "r"
    proc = run_orch("--mock", "--generations", "2", "--per-gen", "4", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out, proc


def test_mock_runs_two_generations(mock_run):
    out, proc = mock_run
    recs = attempts(out)
    assert {r["gen"] for r in recs} == {0, 1, 2}
    assert sum(1 for line in proc.stdout.splitlines() if line.startswith("[gen ")) == 2
    for name in ("archive.json", "pareto.json", "best.py", "state.json", "system_prompt.txt"):
        assert (out / name).exists(), name
    pareto = json.loads((out / "pareto.json").read_text())
    assert pareto["best_score"] < 42_040_320  # the mock's genuine 2n-1 improvement was accepted
    for r in recs:
        if r["gen"] > 0:
            for k in ("parent", "mode", "tokens", "cost_usd", "feedback"):
                assert k in r, (k, r)


def test_mock_exercises_diff_and_verify_paths(mock_run):
    out, _ = mock_run
    recs = [r for r in attempts(out) if r["gen"] > 0]
    fb = [r["feedback"] for r in recs]
    assert any(r.get("edit_format") == "diff" and r["valid"] for r in recs)          # diff applied
    assert any(f.startswith("edit rejected: block 1: SEARCH text not found") for f in fb)
    assert any(f.startswith("INVALID: wrong at n=4") for f in fb)                     # broken diff
    cheat = [r for r in recs if r.get("verify") == "failed verification"]
    assert cheat and all(r["feedback"] == "failed verification" and not r["valid"] for r in cheat)
    assert any(r.get("verify") == "passed verification" for r in recs)
    private = (out / "private" / "verify.jsonl").read_text()
    assert "n=" in private
    archive = json.loads((out / "archive.json").read_text())
    assert not any(e["id"] == cheat[0]["id"] for e in archive)


def test_mock_resume_continues(mock_run, tmp_path):
    out, _ = mock_run
    before = len(attempts(out))
    proc = run_orch("--mock", "--resume", str(out), "--generations", "1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    recs = attempts(out)
    assert len(recs) > before and max(r["gen"] for r in recs) == 3
    state = json.loads((out / "state.json").read_text())
    assert state["gen"] == 3 and state["sessions"] == 1


def test_max_calls_stops_cleanly(tmp_path):
    out = tmp_path / "b"
    proc = run_orch("--mock", "--generations", "5", "--per-gen", "2", "--max-calls", "3", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[stop] --max-calls 3 reached." in proc.stdout
    state = json.loads((out / "state.json").read_text())
    assert state["calls"] == 3 and state["gen"] < 5


def test_api_backend_requires_api_key():
    proc = run_orch("--backend", "api", "--generations", "1")
    assert proc.returncode == 2 and "ANTHROPIC_API_KEY is not set" in proc.stderr


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
def test_system_prompt_describes_contract_and_hides_secrets():
    cfg = load_config()
    s = build_system_prompt(cfg)
    for needle in ["build_circuit(c, n, p)", "16n + 64", "a^-1 mod p", "T*Q at n=32", "log2(T32/T16)",
                   "4294967291", "every input 1..65520", "hidden", "divsteps", "history bits",
                   "<<<<<<< SEARCH", ">>>>>>> REPLACE", "Pareto front", "re-verified"]:
        assert needle in s, needle
    assert str(cfg["hidden_seed"]) not in s and str(cfg["verify"]["seed"]) not in s
    assert build_system_rules(cfg) in s and build_task_spec(cfg) in s


# --------------------------------------------------------------------------
# Diffs
# --------------------------------------------------------------------------
PARENT = "def build_circuit(c, n, p):\n    a = 1\n    b = 2\n    b = 2\n    return a\n"


def test_diff_applies_strictly():
    reply = "idea\n<<<<<<< SEARCH\n    a = 1\n=======\n    a = 3\n>>>>>>> REPLACE\n"
    assert apply_diff(PARENT, reply) == PARENT.replace("a = 1", "a = 3")
    two = reply + "<<<<<<< SEARCH\n    return a\n=======\n    return a + 1\n>>>>>>> REPLACE\n"
    assert apply_diff(PARENT, two).endswith("    return a + 1\n")


@pytest.mark.parametrize("reply,msg", [
    ("<<<<<<< SEARCH\n  a = 1\n=======\n  a = 3\n>>>>>>> REPLACE\n", "not found"),         # wrong indent
    ("<<<<<<< SEARCH\n    b = 2\n=======\n    b = 5\n>>>>>>> REPLACE\n", "occurs 2 times"),
    ("<<<<<<< SEARCH\n    a = 1\n=======\n    a = 3\n", "malformed"),
    ("<<<<<<< SEARCH\n\n=======\n    x = 1\n>>>>>>> REPLACE\n", "empty"),
    ("<<<<<<< SEARCH\n    a = 1\n=======\n    a = 1\n>>>>>>> REPLACE\n", "unchanged"),
    ("just words", "no SEARCH/REPLACE blocks and no ```python block"),
])
def test_bad_diffs_fail_cleanly(reply, msg):
    with pytest.raises(DiffError, match=msg):
        parse_reply(PARENT, reply)


def test_full_program_reply():
    src, fmt = parse_reply(PARENT, "x\n```python\ndef build_circuit(c, n, p):\n    pass\n```\n")
    assert fmt == "full" and src.startswith("def build_circuit")


# --------------------------------------------------------------------------
# Archive
# --------------------------------------------------------------------------
def entry(i, T, Q, verified=None):
    return {"id": f"e{i}", "gen": i, "parent": None, "mode": "x", "source": f"# {i}\n",
            "T": T, "Q": Q, "score": T * Q, "feedback": "", "verified": verified}


def test_pareto_front_and_verified_best():
    a = Archive([entry(0, 100, 10, True), entry(1, 80, 12), entry(2, 120, 9), entry(3, 120, 12), entry(4, 70, 20)])
    assert dominates(entry(9, 1, 1), entry(8, 2, 1))
    assert {e["id"] for e in a.front()} == {"e0", "e1", "e2", "e4"}
    assert a.best()["id"] == "e0"  # e1 has a lower T*Q but is not verified
    assert a.would_be_new_best(900) and not a.would_be_new_best(1000)
    rng = random.Random(0)
    picks = {a.pick_parent(rng, 0.8)["id"] for _ in range(200)}
    assert "e3" in picks and picks >= {"e0", "e1", "e2", "e4"}
    insp = a.inspirations(a.get("e0"), 2, rng)
    assert len(insp) == 2 and all(e["id"] != "e0" for e in insp)
    assert a.duplicate_of("# 3\n") == "e3"


# --------------------------------------------------------------------------
# Claude client (offline: request shape, usage, cost, retries)
# --------------------------------------------------------------------------
def test_cost_from_usage():
    prices = load_config()["llm"]["prices_per_mtok"]
    tok = {"input": 1_000_000, "output": 1_000_000, "cache_read": 1_000_000, "cache_write": 2_000_000,
           "cache_write_5m": 0, "cache_write_1h": 0}
    assert cost_usd(tok, prices, "1h") == pytest.approx(4 + 20 + 0.2 + 16)
    assert cost_usd(tok, prices, "5m") == pytest.approx(4 + 20 + 0.2 + 10)
    tok.update(cache_write_5m=1_000_000, cache_write_1h=1_000_000)
    assert cost_usd(tok, prices, "5m") == pytest.approx(4 + 20 + 0.2 + 5 + 8)


def _fake_message(anthropic):
    from anthropic.types import Message
    return Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                    {"type": "text", "text": "hello ```python\ndef build_circuit(c, n, p):\n    pass\n```"}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 3000, "output_tokens": 5000, "cache_read_input_tokens": 2100,
                  "cache_creation_input_tokens": 0,
                  "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}},
    })


def test_claude_client_request_shape_usage_and_retry(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    import httpx2

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    cfg = load_config()
    llm = dict(cfg["llm"], retry_base_delay=0.0, retry_max_delay=0.0)
    client = ClaudeClient(llm, build_system_rules(cfg), build_task_spec(cfg))
    calls = []
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    class FakeStream:
        def __init__(self, fail):
            self.fail = fail

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            if self.fail:  # mid-stream overloaded error: HTTP 200, error type in the body
                raise anthropic.APIStatusError(
                    "overloaded", response=httpx2.Response(200, request=req),
                    body={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
            yield object()

        def get_final_message(self):
            return _fake_message(anthropic)

    def fake_stream(**params):
        calls.append(params)
        return FakeStream(fail=len(calls) == 1)

    monkeypatch.setattr(client.client.messages, "stream", fake_stream)
    res = client.call("user prompt")

    assert len(calls) == 2 and res["attempts"] == 2  # retried once
    p = calls[0]
    assert p["model"] == "claude-opus-5-5" and p["max_tokens"] == llm["max_tokens"]
    assert p["thinking"]["type"] == "adaptive" and p["output_config"] == {"effort": llm["effort"]}
    assert "temperature" not in p and "top_p" not in p
    assert p["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": llm["cache_ttl"]}
    assert "cache_control" not in p["system"][0]
    assert res["usage"] == {"input": 3000, "output": 5000, "cache_read": 2100, "cache_write": 0,
                            "cache_write_5m": 0, "cache_write_1h": 0}
    assert res["cost_usd"] == pytest.approx((3000 * 4 + 5000 * 20 + 2100 * 0.2) / 1e6)
    assert parse_reply("", res["text"])[1] == "full"

    # A 400 is not retried and surfaces as a fatal error.
    from llm import FatalAPIError

    def bad_request(**params):
        raise anthropic.BadRequestError("bad", response=httpx2.Response(400, request=req), body={})

    monkeypatch.setattr(client.client.messages, "stream", bad_request)
    with pytest.raises(FatalAPIError):
        client.call("x")


def test_usage_to_dict_accepts_sdk_usage():
    pytest.importorskip("anthropic")
    from anthropic.types import Usage
    u = Usage(input_tokens=10, output_tokens=20, cache_read_input_tokens=30, cache_creation_input_tokens=40)
    assert usage_to_dict(u) == {"input": 10, "output": 20, "cache_read": 30, "cache_write": 40,
                                "cache_write_5m": 0, "cache_write_1h": 0}
