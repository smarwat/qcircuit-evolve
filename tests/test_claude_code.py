"""claude-code backend, tested against fake `claude` executables (no real CLI calls)."""

import json
import os
import stat
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

import orchestrator
from archive import Archive
from claude_code_client import (CLIAuthError, ClaudeCodeClient, ClaudeCodeError, UsageLimitError,
                                clean_env, parse_reset_time)
from config import load_config
from conftest import ROOT
from llm import FatalAPIError

IMPROVE_DIFF = ("Run 2n-1 iterations.\n<<<<<<< SEARCH\n    for it in range(2 * n):\n=======\n"
                "    for it in range(2 * n - 1):\n>>>>>>> REPLACE\n")


def success_json(text=IMPROVE_DIFF, model="claude-opus-5-5"):
    return {"type": "result", "subtype": "success", "is_error": False, "api_error_status": None,
            "duration_ms": 42000, "duration_api_ms": 41000, "num_turns": 1, "result": text,
            "stop_reason": "end_turn", "session_id": "sess-1", "total_cost_usd": 0.1234,
            "usage": {"input_tokens": 7000, "output_tokens": 9000, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 1800,
                      "cache_creation": {"ephemeral_5m_input_tokens": 1800, "ephemeral_1h_input_tokens": 0}},
            "modelUsage": {model: {"inputTokens": 7000, "outputTokens": 9000, "costUSD": 0.1234}}}


def error_json(msg, status):
    return {"type": "result", "subtype": "success", "is_error": True, "api_error_status": status,
            "duration_ms": 900, "duration_api_ms": 0, "num_turns": 1, "result": msg,
            "total_cost_usd": 0, "usage": {"input_tokens": 0, "output_tokens": 0}, "modelUsage": {}}


def make_fake(tmp_path, stdout="", stderr="", code=0, record=None, name="claude"):
    """Write an executable fake `claude` that prints fixed output and optionally records
    its argv, cwd, environment, stdin and the cwd listing to a JSON file."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import json, os, sys
        data = sys.stdin.read()
        rec = {str(record) if record else None!r}
        if rec:
            with open(rec, "w") as f:
                json.dump({{"argv": sys.argv[1:], "cwd": os.getcwd(), "env": dict(os.environ),
                           "stdin": data, "listing": os.listdir(".")}}, f)
        sys.stdout.write({stdout!r})
        sys.stderr.write({stderr!r})
        sys.exit({code})
        """))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def client_for(binary, warnings=None):
    cfg = dict(load_config()["claude_code"], binary=binary, retry_base_delay=0.0, max_retries=1)
    return ClaudeCodeClient(cfg, "SYSTEM PROMPT", warn=(warnings.append if warnings is not None else None))


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
def test_success_json(tmp_path):
    warnings = []
    c = client_for(make_fake(tmp_path, json.dumps(success_json())), warnings)
    res = c.call("prompt")
    assert res["text"] == IMPROVE_DIFF
    assert res["model"] == "claude-opus-5-5" and not res["model_mismatch"] and warnings == []
    assert res["num_turns"] == 1 and res["duration_ms"] == 42000
    assert res["cost_usd"] == 0.0 and res["notional_cost_usd"] == 0.1234
    assert res["usage"]["output"] == 9000 and res["usage"]["cache_write_5m"] == 1800


def test_usage_limit_error_json_with_reset(tmp_path):
    reset = int(time.time()) + 3600
    c = client_for(make_fake(tmp_path, json.dumps(error_json(f"Claude AI usage limit reached|{reset}", 429)),
                             code=1))
    with pytest.raises(UsageLimitError) as ei:
        c.call("prompt")
    assert ei.value.reset_at == reset


def test_usage_limit_on_stderr_without_json(tmp_path):
    c = client_for(make_fake(tmp_path, "", "You've hit your limit · resets 3pm (America/Los_Angeles)\n", code=1))
    with pytest.raises(UsageLimitError) as ei:
        c.call("prompt")
    assert ei.value.reset_at is not None and ei.value.reset_at > time.time()


def test_model_mismatch_warns_loudly(tmp_path):
    warnings = []
    c = client_for(make_fake(tmp_path, json.dumps(success_json(model="claude-sonnet-5-5"))), warnings)
    res = c.call("prompt")
    assert res["model_mismatch"] and res["model"] == "claude-sonnet-5-5"
    assert len(warnings) == 1 and "expected claude-opus-5-5" in warnings[0] and "claude-sonnet-5-5" in warnings[0]


def test_missing_model_usage_also_warns(tmp_path):
    warnings = []
    d = success_json()
    d["modelUsage"] = {}
    res = client_for(make_fake(tmp_path, json.dumps(d)), warnings).call("prompt")
    assert res["model_mismatch"] and res["model"] is None and warnings


@pytest.mark.parametrize("stdout", ["this is not json", "[1, 2, 3]", json.dumps({"type": "result"})])
def test_malformed_output(tmp_path, stdout):
    c = client_for(make_fake(tmp_path, stdout, code=0))
    with pytest.raises(ClaudeCodeError, match="malformed output|returned an error"):
        c.call("prompt")


def test_auth_error_is_fatal(tmp_path):
    c = client_for(make_fake(tmp_path, json.dumps(
        error_json("Failed to authenticate. API Error: 401 OAuth access token is invalid.", 401)), code=1))
    with pytest.raises(CLIAuthError, match="claude auth login") as ei:
        c.call("prompt")
    assert isinstance(ei.value, FatalAPIError)


def test_environment_is_stripped_and_cwd_is_empty(tmp_path, monkeypatch):
    for k, v in {"ANTHROPIC_API_KEY": "sk-ant-secret", "ANTHROPIC_AUTH_TOKEN": "tok",
                 "ANTHROPIC_BASE_URL": "http://localhost:1", "CLAUDE_CODE_SIMPLE": "1", "CLAUDECODE": "1",
                 "CLAUDE_CODE_USE_BEDROCK": "1", "QCE_HIDDEN_SEED": "123"}.items():
        monkeypatch.setenv(k, v)
    record = tmp_path / "record.json"
    c = client_for(make_fake(tmp_path, json.dumps(success_json()), record=record))
    c.call("THE USER PROMPT")
    seen = json.loads(record.read_text())
    leaked = [k for k in seen["env"] if k.startswith(("ANTHROPIC", "CLAUDE", "QCE"))]
    assert leaked == []
    assert "sk-ant-secret" not in json.dumps(seen["env"])
    assert set(seen["env"]) <= set(load_config()["claude_code"]["env_allowlist"])
    assert seen["listing"] == []  # a fresh, empty directory
    assert not os.path.realpath(seen["cwd"]).startswith(os.path.realpath(ROOT))
    assert not os.path.exists(seen["cwd"])  # cleaned up afterwards
    assert seen["stdin"] == "THE USER PROMPT"
    argv = seen["argv"]
    assert argv[:3] == ["-p", "--output-format", "json"]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert argv[argv.index("--max-turns") + 1] == "1"
    assert argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--system-prompt") + 1] == "SYSTEM PROMPT"
    assert "--no-session-persistence" in argv


def test_clean_env_is_an_allowlist():
    env = clean_env(["PATH", "HOME"], {"PATH": "/bin", "HOME": "/h", "ANTHROPIC_API_KEY": "x", "FOO": "y"})
    assert env == {"PATH": "/bin", "HOME": "/h"}


def test_parse_reset_time_variants():
    now = time.mktime((2026, 10, 19, 10, 0, 0, 0, 0, -1))
    assert parse_reset_time("Claude AI usage limit reached|1760000000", now) == 1760000000
    assert parse_reset_time("try again in 25 minutes", now) == now + 1500
    t = parse_reset_time("limit reached, resets 3pm", now)
    assert time.localtime(t).tm_hour == 15 and 0 < t - now < 86400
    t = parse_reset_time("resets 9am", now)  # already past today -> tomorrow
    assert 0 < t - now < 86400 and time.localtime(t).tm_hour == 9
    assert parse_reset_time("resets 3:30 pm (UTC)", now) is not None
    assert parse_reset_time("usage limit reached", now) is None


# --------------------------------------------------------------------------
# Orchestrator with the claude-code backend
# --------------------------------------------------------------------------
ENV = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC")}


def run_orch(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "orchestrator.py"), *args],
                          capture_output=True, text=True, timeout=600, cwd=ROOT, env=ENV)


def test_orchestrator_claude_code_end_to_end_with_limit_and_resume(tmp_path):
    out = tmp_path / "run"
    limit = make_fake(tmp_path, json.dumps(error_json("Claude AI usage limit reached", 429)), code=1,
                      name="claude_limit")
    ok = make_fake(tmp_path, json.dumps(success_json()), name="claude_ok")

    # 1) Usage limit with no reset time: state saved, clean exit telling the user to --resume.
    seeds = os.path.join(ROOT, "seeds", "binary_euclid.py")  # the fake diff targets this seed
    p = run_orch("--claude-bin", limit, "--generations", "1", "--out", str(out), "--seeds", seeds)
    assert p.returncode == orchestrator.EXIT_USAGE_LIMIT, p.stdout + p.stderr
    assert "[stop] usage limit hit" in p.stdout and "--resume" in p.stdout
    state = json.loads((out / "state.json").read_text())
    assert state["backend"] == "claude-code" and state["calls"] == 0

    # 2) Resume with a working CLI: one candidate, parsed, evaluated, verified, logged.
    p = run_orch("--claude-bin", ok, "--resume", str(out), "--generations", "1")
    assert p.returncode == 0, p.stdout + p.stderr
    recs = [json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines()]
    cand = [r for r in recs if r["gen"] > 0 and not r.get("usage_limit")]
    assert len(cand) == 1  # per-gen defaults to 1 for this backend
    r = cand[0]
    assert r["backend"] == "claude-code" and r["model"] == "claude-opus-5-5" and r["model_mismatch"] is False
    assert r["cost_usd"] == 0 and r["cost_billed"] is False and r["notional_cost_usd"] == 0.1234
    assert r["num_turns"] == 1 and r["edit_format"] == "diff" and r["valid"]
    assert r["verify"] == "passed verification"
    assert "notional $0.123 (not billed)" in p.stdout


def test_max_calls_stops_the_claude_code_backend(tmp_path):
    ok = make_fake(tmp_path, json.dumps(success_json(text="no code here")))
    out = tmp_path / "mc"
    p = run_orch("--claude-bin", ok, "--generations", "5", "--max-calls", "2", "--out", str(out))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "[stop] --max-calls 2 reached." in p.stdout
    assert json.loads((out / "state.json").read_text())["calls"] == 2


def test_default_backend_is_claude_code_never_api(tmp_path):
    p = run_orch("--claude-bin", str(tmp_path / "missing-claude"), "--generations", "1", "--out", str(tmp_path / "x"))
    assert p.returncode != 0 and "not found on PATH" in (p.stdout + p.stderr)
    assert "ANTHROPIC_API_KEY" not in p.stderr


def test_budget_usd_is_api_only(tmp_path):
    p = run_orch("--budget-usd", "1", "--generations", "1", "--out", str(tmp_path / "y"))
    assert p.returncode == 2 and "--budget-usd only applies to --backend api" in p.stderr
    p = run_orch("--backend", "api", "--generations", "1", "--out", str(tmp_path / "z"))
    assert p.returncode == 2 and "ANTHROPIC_API_KEY is not set" in p.stderr


def test_handle_limit_waits_for_reported_reset(tmp_path, monkeypatch):
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(orchestrator.time, "time", lambda: clock["t"])
    monkeypatch.setattr(orchestrator.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    run = orchestrator.Run(SimpleNamespace(on_limit="wait", max_calls=None, budget_usd=None, backend="claude-code"))
    run.out = str(tmp_path)
    run.archive = Archive([])
    run.state = {"gen": 1, "calls": 0}
    assert run.handle_limit(UsageLimitError("limit", reset_at=clock["t"] + 1800)) is True
    assert clock["t"] >= 1_000_000 + 1800 + 60
    assert (tmp_path / "state.json").exists()
    run.args.on_limit = "exit"
    assert run.handle_limit(UsageLimitError("limit", reset_at=clock["t"] + 1800)) is False
    run.args.on_limit = "wait"
    assert run.handle_limit(UsageLimitError("limit", reset_at=None)) is False
    assert run.handle_limit(UsageLimitError("limit", reset_at=clock["t"] + 7 * 86400)) is False
