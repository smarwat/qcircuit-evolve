import json
import os
import subprocess
import sys

from config import load_config
from conftest import ROOT
from orchestrator import build_system_prompt, extract_code


def test_mock_runs_two_generations(tmp_path):
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "orchestrator.py"), "--mock", "--generations", "2",
         "--children", "3", "--out", str(out)],
        capture_output=True, text=True, timeout=300, cwd=ROOT,
        env={k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    entries = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert {e["gen"] for e in entries} == {0, 1, 2}
    assert any(e["eval"]["valid"] for e in entries if e["gen"] == 2)
    assert any(not e["eval"]["valid"] for e in entries if e["gen"] > 0)  # mock exercises failures
    assert (out / "best.py").exists() and "def build_circuit" in (out / "best.py").read_text()
    assert "Done. Best:" in proc.stdout


def test_system_prompt_describes_contract_and_hides_secrets():
    cfg = load_config()
    s = build_system_prompt(cfg)
    for needle in ["build_circuit(c, n, p)", "16n + 64", "a^-1 mod p", "T*Q at n=32",
                   "log2(T32/T16)", "4294967291", "hidden", "divsteps", "history bits"]:
        assert needle in s, needle
    assert str(cfg["hidden_seed"]) not in s


def test_extract_code_picks_the_candidate_block():
    reply = "idea...\n```python\nx = 1\n```\nand\n```python\ndef build_circuit(c, n, p):\n    pass\n```\n"
    assert extract_code(reply).startswith("def build_circuit")
    assert extract_code("no code here") is None
