"""Hidden test values: loaded from an untracked file, with a documented public fallback."""

import json
import os
import subprocess
import sys

import pytest

import config
from config import check_hidden, load_config
from conftest import ROOT


def test_example_file_is_valid_and_documented():
    with open(os.path.join(ROOT, "hidden_values.example.json")) as f:
        ex = json.load(f)
    check_hidden(ex, "example")
    assert "PUBLIC EXAMPLE" in ex["_comment"]


def test_missing_hidden_file_falls_back_to_example_with_warning(tmp_path):
    env = dict(os.environ, QCE_HIDDEN_FILE=str(tmp_path / "does-not-exist.json"))
    env.pop("QCE_HIDDEN_SEED", None)
    env.pop("QCE_VERIFY_SEED", None)
    code = ("import json; from config import load_config; c = load_config(); "
            "print(json.dumps([c['hidden_source'], c['hidden_seed'], c['verify']['extra_widths']]))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env)
    assert p.returncode == 0, p.stderr
    source, seed, widths = json.loads(p.stdout)
    with open(os.path.join(ROOT, "hidden_values.example.json")) as f:
        ex = json.load(f)
    assert source == "example" and seed == ex["hidden_seed"] and widths == ex["verify_extra_widths"]
    assert "PUBLIC example hidden values" in p.stderr


def test_hidden_file_from_env_is_used(tmp_path, monkeypatch):
    vals = {"hidden_seed": 42, "verify_seed": 43, "verify_extra_widths": [{"n": 10, "p": 1021}]}
    path = tmp_path / "h.json"
    path.write_text(json.dumps(vals))
    monkeypatch.setenv("QCE_HIDDEN_FILE", str(path))
    monkeypatch.delenv("QCE_HIDDEN_SEED", raising=False)
    monkeypatch.delenv("QCE_VERIFY_SEED", raising=False)
    cfg = load_config()
    assert cfg["hidden_source"] == "env-file"
    assert cfg["hidden_seed"] == 42 and cfg["verify"]["seed"] == 43
    assert cfg["verify"]["extra_widths"] == [{"n": 10, "p": 1021}]


@pytest.mark.parametrize("bad", [
    {"hidden_seed": 1, "verify_seed": 2, "verify_extra_widths": [{"n": 10, "p": 1023}]},   # 1023 = 3*11*31
    {"hidden_seed": 1, "verify_seed": 2, "verify_extra_widths": [{"n": 10, "p": 1031}]},   # prime but >= 2^10
    {"hidden_seed": 1, "verify_extra_widths": []},                                         # missing key
])
def test_bad_hidden_values_rejected(bad):
    with pytest.raises(ValueError):
        check_hidden(bad, "test")


def test_hidden_file_is_git_ignored():
    p = subprocess.run(["git", "check-ignore", "-q", "hidden_values.json"], cwd=ROOT)
    assert p.returncode == 0
    assert os.path.basename(config.HIDDEN_FILE) == "hidden_values.json"
