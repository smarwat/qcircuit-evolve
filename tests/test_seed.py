from config import ancilla_budget, load_config


def test_seed_valid_at_every_width(seed_result):
    r = seed_result
    assert r["valid"], r["feedback"]
    cfg = load_config()
    assert [w["n"] for w in r["widths"]] == [w["n"] for w in cfg["widths"]]
    assert all(w["passed"] for w in r["widths"])


def test_seed_counts_match_design(seed_result):
    # (26n + 8) Toffolis per iteration, 2n iterations, doubled by the Bennett uncompute.
    # 10n + 7 ancillas plus the 2n register qubits.
    cfg = load_config()
    for w in seed_result["widths"]:
        n = w["n"]
        assert w["T"] == 2 * (2 * n) * (26 * n + 8)
        assert w["Q"] == 12 * n + 7
        assert w["Q"] - 2 * n <= ancilla_budget(cfg, n)
    assert seed_result["score"] == 107_520 * 391
    assert 1.9 < seed_result["scaling_exponent"] < 2.1


def test_feedback_reports_every_width(seed_result):
    fb = seed_result["feedback"]
    for n in (4, 8, 16, 32):
        assert f"n={n} " in fb
    assert "log2(T32/T16)" in fb
    assert "PRIMARY SCORE" in fb and "42,040,320" in fb
