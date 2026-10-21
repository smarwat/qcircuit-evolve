"""Analyse an orchestrator run directory; write plots and analysis.json into results/.

    .venv-verify/bin/python analysis/run_analysis.py runs/smoke2 results/

Reads only the run logs (attempts.jsonl, archive.json, replies/, candidates/) and
results/resources.json. Produces:
  best_vs_call.png     verified best T*Q at n=32 after each model call
  pareto_T_vs_Q.png    every valid program at n=32, Pareto front highlighted
  loglog_T_vs_n.png    Toffoli count vs n for the seed and the best (from resources.json)
  analysis.json        per-call table, failure categories, timing/cost stats, lineage
"""

import difflib
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_attempts(run):
    with open(os.path.join(run, "attempts.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def category(r):
    fb = r.get("feedback") or ""
    if r.get("usage_limit"):
        return "usage limit (not a model reply)"
    if fb.startswith("API error"):
        return "CLI/API error"
    if r.get("valid"):
        return "valid"
    if fb == "failed verification":
        return "failed hidden verification"
    if fb.startswith("edit rejected"):
        return "diff did not apply" if r.get("edit_format") == "diff" else "no usable code in reply"
    if fb.startswith("duplicate"):
        return "duplicate of an archived program"
    if fb.startswith("reply truncated"):
        return "reply truncated (max tokens)"
    if "syntax error" in fb:
        return "syntax error in the program"
    kind = r.get("failure_kind") or "other"
    return {"wrong": "wrong answer (public test)", "hidden": "wrong answer (hidden test)",
            "invalid": "illegal gate", "timeout": "evaluation timeout", "exception": "generator raised",
            "rejected": "source rejected by sandbox", "gate_cap": "gate cap exceeded",
            "crash": "generator crashed"}.get(kind, kind)


def lineage(archive, best_id):
    by_id = {e["id"]: e for e in archive}
    chain, cur = [], best_id
    while cur is not None:
        chain.append(by_id[cur])
        cur = by_id[cur].get("parent")
    return chain[::-1]


def reply_summary(run, eid, limit=900):
    path = os.path.join(run, "replies", f"{eid}.md")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    for marker in ("```python", "<<<<<<< SEARCH"):
        i = text.find(marker)
        if i >= 0:
            text = text[:i]
    return text.strip()[:limit]


def diff_stats(a, b):
    sm = difflib.SequenceMatcher(None, a.splitlines(), b.splitlines())
    added = removed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return {"lines_added": added, "lines_removed": removed, "similarity": round(sm.ratio(), 3)}


def main(argv):
    run, out = argv[1], argv[2]
    attempts = load_attempts(run)
    with open(os.path.join(run, "archive.json")) as f:
        archive = json.load(f)
    seeds = [r for r in attempts if r["gen"] == 0]
    calls = [r for r in attempts if r["gen"] > 0 and not r.get("usage_limit")]

    # Best T*Q after each call (only verified programs can become best).
    seed_best = min(r["score"] for r in seeds if r.get("valid"))
    best, series = seed_best, [{"call": 0, "id": "seeds", "best_tq": seed_best}]
    for i, r in enumerate(calls, 1):
        if r.get("valid") and r.get("verify") == "passed verification" and r["score"] < best:
            best = r["score"]
        series.append({"call": i, "id": r["id"], "mode": r.get("mode"), "category": category(r),
                       "score": r.get("score"), "T": r.get("T"), "Q": r.get("Q"), "best_tq": best,
                       "api_s": r.get("api_s"), "notional_usd": r.get("notional_cost_usd"),
                       "output_tokens": (r.get("tokens") or {}).get("output"),
                       "model": r.get("model"), "edit_format": r.get("edit_format")})

    cats = {}
    for r in calls:
        cats[category(r)] = cats.get(category(r), 0) + 1
    by_mode = {}
    for r in calls:
        m = by_mode.setdefault(r.get("mode"), {"calls": 0, "valid": 0, "new_best": 0})
        m["calls"] += 1
        m["valid"] += bool(r.get("valid"))
    for s in series[1:]:
        prev = series[s["call"] - 1]["best_tq"]
        if s["best_tq"] < prev:
            by_mode[s["mode"]]["new_best"] += 1
    answered = [r for r in calls if r.get("api_s") is not None]

    def stats(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        return {"n": len(xs), "mean": statistics.mean(xs), "median": statistics.median(xs),
                "min": min(xs), "max": max(xs), "total": sum(xs)}

    final_best = min((e for e in archive if e.get("verified") is True), key=lambda e: (e["score"], e["gen"]))
    chain = lineage(archive, final_best["id"])
    steps = []
    for prev, e in zip([None] + chain[:-1], chain):
        rec = next((r for r in attempts if r["id"] == e["id"]), {})
        steps.append({"id": e["id"], "parent": e.get("parent"), "mode": e.get("mode"), "T": e["T"], "Q": e["Q"],
                      "score": e["score"], "verified": e.get("verified"), "edit_format": rec.get("edit_format"),
                      "api_s": rec.get("api_s"), "output_tokens": (rec.get("tokens") or {}).get("output"),
                      "model_summary": reply_summary(run, e["id"]),
                      "diff_vs_parent": diff_stats(prev["source"], e["source"]) if prev else None})

    front = sorted({(e["T"], e["Q"]) for e in archive
                    if not any(o["T"] <= e["T"] and o["Q"] <= e["Q"] and (o["T"], o["Q"]) != (e["T"], e["Q"])
                               for o in archive)})
    analysis = {
        "run_dir": run, "calls": len(calls), "attempt_records": len(attempts),
        "models_answering": sorted({r.get("model") for r in calls if r.get("model")}),
        "model_mismatches": sum(1 for r in calls if r.get("model_mismatch")),
        "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "valid_rate": sum(1 for r in calls if r.get("valid")) / len(calls) if calls else None,
        "by_mode": by_mode,
        "edit_formats": {k: sum(1 for r in calls if r.get("edit_format") == k) for k in ("diff", "full", None)},
        "seconds_per_call": stats([r.get("api_s") for r in answered]),
        "notional_usd_per_call": stats([r.get("notional_cost_usd") for r in answered]),
        "output_tokens_per_call": stats([(r.get("tokens") or {}).get("output") for r in answered]),
        "context_tokens_per_call": stats([r.get("context_tokens") for r in answered]),
        "seed_best_tq": seed_best, "final_best": {k: final_best[k] for k in ("id", "T", "Q", "score", "gen")},
        "improvement_factor": seed_best / final_best["score"],
        "series": series, "lineage": steps,
        "pareto_front": [{"T": t, "Q": q, "TQ": t * q} for t, q in front],
        "valid_programs": [{"id": e["id"], "T": e["T"], "Q": e["Q"], "TQ": e["score"]} for e in archive],
    }
    with open(os.path.join(out, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=1)

    # --- plots ---------------------------------------------------------------
    xs = [s["call"] for s in series]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(xs, [s["best_tq"] for s in series], where="post", color="#2a6fdb", lw=2, label="verified best")
    v = [s for s in series[1:] if s.get("score")]
    ax.scatter([s["call"] for s in v], [s["score"] for s in v], s=18, color="#999", zorder=3,
               label="valid candidate")
    ax.set_yscale("log")
    ax.set_xlabel("model call")
    ax.set_ylabel("T·Q at n=32 (log)")
    ax.set_title("Best T·Q vs. model call")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "best_vs_call.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter([e["T"] for e in archive], [e["Q"] for e in archive], s=22, color="#bbb", label="valid program")
    ax.plot([t for t, _ in front], [q for _, q in front], "o-", color="#d9480f", label="Pareto front")
    ax.annotate(final_best["id"], (final_best["T"], final_best["Q"]), textcoords="offset points", xytext=(6, 6),
                fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Toffolis at n=32 (log)")
    ax.set_ylabel("qubits at n=32")
    ax.set_title("Valid programs: T vs Q at n=32")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pareto_T_vs_Q.png"), dpi=150)

    res_path = os.path.join(out, "resources.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            res = json.load(f)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for r, color in zip(res, ("#666", "#2a6fdb", "#d9480f")):
            ns = [x["n"] for x in r["rows"]]
            ax.loglog(ns, [x["toffolis"] for x in r["rows"]], "o-", ms=3, color=color,
                      label=f"{r['name']} (slope {r['fit_T']['loglog_slope_n16_32']:.2f}, n=16..32)")
        ax.loglog([4, 32], [4 ** 2 * 60, 32 ** 2 * 60], "--", color="#ccc", label="∝ n² (reference)")
        ax.set_xlabel("n (bits)")
        ax.set_ylabel("Toffoli count")
        ax.set_title("Toffoli count vs n (log-log)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "loglog_T_vs_n.png"), dpi=150)
    print(f"calls={len(calls)} best={final_best['id']} {final_best['score']:,} categories={cats}")


if __name__ == "__main__":
    main(sys.argv)
