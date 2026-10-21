"""Re-run the project's normal and verify tiers on a program with the current hidden values.

Used after rotating the hidden values (the originals were published in git history) to
show the winner was not tuned to the old hidden set. Writes results/reverify.json with
pass/fail and counts only; the new hidden widths, primes and seeds are never written.

    .venv/bin/python analysis/reverify.py results/best.py results/reverify.json
"""

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config  # noqa: E402
from evaluator import evaluate, verify, verify_targets  # noqa: E402


def main(argv):
    src_path, out_path = argv[1], argv[2]
    with open(src_path) as f:
        src = f.read()
    cfg = load_config()
    if cfg["hidden_source"] == "example":
        raise SystemExit("no hidden_values.json: refusing to report a re-verification against public example values")
    t0 = time.time()
    r = evaluate(src, cfg)
    w32 = next(w for w in r["widths"] if w["n"] == cfg["score_width"])
    v = verify(src, cfg, {"T": w32["T"], "Q": w32["Q"]})
    targets = verify_targets(cfg)
    public_n = {w["n"] for w in cfg["widths"]}
    out = {
        "program": os.path.basename(src_path),
        "program_sha256": hashlib.sha256(src.encode()).hexdigest(),
        "hidden_values": "rotated after the originals were published; stored only in the untracked hidden_values.json",
        "normal_tier": {"valid": r["valid"], "score": r["score"], "T": w32["T"], "Q": w32["Q"],
                        "hidden_inputs_at_n32": cfg["hidden_samples"]},
        "verify_tier": {
            "passed": v["passed"],
            "targets": len(targets),
            "hidden_extra_widths": sum(1 for n, p, _ in targets if n not in public_n),
            "inputs_checked": sum(len(xs) for _, _, xs in targets),
            "inputs_at_score_width": len(targets[0][2]),
            "failure_kind": None if v["passed"] else v["private"].get("kind"),
        },
        "seconds": round(time.time() - t0, 1),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0 if r["valid"] and v["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
