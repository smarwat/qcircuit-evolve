"""Program archive: every valid program, the Pareto front over (T, Q) at the score
width, and the verified best (lowest T*Q). Persists to archive.json / pareto.json."""

import hashlib
import json
import os


def source_hash(src):
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def dominates(a, b):
    return a["T"] <= b["T"] and a["Q"] <= b["Q"] and (a["T"] < b["T"] or a["Q"] < b["Q"])


class Archive:
    """Entries are dicts with at least: id, gen, parent, mode, source, score, T, Q,
    feedback, verified (True / None = not verified yet)."""

    def __init__(self, entries=None):
        self.entries = list(entries or [])
        self.hashes = {source_hash(e["source"]): e["id"] for e in self.entries}

    # -- membership ----------------------------------------------------------
    def add(self, entry):
        self.entries.append(entry)
        self.hashes[source_hash(entry["source"])] = entry["id"]

    def duplicate_of(self, src):
        return self.hashes.get(source_hash(src))

    def get(self, eid):
        return next(e for e in self.entries if e["id"] == eid)

    # -- views -----------------------------------------------------------------
    def front(self):
        es = self.entries
        f = [e for e in es if not any(dominates(o, e) for o in es)]
        # Keep one representative per (T, Q) point, preferring verified, then oldest.
        seen, out = set(), []
        for e in sorted(f, key=lambda e: (e["T"], e["Q"], e.get("verified") is not True, e["gen"])):
            if (e["T"], e["Q"]) not in seen:
                seen.add((e["T"], e["Q"]))
                out.append(e)
        return out

    def best(self):
        """Lowest score among verified entries."""
        v = [e for e in self.entries if e.get("verified") is True]
        return min(v, key=lambda e: (e["score"], e["gen"])) if v else None

    def would_be_new_best(self, score):
        b = self.best()
        return b is None or score < b["score"]

    # -- selection -------------------------------------------------------------
    def pick_parent(self, rng, front_prob):
        front = self.front()
        if front and rng.random() < front_prob:
            return rng.choice(front)
        return rng.choice(self.entries)

    def inspirations(self, parent, k, rng):
        pool = [e for e in self.front() if e["id"] != parent["id"]]
        if len(pool) < k:  # top up with the best other programs
            extra = sorted((e for e in self.entries if e["id"] != parent["id"] and e not in pool),
                           key=lambda e: e["score"])
            pool += extra[: k - len(pool)]
        rng.shuffle(pool)
        return pool[:k]

    # -- persistence -------------------------------------------------------------
    def save(self, out_dir):
        _atomic_json(os.path.join(out_dir, "archive.json"), self.entries)
        front = [{k: e.get(k) for k in ("id", "gen", "parent", "mode", "T", "Q", "score", "verified")}
                 for e in self.front()]
        best = self.best()
        _atomic_json(os.path.join(out_dir, "pareto.json"),
                     {"front": front, "best": best and best["id"], "best_score": best and best["score"],
                      "valid_programs": len(self.entries)})
        if best:
            with open(os.path.join(out_dir, "best.py"), "w") as f:
                f.write(best["source"])

    @classmethod
    def load(cls, out_dir):
        with open(os.path.join(out_dir, "archive.json")) as f:
            return cls(json.load(f))


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
