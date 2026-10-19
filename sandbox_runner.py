"""Child-process entry point: run one candidate at one width.

Reads a JSON job from stdin: {source, n, p, num_ancillas, gate_cap, mem_limit}.
Writes to stdout one JSON header line, then (on success) the raw bytes of the
gate array. Anything the candidate prints is discarded. The parent treats every
byte coming back as untrusted.
"""

import json
import linecache
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit import Circuit, GateCapExceeded  # noqa: E402
from sandbox import SourceRejected, check_source, safe_builtins  # noqa: E402


def _limit_memory(nbytes):
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
    except (ImportError, ValueError, OSError):
        pass  # not supported on this OS (e.g. macOS)


def _candidate_traceback(exc, limit=6):
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<candidate>"]
    lines = [f"  line {f.lineno}, in {f.name}: {(f.line or '').strip()}" for f in frames[-limit:]]
    msg = f"{type(exc).__name__}: {exc}"
    return "\n".join(lines + [msg]) if lines else msg


def main():
    job = json.loads(sys.stdin.buffer.read())
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # nothing the candidate does can write to our result pipe

    def reply(header, payload=b""):
        out.write(json.dumps(header).encode() + b"\n")
        out.write(payload)
        out.flush()

    if job.get("mem_limit"):
        _limit_memory(int(job["mem_limit"]))

    src = job["source"]
    try:
        check_source(src)
    except SourceRejected as e:
        reply({"ok": False, "kind": "rejected", "error": str(e)})
        return

    linecache.cache["<candidate>"] = (len(src), None, src.splitlines(True), "<candidate>")
    n, p = job["n"], job["p"]
    c = Circuit(n, p, job["num_ancillas"], job["gate_cap"])
    gates = c.gates  # the stored array is the circuit, whatever c.gates is rebound to
    try:
        ns = {"__builtins__": safe_builtins(), "__name__": "candidate"}
        exec(compile(src, "<candidate>", "exec"), ns)
        fn = ns.get("build_circuit")
        if not callable(fn):
            raise ValueError("build_circuit is not a function")
        r = fn(c, n, p)
        if r is not None and hasattr(r, "__next__"):
            for _ in r:
                pass
    except GateCapExceeded as e:
        reply({"ok": False, "kind": "gate_cap", "error": str(e)})
        return
    except BaseException as e:  # noqa: BLE001 - report anything the candidate raised
        reply({"ok": False, "kind": "exception", "error": _candidate_traceback(e)})
        return

    reply({"ok": True, "itemsize": gates.itemsize, "length": len(gates)}, gates.tobytes())


if __name__ == "__main__":
    main()
