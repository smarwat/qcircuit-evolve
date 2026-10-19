"""The circuit builder handed to candidates inside the sandbox.

Gates are stored flat in `array('i')`, three ints per gate: (target, c1, c2).
  X:       (t, -1, -1)
  CNOT:    (t,  c, -1)
  Toffoli: (t, c1, c2)

Everything here is a convenience for the candidate. The evaluator never trusts
this object: it takes the raw gate array, re-validates every gate, recounts T
and Q, and simulates with its own code.
"""

from array import array


class GateCapExceeded(Exception):
    pass


class Circuit:
    def __init__(self, n, p, num_ancillas, gate_cap):
        self.n = n
        self.p = p
        self.num_ancillas = num_ancillas
        self.num_qubits = 2 * n + num_ancillas
        self.inp = list(range(n))  # input a, little-endian; must be unchanged
        self.out = list(range(n, 2 * n))  # starts 0, must end as a^-1 mod p
        self.anc = list(range(2 * n, 2 * n + num_ancillas))  # start 0, end 0
        self.gates = array("i")
        self.toffoli_count = 0
        self.gate_cap = gate_cap
        self._cap3 = 3 * gate_cap
        self._free = self.anc[::-1]

    # -- gates ---------------------------------------------------------------
    def _bad(self, kind, qs):
        N = self.num_qubits
        for q in qs:
            if not (isinstance(q, int) and 0 <= q < N):
                raise ValueError(f"{kind}: qubit {q!r} out of range [0, {N})")
        raise ValueError(f"{kind}: qubits must be distinct, got {qs}")

    def _room(self, k=3):
        if len(self.gates) + k > self._cap3:
            raise GateCapExceeded(f"gate cap of {self.gate_cap:,} gates exceeded")

    def x(self, t):
        if not (0 <= t < self.num_qubits):
            self._bad("x", (t,))
        self._room()
        self.gates.extend((t, -1, -1))

    def cx(self, c, t):
        N = self.num_qubits
        if not (0 <= c < N and 0 <= t < N) or c == t:
            self._bad("cx", (c, t))
        self._room()
        self.gates.extend((t, c, -1))

    def ccx(self, c1, c2, t):
        N = self.num_qubits
        if not (0 <= c1 < N and 0 <= c2 < N and 0 <= t < N) or c1 == c2 or c1 == t or c2 == t:
            self._bad("ccx", (c1, c2, t))
        self._room()
        self.gates.extend((t, c1, c2))
        self.toffoli_count += 1

    cnot = cx
    toffoli = ccx

    # -- bookkeeping helpers -------------------------------------------------
    def num_gates(self):
        return len(self.gates) // 3

    def mark(self):
        """Current gate index, for use with append_inverse / invert_since."""
        return len(self.gates) // 3

    def append_inverse(self, start, end=None):
        """Append the inverse of gates[start:end] (all gates are self-inverse,
        so this is the same gates in reverse order). Bennett-style uncompute."""
        end = self.mark() if end is None else end
        seg = self.gates[3 * start : 3 * end]
        self._room(len(seg))
        rev = array("i", bytes(len(seg) * seg.itemsize))
        rev[0::3] = seg[0::3][::-1]
        rev[1::3] = seg[1::3][::-1]
        rev[2::3] = seg[2::3][::-1]
        self.gates.extend(rev)
        self.toffoli_count += sum(1 for c2 in seg[2::3] if c2 >= 0)

    def invert_since(self, start):
        """Replace gates[start:] by their inverse in place. Emit a circuit for
        f, then call this to get f^-1 (e.g. build an adder, get a subtractor)."""
        seg = self.gates[3 * start :]
        del self.gates[3 * start :]
        rev = array("i", bytes(len(seg) * seg.itemsize))
        rev[0::3] = seg[0::3][::-1]
        rev[1::3] = seg[1::3][::-1]
        rev[2::3] = seg[2::3][::-1]
        self.gates.extend(rev)

    # -- ancilla allocation (optional) ---------------------------------------
    def alloc(self, k=1):
        """Return k ancilla indices that are not currently allocated. The
        caller is responsible for returning them to 0 before release()."""
        if k > len(self._free):
            raise ValueError(f"alloc({k}): only {len(self._free)} ancillas left of {self.num_ancillas}")
        out = [self._free.pop() for _ in range(k)]
        return out

    def release(self, qubits):
        self._free.extend(sorted(qubits, reverse=True))
