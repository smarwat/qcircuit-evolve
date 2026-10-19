# Seed: reversible binary extended Euclid (Kaliski family), width-generic.
#
# Classical algorithm (u, v integers; x1, x2 residues mod p):
#     u, v, x1, x2 = a, p, 1, 0          invariant: u = x1*a, v = x2*a (mod p)
#     repeat 2n times:
#         A  u even            : u = u/2,       x1 = x1/2 mod p
#         B  u odd, v even     : v = v/2,       x2 = x2/2 mod p
#         C  both odd, u > v   : u = (u-v)/2,   x1 = (x1-x2)/2 mod p
#         D  both odd, u <= v  : v = (v-u)/2,   x2 = (x2-x1)/2 mod p
#     end: u = 1, v = 0, so x1 = a^-1 mod p.
# Every step at least halves u*v < 2^2n, so v reaches 0 within 2n steps;
# after that case B (v = 0 is even) repeats harmlessly, so no "done" flag.
# Halving x mod p instead of Kaliski's doubling means no Montgomery factor
# 2^k to correct at the end.
#
# Reversible form, per iteration: store two history bits
#     h1 = u0 & v0                 (subtract?)
#     h0 = !u0 ^ (h1 & [u > v])    (operate on u/x1? else v/x2)
# swap (u,x1) <-> (v,x2) when h0 = 0, so the operated register is always "u";
# u -= v & h1; shift u right (a free relabel: its low bit is 0);
# x1 = (x1 - (x2 & h1)) mod p; x1 = x1/2 mod p; swap back.
# Then copy x1 to the output and run the whole forward pass in reverse
# (Bennett), which clears u, v, x1, x2 and the 4n history bits.
#
# Cost: (26n + 8) Toffolis per iteration, 2n iterations, x2 for uncompute
#       ~ 104 n^2 Toffolis; 10n + 7 ancillas.


def build_circuit(c, n, p):
    # ---- registers (little-endian lists of qubit indices) -------------------
    U = c.alloc(n)
    V = c.alloc(n)
    X1 = c.alloc(n)
    X2 = c.alloc(n)
    H = c.alloc(4 * n)        # 2 history bits per iteration
    T = c.alloc(n + 1)        # holds classical constants temporarily
    W = c.alloc(n)            # W = (operand & h1)
    c0, z, flag, gt, o, xh = c.alloc(6)

    # ---- arithmetic primitives ----------------------------------------------
    def maj(x, y, w):
        c.cx(w, y)
        c.cx(w, x)
        c.ccx(x, y, w)

    def uma(x, y, w):
        c.ccx(x, y, w)
        c.cx(w, x)
        c.cx(x, y)

    def add(a, b, carry=None):
        # Cuccaro ripple adder: b += a (mod 2^len), carry ^= carry-out. 2m Toffolis.
        m = len(a)
        prev = c0
        for i in range(m):
            maj(prev, b[i], a[i])
            prev = a[i]
        if carry is not None:
            c.cx(a[m - 1], carry)
        for i in range(m - 1, -1, -1):
            uma(c0 if i == 0 else a[i - 1], b[i], a[i])

    def sub(a, b):
        # b -= a (mod 2^len)
        m0 = c.mark()
        add(a, b)
        c.invert_since(m0)

    def compare(a, b, out):
        # out ^= [b > a]: carry-out of (~a + b), then uncompute. 2m Toffolis.
        for q in a:
            c.x(q)
        m0 = c.mark()
        prev = c0
        for i in range(len(a)):
            maj(prev, b[i], a[i])
            prev = a[i]
        m1 = c.mark()
        c.cx(a[-1], out)
        c.append_inverse(m0, m1)
        for q in a:
            c.x(q)

    def load(reg, k, ctrl=None):
        for i in range(len(reg)):
            if (k >> i) & 1:
                if ctrl is None:
                    c.x(reg[i])
                else:
                    c.cx(ctrl, reg[i])

    def add_const(k, b, ctrl=None):
        # b += k (mod 2^len(b)), optionally controlled. Loading k is Toffoli-free.
        t = T[: len(b)]
        load(t, k, ctrl)
        add(t, b)
        load(t, k, ctrl)

    def mod_add(a, b):
        # b = (b + a) mod p for a, b in [0, p)
        add(a, b, z)                         # (b,z) = a + b
        bz = b + [z]
        add_const((1 << (n + 1)) - p, bz)    # (b,z) = a + b - p; z = sign
        c.cx(z, flag)
        add_const(p, bz, flag)               # add p back if negative; z -> 0
        compare(b, a, flag)                  # flag ^= [a > b]
        c.x(flag)                            # flag was [b_new >= a]; now 0

    def mod_sub(a, b):
        # b = (b - a) mod p
        m0 = c.mark()
        mod_add(a, b)
        c.invert_since(m0)

    def mod_half(x, spare):
        # x = x / 2 mod p. Returns the relabelled register and the new spare.
        c.cx(x[0], o)                        # o = x odd
        add_const(p, x + [spare], o)         # x + o*p is even, < 2^(n+1)
        newx = x[1:] + [spare]               # shift right by relabelling
        k = (p - 1) // 2
        load(T[:n], k)
        compare(T[:n], newx, o)              # o ^= [x/2 > (p-1)/2], which equals o
        load(T[:n], k)
        return newx, x[0]

    def cswap(s, A, B):
        for i in range(len(A)):
            c.cx(B[i], A[i])
            c.ccx(s, A[i], B[i])
            c.cx(B[i], A[i])

    # ---- forward pass -------------------------------------------------------
    start = c.mark()
    for i in range(n):
        c.cx(c.inp[i], U[i])
    load(V, p)
    c.x(X1[0])

    for it in range(2 * n):
        h0 = H[2 * it]
        h1 = H[2 * it + 1]
        c.ccx(U[0], V[0], h1)
        c.x(h0)
        c.cx(U[0], h0)
        compare(V, U, gt)                    # gt = [u > v]
        c.ccx(h1, gt, h0)
        compare(V, U, gt)                    # gt = 0
        c.x(h0)                              # h0 now 1 iff we must swap
        cswap(h0, U, V)
        cswap(h0, X1, X2)

        for i in range(n):
            c.ccx(h1, V[i], W[i])
        sub(W, U)
        for i in range(n):
            c.ccx(h1, V[i], W[i])
        U = U[1:] + U[:1]                    # u is even: shift right for free

        for i in range(n):
            c.ccx(h1, X2[i], W[i])
        mod_sub(W, X1)
        for i in range(n):
            c.ccx(h1, X2[i], W[i])
        X1, xh = mod_half(X1, xh)

        cswap(h0, X1, X2)
        cswap(h0, U, V)
        c.x(h0)                              # restore stored history bit
    end = c.mark()

    # ---- copy result, then uncompute everything -----------------------------
    for i in range(n):
        c.cx(X1[i], c.out[i])
    c.append_inverse(start, end)
