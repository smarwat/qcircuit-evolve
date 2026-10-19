# Seed: reversible Bernstein-Yang divsteps inversion, width-generic.
#
# Classical algorithm (f, g signed integers; u, v residues mod p):
#     delta, f, g, u, v = 1, p, a, 0, 1      invariant: f = u*a, g = v*a (mod p)
#     repeat ITERS times:
#         if delta > 0 and g odd:   delta, f, g = 1 - delta, g, (g - f)/2
#                                   u, v       = v, (v - u)/2 mod p
#         elif g odd:               delta, g = 1 + delta, (g + f)/2;  v = (v + u)/2 mod p
#         else:                     delta, g = 1 + delta, g/2;        v = v/2 mod p
#     end: g = 0, f = +-1, so a^-1 = +-u mod p.
# ITERS = floor((49n + 80)/17) is Bernstein-Yang's proven bound for n-bit inputs;
# once g = 0 the "g even" case repeats harmlessly, so no "done" flag.
#
# Reversible form, per iteration: store s = [delta > 0] & g0 and t = g0.
# The swap-and-subtract case is done without swaps or negations:
#     g -= f; f += g     (f becomes old g, g becomes g - f)
#     v -= u; u += v     (same on the coefficients, mod p)
# then g += f, v += u when t & !s; arithmetic-shift g (a relabel plus one
# CNOT for sign extension); v = v/2 mod p. delta is kept as e = delta - 1 so
# that [delta > 0] is just the sign bit of e: case 1 is e = ~e (CNOTs), else
# e += 1. Copy +-u to the output, then run the forward pass in reverse (Bennett).
#
# Cost ~ (46n + 40) Toffolis per iteration, ~2.9n iterations, x2 for uncompute.
# More iterations than binary Euclid, but no n-bit comparator per step.


def build_circuit(c, n, p):
    m = n + 2                                  # signed width of f, g
    iters = (49 * n + 80) // 17
    eb = (iters + 2).bit_length() + 1          # signed width of e = delta - 1

    F = c.alloc(m)
    G = c.alloc(m)
    U = c.alloc(n)
    V = c.alloc(n)
    E = c.alloc(eb)
    H = c.alloc(2 * iters)                     # history: s, t per iteration
    T = c.alloc(max(n + 1, eb))                # classical constants
    W = c.alloc(m)                             # W = operand & control
    c0, z, flag, o, vsp, gsp, cc = c.alloc(7)

    # ---- arithmetic primitives (same as the binary_euclid seed) -------------
    def maj(x, y, w):
        c.cx(w, y)
        c.cx(w, x)
        c.ccx(x, y, w)

    def uma(x, y, w):
        c.ccx(x, y, w)
        c.cx(w, x)
        c.cx(x, y)

    def add(a, b, carry=None):
        # b += a (mod 2^len), carry ^= carry-out. Cuccaro, 2m Toffolis.
        k = len(a)
        prev = c0
        for i in range(k):
            maj(prev, b[i], a[i])
            prev = a[i]
        if carry is not None:
            c.cx(a[k - 1], carry)
        for i in range(k - 1, -1, -1):
            uma(c0 if i == 0 else a[i - 1], b[i], a[i])

    def sub(a, b):
        m0 = c.mark()
        add(a, b)
        c.invert_since(m0)

    def compare(a, b, out):
        # out ^= [b > a]
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
        t = T[: len(b)]
        load(t, k, ctrl)
        add(t, b)
        load(t, k, ctrl)

    def mod_add(a, b):
        # b = (b + a) mod p
        add(a, b, z)
        bz = b + [z]
        add_const((1 << (n + 1)) - p, bz)
        c.cx(z, flag)
        add_const(p, bz, flag)
        compare(b, a, flag)
        c.x(flag)

    def mod_sub(a, b):
        m0 = c.mark()
        mod_add(a, b)
        c.invert_since(m0)

    def mod_half(x, spare):
        c.cx(x[0], o)
        add_const(p, x + [spare], o)
        newx = x[1:] + [spare]
        k = (p - 1) // 2
        load(T[:n], k)
        compare(T[:n], newx, o)
        load(T[:n], k)
        return newx, x[0]

    def and_into(ctrl, src, dst):
        for i in range(len(src)):
            c.ccx(ctrl, src[i], dst[i])

    Wn = W[:n]

    # ---- forward pass -------------------------------------------------------
    start = c.mark()
    load(F, p)
    for i in range(n):
        c.cx(c.inp[i], G[i])
    c.x(V[0])

    for it in range(iters):
        s = H[2 * it]
        t = H[2 * it + 1]
        c.cx(G[0], t)                          # t = g odd
        c.x(E[-1])
        c.ccx(E[-1], G[0], s)                  # s = (e >= 0) & g odd
        c.x(E[-1])
        c.cx(t, cc)
        c.cx(s, cc)                            # cc = t & !s  (s implies t)

        # integers: case s: g -= f; f += g.  case cc: g += f.
        and_into(s, F, W)
        sub(W, G)
        and_into(s, F, W)
        and_into(s, G, W)
        add(W, F)
        and_into(s, G, W)
        and_into(cc, F, W)
        add(W, G)
        and_into(cc, F, W)
        c.cx(G[-1], gsp)                       # g is even: arithmetic shift right
        G, gsp = G[1:] + [gsp], G[0]

        # coefficients mod p: case s: v -= u; u += v.  case cc: v += u.  v /= 2.
        and_into(s, U, Wn)
        mod_sub(Wn, V)
        and_into(s, U, Wn)
        and_into(s, V, Wn)
        mod_add(Wn, U)
        and_into(s, V, Wn)
        and_into(cc, U, Wn)
        mod_add(Wn, V)
        and_into(cc, U, Wn)
        V, vsp = mod_half(V, vsp)

        # e = delta - 1:  case s: e = ~e (delta = 1 - delta).  else e += 1.
        for q in E:
            c.cx(s, q)
        c.x(s)
        add_const(1, E, s)
        c.x(s)

        c.cx(t, cc)
        c.cx(s, cc)                            # cc = 0
    end = c.mark()

    # ---- output +-u, then uncompute -----------------------------------------
    for i in range(n):
        c.cx(U[i], c.out[i])
    sign = F[-1]                               # f = -1: a^-1 = p - u
    for i in range(n):
        c.cx(sign, c.out[i])                   # out = 2^n - 1 - u
    add_const((p + 1) % (1 << n), c.out, sign) # out = p - u
    c.append_inverse(start, end)
