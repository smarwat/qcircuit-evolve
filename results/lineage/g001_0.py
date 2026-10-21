def build_circuit(c, n, p):
    iters = (p * (p - 1)).bit_length()        # <= 2n; u*v halves each step
    V = list(c.inp)                           # v starts as a (restored by Bennett)
    U = c.alloc(n)
    XU = c.alloc(n)
    XV = c.alloc(n)
    H = c.alloc(2 * iters)
    T = c.alloc(n)
    W = c.alloc(n)
    c0, z, o, xh = c.alloc(4)

    def maj(x, y, w):
        c.cx(w, y)
        c.cx(w, x)
        c.ccx(x, y, w)

    def uma(x, y, w):
        c.ccx(x, y, w)
        c.cx(w, x)
        c.cx(x, y)

    def add(a, b, carry=None):
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
        m0 = c.mark()
        add(a, b)
        c.invert_since(m0)

    def compare(a, b, out, ctrl=None):
        # out ^= [b > a] (& ctrl)
        for q in a:
            c.x(q)
        m0 = c.mark()
        prev = c0
        for i in range(len(a)):
            maj(prev, b[i], a[i])
            prev = a[i]
        m1 = c.mark()
        if ctrl is None:
            c.cx(a[-1], out)
        else:
            c.ccx(ctrl, a[-1], out)
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

    def add_const(k, b, ctrl=None, carry=None):
        load(T, k, ctrl)
        add(T, b, carry)
        load(T, k, ctrl)

    def mod_add(a, b):
        # b = (b + a) mod p, z used as sign/flag and returned to 0
        add(a, b, z)                              # (b,z) = a + b
        add_const((1 << n) - p, b, None, z)       # (b,z) += 2^(n+1) - p ...
        c.x(z)                                    # ... top-bit part
        add_const(p, b, z)                        # add p back to low bits if negative
        compare(b, a, z)                          # z ^= [a > b_new]
        c.x(z)

    def mod_sub(a, b):
        m0 = c.mark()
        mod_add(a, b)
        c.invert_since(m0)

    def mod_half(x, spare):
        c.cx(x[0], o)
        add_const(p, x, o, spare)                 # (x,spare) = x + o*p
        newx = x[1:] + [spare]
        k = (p - 1) // 2
        load(T, k)
        compare(T, newx, o)
        load(T, k)
        return newx, x[0]

    def cswap(s, A, B):
        for i in range(len(A)):
            c.cx(B[i], A[i])
            c.ccx(s, A[i], B[i])
            c.cx(B[i], A[i])

    start = c.mark()
    load(U, p)
    c.x(XV[0])

    for it in range(iters):
        h1 = H[2 * it]
        s = H[2 * it + 1]
        c.cx(V[0], h1)                            # h1 = v odd
        compare(V, U, s, h1)                      # s = h1 & [u > v]
        cswap(s, U, V)
        cswap(s, XU, XV)

        for i in range(n):
            c.ccx(h1, U[i], W[i])                 # W = h1 & u
        sub(W, V)
        V = V[1:] + V[:1]                         # v even: shift right

        for i in range(n):                        # W = h1 & xu
            c.cx(XU[i], U[i])
            c.ccx(h1, U[i], W[i])
            c.cx(XU[i], U[i])
        mod_sub(W, XV)
        for i in range(n):
            c.ccx(h1, XU[i], W[i])                # W = 0
        XV, xh = mod_half(XV, xh)
    end = c.mark()

    for i in range(n):
        c.cx(XU[i], c.out[i])
    c.append_inverse(start, end)
