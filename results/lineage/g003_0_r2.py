def build_circuit(c, n, p):
    iters = (p * (p - 1)).bit_length()
    L = max(iters, n) + 1
    U = list(c.inp)
    V = c.alloc(n)
    CU = c.alloc(n)
    CV = c.alloc(L)
    H = c.alloc(iters)
    S = c.alloc(2)

    # ---- Montgomery plan (classical) ----
    B = ((2 * p - 2) << iters) // (p + 1)      # proven bound on final CV value
    widths = []
    needE = L
    needT = 1
    Bt = B
    for j in range(iters):
        Bt = (Bt + p) // 2
        w = Bt.bit_length()
        widths.append(w)
        needE = max(needE, j + 1 + w)
        needT = max(needT, w)
    wY = Bt.bit_length()
    needE = max(needE, iters + wY)
    needT = max(needT, wY)
    extE = needE - L - n
    extT = needT - 2 * n
    EX = c.alloc(extE) if extE > 0 else []
    TX = c.alloc(extT) if extT > 0 else []

    def maj(x, y, w):
        c.cx(w, y)
        c.cx(w, x)
        c.ccx(x, y, w)

    def uma(x, y, w):
        c.ccx(x, y, w)
        c.cx(w, x)
        c.cx(x, y)

    def add(a, b, cin):
        m = len(a)
        prev = cin
        for i in range(m):
            maj(prev, b[i], a[i])
            prev = a[i]
        for i in range(m - 1, -1, -1):
            uma(cin if i == 0 else a[i - 1], b[i], a[i])

    def cadd(ctl, a, b, cin):
        # b += ctl * a (mod 2^m), 3m Toffolis
        m = len(a)
        prev = cin
        for i in range(m):
            maj(prev, b[i], a[i])
            prev = a[i]
        for i in range(m - 1, -1, -1):
            x = cin if i == 0 else a[i - 1]
            c.ccx(x, b[i], a[i])
            c.ccx(ctl, x, b[i])
            c.cx(a[i], x)
            c.cx(a[i], b[i])

    def csub(ctl, a, b, cin):
        m0 = c.mark()
        cadd(ctl, a, b, cin)
        c.invert_since(m0)

    def compare(a, b, out, cin):
        # out ^= [b > a]
        for q in a:
            c.x(q)
        m0 = c.mark()
        prev = cin
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

    def cswap(s, A, B2):
        for i in range(len(A)):
            c.cx(B2[i], A[i])
            c.ccx(s, A[i], B2[i])
            c.cx(B2[i], A[i])

    start = c.mark()
    load(V, p)
    c.x(CV[0])
    for it in range(iters):
        h1 = H[it]
        sp = S[it % 2]
        sc = S[(it + 1) % 2]
        cz = CV[-1]                      # zero during the iteration
        if it == 0:
            c.cx(U[0], h1)               # v = p odd
            c.cx(U[0], sc)
            c.x(sc)                      # s = u even ([a > p] = 0)
        else:
            c.ccx(U[0], V[0], h1)        # h1 = both odd (swap-symmetric)
            c.cx(U[0], sc)
            c.cx(V[0], U[0])
            c.ccx(sp, U[0], sc)          # sc = logical u0
            c.cx(V[0], U[0])
            c.x(sc)
            for q in V:
                c.x(q)
            m0 = c.mark()
            prev = sp                    # carry-in sp: gives [U >= V] if swapped
            for i in range(n):
                maj(prev, U[i], V[i])
                prev = V[i]
            m1 = c.mark()
            c.ccx(h1, V[n - 1], sc)
            c.append_inverse(m0, m1)
            for q in V:
                c.x(q)
            c.ccx(h1, sp, sc)            # logical [u > v] = carry ^ sp
            c.cx(sp, sc)
        wc = min(n, it + 2)              # coefficients <= 2^(it+1) here
        cswap(sc, U, V)                  # merged unswap(prev) + swap(cur)
        if it > 0:
            c.cx(sp, sc)
            c.cx(CV[0], sp)              # previous s = parity of CV
        cswap(sc, CU[:wc], CV[:wc])
        csub(h1, U, V, cz)
        V = V[1:] + V[:1]
        cadd(h1, CV[:wc], CU[:wc], cz)
        CV = CV[-1:] + CV[:-1]           # double partner by relabel
        cswap(sc, CU[:wc], CV[:wc])

    # u = 1, v = 0, CU = p: clear to use as scratch
    c.x(U[0])
    load(CU, p)
    E = CV + CU + EX
    TS = V + U + TX
    cin = S[0]
    f = S[1]
    kk = (p + 1) // 2
    for j in range(iters):
        w = widths[j]
        b = E[j]
        R = E[j + 1:j + 1 + w]
        T = TS[:w]
        load(T, kk, b)
        add(T, R, cin)
        load(T, kk, b)
    Y = E[iters:iters + wY]
    if Bt > p:
        T = TS[:wY]
        load(T, p)
        compare(T, Y, f, cin)
        load(T, p)
        d = (p + 1) ^ 1
        for i in range(wY):
            if (d >> i) & 1:
                c.cx(f, Y[i])
    end = c.mark()
    for i in range(min(n, wY)):
        c.cx(Y[i], c.out[i])
    c.append_inverse(start, end)
