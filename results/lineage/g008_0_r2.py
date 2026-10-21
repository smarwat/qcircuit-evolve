def build_circuit(c, n, p):
    iters = (p * (p - 1)).bit_length()
    L = max(iters, n) + 1
    U = list(c.inp)
    V = c.alloc(n)
    CU = c.alloc(n)
    CV = c.alloc(L)
    H = c.alloc(iters)
    S = c.alloc(2)

    B = ((2 * p - 2) << iters) // (p + 1)
    needE = L
    needT = 1
    Bt = B
    for j in range(iters):
        Bt = (Bt + p) // 2
        w = Bt.bit_length()
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

    def popc(x):
        return bin(x).count('1')

    def inv2r(r):
        mod = 1 << r
        t = 1
        for _ in range(r + 2):
            t = (t * (2 - p * t)) % mod
        return t

    start = c.mark()
    load(V, p)
    c.x(CV[0])
    for it in range(iters):
        h1 = H[it]
        sp = S[it % 2]
        sc = S[(it + 1) % 2]
        cz = CV[-1]
        w = min(n, max(1, iters - it))
        Uw = U[:w]
        Vw = V[:w]
        if it == 0:
            c.cx(U[0], h1)
            c.cx(U[0], sc)
            c.x(sc)
        else:
            c.ccx(U[0], V[0], h1)
            c.cx(U[0], sc)
            c.cx(V[0], U[0])
            c.ccx(sp, U[0], sc)
            c.cx(V[0], U[0])
            c.x(sc)
            for q in Vw:
                c.x(q)
            m0 = c.mark()
            prev = sp
            for i in range(w):
                maj(prev, Uw[i], Vw[i])
                prev = Vw[i]
            m1 = c.mark()
            c.ccx(h1, Vw[w - 1], sc)
            c.append_inverse(m0, m1)
            for q in Vw:
                c.x(q)
            c.ccx(h1, sp, sc)
            c.cx(sp, sc)
        wc = min(n, it + 2)
        cswap(sc, Uw, Vw)
        cswap(sc, CU[:wc], CV[:wc])
        if it > 0:
            c.cx(sp, sc)
            c.cx(sc, sp)
            c.cx(CV[0], sp)
        csub(h1, Uw, Vw, cz)
        V = V[1:] + V[:1]
        cadd(h1, CV[:wc], CU[:wc], cz)
        CV = CV[-1:] + CV[:-1]

    c.x(U[0])
    load(CU, p)
    E = CV + CU + EX
    TS = V + U + TX
    cin = S[0]
    f = S[1]
    lenE = len(E)
    lenTS = len(TS)

    # radix-2^r Montgomery reduction of the 2^iters factor
    j = 0
    Bc = B
    while j < iters:
        best = None
        for r in range(1, min(6, iters - j) + 1):
            mod = 1 << r
            ip = inv2r(r)
            Bn = (Bc + (mod - 1) * p) >> r
            w = Bn.bit_length()
            if w > lenTS or j + r + w > lenE:
                continue
            Ks = [(x + ((-x * ip) % mod) * p) >> r for x in range(mod)]
            kb = max(Ks).bit_length()
            if kb > w:
                continue
            anf = []
            mons = set()
            for i in range(kb):
                co = [(K >> i) & 1 for K in Ks]
                for k in range(r):
                    for x in range(mod):
                        if (x >> k) & 1:
                            co[x] ^= co[x ^ (1 << k)]
                terms = [x for x in range(mod) if co[x]]
                anf.append(terms)
                for x in terms:
                    if popc(x) >= 2:
                        mons.add(x)
            stack = list(mons)
            while stack:
                s = stack.pop()
                hb = s.bit_length() - 1
                par = s ^ (1 << hb)
                if popc(par) >= 2 and par not in mons:
                    mons.add(par)
                    stack.append(par)
            M = len(mons)
            pool_sz = (lenTS - w) + (lenE - (j + r + w)) + 1
            if M > pool_sz:
                continue
            cost = (2.0 * w + 2.0 * M) / r
            if best is None or cost < best[0]:
                best = (cost, r, w, Bn, anf, mons)
        cost, r, w, Bn, anf, mons = best
        X = E[j:j + r]
        R = E[j + r:j + r + w]
        T = TS[:w]
        pool = TS[w:] + E[j + r + w:] + [f]
        order = sorted([(popc(s), s) for s in mons])
        mq = {}
        m0 = c.mark()
        for idx in range(len(order)):
            s = order[idx][1]
            q = pool[idx]
            mq[s] = q
            hb = s.bit_length() - 1
            par = s ^ (1 << hb)
            if popc(par) >= 2:
                a = mq[par]
            else:
                a = X[par.bit_length() - 1]
            c.ccx(a, X[hb], q)
        for i in range(len(anf)):
            for s in anf[i]:
                pc = popc(s)
                if s == 0:
                    c.x(T[i])
                elif pc == 1:
                    c.cx(X[s.bit_length() - 1], T[i])
                else:
                    c.cx(mq[s], T[i])
        m1 = c.mark()
        add(T, R, cin)
        c.append_inverse(m0, m1)
        Bc = Bn
        j += r

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
