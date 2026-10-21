def build_circuit(c, n, p):
    iters = (p * (p - 1)).bit_length()
    L = max(iters, n) + 1
    U = list(c.inp)
    nhu = min(iters, n - 1)
    nfresh = iters - nhu

    B = ((2 * p - 2) << iters) // (p + 1)
    needE = L
    Bt = B
    for j in range(iters):
        Bt = (Bt + p) // 2
        needE = max(needE, j + 1 + Bt.bit_length())
    wY = Bt.bit_length()
    needE = max(needE, iters + wY)
    extE = max(0, needE - L - n)
    lenE = L + n + extE

    def popc(x):
        return bin(x).count('1')

    def inv2r(r):
        mod = 1 << r
        t = 1
        for _ in range(r + 2):
            t = (t * (2 - p * t)) % mod
        return t

    RI = {}
    for r in range(1, min(6, iters) + 1):
        mod = 1 << r
        ip = inv2r(r)
        Ks = [(x + ((-x * ip) % mod) * p) >> r for x in range(mod)]
        kb = max(Ks).bit_length()
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
        RI[r] = (anf, mons, kb)

    def choose(j, Bc, nfree):
        best = None
        for r in range(1, min(6, iters - j) + 1):
            Bn = (Bc + ((1 << r) - 1) * p) >> r
            w = Bn.bit_length()
            if j + r + w > lenE:
                continue
            anf, mons, kb = RI[r]
            if kb > w:
                continue
            M = len(mons)
            avail = nfree + lenE - (j + r + w)
            if w + M > avail:
                continue
            cost = (2.0 * w + 2.0 * M) / r
            if best is None or cost < best[0]:
                best = (cost, r, w, Bn, anf, mons, M)
        return best

    def plan(nfree):
        j = 0
        Bc = B
        tot = 0
        while j < iters:
            b = choose(j, Bc, nfree)
            if b is None:
                return None
            tot += 2 * b[2] + 2 * b[6]
            Bc = b[3]
            j += b[1]
        return tot

    Tloop = 0
    for it in range(iters):
        w = min(n, max(1, iters - it))
        wc = min(n, it + 2)
        if it > 0:
            Tloop += 2
            if w > 1:
                Tloop += 2 * (w - 1) + 2
        Tloop += w + wc + 3 * (w - 1) + 3 * wc
    tcmp = 2 * wY if Bt > p else 0

    baseTSF = n + 1 + (n - 1 - nhu)
    nHf = nfresh - 1 if nfresh >= 2 else nfresh
    anc_base = n + n + L + nHf + 2 + extE
    Qbase = 2 * n + anc_base
    budget = len(c.anc)
    best_extra = None
    best_sc = None
    for extra in range(0, 4 * n + 64):
        if anc_base + extra > budget:
            break
        nT = baseTSF + extra
        if Bt > p and nT < wY:
            continue
        tot = plan(nT + 1)
        if tot is None:
            continue
        sc = (Tloop + tot + tcmp) * (Qbase + extra)
        if best_sc is None or sc < best_sc:
            best_sc = sc
            best_extra = extra

    V = c.alloc(n)
    CU = c.alloc(n)
    CV = c.alloc(L)
    Hf = c.alloc(nHf) if nHf > 0 else []
    S = c.alloc(2)
    EX = c.alloc(extE) if extE > 0 else []
    TX = c.alloc(best_extra) if best_extra > 0 else []

    HQ = []
    for it in range(iters):
        k = iters - it
        if k < n:
            HQ.append(U[k])
        elif nfresh >= 2 and it >= 1:
            HQ.append(Hf[it - 1])
        else:
            HQ.append(Hf[it])

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

    start = c.mark()
    load(V, p)
    c.x(CV[0])
    for it in range(iters):
        h1 = HQ[it]
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
            if w > 1:
                for q in Vw[1:]:
                    c.x(q)
                m0 = c.mark()
                prev = sp
                for i in range(1, w):
                    maj(prev, Uw[i], Vw[i])
                    prev = Vw[i]
                m1 = c.mark()
                c.ccx(h1, Vw[w - 1], sc)
                c.append_inverse(m0, m1)
                for q in Vw[1:]:
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
        csub(h1, Uw[1:], Vw[1:], cz)
        c.cx(h1, Vw[0])
        V = V[1:] + V[:1]
        cadd(h1, CV[:wc], CU[:wc], cz)
        CV = CV[-1:] + CV[:-1]
        if it == 0:
            c.cx(CV[1], h1)

    c.x(U[0])
    load(CU, p)
    E = CV + CU + EX
    TSF = list(V) + [U[0]] + U[nhu + 1:] + TX
    cin = S[0]
    f = S[1]
    nfree = len(TSF) + 1

    j = 0
    Bc = B
    while j < iters:
        cost, r, w, Bn, anf, mons, M = choose(j, Bc, nfree)
        X = E[j:j + r]
        R = E[j + r:j + r + w]
        free = TSF + E[j + r + w:] + [f]
        T = free[:w]
        pool = free[w:]
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
        T = TSF[:wY]
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
