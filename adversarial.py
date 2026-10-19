"""Classical step counters and a search for inputs that need the most iterations.

Evolved inversion circuits run a fixed number of loop iterations. A circuit that
quietly runs too few iterations is wrong only on the inputs that need the most
steps, and uniform random sampling rarely hits those. The verify tier therefore
adds the hardest inputs found by random restarts + hill climbing under several
step metrics (different algorithms have different worst cases).

    python adversarial.py 32 4294967291     # print the hardest inputs found
"""

import random
import sys
from functools import lru_cache

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n):
    """Deterministic Miller-Rabin; exact for n < 3.3e24 (covers every width here)."""
    if n < 2:
        return False
    for q in _MR_BASES:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def steps_binary_euclid(a, p):
    """The seed's variant: u=a, v=p; halve whichever is even, else subtract
    smaller from larger and halve; stop when v == 0."""
    u, v, k = a, p, 0
    while v:
        if not u & 1:
            u >>= 1
        elif not v & 1:
            v >>= 1
        elif u > v:
            u = (u - v) >> 1
        else:
            v = (v - u) >> 1
        k += 1
    return k


def steps_kaliski(a, p):
    """Kaliski's almost-inverse loop (u=p, v=a, while v > 0), as used by
    Roetteler-Naehrig-Svore-Lauter 2017."""
    u, v, k = p, a, 0
    while v > 0:
        if not u & 1:
            u >>= 1
        elif not v & 1:
            v >>= 1
        elif u > v:
            u = (u - v) >> 1
        else:
            v = (v - u) >> 1
        k += 1
    return k


def steps_divsteps(a, p):
    """Bernstein-Yang divsteps from (delta, f, g) = (1, p, a) until g == 0."""
    delta, f, g, k = 1, p, a, 0
    while g:
        if delta > 0 and g & 1:
            delta, f, g = 1 - delta, g, (g - f) >> 1
        else:
            delta, g = 1 + delta, (g + (g & 1) * f) >> 1
        k += 1
    return k


def steps_binary_gcd_plain(a, p):
    """Stein's binary GCD on (a, p) with swap-to-keep-ordered, counting
    subtract-and-shift rounds (another common circuit structure)."""
    u, v, k = a, p, 0
    while u:
        while not u & 1:
            u >>= 1
            k += 1
        if u < v:
            u, v = v, u
        u -= v
        k += 1
    return k


METRICS = {
    "binary_euclid": steps_binary_euclid,
    "kaliski": steps_kaliski,
    "divsteps": steps_divsteps,
    "binary_gcd": steps_binary_gcd_plain,
}


def hill_climb(metric, n, p, rng, restarts=32, iters=2500):
    """Return {input: steps} for the best inputs seen across restarts."""
    best = {}
    for _ in range(restarts):
        x = rng.randrange(1, p)
        sx = metric(x, p)
        for _ in range(iters):
            r = rng.random()
            if r < 0.6:                       # flip one or two bits
                y = x ^ (1 << rng.randrange(n))
                if r < 0.2:
                    y ^= 1 << rng.randrange(n)
            elif r < 0.9:                     # small additive step
                y = x + rng.choice((-1, 1)) * (1 << rng.randrange(max(1, n // 2)))
            else:                             # jump near p - x or x/2 (symmetries of the loop)
                y = p - x if r < 0.95 else (x >> 1) | (rng.getrandbits(1) << (n - 1))
            if not 1 <= y < p:
                continue
            sy = metric(y, p)
            if sy >= sx:
                x, sx = y, sy
        best[x] = sx
    return best


@lru_cache(maxsize=32)
def adversarial_inputs(n, p, seed, per_metric=16, restarts=32, iters=2500):
    """Deterministic tuple of hard inputs: the top `per_metric` per step metric."""
    rng = random.Random(seed * 7919 + n * 104729 + p % 1_000_003)
    out = []
    for name, metric in METRICS.items():
        found = hill_climb(metric, n, p, rng, restarts, iters)
        # Also try the structured candidates hill climbing tends to circle.
        for x in (1, 2, p - 1, p - 2, (p + 1) // 2, (p - 1) // 2, (1 << (n - 1)) - 1):
            if 1 <= x < p:
                found.setdefault(x, metric(x, p))
        ranked = sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))
        out.extend(x for x, _ in ranked[:per_metric])
    seen = set()
    return tuple(x for x in out if not (x in seen or seen.add(x)))


def main(argv):
    n, p = int(argv[1]), int(argv[2])
    rng = random.Random(1)
    for name, metric in METRICS.items():
        found = hill_climb(metric, n, p, rng)
        rand = max(metric(rng.randrange(1, p), p) for _ in range(20000))
        top = sorted(found.items(), key=lambda kv: -kv[1])[:5]
        print(f"{name:14s} hill-climb max {top[0][1]:4d}  random(20k) max {rand:4d}  top: {top}")


if __name__ == "__main__":
    main(sys.argv)
