import random
from array import array

from evaluator import reference_simulate, simulate_bitsliced


def random_circuit(rng, nq, ngates):
    g = array("i")
    for _ in range(ngates):
        k = rng.randrange(3)
        qs = rng.sample(range(nq), 3)
        if k == 0:
            g.extend((qs[0], -1, -1))
        elif k == 1:
            g.extend((qs[0], qs[1], -1))
        else:
            g.extend((qs[0], qs[1], qs[2]))
    return g


def test_bitsliced_matches_reference_on_random_circuits():
    rng = random.Random(1234)
    for trial in range(40):
        nq = rng.randrange(3, 24)
        g = random_circuit(rng, nq, rng.randrange(1, 400))
        K = rng.randrange(1, 70)
        inputs = [[rng.randrange(2) for _ in range(nq)] for _ in range(K)]
        state = [sum(inputs[k][q] << k for k in range(K)) for q in range(nq)]
        simulate_bitsliced(g, state, (1 << K) - 1)
        for k in range(K):
            want = reference_simulate(g, inputs[k])
            got = [(state[q] >> k) & 1 for q in range(nq)]
            assert got == want, (trial, k)


def test_reversed_circuit_is_inverse():
    rng = random.Random(99)
    nq, K = 12, 64
    g = random_circuit(rng, nq, 300)
    rev = array("i")
    for i in range(len(g) - 3, -1, -3):
        rev.extend(g[i : i + 3])
    start = [rng.getrandbits(K) for _ in range(nq)]
    state = list(start)
    simulate_bitsliced(g, state, (1 << K) - 1)
    simulate_bitsliced(rev, state, (1 << K) - 1)
    assert state == start
