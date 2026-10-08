"""
reference_values.py -- independent high-precision reference values of s for test_correctness.py.

For i.i.d. advice with rational marginal probabilities, levels are grouped by
their exact (Fraction) probability, so ties are merged exactly and ordered
exactly; E_P[G] is an exact rational (level averages of Lemma 3.1(iii)), and
E_P[sqrt G] uses exact sums for blocks of at most 64 ranks and the Euler-Maclaurin
expansion of sum_{r<=x} sqrt(r) otherwise, in mpmath at a working precision of
(digits of N) + 60, so the cancellation in the block differences is harmless.
This shares no code with the engines.

    python3 reference_values.py      # prints the table used in test_correctness.py
"""

from __future__ import annotations
from fractions import Fraction as Fr
from math import comb, prod
import mpmath as mp

CASES = [  # label, marginal (exact), m
    ("P=(.5,.2,.2,.1)", [Fr(1, 2), Fr(1, 5), Fr(1, 5), Fr(1, 10)], 1),
    ("Ber(0.75) m=16", [Fr(3, 4), Fr(1, 4)], 16),
    ("Ber(0.75) m=256", [Fr(3, 4), Fr(1, 4)], 256),
    ("Ber(0.75) m=1024", [Fr(3, 4), Fr(1, 4)], 1024),
    ("Ber(0.90) m=16", [Fr(9, 10), Fr(1, 10)], 16),
    ("Ber(0.90) m=256", [Fr(9, 10), Fr(1, 10)], 256),
    ("Ber(0.90) m=1024", [Fr(9, 10), Fr(1, 10)], 1024),
    ("uniform ternary m=16", [Fr(1, 3)] * 3, 16),
    ("uniform ternary m=1024", [Fr(1, 3)] * 3, 1024),
    ("ternary .6/.2/.2 m=16", [Fr(3, 5), Fr(1, 5), Fr(1, 5)], 16),
    ("ternary .6/.2/.2 m=256", [Fr(3, 5), Fr(1, 5), Fr(1, 5)], 256),
    ("Ber(0.999) m=16", [Fr(999, 1000), Fr(1, 1000)], 16),
    ("Ber(0.999) m=256", [Fr(999, 1000), Fr(1, 1000)], 256),
    ("Ber(0.999) m=1024", [Fr(999, 1000), Fr(1, 1000)], 1024),
]


def _compositions(m, d):
    if d == 1:
        yield (m,)
        return
    for first in range(m + 1):
        for rest in _compositions(m - first, d - 1):
            yield (first,) + rest


def levels(marginal, m):
    """[(probability of one key, number of keys)] in decreasing probability."""
    distinct = {}
    for x in marginal:
        distinct[x] = distinct.get(x, 0) + 1
    vals = list(distinct.items())
    out = {}
    for c in _compositions(m, len(vals)):
        mult = comb(m, c[0])
        rest = m - c[0]
        for k in c[1:]:
            mult *= comb(rest, k)
            rest -= k
        cnt = mult * prod(cj ** k for (_, cj), k in zip(vals, c))
        prob = prod(v ** k for (v, _), k in zip(vals, c))
        out[prob] = out.get(prob, 0) + cnt
    return sorted(out.items(), key=lambda kv: -kv[0])


def _G(x):
    """sum_{r=1}^{x} sqrt(r) - zeta(-1/2), asymptotic (x >= 64)."""
    x = mp.mpf(x)
    return (mp.mpf(2) / 3 * x ** 1.5 + x ** 0.5 / 2 + x ** -0.5 / 24
            - x ** -2.5 / 1920 + x ** -4.5 / 9216)


def sqrt_sum(a, b):
    """sum_{r=a+1}^{b} sqrt(r) at the current mpmath precision."""
    if b - a <= 64:
        return mp.fsum(mp.sqrt(r) for r in range(a + 1, b + 1))
    if a < 64:
        return mp.fsum(mp.sqrt(r) for r in range(a + 1, 65)) + _G(b) - _G(64)
    return _G(b) - _G(a)


def reference_s(marginal, m):
    lv = levels(marginal, m)
    N = sum(c for _, c in lv)
    mp.mp.dps = len(str(N)) + 60
    EG = Fr(0)
    EH = mp.mpf(0)
    a = 0
    for prob, cnt in lv:
        b = a + cnt
        EG += prob * cnt * Fr(a + b + 1, 2)
        EH += mp.mpf(prob.numerator) / prob.denominator * sqrt_sum(a, b)
        a = b
    lnEG = mp.log(mp.mpf(EG.numerator)) - mp.log(mp.mpf(EG.denominator))
    return lnEG / mp.log(EH)


if __name__ == "__main__":
    for label, marginal, m in CASES:
        print(f"{label:<26} s = {mp.nstr(reference_s(marginal, m), 15)}")
