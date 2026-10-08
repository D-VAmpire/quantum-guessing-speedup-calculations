"""
exact_commensurate.py -- exact evaluation, no discretization error.

Implements Theorem 4.1 with the level averages of Lemma 3.1(iii):

    f(rho) = ln sum_t pi(t) A_rho(t),      A_rho(t) = n(t)^-1 sum_{r in J_t} r^rho,
    J_t = ( N F_S(t^-), N F_S(t) ],        n(t) = N nu(t),      pi(t) = n(t) e^{-t}.

Route: the paper's finite-rank exact engine (Sec. 4.1, "The finite-rank exact
engine").  i.i.d. advice over a k-letter alphabet has rank d <= k, so the level of
a key depends only on the *counts* of each symbol.  Levels are therefore
enumerated by symbol count, with multinomial coefficients as exact Python
integers, instead of convolving a grid.  This is exact even when the marginals
are not commensurate (generic Ber(p) has rank 2, not rank 1).

Exact integer counts matter: the rank-block endpoints a, b are integers of
astronomical size, and A_{1/2} needs them exactly at the concentrated end, where
the dominant levels have n(t) = 1 (a single key) and any asymptotic formula is
invalid.

Applicable when the coordinates fall into few types with small alphabets, which
covers Table 2 blocks A and B.  For heterogeneous advice (rank Theta(m)) use
binning_general.py.  Levels are identified by their surprisal rounded to `nd`
decimals; certify_table2_AB.py identifies them exactly instead.
"""

from __future__ import annotations
import math
import numpy as np
from math import comb
from common import group_identical, assemble_from_counts

MAX_LEVELS = 5_000_000


def _type_levels(p, M, nd):
    """{level -> exact int count} for M identical coordinates with marginal p."""
    p = np.asarray(p, dtype=float)
    b = p.size
    v = -np.log(p)
    out = {}
    if b == 2:                                   # binomial fast path
        for k in range(M + 1):
            lv = round((M - k) * v[0] + k * v[1], nd)
            out[lv] = out.get(lv, 0) + comb(M, k)
        return out
    if math.comb(M + b - 1, b - 1) > MAX_LEVELS:
        raise NotImplementedError(
            f"level count too large for b={b}, M={M}; use binning.")

    def comps(M, b):
        if b == 1:
            yield (M,)
            return
        for first in range(M + 1):
            for rest in comps(M - first, b - 1):
                yield (first,) + rest

    fact = [math.factorial(i) for i in range(M + 1)]
    for c in comps(M, b):
        mult = fact[M]
        for x in c:
            mult //= fact[x]
        lv = round(float(np.dot(c, v)), nd)
        out[lv] = out.get(lv, 0) + mult
    return out


def level_counts(marginals, nd=9):
    """Sorted [(surprisal level, exact integer key count)] for the product law."""
    levels = {0.0: 1}
    for p, M in group_identical(marginals):
        tl = _type_levels(p, M, nd)
        if len(levels) * len(tl) > MAX_LEVELS:
            raise NotImplementedError("level product too large; use binning.")
        new = {}
        for l1, c1 in levels.items():
            for l2, c2 in tl.items():
                l = round(l1 + l2, nd)
                new[l] = new.get(l, 0) + c1 * c2
        levels = new
    return sorted(levels.items())


def exact_speedup(marginals, nd=9):
    """Exact speedup exponent s, with f(1), f(1/2), for product advice."""
    res = assemble_from_counts(level_counts(marginals, nd))
    res["method"] = "exact (finite-rank engine, A_rho)"
    return res


if __name__ == "__main__":
    def bern(p, m):
        return [[p, 1 - p]] * m
    print(f"{'case':<26}{'s':>10}{'f(1/2)':>9}")
    for p, m in [(0.97, 128), (0.95, 128), (0.90, 256), (0.75, 256)]:
        r = exact_speedup(bern(p, m))
        print(f"Ber({p}) m={m:<16}{r['s']:>10.5f}{r['f_half']:>9.2f}")
