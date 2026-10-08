"""certify_table2_AB.py -- ball-arithmetic certification of Table 2, blocks A and B (App. F.3).

Blocks A and B of Table 2 are exact finite-rank instances: product advice with at
most two binary coordinate types, each with rational (block A) or exactly
representable binary-rational (block B) probabilities.  This script re-evaluates
both columns of those rows with rigorous error control:

  * Level structure (EXACT, big integers).  A key's probability is N/Den with
    N = prod_types l^(n-j) u^j and Den = prod_types d^n.  Levels are grouped by the
    exact integer N, so ties are merged exactly and the ordering by decreasing
    probability is exact (no floating-point tie tolerance as in exact_commensurate.py).
  * E_P[G] (EXACT rational).  A level of cnt keys occupying ranks a+1..b
    contributes N*cnt*(a+b+1)/(2*Den), the level average of Lemma 3.1(iii).
  * E_P[sqrt G] (BALL).  The block sums S(a,b) = sum_{r=a+1}^{b} sqrt(r) are
    evaluated as zeta(-1/2, a+1) - zeta(-1/2, b+1) (Hurwitz zeta) in Arb ball
    arithmetic; short blocks are summed directly.
  * s = ln E[G] / ln E[sqrt G] and the bound of BGMN Theorem 4.3,
        2 (H_{1/2} - ln(1 + log2|K|)) / H_{2/3}   (entropies in nats, see bgmn_bound.py),
    with H_{1/2}, H_{2/3} summed per coordinate from the exact rationals.

Every quantity is an Arb ball (midpoint-radius interval) at ctx.prec bits, so the
printed enclosures are rigorous.  The script then checks that each value printed
in Table 2 is the correctly rounded 3-decimal value of the enclosed number, i.e.
that the whole ball lies strictly inside (printed - 0.0005, printed + 0.0005).
Rows, models and printed values come from table2_config.py; block B's q enters
exactly as stored (a double), so the Monte Carlo error of the fit concerns the
model, not these digits.

Requires python-flint (pip install python-flint; tested with 0.9.0, which bundles
FLINT 3 / Arb).
"""
import time
from fractions import Fraction
from math import comb
from flint import arb, fmpq, ctx
import table2_config as T2

ctx.prec = 3000
HALF = arb(-1) / 2
LN2 = arb(2).log()


def S(a, b):
    """Rigorous enclosure of sum_{r=a+1}^{b} sqrt(r)."""
    if b - a <= 64:
        tot = arb(0)
        for r in range(a + 1, b + 1):
            tot += arb(r).sqrt()
        return tot
    return HALF.zeta(arb(a + 1)) - HALF.zeta(arb(b + 1))


def certify(types):
    """types: list of (n, l, u, d): n binary coordinates with probabilities l/d, u/d.

    Returns (s, bound, log2 E[G], number of distinct levels, number of keys)."""
    levels = {}

    def rec(i, N, cnt):
        if i == len(types):
            levels[N] = levels.get(N, 0) + cnt
            return
        n, l, u, d = types[i]
        for j in range(n + 1):
            rec(i + 1, N * l ** (n - j) * u ** j, cnt * comb(n, j))

    rec(0, 1, 1)
    Den = 1
    for n, l, u, d in types:
        assert l + u == d
        Den *= d ** n
    order = sorted(levels.items(), key=lambda kv: -kv[0])   # exact, decreasing probability
    c = 0
    EG_num = 0                  # E[G] = EG_num / (2*Den), exact integer arithmetic
    EH = arb(0)                 # E[sqrt G] * Den
    for N, cnt in order:
        a, b = c, c + cnt
        EG_num += N * cnt * (a + b + 1)
        EH += arb(N) * S(a, b)
        c = b
    total = c
    assert total == 2 ** sum(t[0] for t in types)
    lnDen = arb(Den).log()
    f1 = arb(EG_num).log() - LN2 - lnDen          # ln E[G]
    fh = EH.log() - lnDen                          # ln E[sqrt G]
    s = f1 / fh
    H12 = arb(0)
    H23 = arb(0)
    for n, l, u, d in types:
        p, q = arb(fmpq(l, d)), arb(fmpq(u, d))
        H12 += 2 * n * (p.sqrt() + q.sqrt()).log()
        H23 += 3 * n * (p ** (arb(2) / 3) + q ** (arb(2) / 3)).log()
    W = sum(t[0] for t in types)                   # log2|K|
    bound = 2 * (H12 - arb(1 + W).log()) / H23
    return s, bound, f1 / LN2, len(order), total


def rows():
    out = []
    for r in T2.BLOCK_A:
        out.append((f"A {r['label']}, beta={r['beta']}", T2.exact_types_A(r), r["s"], r["bound"]))
    for r in T2.BLOCK_B:
        out.append((f"B {r['label']}, 2^{r['T']}", T2.exact_types_B(r), r["s"], r["bound"]))
    return out


def rounds_to(x, printed):
    """True iff the whole ball x lies strictly inside (printed - 0.0005, printed + 0.0005)."""
    v = Fraction(printed)
    lo = arb(fmpq((v - Fraction(1, 2000)).numerator, (v - Fraction(1, 2000)).denominator))
    hi = arb(fmpq((v + Fraction(1, 2000)).numerator, (v + Fraction(1, 2000)).denominator))
    return bool(x > lo) and bool(x < hi)


if __name__ == "__main__":
    allok = True
    worst_log10_rad = -10 ** 9          # log10 of the largest ball radius (radii underflow a double)
    for name, types, ps, pb in rows():
        t = time.time()
        s, bound, log2EG, nlev, total = certify(types)
        ok_s, ok_b = rounds_to(s, ps), rounds_to(bound, pb)
        allok &= ok_s and ok_b
        for x in (s, bound):
            worst_log10_rad = max(worst_log10_rad, float((x.rad().log() / arb(10).log()).mid()))
        print(f"{name:34s} levels={nlev:6d}  s     = {s.str(20)}  printed {ps}  ok={ok_s}")
        print(f"{'':34s} {'':13s}  bound = {bound.str(20)}  printed {pb}  ok={ok_b}  ({time.time() - t:.1f}s)")
    print(f"largest ball radius: about 10^{worst_log10_rad:.0f}")
    print("ALL PRINTED DIGITS OF BLOCKS A AND B CERTIFIED:", allok)
    raise SystemExit(0 if allok else 1)
