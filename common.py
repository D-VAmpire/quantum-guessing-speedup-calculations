"""
common.py -- shared primitives for the speedup exponent s = f(1)/f(1/2).

Notation follows the paper (C. Schubert, N. Paskarbeit, M. J. Kramer, J.-P. Seifert,
M. Margraf, "Pinpointing Super-Quadratic Quantum Enumeration Speedups", PQQS '26,
https://doi.org/10.1145/3839338.3857257):

    K = prod_i A_i            product key space, |A_i| = b_i, N = prod_i b_i
    P = prod_i p_i            product advice: one marginal p_i per coordinate
    s_i(a) = -ln p_i(a)       per-coordinate surprisal (natural log)
    S(x) = sum_i s_i(x_i)     surprisal of a key (additive, since P is a product)
    F_S(t) = P_U[S <= t]      CDF of S under the uniform law U on K
    J_t, A_rho(t)             rank block of level t and its mean rho-th power rank
    f(rho) = ln E_P[G^rho] = ln sum_t pi(t) A_rho(t)          (Lemma 3.1(iii))
    s = f(1) / f(1/2) >= 2

All logs are natural; s is a ratio of logs, hence base-independent.
A "marginal" is a 1-D probability vector; an "instance" is a list of marginals,
one per coordinate.  This module holds no scheme data -- see distributions.py.
"""

from __future__ import annotations
import math
import numpy as np

# --------------------------------------------------------------------------
# log-domain helpers
# --------------------------------------------------------------------------

def logsumexp(a):
    """Numerically stable ln(sum(exp(a))) over a 1-D array (ignores -inf)."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return -np.inf
    m = np.max(a)
    if not np.isfinite(m):
        return m  # all -inf -> -inf ; or +inf
    return m + np.log(np.sum(np.exp(a - m)))


def logcumsumexp(a):
    """Cumulative ln(sum(exp(.))) along a 1-D array; entry k = LSE(a[:k+1]).

    This is the left-to-right prefix LSE of the paper's log-domain recovery,
    Eq. (6): the larger term stays outside the logarithm, nothing overflows."""
    return np.logaddexp.accumulate(np.asarray(a, dtype=float))


# --------------------------------------------------------------------------
# information-theoretic quantities
# --------------------------------------------------------------------------

def surprisals(p):
    """Per-outcome surprisals s(a) = -ln p(a) for one marginal."""
    p = np.asarray(p, dtype=float)
    return -np.log(p)


def renyi_entropy(p, alpha):
    """Renyi entropy H_alpha(p) in nats.  alpha=1 -> Shannon (continuous)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if abs(alpha - 1.0) < 1e-12:
        return float(-np.sum(p * np.log(p)))
    return float((1.0 / (1.0 - alpha)) * np.log(np.sum(p ** alpha)))


def total_renyi(marginals, alpha):
    """H_alpha(P) = sum_i H_alpha(p_i)  (additive for a product law), nats."""
    return float(sum(renyi_entropy(p, alpha) for p in marginals))


def mean_renyi(marginals, alpha):
    """Per-coordinate mean  Hbar_alpha = (1/m) sum_i H_alpha(p_i)."""
    m = len(marginals)
    return total_renyi(marginals, alpha) / m


def lnN(marginals):
    """ln|K| = sum_i ln b_i."""
    return float(sum(np.log(len(p)) for p in marginals))


# --------------------------------------------------------------------------
# level averages A_rho(t)   (Lemma 3.1(iii); evaluation as in App. F.1)
# --------------------------------------------------------------------------
# Keys of equal surprisal occupy consecutive ranks, so with
#     J_t := ( N F_S(t^-), N F_S(t) ],      n(t) := N nu(t) = |J_t|
# the moment is an average over that block, NOT the block-maximal rank:
#     A_rho(t) := n(t)^{-1} sum_{r in J_t} r^rho,     E_P[G^rho] = sum_t pi(t) A_rho(t).
# The block-maximal surrogate (N F_S(t))^rho is the n(t) = 1 special case of the
# bracket Eq. (4) of Theorem 4.1 (see also Remark F.1).

EXACT_BLOCK_MAX = 5_000      # blocks of at most this many ranks are summed exactly


def log_A1_int(a, b):
    """ln A_1 over ranks (a,b], exact integers.  Sum = (b(b+1)-a(a+1))/2."""
    tot = b * (b + 1) - a * (a + 1)
    return math.log(tot) - math.log(2.0) - math.log(b - a)


def log_sum_sqrt(a, b):
    """ln sum_{r=a+1}^{b} sqrt(r).  a, b exact ints (a >= 0, b > a).

    Exact summation for small blocks; for large blocks the two-term
    Euler-Maclaurin form (the upper end of the bracket in App. F.1, tight to a
    relative O(b^-2)), evaluated through the occupancy ratio beta = n/b with
    log1p/expm1: the naive difference b^{3/2} - a^{3/2} would lose about
    log10(1/beta) digits on thin levels (App. F.1)."""
    n = b - a
    if n <= EXACT_BLOCK_MAX:                       # exact
        acc = [0.5 * math.log(r) for r in range(a + 1, b + 1)]
        mx = max(acc)
        return mx + math.log(sum(math.exp(x - mx) for x in acc))
    lb = math.log(b)
    if a == 0:                                     # whole prefix
        t1 = math.log(2.0 / 3.0) + 1.5 * lb
        t2 = -math.log(2.0) + 0.5 * lb
    else:
        beta = n / b                               # exact ints -> float ratio
        if beta < 1e-13:                           # thin level: ~ n*sqrt(b)
            return math.log(n) + 0.5 * lb
        l1mb = math.log1p(-beta)                   # ln(a/b)
        t1 = math.log(2.0/3.0) + 1.5*lb + math.log(-math.expm1(1.5 * l1mb))
        t2 = -math.log(2.0) + 0.5*lb + math.log(-math.expm1(0.5 * l1mb))
    m_ = max(t1, t2)
    return m_ + math.log(math.exp(t1 - m_) + math.exp(t2 - m_))


def log_A_int(a, b, rho):
    """ln A_rho over the integer rank block (a, b]."""
    if rho == 1.0:
        return log_A1_int(a, b)
    if rho == 0.5:
        return log_sum_sqrt(a, b) - math.log(b - a)
    raise ValueError("rho must be 1 or 1/2")


def assemble_from_counts(levels):
    """f(1), f(1/2), s from exact integer level counts (exact engine).

    `levels`: sorted list of (surprisal t, exact int count n(t)).
    Uses pi(t) = n(t) e^{-t}, Eq. (2), so only counts are needed."""
    cum = 0
    t1s, ths = [], []
    for t, n in levels:
        a, b = cum, cum + n
        cum = b
        ln_pi = math.log(n) - t                    # Eq. (2)
        t1s.append(ln_pi + log_A_int(a, b, 1.0))
        ths.append(ln_pi + log_A_int(a, b, 0.5))

    def lse(v):
        mx = max(v)
        return mx + math.log(sum(math.exp(x - mx) for x in v))

    f1, fh = lse(t1s), lse(ths)
    return {"s": f1 / fh, "f1": f1, "f_half": fh, "n_levels": len(levels),
            "N": cum}


# ---- float-domain variants (binned chains: counts are not exact integers) --

def log_A_float(ln_a, ln_b, ln_n, rho):
    """ln A_rho of the block (a, b], n = b - a, from log-counts (binned engine).

    ln_a may be -inf (a = 0).  The result is floored at rank 1 (ln A >= 0):
    G(x) >= 1 always, and without the floor Delta(x) of Lemma 6.1 would be
    undefined at the minimum-surprisal key, which strict rounding always keeps."""
    if not (np.isfinite(ln_b) and np.isfinite(ln_n)):
        return 0.0                                 # no keys counted -> rank 1
    if rho == 1.0:                                 # (a+b+1)/2
        val = np.logaddexp(np.logaddexp(ln_a, ln_b), 0.0) - np.log(2.0)
        return max(float(val), 0.0)
    lb = float(ln_b)
    # small blocks: the asymptotic form is invalid (at n=1, a=0 it errs by 17%),
    # and those levels carry the most advice mass at the concentrated end.
    if lb < 36.0:                                  # b < 2^52, safe as an int
        bi = int(round(float(np.exp(lb))))
        ai = int(round(float(np.exp(ln_a)))) if np.isfinite(ln_a) else 0
        ai = max(0, min(ai, bi - 1))
        if bi >= 1 and (bi - ai) <= EXACT_BLOCK_MAX:
            return max(log_sum_sqrt(ai, bi) - math.log(bi - ai), 0.0)
    beta = float(np.exp(min(float(ln_n) - lb, 0.0)))   # n/b in (0,1]
    if not np.isfinite(ln_a) or beta >= 1.0 - 1e-15:
        t1 = np.log(2.0/3.0) + 1.5*lb
        t2 = -np.log(2.0) + 0.5*lb
    elif beta < 1e-13:
        return max(0.5*lb, 0.0)                    # thin level: A ~ sqrt(b)
    else:
        l1mb = math.log1p(-beta)
        t1 = np.log(2.0/3.0) + 1.5*lb + math.log(-math.expm1(1.5*l1mb))
        t2 = -np.log(2.0) + 0.5*lb + math.log(-math.expm1(0.5*l1mb))
    return max(float(np.logaddexp(t1, t2)) - float(ln_n), 0.0)


def log_A_array(ln_a, ln_b, ln_n, rho):
    """Vectorized log_A_float over arrays of blocks (same values, same branches).

    Blocks that need exact summation (b < 2^52 and at most EXACT_BLOCK_MAX
    ranks) are evaluated by the scalar routine; all others in closed form."""
    ln_a = np.asarray(ln_a, dtype=float)
    ln_b = np.asarray(ln_b, dtype=float)
    ln_n = np.asarray(ln_n, dtype=float)
    out = np.zeros(ln_b.shape)
    ok = np.isfinite(ln_b) & np.isfinite(ln_n)
    if rho == 1.0:
        with np.errstate(invalid="ignore"):
            val = np.logaddexp(np.logaddexp(ln_a, ln_b), 0.0) - np.log(2.0)
        out[ok] = np.maximum(val[ok], 0.0)
        return out
    if rho != 0.5:
        raise ValueError("rho must be 1 or 1/2")
    # blocks the scalar routine sums exactly
    small = ok & (ln_b < 36.0)
    if np.any(small):
        bi = np.rint(np.exp(ln_b[small]))
        ai = np.where(np.isfinite(ln_a[small]), np.rint(np.exp(ln_a[small])), 0.0)
        ai = np.maximum(0.0, np.minimum(ai, bi - 1.0))
        exact = (bi >= 1.0) & ((bi - ai) <= EXACT_BLOCK_MAX)
        idx_small = np.nonzero(small)[0]
        idx_exact = idx_small[exact]
    else:
        idx_exact = np.zeros(0, dtype=int)
    for i in idx_exact:
        out[i] = log_A_float(ln_a[i], ln_b[i], ln_n[i], 0.5)
    rest = ok.copy()
    rest[idx_exact] = False
    if np.any(rest):
        la, lb, ln = ln_a[rest], ln_b[rest], ln_n[rest]
        beta = np.exp(np.minimum(ln - lb, 0.0))
        full = (~np.isfinite(la)) | (beta >= 1.0 - 1e-15)
        thin = (~full) & (beta < 1e-13)
        mid = ~(full | thin)
        res = np.empty(lb.shape)
        # whole prefix / single full block
        t1 = np.log(2.0/3.0) + 1.5*lb
        t2 = -np.log(2.0) + 0.5*lb
        res[full] = (np.logaddexp(t1, t2) - ln)[full]
        # thin level: A ~ sqrt(b)
        res[thin] = 0.5*lb[thin]
        if np.any(mid):
            l1mb = np.log1p(-beta[mid])
            t1m = np.log(2.0/3.0) + 1.5*lb[mid] + np.log(-np.expm1(1.5*l1mb))
            t2m = -np.log(2.0) + 0.5*lb[mid] + np.log(-np.expm1(0.5*l1mb))
            res[mid] = np.logaddexp(t1m, t2m) - ln[mid]
        out[rest] = np.maximum(res, 0.0)
    return out


# --------------------------------------------------------------------------
# brute-force ground truth  (validation only; enumerates all N keys)
# --------------------------------------------------------------------------

def brute_force_speedup(marginals, max_keys=2_000_000):
    """Ground truth: the TRUE optimal guessing moments, by enumeration.

    Sort all N keys by decreasing posterior mass, assign the optimal ranks
    1..N, and average directly:  E_P[G^rho] = sum_x P(x) G(x)^rho.  This is
    exactly the object of Lemma 3.1; by part (i) the value does not depend
    on how ties are broken, so no tie convention enters here at all."""
    import itertools
    alph = [len(p) for p in marginals]
    N = int(np.prod(alph))
    if N > max_keys:
        raise ValueError(f"brute force refused: N={N} > max_keys={max_keys}")
    logp = [np.log(np.asarray(p, dtype=float)) for p in marginals]
    S = np.empty(N)
    for idx, combo in enumerate(itertools.product(*[range(b) for b in alph])):
        S[idx] = -sum(logp[i][combo[i]] for i in range(len(combo)))
    S = np.sort(S)                              # ascending surprisal = optimal order
    P = np.exp(-S)
    r = np.arange(1, N + 1, dtype=float)
    f1 = float(np.log(np.sum(P * r)))
    fh = float(np.log(np.sum(P * np.sqrt(r))))
    return {"s": f1 / fh, "f1": f1, "f_half": fh, "N": N}


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def group_identical(marginals, tol=1e-12):
    """Collapse identical coordinates -> list of (marginal, multiplicity)."""
    groups = []  # (repr_array, count)
    for p in marginals:
        p = np.asarray(p, dtype=float)
        placed = False
        for g in groups:
            if g[0].size == p.size and np.allclose(g[0], p, atol=tol, rtol=0):
                g[1] += 1
                placed = True
                break
        if not placed:
            groups.append([p, 1])
    return [(g[0], g[1]) for g in groups]
