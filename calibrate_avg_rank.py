"""calibrate_avg_rank.py -- calibration of the synthetic Keccak (SASCA) posteriors (Table 2, block B; Sec. 7 (B)).

Maillard et al. (SECRYPT 2024) run each attack 50 times and report the AVERAGE key rank
on a log2 scale.  Table 2, block B, fits an i.i.d. Ber(q) advice on m bits such that 50
simulated runs reproduce the published average 2^T in the median:

    median over experiments of  log2( (1/50) * sum_{i=1}^{50} G(X_i) )  =  T,
    X_1..X_50 ~ P i.i.d. (the correct key of each run drawn from the posterior),

estimated by Monte Carlo (common random numbers, fixed seeds) and solved by bisection.
The fitted q are stored in distributions.SHA3_AVG_RANK_FITS.  Two alternative readings
are provided for comparison (sensitivity table in VERIFICATION.md):

    "EG"  log2 E_P[G] = T        (the calibration of arXiv v1; far too confident, see VERIFICATION.md)
    "R1"  E_P[log2 G] = T        (if the published averages were taken over log2 ranks)

and two heterogeneous families with the 50-run fit ("U": uncertainty on k bits only,
"V": half of the bits c times as error-prone).

Products of a few Bernoulli TYPES are evaluated exactly at the level structure (exact
binomial counts, float log-domain level sums with Euler-Maclaurin block sums); validated
against the Arb certificate (certify_table2_AB.py) to ~1e-15 and against brute force.

Usage:  python3 calibrate_avg_rank.py          # the sensitivity table of VERIFICATION.md (~15 min)
        python3 calibrate_avg_rank.py quick    # only the four fits of Table 2 (~6 min)
Requires numpy only.
"""
from __future__ import annotations
import sys
from math import comb, log, log1p, expm1, sqrt
import numpy as np

LN2 = log(2.0)
A0 = 4000                      # ranks below A0 are summed directly
RUNS = 50                      # runs per published average [Maillard sec 6.2]
REPS, CHUNKS, SEED0 = 50_000, 8, 0   # 4e5 simulated experiments; batch k uses seed SEED0 + k


# ----------------------------------------------------------------------------- exact levels
def _lse(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))


def _levels(types):
    """types: [(n, q)], n bits correct w.p. q. Returns flat-level arrays in rank order."""
    lp = np.zeros(1)
    cnt = np.ones(1)
    for n, q in types:
        j = np.arange(n + 1)
        lpj = (n - j) * log(q) + j * log1p(-q)
        cj = np.array([float(comb(n, int(k))) for k in j])
        lp = (lp[:, None] + lpj[None, :]).ravel()
        cnt = (cnt[:, None] * cj[None, :]).ravel()
    order = np.argsort(-lp, kind="stable")
    lp, cnt = lp[order], cnt[order]
    b = np.cumsum(cnt)
    a = np.concatenate([[0.0], b[:-1]])
    return lp, cnt, a, b, order


def _direct(a, b, f):
    r = np.arange(int(a) + 1, int(b) + 1, dtype=float)
    return float(np.sum(f(r)))


def _em_setup(a, cnt, b, f):
    tot = 0.0
    if a < A0:
        hi = min(b, A0)
        tot += _direct(a, hi, f)
        if b <= A0:
            return tot, None, None
        a_eff, cnt_eff = float(A0), b - A0
    else:
        a_eff, cnt_eff = a, cnt
    x = log1p(-cnt_eff / b) if cnt_eff < 0.5 * b else log(a_eff) - log(b)
    return tot, a_eff, x


def _S_sqrt(a, cnt, b):
    tot, a_eff, x = _em_setup(a, cnt, b, np.sqrt)
    if a_eff is None:
        return tot
    return (tot + (2.0 / 3.0) * b ** 1.5 * (-expm1(1.5 * x)) + 0.5 * sqrt(b) * (-expm1(0.5 * x))
            + (1.0 / sqrt(b) - 1.0 / sqrt(a_eff)) / 24.0)


def _S_log(a, cnt, b):
    tot, a_eff, x = _em_setup(a, cnt, b, np.log)
    if a_eff is None:
        return tot
    return tot + (b - a_eff) * (log(b) - 1.0) - a_eff * x - 0.5 * x + (1.0 / 12.0) * (1.0 / b - 1.0 / a_eff)


def moments(types, want_log=False):
    """ln E[G], ln E[sqrt G], s, and optionally E[ln G], exactly at the level structure."""
    lp, cnt, a, b, _ = _levels(types)
    lnEG = _lse(lp + np.log(cnt) + np.log((a + b + 1.0) / 2.0))
    small = a < A0
    big = ~small
    ab, cb, bb = a[big], cnt[big], b[big]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(cb < 0.5 * bb, np.log1p(-cb / bb), np.log(ab) - np.log(bb))
    S12 = np.empty_like(a)
    S12[big] = ((2.0 / 3.0) * bb ** 1.5 * (-np.expm1(1.5 * x)) + 0.5 * np.sqrt(bb) * (-np.expm1(0.5 * x))
                + (1.0 / np.sqrt(bb) - 1.0 / np.sqrt(ab)) / 24.0)
    idx = np.nonzero(small)[0]
    S12[idx] = [_S_sqrt(a[i], cnt[i], b[i]) for i in idx]
    lnEH = _lse(lp + np.log(S12))
    out = {"lnEG": lnEG, "lnEH": lnEH, "s": lnEG / lnEH}
    if want_log:
        SL = np.empty_like(a)
        SL[big] = (bb - ab) * (np.log(bb) - 1.0) - ab * x - 0.5 * x + (1.0 / 12.0) * (1.0 / bb - 1.0 / ab)
        SL[idx] = [_S_log(a[i], cnt[i], b[i]) for i in idx]
        with np.errstate(divide="ignore"):
            out["ElnG"] = float(np.exp(_lse(lp + np.log(SL))))
    return out


# ----------------------------------------------------------------------------- 50-run statistic
class _Sampler:
    def __init__(self, types):
        lp, cnt, a, b, order = _levels(types)
        self.pos = np.empty_like(order)
        self.pos[order] = np.arange(len(order))      # flat (j_1, ..., j_T) index -> rank-order position
        self.a, self.c = a, cnt
        self.cdfs, self.dims = [], []
        for n, q in types:
            j = np.arange(n + 1)
            lpmf = np.array([log(comb(n, int(k))) for k in j]) + j * log1p(-q) + (n - j) * log(q)
            pmf = np.exp(lpmf - lpmf.max())
            cdf = np.cumsum(pmf)
            self.cdfs.append(cdf / cdf[-1])
            self.dims.append(n + 1)

    def sums(self, reps, seed, runs):
        rng = np.random.default_rng(seed)
        flat = np.zeros((reps, runs), dtype=np.int64)
        for cdf, d in zip(self.cdfs, self.dims):
            jt = np.minimum(np.searchsorted(cdf, rng.random((reps, runs))), d - 1)
            flat = flat * d + jt
        p = self.pos[flat]
        G = self.a[p] + 1.0 + np.floor(rng.random((reps, runs)) * self.c[p])
        return G.sum(axis=1)


def avg_rank_stat(types, runs=RUNS, reps=REPS, chunks=CHUNKS, seed0=SEED0):
    """median over experiments of log2(mean rank of `runs` runs)."""
    S = _Sampler(types)
    sums = np.concatenate([S.sums(reps, seed0 + k, runs) for k in range(chunks)])
    return float(np.log2(np.median(sums) / runs))


def _bisect(fun, target, lo, hi, iters=60, tol=1e-12):
    flo = fun(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = fun(mid)
        if (fm > target) == (flo > target):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------- fits
def fit_avg_rank(m, T, runs=RUNS, reps=REPS, chunks=CHUNKS, seed0=SEED0):
    """q of the i.i.d. Ber(q)^m advice whose 50-run average rank is 2^T in the median (Table 2)."""
    return _bisect(lambda q: avg_rank_stat([(m, q)], runs, reps, chunks, seed0), T, 0.5 + 1e-9, 1 - 1e-12)


def fit_EG(m, T):
    """q with log2 E[G] = T (arXiv v1 calibration; comparison column 1 in VERIFICATION.md)."""
    return _bisect(lambda q: moments([(m, q)])["lnEG"] / LN2, T, 0.5 + 1e-12, 1 - 1e-15, iters=200, tol=1e-15)


def fit_R1(m, T):
    """q with E[log2 G] = T (comparison column 3 in VERIFICATION.md)."""
    return _bisect(lambda q: moments([(m, q)], want_log=True)["ElnG"] / LN2, T, 0.5 + 1e-12, 1 - 1e-15,
                   iters=200, tol=1e-15)


def fit_U(m, T, k, **kw):
    """uncertainty on k of the m bits only (the rest known), 50-run fit."""
    return _bisect(lambda q: avg_rank_stat([(k, q)], **kw), T, 0.5 + 1e-9, 1 - 1e-12)


def fit_V(m, T, c, **kw):
    """half of the bits c times as error-prone as the other half, 50-run fit; returns eps."""
    h = m // 2
    hi = min(0.5 / c, 0.5) - 1e-12
    return _bisect(lambda e: avg_rank_stat([(h, 1 - c * e), (m - h, 1 - e)], **kw), T, 1e-12, hi)


ROWS = [("ML-KEM coin 256b", 256, 69), ("ML-KEM coin 256b", 256, 128),
        ("ML-DSA rho' 512b", 512, 69), ("ML-DSA rho' 512b", 512, 128)]

if __name__ == "__main__":
    quick = "quick" in sys.argv[1:]
    for label, m, T in ROWS:
        q = fit_avg_rank(m, T)
        line = f"{label} 2^{T}: 50-run fit q={q!r} s={moments([(m, q)])['s']:.4f}"
        if not quick:
            line += f" | EG s={moments([(m, fit_EG(m, T))])['s']:.4f} | R1 s={moments([(m, fit_R1(m, T))])['s']:.4f}"
            kw = dict(seed0=100)
            qU = fit_U(m, T, (3 * m) // 4, **kw)
            e = fit_V(m, T, 10.0, **kw)
            line += (f" | U(3m/4) s={moments([((3 * m) // 4, qU)])['s']:.4f}"
                     f" | V(10) s={moments([(m // 2, 1 - 10 * e), (m - m // 2, 1 - e)])['s']:.4f}")
        print(line, flush=True)
