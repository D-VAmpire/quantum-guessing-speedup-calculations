"""
lclt_diagnostics.py -- computable quantities of the local CLT (Lemma 6.3, App. C)
and the diagnostics of Algorithm 2, step 6.

Under the binned tilt U~_tau, coordinate i puts weight p_{i,a}^(tau) proportional
to exp(tau s~_i(a)) on its binned surprisals s~_i(a) (raw surprisals for eta = 0).
With Y_i the centered surprisal of coordinate i under these weights,

    sigma_i^2 = Var Y_i,  beta_i = E|Y_i|^3,  sigma~^2 = sum_i sigma_i^2,  beta = sum_i beta_i,
    delta_1 = min{ 3 sigma~^2 / (2 beta),  sqrt(3) / (2 max_i sigma_i) },
    Lambda_eta = min_{delta_1 <= |t| <= pi/eta} sum_i sum_{a<b} 4 p_a p_b sin^2(t Delta_iab / 2)   (Eq. (10)),

the cubic remainder R(t) = ln psi(t) + sigma~^2 t^2 / 2 of the characteristic
function psi = prod_i psi_i (principal branch per coordinate), and the Gaussian
tail term sqrt(2/pi) Phibar(sigma~ delta_1) of App. C.  The empty-lattice fraction
is the share of grid points of S~ in the bulk of the uniform law that carry no keys.

The pairwise sums are formed explicitly, which is meant for small alphabets.
"""

from __future__ import annotations
import math
import numpy as np
from common import group_identical


def _coordinate(p, eta, tau):
    s = -np.log(np.asarray(p, dtype=float))
    if eta > 0:
        s = eta * (np.floor(s / eta) + 1.0)          # strict upward rounding (Sec. 5)
    w = np.exp(tau * (s - s.min()))
    return s, w / w.sum()


def scalars(marginals, eta, tau):
    """sigma~^2, beta, max_i sigma_i, delta_1 under the binned tilt U~_tau."""
    S2 = B3 = 0.0
    smax = 0.0
    for p, M in group_identical(marginals):
        s, w = _coordinate(p, eta, tau)
        y = s - np.dot(w, s)
        v = float(np.dot(w, y * y))
        S2 += M * v
        B3 += M * float(np.dot(w, np.abs(y) ** 3))
        smax = max(smax, math.sqrt(v))
    d1 = min(3.0 * S2 / (2.0 * B3), math.sqrt(3.0) / (2.0 * smax))
    return {"sigma2": S2, "beta": B3, "sigma_max": smax, "delta1": d1}


def lattice_span_exponent(marginals, eta, tau, step=1e-3):
    """Lambda_eta of Eq. (10) on the band delta_1 <= |t| <= pi/eta, and the minimizing t.

    Dense grid of spacing `step`, then a local refinement around the grid minimum."""
    d1 = scalars(marginals, eta, tau)["delta1"]
    top = math.pi / eta
    if d1 >= top:
        return math.inf, None
    terms = []                                   # (multiplicity * 4 p_a p_b, gap)
    for p, M in group_identical(marginals):
        s, w = _coordinate(p, eta, tau)
        for a in range(s.size):
            for b in range(a + 1, s.size):
                terms.append((M * 4.0 * w[a] * w[b], abs(s[a] - s[b])))
    coef = np.array([c for c, _ in terms])
    gap = np.array([g for _, g in terms])

    def lam(t):
        t = np.atleast_1d(t)
        return (coef[None, :] * np.sin(np.outer(t, gap) / 2.0) ** 2).sum(axis=1)

    best_t, best = None, math.inf
    grid = np.arange(d1, top, step)
    for chunk in np.array_split(grid, max(1, grid.size // 200_000)):
        v = lam(chunk)
        i = int(np.argmin(v))
        if v[i] < best:
            best, best_t = float(v[i]), float(chunk[i])
    fine = np.linspace(max(d1, best_t - step), min(top, best_t + step), 4001)
    v = lam(fine)
    i = int(np.argmin(v))
    return float(v[i]), float(fine[i])


def cubic_remainder_ratio(marginals, eta, tau, t_min=0.2, n=20001):
    """max over t in [t_min, delta_1] of |R(t)| / (beta |t|^3 / 6) and the maximizing t.

    t_min keeps the cancellation in R away from t = 0, where R = O(t^3)."""
    sc = scalars(marginals, eta, tau)
    tt = np.linspace(t_min, sc["delta1"], n)
    lnpsi = np.zeros(tt.size, dtype=complex)
    for p, M in group_identical(marginals):
        s, w = _coordinate(p, eta, tau)
        y = s - np.dot(w, s)
        psi_i = (w[None, :] * np.exp(1j * np.outer(tt, y))).sum(axis=1)
        lnpsi += M * np.log(psi_i)
    R = lnpsi + sc["sigma2"] * tt ** 2 / 2.0
    ratio = np.abs(R) / (sc["beta"] * tt ** 3 / 6.0)
    i = int(np.argmax(ratio))
    return float(ratio[i]), float(tt[i]), sc["delta1"]


def gaussian_tail(marginals, eta, tau):
    """sqrt(2/pi) Phibar(sigma~ delta_1), the Gaussian-tail term of App. C."""
    sc = scalars(marginals, eta, tau)
    x = math.sqrt(sc["sigma2"]) * sc["delta1"]
    return math.sqrt(2.0 / math.pi) * 0.5 * math.erfc(x / math.sqrt(2.0))


def empty_lattice_fraction(marginals, eta, bulk=(0.01, 0.99)):
    """Share of grid points of S~ inside the bulk of the uniform law (between the
    given quantiles) that carry no keys at all (exact support by integer counting)."""
    support = np.array([1], dtype=np.int64)
    counts = np.array([1.0])
    for p in marginals:
        s = -np.log(np.asarray(p, dtype=float))
        k = (np.floor(s / eta) + 1).astype(np.int64)
        ind = np.zeros(int(k.max()) + 1, dtype=np.int64)
        cnt = np.zeros(int(k.max()) + 1)
        np.add.at(ind, k, 1)
        np.add.at(cnt, k, 1.0 / k.size)
        support = (np.convolve(support, ind) > 0).astype(np.int64)
        counts = np.convolve(counts, cnt)
    F = np.cumsum(counts) / counts.sum()
    lo, hi = int(np.searchsorted(F, bulk[0])), int(np.searchsorted(F, bulk[1]))
    window = support[lo:hi + 1]
    return 1.0 - window.mean()
