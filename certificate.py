"""
certificate.py -- the computable certificate of Proposition 6.8 (proofs in App. E).

Run the binned engine with strict upward rounding and bound |s_eta - s| from the
engine's own arrays.  For each level t of S~ (grid index k, t = k eta):

    D(t)     = max_{u in {t - m eta, ..., t - eta}} ln( F~(u + m eta) / F~(u) )
               (infinite if some F~(u) = 0),
    lo_rho(t) = max{ (N F~(t - m eta))^rho,  1,  A~_rho(t) e^{-rho D(t)} },
    hi_rho(t) = min{ (N F~(t + m eta))^rho,  e^{rho t},  A~_rho(t) e^{+rho D(t)} },
    sum_t pi~ lo_rho <= e^{f(rho)} <= sum_t pi~ hi_rho                     (Eq. (12)),
    B = (E_1 + s_eta E_{1/2}) / (f_eta(1/2) - E_{1/2})                     (Eq. (13)),

valid whenever f_eta(1/2) > E_{1/2}.  The bound uses the engine assumptions and
strict rounding only: no local CLT, lattice-span condition, eps_0, saddlepoint
expansion or Arikan scale.

Truncated merges (App. E, "Truncated merges").  The engine (binning_general.py)
discards merge-tree entries below a floor, so its arrays are dominated by the
exact binned laws, nu^ <= nu~ and pi^ <= pi~, with exactly known weighted losses
D_nu = sum_t (nu~ - nu^)(t) e^{phi t} and D_pi = sum_t (pi~ - pi^)(t) e^{psi t}
(phi = phi_rho < 0, psi = phi_rho + 1 > 0).  They give the enclosure
    L(t) := F^(t) <= F~(t) <= U(t) := min(1, F^(t) + D_nu e^{|phi| t}),
which enters the lower candidates through L (also inside A and D) and the upper
candidates through U, and the additive term D_pi max_t e^{-psi t} hi_rho(t) on the
upper end of Eq. (12).  Without truncation both losses vanish and L = U = F^.

Eq. (13) is a theorem about exact arithmetic; the arrays here are floating point,
so this certifies the binning error of the exact estimator, and the floating-point
error of Lemma 4.5 comes on top (the paper's Sec. 6.4: no rigorous end-to-end
enclosure without interval arithmetic).
"""

from __future__ import annotations
import numpy as np
from common import group_identical, lnN, logsumexp, logcumsumexp, log_A_array
from binning_general import (binned_chains, predicted_grid, TRUNCATION_FLOOR)


def _window_max(g, w):
    """out[k] = max(g[k-w : k]) over indices >= 0 (empty window -> -inf)."""
    x = np.concatenate((np.full(w, -np.inf), g))
    tab, span = x, 1
    while 2 * span <= w:
        tab = np.maximum(tab[:-span], tab[span:])
        span *= 2
    G = g.size
    return np.maximum(tab[:G], tab[w - span:w - span + G])


def _at(lnF, idx):
    """ln F~ at grid indices idx: F~ = 0 below the grid, F~ = 1 above it."""
    out = np.zeros(idx.shape)
    inside = (idx >= 0) & (idx < lnF.size)
    out[idx < 0] = -np.inf
    out[inside] = lnF[idx[inside]]
    return out


def _level_terms(lnF, ln_n, ln_N, m, rho, side):
    """ln A~_rho and the window D of Prop. 6.8 from one end of the F~ enclosure.

    side='lo': A is capped at (N F(t))^rho, so A e^{-rho D} <= (N F(t - m eta))^rho;
    side='hi': A is floored at (N F(t^-))^rho, so A e^{+rho D} >= (N F(l + m eta))^rho
    for every window cell l.  Both keep the A-D candidates valid for this end."""
    k = np.arange(lnF.size)
    ln_b = ln_N + lnF
    ln_a = np.concatenate(([-np.inf], ln_b[:-1]))
    lnA = log_A_array(ln_a, ln_b, np.minimum(ln_n, ln_b), rho)
    if side == "lo":
        lnA = np.minimum(lnA, rho * ln_b)
    else:
        lnA = np.maximum(lnA, rho * ln_a)
    with np.errstate(invalid="ignore"):
        g = _at(lnF, k + m) - lnF
    g = np.where(np.isfinite(lnF), g, np.inf)          # F~(u) = 0 -> D infinite
    D = _window_max(g, m)
    return lnA, D


def certify(marginals, eta, grid_cap=40_000_000, floor=TRUNCATION_FLOOR):
    """Certified bound B on |s_eta - s| (Prop. 6.8) at bin width eta.

    Returns s_eta, B, the per-moment bounds E_1, E_half, f_eta(1), f_eta(1/2),
    the enclosure [s_eta - B, s_eta + B], the truncation losses, and per rho the
    ratio R of the mass-chain term (mass_term_ratio): relative to the upper end
    of Eq. (12), that term is the lost tilted mass times R."""
    m = len(marginals)
    G = predicted_grid(marginals, eta, "up")
    if G > grid_cap:
        raise MemoryError(f"grid ~{G:.2e} > cap at eta={eta:g}")
    ln_N = lnN(marginals)
    k = np.arange(G)
    t = k * eta
    out = {}
    for rho in (1.0, 0.5):
        ch = binned_chains(marginals, eta, rho, "up", floor)
        ln_nu, ln_pi = ch["ln_nu"], ch["ln_pi"]
        phi, psi = ch["phi"], ch["psi"]
        # enclosure of F~ (App. E): lower end L = F^, upper end U
        lnL = np.minimum(logcumsumexp(ln_nu), 0.0)
        lnU = np.minimum(np.logaddexp(lnL, ch["ln_D_nu"] + abs(phi) * t), 0.0)
        # block sizes n = N (F(t) - F(t^-)): from the chain for L; for U the
        # added D_nu e^{|phi| t} contributes D_nu e^{|phi| t} (1 - e^{-|phi| eta})
        ln_n_L = ln_N + ln_nu
        with np.errstate(divide="ignore"):
            extra = ch["ln_D_nu"] + abs(phi) * t + np.log(-np.expm1(-abs(phi) * eta))
        ln_n_U = ln_N + np.logaddexp(ln_nu, extra)
        lnA_L, D_L = _level_terms(lnL, ln_n_L, ln_N, m, rho, "lo")
        lnA_U, D_U = _level_terms(lnU, ln_n_U, ln_N, m, rho, "hi")
        # the estimator itself, Eq. (7), from the computed arrays
        lnA_est = log_A_array(np.concatenate(([-np.inf], ln_N + lnL[:-1])),
                              ln_N + lnL, ln_n_L, rho)
        # candidates of Eq. (11); the D-candidates are dropped where D is infinite
        with np.errstate(invalid="ignore"):
            c1 = rho * (ln_N + _at(lnL, k - m))
            c3 = np.where(np.isfinite(D_L), lnA_L - rho * D_L, -np.inf)
            d1 = rho * (ln_N + _at(lnU, k + m))
            d3 = np.where(np.isfinite(D_U), lnA_U + rho * D_U, np.inf)
        ln_lo = np.maximum(np.maximum(c1, 0.0), c3)
        d2 = rho * t
        ln_hi = np.minimum(np.minimum(d1, d2), d3)

        keep = np.isfinite(ln_pi)
        f_eta = logsumexp(ln_pi[keep] + lnA_est[keep])
        lnLsum = logsumexp(ln_pi[keep] + ln_lo[keep])
        lnHsum = logsumexp(ln_pi[keep] + ln_hi[keep])
        # R = peak / average of g(t) = e^{-psi t} hi_rho(t) under the tilted mass
        # chain; the mass-chain term is (lost tilted mass) x R of the upper end
        ln_R = np.max(ln_hi - psi * t) - (lnHsum - logsumexp(ln_pi[keep] + psi * t[keep]))
        if np.isfinite(ch["ln_D_pi"]):                  # mass-chain term of App. E
            lnHsum = np.logaddexp(lnHsum, ch["ln_D_pi"] + np.max(ln_hi - psi * t))
        E = max(f_eta - lnLsum, lnHsum - f_eta, 0.0)
        out[rho] = {"f_eta": f_eta, "lnL": lnLsum, "lnH": lnHsum, "E": E,
                    "ln_D_nu": ch["ln_D_nu"], "ln_D_pi": ch["ln_D_pi"], "R": float(np.exp(ln_R))}

    f1, fh = out[1.0]["f_eta"], out[0.5]["f_eta"]
    E1, Eh = out[1.0]["E"], out[0.5]["E"]
    s_eta = f1 / fh
    B = (E1 + s_eta * Eh) / (fh - Eh) if fh > Eh else float("inf")
    return {"s": s_eta, "B": B, "E_1": E1, "E_half": Eh,
            "f1": f1, "f_half": fh, "eta": eta, "grid_points": G,
            "s_lo": s_eta - B, "s_hi": s_eta + B,
            "truncation": {rho: (out[rho]["ln_D_nu"], out[rho]["ln_D_pi"]) for rho in out},
            "mass_term_ratio": {rho: out[rho]["R"] for rho in out},
            "method": f"certified binning (eta={eta:g})"}


if __name__ == "__main__":
    from common import brute_force_speedup
    marg = [[0.6, 0.25, 0.15]] * 8
    s = brute_force_speedup(marg)["s"]
    for eta in (0.2, 0.05, 0.01):
        c = certify(marg, eta)
        print(f"eta={eta:<5} s_eta={c['s']:.6f}  B={c['B']:.3e}  "
              f"[{c['s_lo']:.4f}, {c['s_hi']:.4f}]  contains exact {s:.6f}: "
              f"{c['s_lo'] <= s <= c['s_hi']}")
