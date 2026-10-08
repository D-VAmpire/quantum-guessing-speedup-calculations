"""
binning_general.py -- the binned engine for non-commensurate advice (Algorithm 2).

Paper: Sec. 5 ("Binning: the general case") and the stable evaluation of Sec. 4.2.
A measured posterior is generically NOT commensurate, so the per-coordinate
surprisals are snapped onto a common grid eta*Z and the engine runs on the
binned laws.  We bin the *surprisals* (not the probabilities): every sum S~
then lies on eta*Z, with support O(R/eta).

Two chains are carried, because binning breaks Eq. (2):
    count law  nu~_i(k eta) = #{a : s~_i(a) = k eta} / b_i      -> nu~, cumulated into F~
    mass law   pi~_i(k eta) = sum_{a : s~_i(a) = k eta} p_i(a)  -> pi~ (true weights P)
and the estimator is Eq. (7):  f_eta(rho) = ln sum_t pi~(t) A~_rho(t),
with A~_rho(t) built from (F~(t^-), F~(t)) exactly as in Lemma 3.1.

Stable evaluation (Algorithm 1 steps a-c, run per rho as in Algorithm 2 step 3):
  (a) tilt every per-coordinate law, the count chain by phi_rho = -1/(1+rho) and
      the mass chain by phi_rho + 1 (Lemma 4.2: tilting commutes with convolution);
  (b) merge the tilted laws up a balanced tree (repeated squaring for identical
      coordinates), each node renormalized to maximum 1 and truncated: entries
      below TRUNCATION_FLOOR of the node maximum are discarded (Lemma 4.4(ii)
      truncates to a Chernoff window instead; in both cases the computed arrays
      are dominated by the exact ones);
  (c) recover ln nu~, ln F~ and ln pi~ in the log domain, Eq. (6), so the
      e^{-Theta(m)} magnitudes never materialize as floats.
Convolutions use numpy only (direct for short vectors, numpy.fft otherwise), so
results do not depend on which optional packages are installed.  The exact
weighted truncation losses D_nu, D_pi of App. E are returned for the certificate
(certificate.py).

Rounding modes:
  'up'      : s~ = eta (floor(s/eta) + 1), strictly up, 0 < S~ - S <= m eta (Sec. 5);
              required by the certificate (Prop. 6.8).
  'nearest' : s~ = eta round(s/eta); admissible (Sec. 5, footnote), better on
              average; used with Richardson extrapolation for point values.

compute_to_precision() refines eta (halving) and Richardson-extrapolates until
successive extrapolants agree to the target.  This is an empirical convergence
check, not a guarantee; the rigorous bound is the certificate of Prop. 6.8
(certificate.py).  See METHODS.md for how this differs from Algorithm 2's schedule.
"""

from __future__ import annotations
import math
import numpy as np
from common import (group_identical, lnN, mean_renyi, logsumexp, logcumsumexp,
                    log_A_array)

TRUNCATION_FLOOR = 1e-13    # merge nodes discard entries below this fraction of their maximum
DIRECT_MAX = 1 << 16        # convolve directly when len(a) * len(b) is at most this


def phi_rho(rho):
    """Working tilt of the count chain for f(rho), Sec. 4.2: phi_rho = -1/(1+rho)."""
    return -1.0 / (1.0 + rho)


# --------------------------------------------------------------------------
# per-coordinate binned laws
# --------------------------------------------------------------------------

def bin_indices(p, eta, round_mode):
    """Grid indices k with s~_i(a) = k eta for one marginal."""
    v = -np.log(np.asarray(p, dtype=float))
    if round_mode == "up":                   # strictly up: surprisals on the grid move a full eta
        return (np.floor(v / eta) + 1).astype(np.int64)
    if round_mode == "nearest":
        return np.rint(v / eta).astype(np.int64)
    raise ValueError("round_mode must be 'up' or 'nearest'")


def coordinate_laws(p, eta, round_mode):
    """Binned count law (weights 1/b_i) and mass law (weights p_i(a)) on k = 0..kmax."""
    p = np.asarray(p, dtype=float)
    k = bin_indices(p, eta, round_mode)
    nu = np.zeros(int(k.max()) + 1)
    pi = np.zeros(int(k.max()) + 1)
    np.add.at(nu, k, 1.0 / p.size)
    np.add.at(pi, k, p)
    return nu, pi


def predicted_grid(marginals, eta, round_mode):
    """Length of the full grid of S~ = 1 + sum_i max_a k_i(a)."""
    return 1 + sum(int(bin_indices(p, eta, round_mode).max()) for p in marginals)


# --------------------------------------------------------------------------
# tilted, truncated balanced merge (Algorithm 1 steps a-c)
# --------------------------------------------------------------------------
# A node (vec, ln_scale, start) represents the sequence vec[j] * e^{ln_scale}
# at grid index start + j, with max(vec) = 1.

def _convolve(a, b):
    if a.size * b.size <= DIRECT_MAX:
        return np.convolve(a, b)
    n = a.size + b.size - 1
    nfft = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)[:n]


def _trim(vec, ln_scale, start):
    nz = np.flatnonzero(vec)
    return vec[nz[0]:nz[-1] + 1], ln_scale, start + int(nz[0])


def _leaf(law, eta, phi):
    k = np.arange(law.size)
    with np.errstate(divide="ignore"):
        lw = np.log(law) + phi * eta * k
    mx = float(lw.max())
    return _trim(np.exp(lw - mx), mx, 0)


def _merge(x, y, floor):
    c = _convolve(x[0], y[0])
    mx = float(c.max())
    c /= mx
    c[c < floor] = 0.0                       # truncation (also clears FFT round-off noise)
    return _trim(c, x[1] + y[1] + math.log(mx), x[2] + y[2])


def _power(node, M, floor):
    """node convolved with itself M times, by repeated squaring."""
    result, base = None, node
    while M:
        if M & 1:
            result = base if result is None else _merge(result, base, floor)
        M >>= 1
        if M:
            base = _merge(base, base, floor)
    return result


def tilted_chain(laws, eta, phi, floor=TRUNCATION_FLOOR):
    """Merge tilted per-coordinate laws and recover the untilted chain in log domain.

    `laws`: list of (law, multiplicity), each law a vector on k = 0..len-1.
    Returns ln_mu on the full grid k = 0..K (-inf where nothing is kept),
    ln_D = ln of the exact weighted loss D = sum_k (mu - mu_hat)(k) e^{phi k eta}
    (App. E; -inf if nothing was lost), and Lambda = ln sum_k mu(k) e^{phi k eta},
    the analytic CGF assembled per coordinate (Lemma 4.2)."""
    nodes, Lam, K = [], 0.0, 0
    for law, M in laws:
        leaf = _leaf(law, eta, phi)
        Lam += M * (leaf[1] + math.log(float(leaf[0].sum())))
        K += M * (law.size - 1)
        nodes.append(_power(leaf, M, floor))
    while len(nodes) > 1:                        # balanced tree over the coordinate types
        nodes = [_merge(nodes[i], nodes[i + 1], floor) if i + 1 < len(nodes) else nodes[i]
                 for i in range(0, len(nodes), 2)]
    vec, ln_scale, start = nodes[0]
    ln_mu = np.full(K + 1, -np.inf)
    k = np.arange(start, start + vec.size)
    with np.errstate(divide="ignore"):
        ln_mu[k] = np.log(vec) + ln_scale - phi * eta * k     # Eq. (6), first line
    rel = -math.expm1(ln_scale + math.log(float(vec.sum())) - Lam)   # lost fraction of e^Lambda
    ln_D = Lam + math.log(rel) if rel > 0.0 else -np.inf
    return ln_mu, ln_D, Lam


def binned_chains(marginals, eta, rho, round_mode, floor=TRUNCATION_FLOOR):
    """Count and mass chains of the binned law at the working tilts for f(rho).

    Returns dict with ln_nu, ln_pi (full grid), ln_D_nu, ln_D_pi, phi, psi."""
    laws = [(coordinate_laws(p, eta, round_mode), M) for p, M in group_identical(marginals)]
    phi = phi_rho(rho)
    psi = phi + 1.0
    ln_nu, ln_D_nu, _ = tilted_chain([(nu, M) for (nu, _), M in laws], eta, phi, floor)
    ln_pi, ln_D_pi, _ = tilted_chain([(pi, M) for (_, pi), M in laws], eta, psi, floor)
    return {"ln_nu": ln_nu, "ln_pi": ln_pi, "ln_D_nu": ln_D_nu, "ln_D_pi": ln_D_pi,
            "phi": phi, "psi": psi}


def assemble(ln_nu, ln_pi, ln_N, rho):
    """f_eta(rho) of Eq. (7) from the chains; also returns ln F~ and ln A~_rho."""
    lnF = np.minimum(logcumsumexp(ln_nu), 0.0)        # prefix LSE, Eq. (6)
    ln_b = ln_N + lnF
    ln_a = np.concatenate(([-np.inf], ln_b[:-1]))
    lnA = log_A_array(ln_a, ln_b, ln_N + ln_nu, rho)
    keep = np.isfinite(ln_pi)
    f = logsumexp(ln_pi[keep] + lnA[keep])
    return f, lnF, lnA


# --------------------------------------------------------------------------
# the binned estimator
# --------------------------------------------------------------------------

def binning_speedup(marginals, eta, round_mode="nearest", grid_cap=40_000_000,
                    floor=TRUNCATION_FLOOR):
    """Binned estimator s_eta = f_eta(1)/f_eta(1/2) at a fixed bin width eta."""
    ng = predicted_grid(marginals, eta, round_mode)
    if ng > grid_cap:
        raise MemoryError(
            f"grid ~{ng:.2e} points > cap {grid_cap:.0e} at eta={eta:g}; "
            f"use a coarser eta (+Richardson) or exact_commensurate.py.")
    ln_N = lnN(marginals)
    f = {}
    for rho in (1.0, 0.5):
        ch = binned_chains(marginals, eta, rho, round_mode, floor)
        f[rho] = assemble(ch["ln_nu"], ch["ln_pi"], ln_N, rho)[0]
    return {"s": f[1.0] / f[0.5], "f1": f[1.0], "f_half": f[0.5], "eta": eta,
            "grid_points": ng, "method": f"binning({round_mode}, eta={eta:g})"}


def certificate_constant(marginals, s):
    """C(s) = (2 + s)/Hbar_{2/3} of the scaling law, Eq. (8)."""
    return (2.0 + s) / mean_renyi(marginals, 2.0 / 3.0)


def compute_to_precision(marginals, target=1e-6, eta0=0.02,
                         grid_cap=40_000_000, max_refine=14, verbose=False):
    """Binned s at target absolute precision, adaptively (point value only).

    Halves eta from eta0 and Richardson-extrapolates the nearest-rounding
    estimates, 2 s_eta - s_{2 eta}, until successive extrapolants agree to
    `target` or the grid budget is reached.  Returns the extrapolant, the
    EMPIRICAL error estimate (|successive extrapolants|), and the scaling-law
    value C(s) eta for reference.  For a rigorous bound use certificate.certify."""
    def richardson(eta):
        fine = binning_speedup(marginals, eta, "nearest", grid_cap)
        coarse = binning_speedup(marginals, 2.0 * eta, "nearest", grid_cap)
        return 2.0 * fine["s"] - coarse["s"], fine

    eta = eta0
    prev, s_ex, fine = None, None, None
    emp_err = np.inf
    for _ in range(max_refine):
        try:
            s_ex, fine = richardson(eta)
        except MemoryError:
            break
        if prev is not None:
            emp_err = abs(s_ex - prev)
            if verbose:
                print(f"      eta={eta:.2e}  s={s_ex:.8f}  emp_err={emp_err:.1e}")
            if emp_err < target:
                break
        prev = s_ex
        if predicted_grid(marginals, eta / 2, "nearest") > grid_cap:
            break
        eta /= 2.0

    C = certificate_constant(marginals, s_ex)
    return {
        "s": s_ex,
        "s_eta_nearest": fine["s"],
        "eta": eta,
        "empirical_error": emp_err,
        "C_of_s": C,
        "scaling_law_bound": C * eta,
        "grid_points": fine["grid_points"],
        "f1": fine["f1"], "f_half": fine["f_half"],
        "method": "binning (nearest) + Richardson",
    }


if __name__ == "__main__":
    # demo: 256 i.i.d. coordinates with a 5-point binomial posterior
    from math import comb
    binom4 = [comb(4, i) / 16 for i in range(5)]     # Bin(4,1/2)
    r = compute_to_precision([binom4] * 256, target=1e-6)
    print(f"Bin(4,1/2) x256:  s={r['s']:.6f}  (nearest {r['s_eta_nearest']:.6f})  "
          f"eta={r['eta']:.2e}  grid={r['grid_points']}")
