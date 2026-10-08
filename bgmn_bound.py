"""
bgmn_bound.py -- the comparison bound of Bashiri, Glaser, May and Nowakowski (BGMN).

K. Bashiri, T. Glaser, A. May, J. Nowakowski, "Super-Quadratic Quantum Speed-ups
and Guessing Many Likely Keys", EUROCRYPT 2026, LNCS, pp. 457-480,
https://doi.org/10.1007/978-3-032-25291-3_16.  This is the "BGMN Thm 4.3" column
of Table 2.

BGMN bound both runtimes through Arikan's guessing inequality,
    classical  T_C ~ 2^{H_{1/2}(P)},   quantum  T_Q ~ 2^{H_{2/3}(P)/2},
which gives the asymptotic, one-sided estimate
    s_asym = 2 H_{1/2}(P) / H_{2/3}(P) = 2 Hbar_{1/2} / Hbar_{2/3}
(the paper's Eq. (1) to leading order), and their finite-|K| Theorem 4.3
    s >= 2 (H_{1/2}(P) - log(1 + log|K|)) / H_{2/3}(P),
with the Arikan slack charged to the numerator.

Base convention: BGMN write everything in base 2.  Converting the entropies to
nats multiplies numerator and denominator by ln 2 and turns log2(1 + log2|K|)
into ln(1 + log2|K|): the INNER log stays base 2.  Everything here is in nats.
Both forms are lower bounds on the exact exponent computed by the engines.
"""

from __future__ import annotations
import numpy as np
from common import total_renyi, lnN


def bgmn_bound(marginals):
    """BGMN Theorem 4.3 bound and its asymptotic form, plus their ingredients."""
    m = len(marginals)
    H_half = total_renyi(marginals, 0.5)      # H_{1/2}(P), nats
    H_23 = total_renyi(marginals, 2.0 / 3.0)  # H_{2/3}(P), nats
    lK = lnN(marginals)                       # ln|K|
    log2K = lK / np.log(2.0)                  # log2|K|  (inner log, base 2)
    return {
        "s_bgmn": 2.0 * (H_half - np.log(1.0 + log2K)) / H_23,   # Thm 4.3
        "s_bgmn_asymptotic": 2.0 * H_half / H_23,
        "Hbar_half": H_half / m,
        "Hbar_23": H_23 / m,
        "H_half_total": H_half,
        "H_23_total": H_23,
        "lnN": lK,
    }


if __name__ == "__main__":
    for d in (0.01, 0.05, 0.10, 0.20):
        r = bgmn_bound([[1.0 - d, d]] * 128)
        print(f"Ber(1-{d}) x128:  BGMN Thm 4.3 = {r['s_bgmn']:.5f}   "
              f"asymptotic = {r['s_bgmn_asymptotic']:.5f}")
