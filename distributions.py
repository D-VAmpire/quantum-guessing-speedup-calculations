"""
distributions.py -- advice models of the paper's Sec. 7 and Table 2 (data only).

Nothing here computes a speedup.  A "marginal" is a probability vector over one
coordinate's alphabet; an "instance" is a list of marginals.  Parameter sources
(sections and tables are cited at each parameter):

  [Seed]     V. Gauthier-Umana, A. M. Ochoa-Toro, R. Villanueva-Polanco, IEEE Access 14
             (2026) 17264-17277, doi:10.1109/ACCESS.2026.3659061   (cold boot on seeds)
  [Banegas]  G. Banegas, R. Villanueva-Polanco, Cryptogr. Commun. 17(2) (2025) 311-335,
             doi:10.1007/s12095-022-00625-z                        (cold boot on cipher keys)
  [Maillard] J. Maillard, T. Hiscock, M. Lecomte, C. Clavier, SECRYPT 2024, 518-527,
             doi:10.5220/0012787200003767                          (Keccak SASCA)
"""

from __future__ import annotations
import numpy as np

# ==========================================================================
# Table 2, block A -- cold boot (asymmetric channel) on seeds and cipher keys
# ==========================================================================

def bernoulli(correct):
    """One binary coordinate with P(correct symbol) = correct."""
    return [correct, 1.0 - correct]


def coldboot_symmetric(delta, W):
    """Symmetric channel: each of W bits correct w.p. 1 - delta  [Seed Sec. IV.1.b].
    Used only for the symmetric comparison in Sec. 7 (2.532 at W = m = 128, beta = 0.05)."""
    return [bernoulli(1.0 - delta)] * W


def coldboot_asymmetric(alpha, beta, W):
    """Asymmetric cold-boot channel  [Seed Sec. IV.1.c; Banegas Sec. 2.2].

    alpha = P(0 -> 1), beta = P(1 -> 0).  Under a uniform prior the dumped bit is 1
    with probability f1 = (1 - beta + alpha)/2, and conditioning on it gives two
    posterior types.  Sec. 7 (A) reports the expected composition: the first
    type on n1 = round(W f1) of the W coordinates."""
    D1 = (1.0 - beta) + alpha          # 2 P(observe 1)
    D0 = beta + (1.0 - alpha)          # 2 P(observe 0)
    f1 = 0.5 * (1.0 - beta) + 0.5 * alpha
    post_given1 = [(1.0 - beta) / D1, alpha / D1]  # [P(s*=1|~1), P(s*=0|~1)]
    post_given0 = [beta / D0, (1.0 - alpha) / D0]  # [P(s*=1|~0), P(s*=0|~0)]
    n1 = int(round(f1 * W))
    return [post_given1] * n1 + [post_given0] * (W - n1)


COLDBOOT_ALPHA = 0.001     # [Seed Sec. VII.A], [Banegas Sec. 2.2]

# Key lengths [Banegas Table 3].  At equal length the advice is identical, so the
# 128-bit ciphers (GIFT-128, LowMC-128, Picnic-L1) coincide with AES-128 and
# Picnic-L3/L5 with 192/256-bit keys; Table 2 lists AES-128, AES-256, PRESENT-80.
SYMMETRIC_CIPHER_KEYBITS = {
    "AES-128": 128, "AES-256": 256, "PRESENT-80": 80,
    "GIFT-128": 128, "LowMC-128": 128,
    "Picnic-L1": 128, "Picnic-L3": 192, "Picnic-L5": 256,
}


def cipher_coldboot(cipher, beta, asymmetric=True, alpha=COLDBOOT_ALPHA):
    """Cold-boot advice on a block-cipher key of the given cipher."""
    n = SYMMETRIC_CIPHER_KEYBITS[cipher]
    return (coldboot_asymmetric(alpha, beta, n) if asymmetric
            else coldboot_symmetric(beta, n))


# ==========================================================================
# Table 2, block B -- synthetic Bernoulli models fitted to Keccak SASCA ranks
# ==========================================================================
# [Maillard] report the residual key rank of their SASCA, averaged over 50 attack
# runs (log2 scale).  Sec. 7 (B) fits i.i.d. Ber(q) advice on the seed bits so that
# 50 simulated runs reproduce the reported average 2^T in the median.  The fits are
# Monte Carlo (calibrate_avg_rank.fit_avg_rank: 4e5 simulated experiments in 8
# batches, seeds 0-7, bisection; MC standard error of s below 1e-4) and are stored,
# so Table 2 does not depend on numpy's random-number stream.  A stored double is
# an exact binary rational, which certify_table2_AB.py uses as is.
#
# The classifier accuracies of [Maillard, Table 3] are inputs to belief
# propagation, not its output, and must not be used as the advice (Sec. 7 (B)).

KYBER_COIN_BITS = 256           # ML-KEM encryption coin   [Maillard Sec. 7.1]
DILITHIUM_RHO_BITS = 512        # ML-DSA masking seed rho'  [Maillard Sec. 7.2]

SHA3_AVG_RANK_FITS = {          # (seed bits, T) -> q
    (256, 69): 0.9707246174519626,
    (256, 128): 0.9167710427724212,
    (512, 69): 0.9888678286542063,
    (512, 128): 0.9712913990757392,
}


def sha3_sasca_from_avg_rank(log2_rank, seed_len, refit=False):
    """i.i.d. Ber(q) seed advice fitted to a reported average rank 2^log2_rank.

    Returns (instance, q).  Uses the stored fit for the Table 2 operating points;
    otherwise (or with refit=True) runs calibrate_avg_rank.fit_avg_rank (minutes)."""
    key = (seed_len, log2_rank)
    if refit or key not in SHA3_AVG_RANK_FITS:
        from calibrate_avg_rank import fit_avg_rank
        q = fit_avg_rank(seed_len, log2_rank)
    else:
        q = SHA3_AVG_RANK_FITS[key]
    return [bernoulli(q)] * seed_len, q


# ==========================================================================
# Table 2, block C -- AES template attack (256-ary posteriors; binned engine)
# ==========================================================================
# Leakage model: L = phi(Sbox(pt XOR k)) + N(0, sigma^2) per key byte, with phi the
# Hamming weight (HW) or the identity, SNR = Var[phi(Sbox(X))]/sigma^2 (= 2/sigma^2
# for HW).  Sixteen independent byte posteriors form the advice.  Posteriors are
# floored at 1e-10 and renormalized: this bounds the surprisal range (far-off
# hypotheses otherwise get surprisals of order 255^2/(2 sigma^2)) at a mass
# perturbation of about 2.6e-8.  The instances behind the paper's numbers are
# stored in data/template_posteriors.npz (template_data.py).

AES_SBOX = bytes.fromhex(
    "637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0"
    "b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b275"
    "09832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cf"
    "d0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2"
    "cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdb"
    "e0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08"
    "ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9e"
    "e1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16")

_HW = np.array([bin(i).count("1") for i in range(256)], dtype=float)


def template_aes_byte_posterior(sigma, n_traces=1, model="HW", rng=None):
    """One key-byte posterior (256-vector) from n_traces single-byte leakages."""
    rng = rng or np.random.default_rng(0)
    kt = int(rng.integers(256))                       # true key byte
    sbox = np.frombuffer(AES_SBOX, dtype=np.uint8).astype(int)
    phi = _HW[sbox] if model == "HW" else sbox.astype(float)   # HW or identity
    loglik = np.zeros(256)
    for _ in range(n_traces):
        pt = int(rng.integers(256))
        inter_true = sbox[pt ^ kt]
        L = phi[inter_true] + rng.normal(0.0, sigma)
        pred = phi[np.bitwise_xor(pt, np.arange(256))]         # per-hypothesis phi
        loglik += -(L - pred) ** 2 / (2.0 * sigma ** 2)
    loglik -= loglik.max()
    post = np.exp(loglik)
    return post / post.sum()


def template_aes_instance(sigma, n_bytes=16, n_traces=1, model="HW", seed=0,
                          floor=1e-10):
    """16-byte AES key advice: independent per-byte template posteriors."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_bytes):
        p = template_aes_byte_posterior(sigma, n_traces, model, rng)
        if floor:
            p = np.maximum(p, floor)
            p = p / p.sum()
        out.append(p)
    return out


def sigma_from_snr(snr, model="HW"):
    """Map an SNR to the Gaussian noise sigma: SNR = Var(phi)/sigma^2."""
    sbox = np.frombuffer(AES_SBOX, dtype=np.uint8).astype(int)
    phi = _HW[sbox] if model == "HW" else sbox.astype(float)
    return float(np.sqrt(np.var(phi) / snr))
