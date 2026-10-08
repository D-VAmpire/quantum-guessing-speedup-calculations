"""
table2_config.py -- Table 2 of the paper: rows, model parameters, printed values.

Single source of truth for table2.py, certify_table2_AB.py, paper_numbers.py and
test_vs_budroni_martensson.py.  Columns: s and the bound of BGMN Theorem 4.3
(bgmn_bound.py), both printed to three decimals.

Block A  cold boot, asymmetric channel alpha = P(0->1) = 0.001, decay rate beta = P(1->0).
         Expected composition: n1 = round(W f1) coordinates of the type "observed 1",
         f1 = (1 - beta + alpha)/2 (Sec. 7 (A)).  Exact (finite-rank engine).
Block B  i.i.d. Ber(q) on the seed bits, q fitted so that 50 simulated runs reproduce
         the reported average rank 2^T in the median (Sec. 7 (B)); q is stored in
         distributions.SHA3_AVG_RANK_FITS.  Exact within the model.
Block C  AES template attack, 16 bytes over 256 values, single trace, HW leakage at the
         illustrative SNRs 1 and 5 (sigma = sqrt(2/SNR)), posteriors floored at 1e-10; three simulated
         posteriors (seeds 0-2, stored in data/template_posteriors.npz).
         s: mean of the three binned point values (nearest rounding, Richardson from
         eta0 = 0.02, halving, until successive extrapolants agree to 1e-6);
         bound: mean of the three bounds; caption: each posterior enclosed to within
         +-0.003 by Prop. 6.8 at eta = 0.005 (certificate.py).
"""

from __future__ import annotations
from fractions import Fraction
import distributions as D
import template_data

ALPHA = D.COLDBOOT_ALPHA

BLOCK_A = [  # cipher (seed length), W, beta, printed s, printed bound
    dict(label="AES-128, 128-bit seed", W=128, beta=0.01, s="3.918", bound="2.310"),
    dict(label="AES-128, 128-bit seed", W=128, beta=0.05, s="2.763", bound="2.187"),
    dict(label="AES-128, 128-bit seed", W=128, beta=0.25, s="2.199", bound="1.975"),
    dict(label="AES-256, 256-bit seed", W=256, beta=0.01, s="3.639", bound="2.757"),
    dict(label="AES-256, 256-bit seed", W=256, beta=0.05, s="2.713", bound="2.387"),
    dict(label="AES-256, 256-bit seed", W=256, beta=0.20, s="2.245", bound="2.101"),
    dict(label="PRESENT-80", W=80, beta=0.05, s="2.819", bound="1.979"),
    dict(label="PRESENT-80", W=80, beta=0.20, s="2.263", bound="1.895"),
]

BLOCK_B = [  # seed bits m, log2 of the reported average rank T, printed s, printed bound
    dict(label="ML-KEM coin (256 b)", m=256, T=69, s="2.711", bound="2.467"),
    dict(label="ML-KEM coin (256 b)", m=256, T=128, s="2.342", bound="2.212"),
    dict(label="ML-DSA rho' (512 b)", m=512, T=69, s="3.117", bound="2.861"),
    dict(label="ML-DSA rho' (512 b)", m=512, T=128, s="2.697", bound="2.560"),
]

BLOCK_C = [  # SNR, printed s, printed bound
    dict(label="HW leakage", snr=1.0, s="2.018", bound="1.900"),
    dict(label="HW leakage", snr=5.0, s="2.032", bound="1.904"),
]
BLOCK_C_SEEDS = (0, 1, 2)
BLOCK_C_TARGET = 1e-6          # point values: Richardson target
BLOCK_C_ETA0 = 0.02            # point values: first bin width
BLOCK_C_CERT_ETA = 0.005       # certificate bin width (caption)


def composition(W, beta, alpha=ALPHA):
    """(n1, n0) of block A, exactly as distributions.coldboot_asymmetric."""
    n1 = int(round((0.5 * (1.0 - beta) + 0.5 * alpha) * W))
    return n1, W - n1


def advice_A(row):
    return D.coldboot_asymmetric(ALPHA, row["beta"], row["W"])


def exact_types_A(row):
    """Block A as exact rational types [(n, l, u, d)]: n binary coordinates with
    probabilities l/d (the likely symbol) and u/d, for certify_table2_AB.py."""
    a = Fraction(1, 1000)
    assert float(a) == ALPHA
    b = Fraction(str(row["beta"]))
    n1, n0 = composition(row["W"], row["beta"])
    p1 = (1 - b) / (1 - b + a)          # observed 1: P(s* = 1)
    p0 = (1 - a) / (1 - a + b)          # observed 0: P(s* = 0)
    return [(n1, p1.numerator, p1.denominator - p1.numerator, p1.denominator),
            (n0, p0.numerator, p0.denominator - p0.numerator, p0.denominator)]


def q_B(row):
    return D.SHA3_AVG_RANK_FITS[(row["m"], row["T"])]


def advice_B(row):
    return D.sha3_sasca_from_avg_rank(row["T"], row["m"])[0]


def exact_types_B(row):
    Q = Fraction(q_B(row))              # a double is an exact binary rational
    return [(row["m"], Q.numerator, Q.denominator - Q.numerator, Q.denominator)]


def advice_C(row, seed):
    return template_data.load(row["snr"], 1, seed)
