"""
paper_numbers.py -- recompute the numbers stated in the text of the paper and
compare them with the printed values.

Table 2 itself is reproduced by table2.py and its blocks A/B are certified by
certify_table2_AB.py; this script covers the remaining computed numbers of the
PQQS '26 version and of the arXiv full version (marked "full version").  The
running example of the full version has its own script,
running_example/toy_numbers.py.  Each check prints PASS/FAIL, the recomputed
value and, where useful, a note.  The models behind the numbers are fixed here
and in table2_config.py.

Usage:  python3 paper_numbers.py        (exit status 0 iff every check passes)
"""

from __future__ import annotations
import math
import time
from math import comb
import numpy as np

import distributions as D
import table2_config as T2
import template_data
import lclt_diagnostics as LC
from exact_commensurate import exact_speedup, level_counts
from binning_general import binning_speedup, compute_to_precision, certificate_constant
from certificate import certify
from bgmn_bound import bgmn_bound
from common import brute_force_speedup, renyi_entropy, mean_renyi, lnN

CHECKS = []


def check(where, claim):
    def deco(fn):
        CHECKS.append((where, claim, fn))
        return fn
    return deco


def _heterogeneous(seed, m=8, k=3):
    """Seeded heterogeneous advice: m coordinates, k symbols, weights uniform in [0.5, 3]."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(m):
        w = rng.uniform(0.5, 3.0, k)
        out.append(list(w / w.sum()))
    return out


# ------------------------------------------------------------------ Sec. 7 and abstract

@check("Abstract; Sec. 7", "largest exponents 3.92 (cold boot) and 3.12 (synthetic models)")
def _():
    a = max(exact_speedup(T2.advice_A(r))["s"] for r in T2.BLOCK_A)
    b = max(exact_speedup(T2.advice_B(r))["s"] for r in T2.BLOCK_B)
    return f"{a:.4f}, {b:.4f}", f"{a:.2f}" == "3.92" and f"{b:.2f}" == "3.12", ""


@check("Sec. 7", "above 2.7 across the practical cold-boot range")
def _():
    vals = [exact_speedup(D.cipher_coldboot(c, beta))["s"]
            for c in ("AES-128", "AES-256", "PRESENT-80")
            for beta in (0.01, 0.02, 0.03, 0.04, 0.05)]
    return f"min {min(vals):.4f}", min(vals) > 2.7, \
        "practical range read as beta <= 0.05 (asymmetric channel, Table 2 ciphers)"


@check("Sec. 7 (A), footnote", "s in [3.77, 4.11] (AES-128, beta = 0.01) and [2.70, 2.85] "
       "(beta = 0.05) over the observed bit counts (95%)")
def _():
    out, ok = [], True
    for beta, want in ((0.01, ("3.77", "4.11")), (0.05, ("2.70", "2.85"))):
        a, W = T2.ALPHA, 128
        D1, D0 = (1 - beta) + a, beta + (1 - a)
        t1, t0 = [(1 - beta) / D1, a / D1], [beta / D0, (1 - a) / D0]
        f1 = 0.5 * (1 - beta) + 0.5 * a
        sd = math.sqrt(W * f1 * (1 - f1))
        vals = []
        for n1 in range(max(0, int(W * f1 - 7 * sd)), min(W, int(W * f1 + 7 * sd)) + 1):
            prob = comb(W, n1) * f1 ** n1 * (1 - f1) ** (W - n1)
            vals.append((exact_speedup([t1] * n1 + [t0] * (W - n1))["s"], prob))
        vals.sort()
        cdf = np.cumsum([v[1] for v in vals]) / sum(v[1] for v in vals)
        lo = vals[int(np.searchsorted(cdf, 0.025))][0]
        hi = vals[int(np.searchsorted(cdf, 0.975))][0]
        out.append(f"[{lo:.3f}, {hi:.3f}]")
        ok &= (f"{lo:.2f}", f"{hi:.2f}") == want
    return ", ".join(out), ok, "2.5%/97.5% quantiles of s over n1 ~ Bin(W, f1); reported rows use n1 = round(W f1)"


@check("Sec. 7", "asymmetric 2.763 vs symmetric 2.532 at W = m = 128, beta = 0.05")
def _():
    a = exact_speedup(D.cipher_coldboot("AES-128", 0.05, asymmetric=True))["s"]
    s = exact_speedup(D.cipher_coldboot("AES-128", 0.05, asymmetric=False))["s"]
    return f"{a:.4f} vs {s:.4f}", (f"{a:.3f}", f"{s:.3f}") == ("2.763", "2.532"), ""


@check("Table 2 caption; Sec. 7", "block C: each posterior enclosed to within +-0.003 at eta = 0.005; "
       "all three in [2.014, 2.022] at SNR = 1")
def _():
    res, ok = {}, True
    for snr in (1.0, 5.0):
        cs = [certify(template_data.load(snr, 1, seed), T2.BLOCK_C_CERT_ETA) for seed in T2.BLOCK_C_SEEDS]
        ok &= max(c["B"] for c in cs) < 0.003
        lo = math.floor(min(c["s_lo"] for c in cs) * 1000) / 1000
        hi = math.ceil(max(c["s_hi"] for c in cs) * 1000) / 1000
        res[snr] = (lo, hi, max(c["B"] for c in cs))
    ok &= (res[1.0][0], res[1.0][1]) == (2.014, 2.022)
    return (f"SNR 1: [{res[1.0][0]:.3f}, {res[1.0][1]:.3f}], max B {res[1.0][2]:.4f}; "
            f"SNR 5: [{res[5.0][0]:.3f}, {res[5.0][1]:.3f}], max B {res[5.0][2]:.4f}"), ok, ""


@check("Sec. 7", "uniform advice on N = 2^128 gives s = 2.0027; excess nearer 0.015 than 0.018")
def _():
    N = 2.0 ** 128
    zeta_half = -0.2078862249773545          # zeta(-1/2)
    lnEG = math.log((N + 1) / 2)
    lnEH = math.log((2 / 3) * N ** 1.5 + 0.5 * N ** 0.5 + zeta_half) - math.log(N)
    s_u = lnEG / lnEH
    excess = 2.018 - s_u
    return f"{s_u:.5f}; excess {excess:.4f}", \
        f"{s_u:.4f}" == "2.0027" and abs(excess - 0.015) < abs(excess - 0.018), ""


@check("Sec. 7", "Hbar_1/2 / Hbar_2/3 is at most about 1.02 for single-trace posteriors")
def _():
    r = {}
    for snr in (0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
        if snr in (1.0, 5.0):                       # the stored Table 2 posteriors
            inst = [template_data.load(snr, 1, s) for s in T2.BLOCK_C_SEEDS]
        else:
            sig = D.sigma_from_snr(snr, "HW")
            inst = [D.template_aes_instance(sig, seed=s) for s in T2.BLOCK_C_SEEDS]
        r[snr] = max(mean_renyi(x, 0.5) / mean_renyi(x, 2 / 3) for x in inst)
    worst = max(r, key=r.get)
    return (f"SNR 1: {r[1.0]:.4f}, SNR 5: {r[5.0]:.4f}, largest {r[worst]:.4f} at SNR {worst:g}",
            f"{r[worst]:.2f}" == "1.02",
            "maximum over seeds 0-2 per SNR; the arXiv v1 text quoted 1.015, the SNR-5 value")


@check("Sec. 7", "s ~ 2 across the tested SNR range (single trace)")
def _():
    vals = []
    for snr in (0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
        sig = D.sigma_from_snr(snr, "HW")
        vals.append(np.mean([compute_to_precision(D.template_aes_instance(sig, seed=sd), target=1e-5)["s"]
                             for sd in T2.BLOCK_C_SEEDS]))
    return f"s = {min(vals):.3f} ... {max(vals):.3f} (SNR 0.2-100)", 2.0 <= min(vals) and max(vals) < 2.05, \
        "instances regenerated from the model (not stored)"


@check("Sec. 7", "Ber(beta): ratio 1.55 at beta = 0.01, growth like (2/3) beta^(-1/6)")
def _():
    ratio = lambda b: renyi_entropy([1 - b, b], 0.5) / renyi_entropy([1 - b, b], 2 / 3)
    r1, r6 = ratio(0.01), ratio(1e-6)
    lead = (2 / 3) * 1e-6 ** (-1 / 6)
    return f"{r1:.4f}; at 1e-6: {r6:.3f} vs {lead:.3f}", f"{r1:.2f}" == "1.55" and abs(r6 / lead - 1) < 0.01, ""


@check("Sec. 7", "ten traces at SNR 1: s reaches 2.28 at a classical cost of about 2^73 (mean of three)")
def _():
    s_vals, g_vals = [], []
    for seed in T2.BLOCK_C_SEEDS:
        r = compute_to_precision(template_data.load(1.0, 10, seed), target=1e-6)
        s_vals.append(r["s"])
        g_vals.append(r["f1"] / math.log(2))
    s, g = float(np.mean(s_vals)), float(np.mean(g_vals))
    return f"s = {s:.4f}, log2 E[G] = {g:.2f}", f"{s:.2f}" == "2.28" and round(g) == 73, \
        "per seed: " + ", ".join(f"{x:.4f}" for x in s_vals)


# ------------------------------------------------------------------ Sec. 5 and 6

@check("Sec. 5, footnote", "observed |s_eta - s| orders of magnitude below the certified bound, "
       "shrinking faster than linearly (both rounding modes)")
def _():
    worst_ratio, slopes = math.inf, []
    for seed in (0, 1):
        marg = _heterogeneous(seed)
        s = brute_force_speedup(marg)["s"]
        etas = [0.1 / 2 ** j for j in range(7)]
        Bs = [certify(marg, eta)["B"] for eta in etas]
        for mode in ("up", "nearest"):
            errs = [max(abs(binning_speedup(marg, eta, mode)["s"] - s), 1e-15) for eta in etas]
            worst_ratio = min(worst_ratio, min(B / e for B, e in zip(Bs, errs)))
            slopes.append(np.polyfit(np.log(etas), np.log(errs), 1)[0])
    return f"min B/error {worst_ratio:.0f}; error slopes {', '.join(f'{x:.2f}' for x in slopes)}", \
        worst_ratio >= 10 and min(slopes) > 1.0, \
        "seeded heterogeneous advice (m = 8, 3 symbols), eta = 0.1 ... 0.1/64"


@check("Sec. 6.1", "binning reorders keys: at eta = 1, (1.0, 0.7) rounds to 3 and (0.9, 0.9) to 2")
def _():
    up = lambda v: math.floor(v) + 1
    a, b = up(1.0) + up(0.7), up(0.9) + up(0.9)
    return f"{a}, {b}", (a, b) == (3, 2), ""


@check("Remark 6.2", "eta = 0.01, m = 64, 57 x Ber(0.8) + 7 x Ber(0.5025): binned gaps 138 and 1, "
       "Lambda_eta = 0.0036, 94% of the bulk lattice points empty")
def _():
    marg = [[0.8, 0.2]] * 57 + [[0.5025, 0.4975]] * 7
    gaps = sorted({int(abs(np.diff(np.floor(-np.log(np.array(p)) / 0.01) + 1))[0]) for p in marg})
    lam, t = LC.lattice_span_exponent(marg, 0.01, 0.0)
    empty = LC.empty_lattice_fraction(marg, 0.01)
    return f"gaps {gaps}, Lambda_eta {lam:.4f} at t = {t:.3f}, empty {100 * empty:.1f}%", \
        gaps == [1, 138] and f"{lam:.4f}" == "0.0036" and round(100 * empty) == 94, \
        "Lambda_eta at tau = 0 (uniform weights); bulk = central 98% of the uniform law"


@check("Sec. 6.4", "certificate within a factor 1.14-1.45 of C(s) eta at eta = 0.1 and 0.05 "
       "(the PQQS text says eta <= 0.1; m <= 10), closing monotonically with m")
def _():
    ratios = {}
    for m in (6, 8, 10):
        marg = [[0.5, 0.3, 0.2]] * m
        C = certificate_constant(marg, brute_force_speedup(marg)["s"])
        for eta in (0.1, 0.05):
            ratios[(m, eta)] = certify(marg, eta)["B"] / (C * eta)
    lo, hi = min(ratios.values()), max(ratios.values())
    mono_m = all(ratios[(6, e)] > ratios[(8, e)] > ratios[(10, e)] for e in (0.1, 0.05))
    note = ("i.i.d. p = (0.5, 0.3, 0.2); per (m, eta): " +
            ", ".join(f"{m}/{e}: {v:.2f}" for (m, e), v in ratios.items()) +
            "; not monotone in eta at m = 8")
    return f"{lo:.2f} ... {hi:.2f}", (f"{lo:.2f}", f"{hi:.2f}") == ("1.14", "1.45") and mono_m, note


@check("Sec. 6.4", "on heterogeneous advice the certificate scales linearly in eta across seven halvings")
def _():
    marg = _heterogeneous(0)
    Bs = [certify(marg, 0.2 / 2 ** j)["B"] for j in range(8)]
    q = [Bs[j + 1] / Bs[j] for j in range(7)]
    return "halving ratios " + ", ".join(f"{x:.3f}" for x in q), \
        all(0.3 < x < 0.6 for x in q) and all(0.45 < x < 0.55 for x in q[2:]), \
        "seeded heterogeneous advice (m = 8, 3 symbols), eta = 0.2 ... 0.2/128"


@check("Sec. 6.4 (full version)", "at finer eta the ratio leaves the 1.14-1.45 band and eventually grows")
def _():
    out, ok = [], True
    for m in (6, 8, 10):
        marg = [[0.5, 0.3, 0.2]] * m
        C = certificate_constant(marg, brute_force_speedup(marg)["s"])
        r = [certify(marg, eta)["B"] / (C * eta) for eta in (0.05, 0.025, 0.0125, 0.00625, 0.003125)]
        out.append(f"m = {m}: " + ", ".join(f"{x:.2f}" for x in r))
        ok &= any(not 1.14 <= round(x, 2) <= 1.45 for x in r[1:]) and r[-1] > 2 * r[0]
    return "; ".join(out), ok, \
        "eta = 0.05, 0.025, ..., 0.003125; at the finest eta the ratio doubles per halving, B no longer decreases"


@check("Remark 6.9", "f(1/2) ~ (1/2) m Hbar_2/3 overstates the true f(1/2) by ~20% at m = 8")
def _():
    marg = [[0.5, 0.3, 0.2]] * 8
    fh = exact_speedup(marg)["f_half"]
    ar = 0.5 * 8 * mean_renyi(marg, 2 / 3)
    return f"{ar:.3f} vs {fh:.3f}: {100 * (ar / fh - 1):.1f}%", round(100 * (ar / fh - 1)) == 20, \
        "i.i.d. p = (0.5, 0.3, 0.2)"


# ------------------------------------------------------------------ appendices

@check("App. C", "|R| / (beta |t|^3 / 6) reaches 1.08 on i.i.d. Ber(0.8) advice at m = 256")
def _():
    marg = [[0.8, 0.2]] * 256
    vals = [LC.cubic_remainder_ratio(marg, eta, -1.0) for eta in (0.0, 0.001, 0.01, 0.05)]
    return ", ".join(f"{r:.4f}" for r, _, _ in vals), all(f"{r:.2f}" == "1.08" for r, _, _ in vals), \
        f"tau = -1, eta = 0 / 0.001 / 0.01 / 0.05; maximum at t = delta_1 = {vals[0][2]:.3f}"


@check("App. C", "Gaussian tail below 1e-15 on the tested ensembles at m = 256")
def _():
    ens = {"Ber(0.7)": [[0.7, 0.3]] * 256, "Ber(0.8)": [[0.8, 0.2]] * 256, "Ber(0.9)": [[0.9, 0.1]] * 256,
           "tern(.6,.25,.15)": [[0.6, 0.25, 0.15]] * 256,
           "228 Ber(0.8) + 28 Ber(0.5025)": [[0.8, 0.2]] * 228 + [[0.5025, 0.4975]] * 28}
    worst = max(LC.gaussian_tail(m, eta, tau) for m in ens.values()
                for eta in (0.0, 0.01) for tau in (-1.0, -0.5, 0.0))
    return f"max {worst:.1e}", worst < 1e-15, "ensembles " + "; ".join(ens) + "; tau in {-1, -0.5, 0}, eta in {0, 0.01}"


@check("App. F.3", "the rows of blocks A and B have at most 16640 levels each")
def _():
    n = [len(level_counts(T2.advice_A(r))) for r in T2.BLOCK_A] + \
        [len(level_counts(T2.advice_B(r))) for r in T2.BLOCK_B]
    return f"max {max(n)}", max(n) == 16640, "certify_table2_AB.py counts the levels exactly"


@check("App. E (full version; App. D in arXiv v1)", "the certificate returns 15.3 on Ber(0.999) advice, m = 256")
def _():
    marg = [[0.999, 0.001]] * 256
    c = certify(marg, 0.02)
    s = exact_speedup(marg)["s"]
    return f"B = {c['B']:.2f} (eta = 0.02), interval contains s = {s:.4f}: {c['s_lo'] <= s <= c['s_hi']}", \
        f"{c['B']:.1f}" == "15.3" and c["s_lo"] <= s <= c["s_hi"], "B = infinity at eta >= 0.05"


@check("App. G (full version)", "Monte Carlo standard error of s below 1e-4 for the block B fits")
def _():
    import calibrate_avg_rank as CA
    se = []
    for (m, T), q in D.SHA3_AVG_RANK_FITS.items():
        smp = CA._Sampler([(m, q)])
        stat = [math.log2(np.median(smp.sums(CA.REPS, CA.SEED0 + k, CA.RUNS)) / CA.RUNS)
                for k in range(CA.CHUNKS)]
        se_T = float(np.std(stat, ddof=1)) / math.sqrt(CA.CHUNKS)
        dq = 2e-4 * (1 - q)
        dT = (CA.avg_rank_stat([(m, q + dq)]) - CA.avg_rank_stat([(m, q - dq)])) / (2 * dq)
        ds = (CA.moments([(m, q + dq)])["s"] - CA.moments([(m, q - dq)])["s"]) / (2 * dq)
        se.append(abs(ds) * se_T / abs(dT))
    return ", ".join(f"{x:.1e}" for x in se), max(se) < 1e-4, \
        "rows 256/2^69, 256/2^128, 512/2^69, 512/2^128; spread of the fitted statistic over the 8 batches " \
        "(seeds 0-7), carried to q and s by central differences"


NOT_COVERED = [
    ("arXiv v1, App. E.2 / Remarks 6.10-6.11 (the PQQS version states no measured values in App. F.2 / Remarks 6.9-6.10, nor will arXiv v2)",
     "floating-point floor: measured root error ~1e-14, ~2e-16 at m = 256, "
     "crossing near eta ~ 1e-11, floor ~1e-10"),
    ("arXiv v1, App. E.2", "merge-tree round-off: growth exponent 0.96, 3.3e-15 at m = 512, "
     "m/log m ~ 32 at m = 256"),
    ("arXiv v1, App. E.2", "atom spacing ~1e-120 at m = 256"),
]

if __name__ == "__main__":
    t0 = time.time()
    n_ok = 0
    for where, claim, fn in CHECKS:
        t = time.time()
        got, ok, note = fn()
        n_ok += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {where}: {claim}\n       computed: {got}"
              + (f"\n       note: {note}" if note else "") + f"   ({time.time() - t:.1f} s)")
    print("\nNot reproduced by this code (the round-off numbers need an experiment comparing "
          "the float64 merge tree with exact integer convolution; the atom spacing is an "
          "order-of-magnitude statement about an unnamed instance):")
    for where, what in NOT_COVERED:
        print(f"  - {where}: {what}")
    print(f"\n{n_ok}/{len(CHECKS)} CHECKS PASSED   ({time.time() - t0:.0f} s)")
    raise SystemExit(0 if n_ok == len(CHECKS) else 1)
