"""
test_correctness.py -- validation of the engines against ground truth.

Run:  python3 test_correctness.py      (exit status 0 iff all tests pass)

 1) exact engine == brute-force optimal moments
 2) exact engine == independent high-precision reference values (reference_values.py)
 3) uniform advice gives s > 2 (the block-maximal surrogate would give exactly 2)
 4) the bracket 1/(1+rho) <= A_rho / (N F_S)^rho <= 1 of Theorem 4.1 at every level
 5) binned estimator -> exact value as eta shrinks (non-commensurate advice)
 6) exact s >= BGMN Theorem 4.3 bound
 7) binned engine == exact engine on skewed, large-m advice (regression: these
    cases failed when the binned laws were convolved without tilting)
 8) binned engine == plain direct convolution on template-attack advice
 9) the certificate of Prop. 6.8 brackets the true s; B shrinks with eta;
    on Ber(0.999)^256 it is valid but vacuous (B = 15.3, App. E of the full version)
"""

from __future__ import annotations
import math
import numpy as np
from common import brute_force_speedup, log_A_int, logsumexp, logcumsumexp, log_A_array, lnN
from exact_commensurate import exact_speedup, level_counts
from binning_general import binning_speedup, compute_to_precision, coordinate_laws
from certificate import certify
from bgmn_bound import bgmn_bound
import template_data

PASS, FAIL = "PASS", "FAIL"

# Printed by reference_values.py (exact level structure, mpmath; 15 digits there).
REFERENCE = [("P=(.5,.2,.2,.1)", [[0.5, 0.2, 0.2, 0.1]], 2.25514922188719),
             ("Ber(0.75) m=16", [[0.75, 0.25]] * 16, 2.14724805561377),
             ("Ber(0.75) m=256", [[0.75, 0.25]] * 256, 2.07688002384561),
             ("Ber(0.75) m=1024", [[0.75, 0.25]] * 1024, 2.07292443396448),
             ("Ber(0.90) m=16", [[0.9, 0.1]] * 16, 2.46973256383767),
             ("Ber(0.90) m=256", [[0.9, 0.1]] * 256, 2.28908096682828),
             ("Ber(0.90) m=1024", [[0.9, 0.1]] * 1024, 2.27947574673195),
             ("uniform ternary m=16", [[1/3] * 3] * 16, 2.01404949749956),
             ("uniform ternary m=1024", [[1/3] * 3] * 1024, 2.00020954702511),
             ("ternary .6/.2/.2 m=16", [[0.6, 0.2, 0.2]] * 16, 2.09357616086339),
             ("ternary .6/.2/.2 m=256", [[0.6, 0.2, 0.2]] * 256, 2.05325025295140),
             ("Ber(0.999) m=16", [[0.999, 0.001]] * 16, 4.24708267967630),
             ("Ber(0.999) m=256", [[0.999, 0.001]] * 256, 5.50038050461391),
             ("Ber(0.999) m=1024", [[0.999, 0.001]] * 1024, 4.68000884071396)]


def check(name, cond, extra=""):
    print(f"  [{PASS if cond else FAIL}] {name}   {extra}")
    return cond


def bern(p, m):
    return [[p, 1 - p]] * m


def direct_binned_s(marginals, eta, round_mode="nearest"):
    """Oracle for test 8: untilted chains by plain direct convolution (entrywise
    accurate for nonnegative vectors), assembled exactly like the engine."""
    nu_c, pi_c = np.array([1.0]), np.array([1.0])
    for p in marginals:
        nu, pi = coordinate_laws(p, eta, round_mode)
        nu_c, pi_c = np.convolve(nu_c, nu), np.convolve(pi_c, pi)
    ln_N = lnN(marginals)
    with np.errstate(divide="ignore"):
        ln_nu, ln_pi = np.log(nu_c), np.log(pi_c)
    lnF = np.minimum(logcumsumexp(ln_nu), 0.0)
    ln_b = ln_N + lnF
    ln_a = np.concatenate(([-np.inf], ln_b[:-1]))
    keep = np.isfinite(ln_pi)
    f = {rho: logsumexp(ln_pi[keep] + log_A_array(ln_a, ln_b, ln_N + ln_nu, rho)[keep])
         for rho in (1.0, 0.5)}
    return f[1.0] / f[0.5]


def main():
    ok = True

    print("1) exact engine == brute-force optimal moments")
    for lbl, marg in [("Ber(0.75) m=12", bern(0.75, 12)),
                      ("Ber(0.9)  m=14", bern(0.90, 14)),
                      ("Ber(0.99) m=12", bern(0.99, 12)),
                      ("Ber(0.999) m=14", bern(0.999, 14)),
                      ("tern(.6,.2,.2) m=8", [[0.6, 0.2, 0.2]] * 8),
                      ("uniform ternary m=10", [[1/3]*3] * 10),
                      ("4-ary m=7", [[0.5, 0.2, 0.2, 0.1]] * 7)]:
        bf = brute_force_speedup(marg)["s"]
        ex = exact_speedup(marg)["s"]
        ok &= check(lbl, abs(bf - ex) < 1e-6,
                    f"brute={bf:.8f} exact={ex:.8f} |d|={abs(bf-ex):.1e}")

    print("2) exact engine == reference values (reference_values.py)")
    for lbl, marg, ref in REFERENCE:
        s = exact_speedup(marg)["s"]
        # s is ill-conditioned when f(1/2) is small (Ber(0.999), m = 16): tolerance 1e-7
        ok &= check(lbl, abs(s - ref) < 1e-7, f"s={s:.10f} ref={ref:.10f} |d|={abs(s-ref):.1e}")

    print("3) uniform advice gives s > 2")
    for b, m in [(3, 16), (3, 1024), (2, 64)]:
        s = exact_speedup([[1.0/b]*b]*m)["s"]
        ok &= check(f"uniform b={b} m={m}", s > 2.0 + 1e-9, f"s={s:.6f}")

    print("4) bracket 1/(1+rho) <= A_rho/(N F_S)^rho <= 1 at every level (Theorem 4.1), = 1 if n(t) = 1")
    for lbl, marg in [("Ber(0.9) m=64", bern(0.90, 64)),
                      ("uniform ternary m=32", [[1/3]*3]*32),
                      ("tern(.6,.2,.2) m=16", [[0.6, 0.2, 0.2]]*16)]:
        good, a = True, 0
        for _, n in level_counts(marg):
            b = a + n
            for rho in (1.0, 0.5):
                kap = math.exp(log_A_int(a, b, rho) - rho * math.log(b))
                good &= 1.0 / (1.0 + rho) - 1e-12 <= kap <= 1.0 + 1e-12
                if n == 1:
                    good &= abs(kap - 1.0) < 1e-12
            a = b
        ok &= check(lbl, good, f"{len(level_counts(marg))} levels")

    print("5) binned estimator -> exact value as eta shrinks (non-commensurate advice)")
    marg = [[0.6, 0.25, 0.15]] * 8
    ref = brute_force_speedup(marg)["s"]
    for eta in (0.2, 0.05, 0.01):
        s = binning_speedup(marg, eta, "up")["s"]
        print(f"      eta={eta:<5} up-rounded s_eta={s:.9f}  |s_eta - s|={abs(s-ref):.2e}")
    r = compute_to_precision(marg, target=1e-6)
    ok &= check("binning + Richardson vs exact", abs(r["s"] - ref) < 1e-6,
                f"s={r['s']:.9f} exact={ref:.9f} |d|={abs(r['s']-ref):.1e}")
    rng = np.random.default_rng(0)                       # heterogeneous advice: binning error is real
    marg = [list(w / w.sum()) for w in rng.uniform(0.5, 3.0, (8, 3))]
    ref = brute_force_speedup(marg)["s"]
    errs = [abs(binning_speedup(marg, eta, "nearest")["s"] - ref) for eta in (0.1, 0.01)]
    ok &= check("heterogeneous ternary m=8, nearest", errs[1] < errs[0] and errs[1] < 1e-4,
                f"|s_eta - s| = {errs[0]:.1e} (eta 0.1), {errs[1]:.1e} (eta 0.01)")

    print("6) exact s >= BGMN Theorem 4.3 bound")
    for lbl, marg in [("Ber(0.9) m=64", bern(0.90, 64)),
                      ("Ber(0.97) m=256", bern(0.97, 256))]:
        s = exact_speedup(marg)["s"]
        g = bgmn_bound(marg)["s_bgmn"]
        ok &= check(lbl, s >= g - 1e-9, f"exact={s:.5f} BGMN={g:.5f}")

    print("7) binned engine == exact engine on skewed, large-m advice")
    # i.i.d. binary advice: binning keeps the level order, so s_eta = s exactly
    for lbl, marg, eta in [("Ber(0.999) m=256", bern(0.999, 256), 0.05),
                           ("Ber(0.999) m=1024", bern(0.999, 1024), 0.05),
                           ("Ber(0.97) m=512", bern(0.97, 512), 0.01)]:
        b = binning_speedup(marg, eta, "nearest")["s"]
        e = exact_speedup(marg)["s"]
        ok &= check(lbl, abs(b - e) < 1e-8, f"binned={b:.9f} exact={e:.9f} |d|={abs(b-e):.1e}")
    two_type = [[990/991, 1/991]] * 63 + [[999/1009, 10/1009]] * 65   # Table 2, AES-128, beta = 0.01
    b = binning_speedup(two_type, 0.001, "nearest")["s"]
    e = exact_speedup(two_type)["s"]
    ok &= check("two-type cold boot m=128, eta=0.001", abs(b - e) < 1e-6,
                f"binned={b:.9f} exact={e:.9f} |d|={abs(b-e):.1e}")

    print("8) binned engine == plain direct convolution (same binning, eta = 0.01)")
    cases = [(f"template SNR {snr:g}, {tr} trace(s), seed {sd}", template_data.load(snr, tr, sd))
             for snr, tr, sd in [(1.0, 10, 0), (1.0, 10, 2), (5.0, 1, 0)]]
    cases.append(("two-type cold boot m=128", two_type))
    for lbl, marg in cases:
        b = binning_speedup(marg, 0.01, "nearest")["s"]
        d = direct_binned_s(marg, 0.01, "nearest")
        ok &= check(lbl, abs(b - d) < 1e-9, f"engine={b:.9f} direct={d:.9f} |d|={abs(b-d):.1e}")

    print("9) the certificate (Prop. 6.8) brackets the true s")
    for lbl, marg in [("tern(.6,.25,.15) m=8", [[0.6, 0.25, 0.15]] * 8),
                      ("Ber(0.7) m=12", bern(0.70, 12)),
                      ("4-ary m=7", [[0.5, 0.2, 0.2, 0.1]] * 7)]:
        st = brute_force_speedup(marg)["s"]
        for eta in (0.2, 0.05):
            c = certify(marg, eta)
            ok &= check(f"{lbl} eta={eta}",
                        c["s_lo"] - 1e-9 <= st <= c["s_hi"] + 1e-9,
                        f"[{c['s_lo']:.4f}, {c['s_hi']:.4f}] contains {st:.5f} "
                        f"(B={c['B']:.2e})")
    c1 = certify([[0.6, 0.25, 0.15]] * 8, 0.2)
    c2 = certify([[0.6, 0.25, 0.15]] * 8, 0.02)
    ok &= check("B shrinks with eta", c2["B"] < c1["B"],
                f"B(0.2)={c1['B']:.3e} -> B(0.02)={c2['B']:.3e}")
    c = certify(bern(0.999, 256), 0.02)
    st = exact_speedup(bern(0.999, 256))["s"]
    ok &= check("Ber(0.999) m=256 eta=0.02: valid, vacuous",
                c["s_lo"] <= st <= c["s_hi"] and f"{c['B']:.1f}" == "15.3",
                f"s_eta={c['s']:.4f} B={c['B']:.2f} exact={st:.4f}")

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
