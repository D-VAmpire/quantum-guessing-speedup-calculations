"""
toy_numbers.py -- recompute the numbers stated about the running example and
compare them with the printed values.

From version 2 on, the arXiv full version illustrates Secs. 3 and 4 on a toy
advice with two dyadic coordinates (its Fig. 1, drawn by
make_fig_running_example.py); the PQQS '26 version has no running example.
The checks cover the levels and rank blocks, E_P[G], E_P[sqrt G], s and the
uniform baseline (Sec. 3), the count polynomials (Sec. 4.1), the
block-maximal surrogate (after Thm 4.1) and the split
f(rho) = rho H_{1/(1+rho)}(P) + c_rho of Cor. 4.3.  The moments are exact
rationals; square roots and logarithms are in double precision.  The last
check recomputes s with the exact engine and by brute force, so it needs
numpy, like the rest of the repository.

Usage:  python3 running_example/toy_numbers.py   (exit status 0 iff every check passes)
"""

import os
import sys
from fractions import Fraction as Fr
from math import log, sqrt

P1 = [1, 3, 3, 3, 4, 4]          # surprisal of each symbol in units of h = ln 2: p1 = (1/2, 1/8, 1/8, 1/8, 1/16, 1/16)
P2 = [1, 2, 3, 4, 5, 6, 6]       # p2 = (1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/64)
HL = 5                           # the highlighted level t = 5h
assert sum(Fr(1, 2**k) for k in P1) == 1 and sum(Fr(1, 2**k) for k in P2) == 1


def count_poly(bits):
    """Coefficients c[k] = number of symbols of surprisal k h."""
    c = [0] * (max(bits) + 1)
    for k in bits:
        c[k] += 1
    return c


def poly_mul(a, b):
    c = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i + j] += x * y
    return c


def poly_str(c):
    return "+".join(("" if x == 1 else str(x)) + ("z" if k == 1 else f"z^{k}")
                    for k, x in enumerate(c) if x)


def renyi(bits, alpha):
    """Renyi entropy (nats) of a marginal with symbol probabilities 2^-k."""
    return log(sum(2.0 ** (-alpha * k) for k in bits)) / (1 - alpha)


def two(*xs):
    return tuple(f"{x:.2f}" for x in xs)


def main():
    c1, c2 = count_poly(P1), count_poly(P2)
    cS = poly_mul(c1, c2)                        # level sizes n(t) = N nu(t) at t = k h
    levels = {k: n for k, n in enumerate(cS) if n}
    N = sum(levels.values())

    # Lemma 3.1 (iii): E_P[G^rho] = sum_t pi(t) A_rho(t), with J_t = {lo, ..., hi}
    EG, EH, EGs, EHs, r, blocks = Fr(0), 0.0, Fr(0), 0.0, 0, {}
    print(f"N = {N}, levels t/h = {sorted(levels)}")
    for t, n in sorted(levels.items()):
        lo, hi, pi = r + 1, r + n, Fr(n, 2**t)
        EG += pi * Fr(lo + hi, 2)
        EH += float(pi) * sum(sqrt(x) for x in range(lo, hi + 1)) / n
        EGs += pi * hi                           # block-maximal rank N F_S(t)
        EHs += float(pi) * sqrt(hi)
        blocks[t] = (lo, hi, pi)
        print(f"t = {t:2d}h  n(t) = {n}  J_t = [{lo}, {hi}]  P per key = 2^-{t}  pi(t) = {pi}")
        r = hi
    f1, fh = log(float(EG)), log(EH)
    s = f1 / fh
    s_uni = log((N + 1) / 2) / log(sum(sqrt(x) for x in range(1, N + 1)) / N)
    s_bm = log(float(EGs)) / log(EHs)
    H12 = renyi(P1, 1 / 2) + renyi(P2, 1 / 2)    # Renyi entropies add over the coordinates
    H23 = renyi(P1, 2 / 3) + renyi(P2, 2 / 3)
    c_1, c_h = f1 - H12, fh - H23 / 2            # Cor. 4.3 at rho = 1 and rho = 1/2
    low_1, low_h = -log(1 + log(N)), -log(1 + log(N)) / 2
    ticks = [blocks[t][1] for t in sorted(blocks)]

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from exact_commensurate import exact_speedup
    from common import brute_force_speedup
    advice = [[2.0 ** -k for k in P1], [2.0 ** -k for k in P2]]
    s_ex, s_bf = exact_speedup(advice)["s"], brute_force_speedup(advice)["s"]

    checks = [
        ("Sec. 3", "N = 6*7 = 42 keys on nine levels",
         f"N = {N}, {len(levels)} levels t/h = {min(levels)}..{max(levels)}",
         N == len(P1) * len(P2) == 42 and len(levels) == 9),
        ("Sec. 3", "level t = 5h: n(t) = 6 keys of probability 2^-5, J_t = {7, ..., 12}, pi(t) = 6*2^-5",
         f"n(t) = {levels[HL]}, J_t = [{blocks[HL][0]}, {blocks[HL][1]}], pi(t) = {blocks[HL][2]}",
         levels[HL] == 6 and blocks[HL][:2] == (7, 12) and blocks[HL][2] == Fr(6, 2**5)),
        ("Sec. 3", "E_P[G] = 3741/512 ~ 7.31, E_P[sqrt G] ~ 2.37, s ~ 2.30",
         f"{EG} = {float(EG):.7f}, {EH:.7f}, s = {s:.7f}",
         EG == Fr(3741, 512) and two(float(EG), EH, s) == ("7.31", "2.37", "2.30")),
        ("Sec. 3", "uniform advice on 42 keys gives s ~ 2.07",
         f"{s_uni:.7f}", two(s_uni) == ("2.07",)),
        ("Fig. 1", "rank ticks N F_S(t) = 1, 2, 6, 12, 18, 25, 30, 38, 42",
         ", ".join(map(str, ticks)), ticks == [1, 2, 6, 12, 18, 25, 30, 38, 42]),
        ("Sec. 4.1", "(z+3z^3+2z^4)(z+z^2+z^3+z^4+z^5+2z^6) = z^2+z^3+4z^4+6z^5+6z^6+7z^7+5z^8+8z^9+4z^10",
         f"({poly_str(c1)})({poly_str(c2)}) = {poly_str(cS)}",
         (poly_str(c1), poly_str(c2), poly_str(cS)) ==
         ("z+3z^3+2z^4", "z+z^2+z^3+z^4+z^5+2z^6", "z^2+z^3+4z^4+6z^5+6z^6+7z^7+5z^8+8z^9+4z^10")),
        ("after Thm 4.1", "the block-maximal surrogate gives s ~ 2.27",
         f"{s_bm:.7f}", two(s_bm) == ("2.27",)),
        ("after Cor. 4.3", "c_1 ~ -1.26 and c_1/2 ~ -0.69, against lower ends -1.56 and -0.78; "
         "2 H_1/2(P)/H_2/3(P) ~ 2.10",
         f"{c_1:.5f}, {c_h:.5f}; {low_1:.5f}, {low_h:.5f}; {2 * H12 / H23:.5f} "
         f"(H_1/2(P) = {H12:.5f}, H_2/3(P) = {H23:.5f})",
         two(c_1, c_h, low_1, low_h, 2 * H12 / H23) == ("-1.26", "-0.69", "-1.56", "-0.78", "2.10")
         and low_1 <= c_1 <= 0 and low_h <= c_h <= 0),
        ("Thm 4.1, Lemma 3.1", "the exact engine and brute force reproduce s",
         f"exact engine {s_ex:.13f} (|d| = {abs(s_ex - s):.0e}), brute force {s_bf:.13f} "
         f"(|d| = {abs(s_bf - s):.0e}), here {s:.13f}\n       note: the exact engine identifies "
         f"levels by their surprisal rounded to 9 decimals (exact_commensurate.py), hence |d| ~ 1e-9",
         abs(s_ex - s) < 1e-8 and abs(s_bf - s) < 1e-12),
    ]
    print()
    n_ok = 0
    for where, claim, got, ok in checks:
        n_ok += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {where}: {claim}\n       computed: {got}")
    print(f"\n{n_ok}/{len(checks)} CHECKS PASSED")
    return n_ok == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
