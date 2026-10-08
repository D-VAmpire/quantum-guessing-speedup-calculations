"""
test_vs_budroni_martensson.py -- independent cross-check of the exponents against
the reference implementation of Budroni and Martensson.

Reference: A. Budroni and E. Martensson, "Further improvements of the estimation
    of key enumeration with applications to solving LWE", Cryptography and
    Communications 16(5):1163-1182, 2024, doi:10.1007/s12095-024-00722-1
    (extends their ISIT 2023 paper).
Code:  https://github.com/ErikMaartensson/ImprovedKeyEnumeration (EnumerationLarge.py).
    That repository states no license, so this file does not contain their code.
    On first use it downloads EnumerationLarge.py at a pinned commit into .cache/,
    checks its SHA-256, and executes only the four function definitions it needs
    (sumOfSqrts, sums, exactNumberOfEnumeration, exactNumberOfEnumerationQuantum);
    the file's driver and plotting code is never run.

Why this is a valid cross-check
-------------------------------
Their `exactNumberOfEnumeration(p, freq, n)` sorts the type classes of an i.i.d.
product law by decreasing probability and accumulates, per class,
    (partialSum*fk + fk*(fk+1)/2) * P_class
  = P(one key) * sum_{r=a+1}^{a+fk} r
  = sum over keys in the class of P(key) * rank(key),
so the total is exactly E_P[G] = e^{f(1)}.  `...Quantum` replaces the rank sum by
`sumOfSqrts(a+fk) - sumOfSqrts(a)`, giving exactly E_P[sqrt(G)] = e^{f(1/2)}.
These are the two guessing moments with the same likelihood order and the same
tie handling: they sum r and sqrt(r) exactly across each equal-probability block,
which is the level average A_rho of Lemma 3.1(iii) written per key.  Agreement
therefore confirms the tie handling, not only the arithmetic.

Scope: their method assumes ONE i.i.d. coordinate law (p, freq) repeated n times,
as the paper's related-work paragraph says, so it covers block B of Table 2 and
symmetric cold-boot advice but not the asymmetric channel (two coordinate types)
or the template attack (16 distinct 256-ary posteriors).  In float64 it also
overflows around n >~ 1024.  Requires scipy (for zeta) and, on first use,
network access to GitHub; without either it reports SKIPPED.

Run:  python3 test_vs_budroni_martensson.py
"""

from __future__ import annotations
import ast
import hashlib
import math
import os
import urllib.request
import warnings
import numpy as np

from exact_commensurate import exact_speedup
from common import brute_force_speedup
import table2_config as T2

try:
    from scipy.special import zeta
except ImportError:                      # pragma: no cover
    zeta = None

warnings.filterwarnings("ignore")
TOL = 1e-6

# --------------------------------------------------------------------------
# Budroni and Martensson's code, fetched at a pinned commit (not redistributed)
# --------------------------------------------------------------------------
BM_COMMIT = "86f5c55244b9f5f7c988e51125cf5ff33c41206e"
BM_URL = ("https://raw.githubusercontent.com/ErikMaartensson/ImprovedKeyEnumeration/"
          + BM_COMMIT + "/EnumerationLarge.py")
BM_SHA256 = "5dbc861fe270d67a151d77fd45b52a5f50e6c90b754ca948c610b989ceb1c045"
BM_FUNCTIONS = ("sumOfSqrts", "sums", "exactNumberOfEnumeration",
                "exactNumberOfEnumerationQuantum")
BM_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache",
                        "EnumerationLarge.py")
BM = {}


def load_bm():
    """Load the four B-M functions into BM; return a reason string on failure."""
    if not os.path.exists(BM_CACHE):
        try:
            with urllib.request.urlopen(BM_URL, timeout=30) as response:
                data = response.read()
        except OSError as err:
            return f"cannot download {BM_URL} ({err})"
        if hashlib.sha256(data).hexdigest() != BM_SHA256:
            return "the downloaded file does not match the pinned SHA-256"
        os.makedirs(os.path.dirname(BM_CACHE), exist_ok=True)
        with open(BM_CACHE, "wb") as fh:
            fh.write(data)
    with open(BM_CACHE, "rb") as fh:
        data = fh.read()
    if hashlib.sha256(data).hexdigest() != BM_SHA256:
        return f"{BM_CACHE} does not match the pinned SHA-256 (delete it to re-download)"
    tree = ast.parse(data.decode("utf-8"))
    defs = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in BM_FUNCTIONS]
    if sorted(node.name for node in defs) != sorted(BM_FUNCTIONS):
        return "EnumerationLarge.py lacks an expected function"
    BM.update({"np": np, "math": math, "zeta": zeta})
    exec(compile(ast.Module(body=defs, type_ignores=[]), "EnumerationLarge.py", "exec"), BM)
    return None


def bm_speedup(p, freq, n):
    """s = ln E[G] / ln E[sqrt G] from the reference implementation."""
    c = BM["exactNumberOfEnumeration"](p, freq, n)
    q = BM["exactNumberOfEnumerationQuantum"](p, freq, n)
    if not (c > 0 and q > 0 and math.isfinite(c) and math.isfinite(q)):
        raise ValueError("non-finite / underflowed")
    return math.log(c) / math.log(q)


# --------------------------------------------------------------------------
def main():
    if zeta is None:
        print("SKIPPED: scipy is not installed")
        return 0
    reason = load_bm()
    if reason:
        print(f"SKIPPED: {reason}")
        return 0
    ok = True

    def chk(name, a, b, extra=""):
        nonlocal ok
        good = abs(a - b) < TOL
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<34}"
              f"ours={a:.7f} B-M={b:.7f} |d|={abs(a-b):.1e} {extra}")

    print("1) small i.i.d. cases: ours == B-M == brute force")
    for q, n in [(0.75, 10), (0.90, 12), (0.70, 14), (0.95, 8)]:
        o = exact_speedup([[q, 1-q]]*n)["s"]
        b = bm_speedup([q, 1-q], [1, 1], n)
        bf = brute_force_speedup([[q, 1-q]]*n)["s"]
        chk(f"Ber({q}) n={n}", o, b, f"(brute {bf:.7f})")

    print("2) Table 2, block B (50-run average-rank fits)")
    for r in T2.BLOCK_B:
        q, n = T2.q_B(r), r["m"]
        o = exact_speedup([[q, 1-q]]*n)["s"]
        chk(f"{r['label']} 2^{r['T']}", o, bm_speedup([q, 1-q], [1, 1], n), f"(printed {r['s']})")

    print("3) symmetric cold-boot advice (i.i.d., so also in scope)")
    for n in (128, 256):
        for beta in (0.05, 0.10, 0.25):
            o = exact_speedup([[1-beta, beta]]*n)["s"]
            chk(f"n={n} beta={beta}", o, bm_speedup([1-beta, beta], [1, 1], n))

    print("4) multi-symbol advice (ternary)")
    for p0, n in [(0.6, 64), (0.8, 128)]:
        rest = (1 - p0) / 2
        o = exact_speedup([[p0, rest, rest]]*n)["s"]
        chk(f"ternary p0={p0} n={n}", o, bm_speedup([p0, rest], [1, 2], n))

    print("5) beyond their float64 range -> ours still returns a value")
    for beta, n in [(0.05, 1024), (0.01, 1024), (0.001, 1024)]:
        o = exact_speedup([[1-beta, beta]]*n)["s"]
        try:
            bm_speedup([1-beta, beta], [1, 1], n)
            status = "B-M also fine"
        except (OverflowError, ValueError) as e:
            status = f"B-M {type(e).__name__} (expected)"
        print(f"  [info] beta={beta} n={n}: ours={o:.5f}   {status}")

    print("\n" + ("ALL CROSS-CHECKS PASSED" if ok else "SOME CROSS-CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
