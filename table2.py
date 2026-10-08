"""
table2.py -- reproduce Table 2 of the paper and compare with the printed digits.

Blocks A and B: exact finite-rank engine (exact_commensurate.py).
Block C: binned engine (binning_general.py) on the stored posteriors, mean over
the three simulated posteriors; the caption's certificates are in paper_numbers.py.
Every row's configuration is in table2_config.py; certify_table2_AB.py certifies
the digits of blocks A and B in ball arithmetic.

Usage:  python3 table2.py          (exit status 0 iff every printed digit matches)
"""

from __future__ import annotations
import time
import numpy as np
import table2_config as T2
from exact_commensurate import exact_speedup
from binning_general import compute_to_precision
from bgmn_bound import bgmn_bound


def rows():
    """[(block, label, parameter, s, bound, printed s, printed bound)]"""
    out = []
    for r in T2.BLOCK_A:
        marg = T2.advice_A(r)
        out.append(("A", r["label"], f"beta={r['beta']:.2f}", exact_speedup(marg)["s"],
                    bgmn_bound(marg)["s_bgmn"], r["s"], r["bound"]))
    for r in T2.BLOCK_B:
        marg = T2.advice_B(r)
        out.append(("B", r["label"], f"2^{r['T']}", exact_speedup(marg)["s"],
                    bgmn_bound(marg)["s_bgmn"], r["s"], r["bound"]))
    for r in T2.BLOCK_C:
        s_vals, b_vals = [], []
        for seed in T2.BLOCK_C_SEEDS:
            marg = T2.advice_C(r, seed)
            s_vals.append(compute_to_precision(marg, target=T2.BLOCK_C_TARGET,
                                               eta0=T2.BLOCK_C_ETA0)["s"])
            b_vals.append(bgmn_bound(marg)["s_bgmn"])
        out.append(("C", r["label"], f"SNR={r['snr']:g}", float(np.mean(s_vals)),
                    float(np.mean(b_vals)), r["s"], r["bound"]))
    return out


if __name__ == "__main__":
    t0 = time.time()
    ok_all = True
    print(f"{'':2}{'setting':<24}{'parameter':<11}{'s':>10}{'printed':>9}"
          f"{'BGMN 4.3':>11}{'printed':>9}  match")
    for blk, label, par, s, b, ps, pb in rows():
        ok = f"{s:.3f}" == ps and f"{b:.3f}" == pb
        ok_all &= ok
        print(f"{blk:2}{label:<24}{par:<11}{s:>10.5f}{ps:>9}{b:>11.5f}{pb:>9}  {'yes' if ok else 'NO'}")
    print(f"\nALL 28 PRINTED VALUES REPRODUCED: {ok_all}   ({time.time() - t0:.0f} s)")
    raise SystemExit(0 if ok_all else 1)
