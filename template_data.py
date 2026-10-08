"""
template_data.py -- stored template-attack posteriors behind the paper's numbers.

data/template_posteriors.npz holds the simulated 16 x 256 posteriors of
distributions.template_aes_instance (HW leakage, floor 1e-10) for
    Table 2, block C:        SNR 1 and 5, one trace, seeds 0-2
    Sec. 7, multi-trace:     SNR 1, ten traces, seeds 0-2
so that these numbers do not depend on numpy's random-number stream, which numpy
does not guarantee across versions.  load() falls back to regenerating an
instance that is not stored.

    python3 template_data.py          # check the stored file against regeneration
    python3 template_data.py --write  # (re)write the file from the model
"""

from __future__ import annotations
import hashlib
import os
import sys
import numpy as np
import distributions as D

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "template_posteriors.npz")
STORED = [(1.0, 1, s) for s in (0, 1, 2)] + [(5.0, 1, s) for s in (0, 1, 2)] \
         + [(1.0, 10, s) for s in (0, 1, 2)]
_cache = None


def key(snr, n_traces, seed):
    return f"snr{snr:g}_traces{n_traces}_seed{seed}"


def regenerate(snr, n_traces, seed):
    """The instance from the model (HW leakage, 16 bytes, floor 1e-10)."""
    sigma = D.sigma_from_snr(snr, "HW")
    return D.template_aes_instance(sigma, n_traces=n_traces, model="HW", seed=seed)


def load(snr, n_traces=1, seed=0):
    """Stored instance as a list of 16 marginals (regenerated if not stored)."""
    global _cache
    if _cache is None:
        _cache = dict(np.load(PATH)) if os.path.exists(PATH) else {}
    k = key(snr, n_traces, seed)
    if k in _cache:
        return [row for row in _cache[k]]
    return regenerate(snr, n_traces, seed)


def sha256(path=PATH):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


if __name__ == "__main__":
    if "--write" in sys.argv[1:]:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        np.savez_compressed(PATH, **{key(*c): np.array(regenerate(*c)) for c in STORED})
        print("wrote", PATH)
    print("sha256", sha256())
    worst = max(float(np.max(np.abs(np.array(load(*c)) - np.array(regenerate(*c)))))
                for c in STORED)
    if worst == 0.0:
        verdict = "identical"
    elif worst < 1e-12:
        verdict = "equal up to floating-point rounding of the platform's math library"
    else:
        verdict = "DIFFERENT: numpy's random stream has changed; the stored file is authoritative"
    print(f"max |stored - regenerated| = {worst:.1e}  ({verdict})")
