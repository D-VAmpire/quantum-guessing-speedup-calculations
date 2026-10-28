# speedup_calc: reference implementation

> Notice: This companion repository, including this README, was set up using Anthropic's Claude AI Tools.

This code computes the guessing-moment exponent `s = f(1)/f(1/2)`, with `f(rho) = ln E_P[G^rho]`, for product-distribution advice. It also reproduces the numbers of the paper

> C. Schubert, N. Paskarbeit, M. J. Kramer, J.-P. Seifert, M. Margraf. *Pinpointing Super-Quadratic Quantum Enumeration Speedups: Exact and Certified Evaluation of the Guessing-Moment Exponent under Product-Distribution Advice.* Post-Quantum and Quantum-based Security (PQQS '26). https://doi.org/10.1145/3839338.3857257

Section, result and equation numbers in this repository refer to the PQQS paper. The arXiv full version ([arXiv:2609.28226](https://arxiv.org/abs/2609.28226)) shares them from its version 2 on. Version 1 predates the camera-ready revision and numbers several results, equations and appendices differently; for example, the certificate is Prop. 6.9 there, with its proofs in App. D. Only the arXiv full version has the running example of `running_example/`, as Fig. 1 from version 2 on. If you use this code, please cite the PQQS paper (see [Citation](#citation)).

Requirements: Python 3.10 or later and numpy (`pip install -r requirements.txt`). Three scripts need an optional package, listed in the table below and in `requirements.txt`.

## Reproduce the paper

| command | reproduces | paper | needs |
|---|---|---|---|
| `python3 table2.py` | all 28 values of Table 2, compared with the printed digits | Table 2 | numpy |
| `python3 certify_table2_AB.py` | every printed digit of blocks A and B, in ball arithmetic | App. F.3 | python-flint |
| `python3 paper_numbers.py` | the other computed numbers in the text (20 checks, PASS/FAIL) | Secs. 5–7, Apps. C, E, F.3 | numpy |
| `python3 running_example/toy_numbers.py` | the numbers stated about the running example (9 checks, PASS/FAIL) | arXiv v2: Fig. 1, Secs. 3–4 | numpy |
| `python3 running_example/make_fig_running_example.py` | the running-example figure as standalone TikZ, to compile with `pdflatex` | arXiv v2: Fig. 1 | LaTeX: standalone, tikz, libertine, newtx, zi4 |
| `python3 calibrate_avg_rank.py quick` | the fitted `q` of block B (Monte Carlo) | Sec. 7 (B) | numpy |
| `python3 test_correctness.py` | engines vs brute force, reference values, exact engine, direct convolution | — | numpy |
| `python3 test_vs_budroni_martensson.py` | cross-check against Budroni and Mårtensson's code, downloaded on first run ([below](#third-party-code)) | Sec. 1, related work | scipy, network |
| `python3 reference_values.py` | independent high-precision values used by the tests | — | mpmath |
| `python3 template_data.py` | the stored block C posteriors against regeneration from the model | Table 2 C, Sec. 7 | numpy |

Every script exits with status 0 when all of its checks pass. Run times on a laptop:

- `table2.py`, `paper_numbers.py` and the two `running_example/` scripts: seconds.
- `certify_table2_AB.py` and `test_correctness.py`: about a minute each.
- `calibrate_avg_rank.py quick`: a few minutes.

## Use

```python
from exact_commensurate import exact_speedup        # few coordinate types, small alphabets
from binning_general import compute_to_precision    # any product advice (point value)
from certificate import certify                     # Prop. 6.8: certified |s_eta - s| <= B
from bgmn_bound import bgmn_bound                   # comparison bound (BGMN Thm 4.3)

advice = [[0.9, 0.1]] * 128                         # one probability vector per coordinate
exact_speedup(advice)["s"]                          # 2.30112...
compute_to_precision(advice, target=1e-6)["s"]      # same value via binning
certify(advice, eta=0.01)                           # s_eta, B, [s_eta - B, s_eta + B]
```

## Files

| file | role | paper |
|---|---|---|
| `common.py` | log-domain helpers, level averages `A_rho`, exact assembly, brute force | Lemma 3.1, App. F.1 |
| `exact_commensurate.py` | exact engine: levels by symbol counts, exact integers | Thm 4.1, Sec. 4.1 (finite-rank engine) |
| `binning_general.py` | binned engine: tilted, truncated balanced merge, log-domain recovery, Richardson | Alg. 1 (a)–(c), Alg. 2, Lemma 4.2 |
| `certificate.py` | computable certificate, including the truncation terms | Prop. 6.8, App. E |
| `bgmn_bound.py` | bound of Bashiri, Glaser, May and Nowakowski | Table 2 (BGMN Thm 4.3), Eq. (1) |
| `lclt_diagnostics.py` | `Lambda_eta`, `delta_1`, cubic remainder, Gaussian tail, empty-lattice fraction | Lemma 6.3, Remark 6.2, App. C |
| `distributions.py` | advice models and sourced parameters (no computation) | Sec. 7 |
| `table2_config.py` | Table 2 rows, model parameters, printed values (single source) | Table 2 |
| `template_data.py`, `data/template_posteriors.npz` | stored simulated posteriors for block C and the ten-trace example | Table 2 C, Sec. 7 |
| `table2.py`, `paper_numbers.py` | reproduction scripts | Table 2, text |
| `running_example/` | the running example: `toy_numbers.py` checks its numbers, `make_fig_running_example.py` draws its figure | arXiv v2: Fig. 1, Secs. 3–4 |
| `certify_table2_AB.py` | Arb certification of blocks A and B | App. F.3 |
| `calibrate_avg_rank.py` | block B calibration and its sensitivity study | Sec. 7 (B) |
| `test_correctness.py`, `reference_values.py`, `test_vs_budroni_martensson.py` | validation | — |
| `METHODS.md` | how the code realizes the paper's algorithms, and where it differs | Secs. 4–6 |
| `VERIFICATION.md` | validation results, certified Table 2 values, block B calibration study | — |
| `LICENSE.md`, `CITATION.cff`, `requirements.txt` | license, citation metadata, dependencies | — |

## Conventions

- **Logarithms.** All logs are natural. `s` is a ratio of logs, so it does not depend on the base.
- **Ties.** Keys of equal surprisal fill a rank block, and the moments use the block average `A_rho` (Lemma 3.1(iii)), not the block-maximal rank.
- **BGMN bound.** The Thm 4.3 bound is `2 (H_{1/2} - ln(1 + log2|K|)) / H_{2/3}` with entropies in nats. The inner logarithm is base 2, as in BGMN's base-2 statement.
- **Floors.** Template posteriors are floored at `1e-10` and renormalized (see `distributions.py`).

## Limitations

- **Floating point.** The exact engine and the binned engine run in double precision. The certificate bounds the binning error of the exact-arithmetic estimator, and the floating-point error of Lemma 4.5 comes on top. Nothing here is an end-to-end interval enclosure, except `certify_table2_AB.py`.
- **Certificate slack.** `B` is a rigorous bound and can be far above the actual error. For i.i.d. ternary advice (0.6, 0.25, 0.15) with m = 8 at `eta = 0.05`, `B = 0.25` while `|s_eta - s|` is below 1e-15. Report `s_eta` together with the interval, and do not present `B` as the achieved accuracy.
- **Exact engine size.** The exact engine handles few coordinate types with small alphabets. Its level counts are capped at 5·10^6, and larger instances should be binned.
- **Not reproduced.** The round-off measurements quoted in App. E.2 of the legacy version arXiv v1 are not produced by this code: the root error ≈ 1e-14, the floor ~1e-10, the growth exponent 0.96 and 3.3e-15 at m = 512. The PQQS version (App. F.2) states no measured values there, and neither does arXiv v2. `paper_numbers.py` lists them.

## Third-party code

`test_vs_budroni_martensson.py` compares the exact engine with the code of

> A. Budroni, E. Mårtensson. *Further improvements of the estimation of key enumeration with applications to solving LWE.* Cryptography and Communications 16(5):1163–1182, 2024. https://doi.org/10.1007/s12095-024-00722-1

Their code (https://github.com/ErikMaartensson/ImprovedKeyEnumeration) has no license, so this repository does not include it. On first run, the test downloads `EnumerationLarge.py` at a pinned commit into `.cache/`, checks its SHA-256, and runs only the four functions it needs.

## Citation

Please cite the PQQS paper; `CITATION.cff` has the same information.

```bibtex
@inproceedings{schubert2026pinpointing,
  author    = {Schubert, Carsten and Paskarbeit, Niklas and Kramer, Maximilian J. and Seifert, Jean-Pierre and Margraf, Marian},
  title     = {Pinpointing Super-Quadratic Quantum Enumeration Speedups: Exact and Certified Evaluation of the Guessing-Moment Exponent under Product-Distribution Advice},
  booktitle = {Post-Quantum and Quantum-based Security (PQQS '26)},
  series    = {PQQS '26},
  location  = {Milpitas, CA, USA},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  year      = {2026},
  isbn      = {9798400729201},
  doi       = {10.1145/3839338.3857257}
}
```

## License

Copyright (c) 2026 Carsten Schubert, Niklas Paskarbeit, Maximilian Johannes Kramer and Technische Universität Berlin.

The code, data and documentation in this repository are licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). Use, changes and sharing are allowed for any noncommercial purpose, including research and teaching at universities and other public institutions. For commercial use, please contact the corresponding author of the paper. The paper itself is published under CC BY-NC-ND 4.0.
