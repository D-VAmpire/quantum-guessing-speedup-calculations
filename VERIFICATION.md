# Verification

Section, result and equation numbers refer to the PQQS '26 paper (https://doi.org/10.1145/3839338.3857257). The arXiv full version shares them from its version 2 on; version 1 numbers several of them differently (see README).

Last full run: 2026-10-08. Environments:

- Python 3.13.16 with numpy 2.5.3, scipy 1.18.1, python-flint 0.9.0 and mpmath 1.4.1.
- The same Python without scipy. All outputs were identical: the engines use numpy only.
- An arm64 Linux machine: Python 3.10.12, numpy 2.2.6, mpmath 1.3.0, no scipy, no python-flint. `test_correctness.py`, `table2.py`, `paper_numbers.py` and `reference_values.py` pass there with identical results. The cross-check and the Arb certification need scipy and python-flint and were run in the first environment.

## 1. Engines (`test_correctness.py`, all pass)

1. **Brute force.** The exact engine equals brute-force optimal moments within 1e-6 on seven enumerable models (largest difference 4e-8, at Ber(0.999), m = 14).
2. **Reference values.** The exact engine equals the 14 independent high-precision values of `reference_values.py` within 1e-7. These use exact rational levels, exact `E[G]` and Euler–Maclaurin block sums in mpmath. For example, `P = (.5,.2,.2,.1)` gives 2.25514922189, and Ber(0.999) with m = 256 gives 5.50038050461. The worst case is 3e-8, at Ber(0.999) with m = 16, where `f(1/2)` is small and `s` is ill-conditioned.
3. **Uniform advice.** Uniform advice gives `s > 2` (2.014049 for ternary advice with m = 16). The block-maximal surrogate would give exactly 2.
4. **Level bracket.** Theorem 4.1's bracket `1/(1+rho) <= A_rho/(N F_S)^rho <= 1` holds at every level, with 1 at singleton levels.
5. **Binning.** The binned estimator converges to the exact value: Richardson to 1e-8, and heterogeneous ternary advice to 1.3e-6 at `eta = 0.01`.
6. **BGMN.** The exact `s` is at least the BGMN Theorem 4.3 bound.
7. **Skewed, large-m advice.** The binned engine equals the exact engine to 1e-10 on Ber(0.999) with m = 256 and m = 1024, and on Ber(0.97) with m = 512. On the two-type cold-boot advice of Table 2 (AES-128, `beta = 0.01`) it agrees to 1.4e-10 at `eta = 0.001`.
8. **Direct convolution.** The binned engine equals plain direct convolution under the same binning to 1e-13. This covers template advice with one or ten traces and the two-type cold-boot advice.
9. **Certificate.** The certificate brackets the brute-force value on three models at `eta = 0.2` and `0.05`, and `B` shrinks with `eta`. On Ber(0.999) with m = 256 and `eta = 0.02`, it encloses the exact 5.5004 with `B = 15.27`: valid, but vacuous, as the full version states (App. E; App. D in arXiv v1).

## 2. Cross-check with Budroni and Mårtensson (`test_vs_budroni_martensson.py`)

Their routines compute `E_P[G]` and `E_P[sqrt G]` for i.i.d. product advice by summing `r` and `sqrt(r)` across equal-probability blocks: the same moments, likelihood order and tie handling. Every applicable case agrees:

- the four rows of Table 2 block B, at 2.7108438, 2.3422960, 3.1172188 and 2.6972980, to 6e-12;
- six symmetric cold-boot cases and two ternary cases, to 5e-12;
- four small cases, to 8e-10; these also agree with brute force.

Their float64 code overflows at n = 1024, where the exact engine still returns values.

Their repository has no license, so this folder does not contain their code: the test downloads `EnumerationLarge.py` at a pinned commit, checks its SHA-256, and runs only the four functions it needs.

## 3. Table 2

`table2.py` reproduces all 28 printed values; blocks A and B use the exact engine, block C the binned engine on the stored posteriors. `certify_table2_AB.py` certifies every printed digit of blocks A and B in ball arithmetic at 3000 bits: all enclosures have radius below 1e-899 and lie strictly inside `printed ± 0.0005`. The float engine agrees with the enclosures to 1.8e-10 in `s` (AES-256, `beta = 0.01`) and to 1e-14 in the bound.

| row | levels | certified `s` | certified bound (BGMN Thm 4.3) | printed |
|---|---|---|---|---|
| AES-128, β = 0.01 | 4224 | 3.918154377633264 | 2.309597527047816 | 3.918 / 2.310 |
| AES-128, β = 0.05 | 4216 | 2.762633923519614 | 2.186941102411738 | 2.763 / 2.187 |
| AES-128, β = 0.25 | 3969 | 2.198738589054008 | 1.975003664803982 | 2.199 / 1.975 |
| AES-256, β = 0.01 | 16640 | 3.639085924403313 | 2.757142684965585 | 3.639 / 2.757 |
| AES-256, β = 0.05 | 16605 | 2.713394616583928 | 2.387370030855704 | 2.713 / 2.387 |
| AES-256, β = 0.20 | 16016 | 2.245361607431992 | 2.100616941676912 | 2.245 / 2.101 |
| PRESENT-80, β = 0.05 | 1677 | 2.818615641327919 | 1.978997471114529 | 2.819 / 1.979 |
| PRESENT-80, β = 0.20 | 1617 | 2.262651116331717 | 1.894808117837651 | 2.263 / 1.895 |
| ML-KEM coin, 2^69 | 257 | 2.710843796747255 | 2.466599973011690 | 2.711 / 2.467 |
| ML-KEM coin, 2^128 | 257 | 2.342296004570992 | 2.211590776003062 | 2.342 / 2.212 |
| ML-DSA ρ', 2^69 | 513 | 3.117218766683409 | 2.861099410449092 | 3.117 / 2.861 |
| ML-DSA ρ', 2^128 | 513 | 2.697297990491656 | 2.559584337060312 | 2.697 / 2.560 |

Block C, means over the three stored posteriors:

| SNR | `s` | BGMN bound | printed |
|---|---|---|---|
| 1 | 2.01768 | 1.89955 | 2.018 / 1.900 |
| 5 | 2.03245 | 1.90390 | 2.032 / 1.904 |

The certificates at `eta = 0.005` give a bound `B` of 0.0027–0.0028 for each posterior:

- SNR 1: [2.0144, 2.0198], [2.0147, 2.0201], [2.0159, 2.0212];
- SNR 5: [2.0317, 2.0372], [2.0300, 2.0355], [2.0274, 2.0329].

`template_data.py` compares the stored posteriors with their regeneration from the model. They are bit-identical with numpy 2.5.3 on x86-64, and agree to 6e-17 with numpy 2.2.6 on arm64. The difference is last-bit rounding in the math library, not a different random stream. All values above are identical either way.

## 4. Block B calibration (Sec. 7 (B))

Maillard et al. average key ranks over 50 runs and plot them on a log2 scale. Table 2 uses the 50-run fit: 50 simulated runs reproduce the reported average in the median (Monte Carlo, 4e5 experiments, MC standard error of `s` below 1e-4). `calibrate_avg_rank.py` reproduces the four stored `q` bit for bit.

The table below is `calibrate_avg_rank.py`'s full output (re-run 2026-10-08). It compares the 50-run fit with other readings of the statistic and with heterogeneous posteriors:

- **`E[G]` fit.** The fit `log2 E[G] = T` was used in arXiv v1. Its `E[G]` is carried by error patterns of total probability about 3.5e-14 (256 bits, 2^69). 50 runs on it would report about 2^41.
- **`E[log2 G]` fit.** This reads the published averages as averages of log2 ranks.

| row | `E[G]` fit (arXiv v1) | 50-run fit (Table 2) | `E[log2 G]` fit | uncertainty on 3/4 of the bits | half the bits 10× error-prone |
|---|---|---|---|---|---|
| ML-KEM 2^69 | 3.1136 | 2.7108 | 2.5034 | 2.5493 | 2.6356 |
| ML-KEM 2^128 | 2.5131 | 2.3423 | 2.2465 | 2.2044 | 2.2916 |
| ML-DSA 2^69 | 3.9735 | 3.1172 | 2.8338 | 2.9403 | 3.0410 |
| ML-DSA 2^128 | 3.1787 | 2.6973 | 2.5387 | 2.5437 | 2.6314 |

**Run count.** For ML-KEM 2^69, 20, 50 and 100 runs give `s` = 2.6748, 2.7108 and 2.7295.

**Shape.** To leading order, the i.i.d. posterior maximizes `2 Hbar_{1/2}/Hbar_{2/3}` at fixed `H_{1/2}`.
- Along Ber(p), `h_{2/3}` is strictly convex in `h_{1/2}`. With `w = (p/(1-p))^{1/6}`, the slope `dh_{2/3}/dh_{1/2} = 2w(1+w)(1+w^3)/((1+w+w^2)(1+w^4))` is increasing; then Jensen applies.
- At finite size, mild heterogeneity can raise `s` slightly. With half the bits `c` times as error-prone, ML-DSA 2^69 gives 3.1172, 3.1204, 3.1224, 3.1196 and 3.1075 for `c` = 1, 1.25, 1.5, 2 and 3. The largest increase found is about 0.005.
- ML-KEM 2^69 decreases monotonically in `c`: 2.7108, 2.7090, 2.7071, 2.7026, 2.6917.
- These series were re-run on 2026-10-08 with `fit_V(m, 69, c, seed0=100)`.
- Uncertainty on fewer bits always lowers `s`.

## 5. Numbers in the text (`paper_numbers.py`, 22/22 pass)

The two checks marked "full version" for Sec. 6.4 and App. G were added on 2026-10-28 and pass with Python 3.10.12 and numpy 2.2.6.

| where | printed | recomputed |
|---|---|---|
| Abstract, Sec. 7 | 3.92, 3.12 | 3.9182, 3.1172 |
| Sec. 7 | above 2.7 for practical cold boot | min 2.7134 (β ≤ 0.05) |
| Sec. 7 (A), footnote | [3.77, 4.11], [2.70, 2.85] | [3.767, 4.113], [2.700, 2.847] |
| Sec. 7 | 2.763 vs 2.532 (symmetric) | 2.7626 vs 2.5317 |
| Table 2 caption, Sec. 7 | ±0.003 at `eta = 0.005`; [2.014, 2.022] at SNR 1 | B ≤ 0.0028; [2.014, 2.022] |
| Sec. 7 | uniform `N = 2^128`: 2.0027; excess ≈ 0.015 | 2.00268; 0.0153 |
| Sec. 7 | `Hbar_{1/2}/Hbar_{2/3}` at most about 1.02 | max 1.0221 (SNR 20); SNR 1: 1.0070, SNR 5: 1.0155 |
| Sec. 7 | `s` ≈ 2 across the SNR range | 2.004–2.042 (SNR 0.2–100) |
| Sec. 7 | 1.55 at β = 0.01, growth like (2/3)β^(−1/6) | 1.5524; 6.705 vs 6.667 at 1e-6 |
| Sec. 7 | ten traces, SNR 1: 2.28 at about 2^73 | 2.2809 (2.2615, 2.2331, 2.3480), 2^73.3 |
| Sec. 5, footnote | error far below the bound, faster than linear | min B/error 2334; slopes 1.8–2.1 |
| Sec. 6.1 | 3 and 2 | 3 and 2 |
| Remark 6.2 | gaps 138, 1; `Lambda_eta` = 0.0036; 94% empty | 138, 1; 0.0036 (τ = 0, t = 4.553); 94.0% |
| Sec. 6.4 | 1.14–1.45 | 1.14–1.45 (i.i.d. (0.5, 0.3, 0.2), m = 6, 8, 10) |
| Sec. 6.4 | linear across seven halvings | ratios 0.372 … 0.518 |
| Sec. 6.4 (full version) | at finer η the ratio leaves the band and eventually grows | η = 0.025: 2.44, 1.47, 1.01 (m = 6, 8, 10); η = 0.003125: 9.72, 5.23, 3.34 |
| Remark 6.9 | ≈ 20% at m = 8 | 20.2% |
| App. C | 1.08 | 1.0804 (η = 0), 1.0799–1.0814 (η ≤ 0.05) |
| App. C | Gaussian tail below 1e-15 | max 7.4e-19 |
| App. F.3 | at most 16640 levels | 16640 |
| App. E (full version; App. D in arXiv v1) | 15.3 | 15.27 at `eta = 0.02` |
| App. G (full version) | Monte Carlo standard error of `s` below 1e-4 | 8.1e-5, 3.6e-5, 8.9e-5, 4.9e-5 (block B rows) |

Two remarks on these numbers:
- **Sec. 6.4.** The ratio closes monotonically in `m`, as the PQQS text states, but not in `eta` at m = 8: 1.26 at `eta = 0.1`, 1.30 at 0.05. arXiv v1 said "with m and eta". The PQQS text's "for eta <= 0.1" holds only at the tested 0.1 and 0.05: at `eta = 0.025` the ratio is 2.44 for m = 6 and 1.01 for m = 10, and at finer `eta` it grows, because `B` stops decreasing. arXiv v2 states this.
- **The ratio in Sec. 7.** arXiv v1 printed "≈ 1.015", the SNR-5 value. The PQQS text states the bound over the tested SNR range, at most about 1.02.

Not reproduced here are the round-off measurements of arXiv v1, App. E.2; see README.

The running example of the arXiv full version (Fig. 1 from version 2 on; the PQQS version has none) is checked by `running_example/toy_numbers.py`, added on 2026-10-28 after the last full run: 9/9 pass with Python 3.10.12 and numpy 2.2.6. It recomputes the levels and rank blocks, `E_P[G] = 3741/512`, `E_P[sqrt G] = 2.3731`, `s = 2.3013` and the uniform 2.0730 (Sec. 3), the count polynomials (Sec. 4.1), the block-maximal 2.2684 (after Thm 4.1), and `c_1 = -1.2595`, `c_1/2 = -0.6862` with the Rényi ratio 2.0952 (Cor. 4.3). Brute force reproduces `s` to 1e-15. The exact engine is off by 7e-10, because it identifies levels by their surprisal rounded to 9 decimals.

## 6. Changes relative to arXiv v1

- **Block B.** The values changed with the calibration: `E[G]` fit replaced by the 50-run fit (Section 4 above).
- **Abstract.** "up to 3.97" became "3.92 for cold-boot advice and 3.12 in the synthetic models".
- **Block C, removed claim.** v1 stated "2.011 even at a residual 0.54 bits per byte". That value is wrong: identity leakage with three traces at `sigma = 8` gives `s = 2.648` at `E[G]` ≈ 2^10.
- **Block C, certificate.** "[2, 2.044] at SNR = 1" (η = 0.05, clipped at 2) became [2.014, 2.022] at η = 0.005.
- **App. C.** The counterexample ratio 1.09 became 1.08, after `delta_1`'s second arm changed to `sqrt(3)/(2 max_i sigma_i)`.
