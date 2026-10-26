# How the code realizes the paper, and where it differs

Section, result and equation numbers are those of the PQQS '26 paper (https://doi.org/10.1145/3839338.3857257). The arXiv full version shares them from its version 2 on; version 1 numbers several of them differently (see README).

## The quantity

`f(rho) = ln E_P[G^rho] = ln sum_t pi(t) A_rho(t)` (Lemma 3.1(iii)), and `s = f(1)/f(1/2)`.

- **Levels.** `t` runs over the surprisal levels. `pi(t)` is the advice mass at level `t`, and `A_rho(t)` is the mean `rho`-th power of the ranks in the level's rank block `J_t = (N F_S(t^-), N F_S(t)]`.
- **Rank-block averages (`common.py`).** These follow App. F.1.
  - `rho = 1` uses the closed form.
  - `rho = 1/2` sums blocks of at most 5000 ranks exactly. Larger blocks use the upper end of the Euler–Maclaurin bracket, evaluated through the occupancy ratio `n/b` with `expm1`/`log1p`.
  - Results are floored at rank 1.

## Exact engine (`exact_commensurate.py`): Theorem 4.1, finite-rank engine of Sec. 4.1

- **Enumeration.** For advice whose coordinates fall into few types, levels are enumerated by the symbol counts of each type, with multinomial coefficients as exact Python integers. Their rank-block endpoints are therefore exact integers.
  - This is the finite-rank engine at `d <= k`, realized by count enumeration instead of the `Z^d` FFT merge.
  - It is exact even for non-commensurate marginals (Ber(p) has rank 2).
- **Level identity.** Levels are identified by their surprisal rounded to 9 decimals. `certify_table2_AB.py` identifies them exactly by integer numerators.
- **Conditioning.** When `f(1/2)` is small (near-deterministic advice), `s` is ill-conditioned in floating point, at a relative level of about `(1 + s) u / f(1/2)`. Example: Ber(0.999) with m = 16 agrees with the 60-digit reference to 3e-8.

## Binned engine (`binning_general.py`): Algorithm 2 with the stable evaluation of Algorithm 1

1. **Binning.** Surprisals are rounded onto `eta Z`, either strictly up (Sec. 5, required by the certificate) or to nearest (admissible per the footnote in Sec. 5). Each coordinate gets a binned count law and a binned mass law.
2. **Tilting (step a).** For each `rho`, the count laws are tilted by `phi_rho = -1/(1+rho)` and the mass laws by `phi_rho + 1` (Algorithm 2, step 3; Lemma 4.2).
3. **Merge (step b).**
   - The tilted laws are convolved up a balanced tree; identical coordinates are combined by repeated squaring.
   - Every node is renormalized to maximum 1 and truncated: entries below `TRUNCATION_FLOOR = 1e-13` of the node maximum are set to zero. This also removes FFT round-off noise.
   - Lemma 4.4(ii) truncates to Chernoff windows instead. Either way, in exact arithmetic the computed arrays are dominated by the exact ones, which is all the certificate needs (App. E).
   - Convolutions are direct for short vectors and use `numpy.fft` otherwise, so results do not depend on optional packages.
4. **Log-domain recovery (step c).** `ln nu~`, `ln F~` (prefix LSE) and `ln pi~` are recovered by Eq. (6). The `e^{-Theta(m)}` magnitudes stay in the log domain.
5. **Assembly.** `f_eta(rho)` is assembled by Eq. (7), each `rho` from its own chains.

### Point values

`compute_to_precision` uses nearest rounding and Richardson extrapolation `2 s_eta - s_{2 eta}`. It halves `eta` from 0.02 until successive extrapolants agree to the target.

- This is an empirical convergence check, cheaper than Algorithm 2's certified schedule, and it is not a guarantee.
- Table 2 block C uses it for the point values, while the caption's accuracy statement comes from the certificate.

### Earlier implementation (until 2026-10-07)

The earlier implementation convolved the binned laws without tilting, with scipy's FFT when scipy was installed and numpy's direct convolution otherwise.

- **Failure mode.** The FFT path has absolute round-off of about `u` times the largest entry. It therefore lost the small entries of the count chain at the likely end. For Ber(0.999)^256 at `eta = 0.05`, the three smallest entries came out as 6.4e-19, 5.8e-19, 0 instead of 8.6e-78, 2.2e-75, 2.8e-73.
- **Wrong results.** Skewed binned advice then came out wrong:
  - Ten traces at SNR 1 gave about 2.11 instead of 2.28.
  - The certificate on Ber(0.999)^256 enclosed about 2.47 although s = 5.50.
- **Unaffected.** Table 2 was not affected: blocks A and B use the exact engine, and the single-trace block C values agree to every printed digit under both paths.
- **Where the paper numbers came from.** The paper's ten-trace numbers agree with the direct-convolution path, which the old code took when scipy was not installed.
- **Today.** Tilting, the working-precision fix the paper itself prescribes, removes the problem at FFT speed. `test_correctness.py` (groups 7–8) guards against regressions.

## Certificate (`certificate.py`): Prop. 6.8 and App. E

For each `rho`, the certificate is assembled from the strictly up-rounded chains as follows.

- **Enclosure of `F~`.** The truncation losses give the enclosure `L = F^ <= F~ <= U = min(1, F^ + D_nu e^{|phi| t})`, where `D_nu` is the exact weighted count loss (App. E, "Truncated merges").
- **Lower candidates.** `(N F~(t - m eta))^rho`, `1` and `A~ e^{-rho D}` are formed from `L` throughout, including inside `A` and `D`.
- **Upper candidates.** `(N F~(t + m eta))^rho`, `e^{rho t}` and `A~ e^{+rho D}` are formed from `U` throughout.
  - `A` is capped at `(N L(t))^rho` on the lower side and floored at `(N U(t^-))^rho` on the upper side. This keeps the two inequalities that make the `D`-candidates valid for either end of the enclosure, whatever the floating-point error in the block sizes.
- **Mass-chain term.** The upper end of Eq. (12) gets the additive term `D_pi max_t e^{-psi t} hi_rho(t)`.
  - Relative to the upper end, this term is the lost tilted mass times `R`, the ratio of the peak of `e^{-psi t} hi_rho(t)` to its average under the tilted mass chain. `certify` reports `R` as `mass_term_ratio`.
  - At `eta = 0.05`, `R` lies between 1.19 and 2.31 (`rho = 1/2` and `1`) on four instances: the block C posterior at SNR 1 (seed 0), 64 × Ber(0.9), i.i.d. ternary (0.6, 0.25, 0.15) with m = 8, and 32 × Ber(0.99) + 32 × Ber(0.7). The code takes the peak over the whole grid; over the atoms of the chain alone, `R` lies between 1.19 and 1.57.
  - The paper states no value of `R`. App. E bounds the added term analytically, which makes it negligible whenever `m eta = O(log(m/eta))`.
- **Bound.** `E_rho`, `B` and the enclosure `[s_eta - B, s_eta + B]` follow from Eq. (13).
- **Size of the corrections.** With the floor at 1e-13, both losses are at the 1e-13 relative level, and the results agree with the untruncated certificate to the printed digits.

The arrays are floating point, so this bounds the binning error of the exact-arithmetic estimator. The floating-point error of Lemma 4.5 comes on top: there is no interval-arithmetic route here (Sec. 6.4).

## Diagnostics (`lclt_diagnostics.py`): Lemma 6.3, Remark 6.2, App. C

These are computed under the binned tilt `U~_tau`, with weights proportional to `exp(tau s~)`:

- `sigma~^2`, `beta`, `delta_1` and `Lambda_eta` of Eq. (10). The minimum over the band uses a grid with local refinement.
- The cubic remainder ratio `|R| / (beta |t|^3 / 6)`, with the principal branch taken per coordinate.
- The Gaussian tail `sqrt(2/pi) Phibar(sigma~ delta_1)`.
- The empty-lattice fraction, from the exact support of `S~`.

Algorithm 2, step 6 reports these alongside a binned run. Here they are separate functions, used by `paper_numbers.py`. The pairwise sums are explicit, which is meant for small alphabets.

## Model choices behind the paper's numbers

- **Table 2.** The rows are in `table2_config.py`, the advice models in `distributions.py`, and the stored posteriors in `data/template_posteriors.npz`.
- **Other numbers in the text.** The models behind them are fixed in `paper_numbers.py`. The paper does not state some of them:
  - Sec. 6.4 and Remark 6.9 use i.i.d. `(0.5, 0.3, 0.2)` with `m = 6, 8, 10`.
  - The "seven halvings" and the Sec. 5 footnote use seeded heterogeneous ternary advice with `m = 8`.
  - Remark 6.2 uses `tau = 0`.
  - App. C uses `tau = -1`.

## Not implemented

- Interval arithmetic.
- The `Z^d` FFT form of the finite-rank engine.
- Algorithm 2's step-5 schedule `eta_1 = eta_0 delta / B`. Refine `eta` by hand with `certify`.
- The round-off experiment behind App. F.2's measured constants.
