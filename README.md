# qDMC: Multiscale Financial Cross-Correlation Analysis

**A Python research implementation of the multifractal detrended multiple cross-correlation coefficient, with pairwise analysis, numerical diagnostics, and financial-network visualization.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Method](https://img.shields.io/badge/Method-qDMC-176B87)](#scientific-method)
[![Main paper](https://img.shields.io/badge/Physica_A-2026-8B5E3C)](https://doi.org/10.1016/j.physa.2026.131424)

This repository studies how financial cross-correlations vary with **temporal scale** and **fluctuation order**. It implements the pairwise coefficient $\rho_q(n)$ and builds the multiple coefficient $qDMC_X^2(n)$ for a target series $Y$ and a collection of explanatory series $X_1,\ldots,X_k$.

The financial workflow includes S&P index-data preparation and network labels for ESG, SDG, clean-energy, and artificial-intelligence indices. The underlying coefficient functions also accept other numerical time series.

**Main methodological reference:** da Silva Filho, de Castro, and Guedes (2026), *Multifractal detrended multiple cross-correlation coefficient (qDMC)*. [Read the paper](https://doi.org/10.1016/j.physa.2026.131424).

## Contents

- [Research scope](#research-scope)
- [Scientific method](#scientific-method)
- [Repository map](#repository-map)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Financial-data workflow](#financial-data-workflow)
- [Diagnostics and visualization](#diagnostics-and-visualization)
- [Implementation details](#implementation-details)
- [References and citation](#references-and-citation)

## Research scope

The implementation supports:

- Polynomial detrending and forward/reverse segmentation of integrated profiles.
- Pairwise cross-correlation over configurable values of $q$ and $n$.
- Multiple cross-correlation constructed from pairwise coefficients.
- Matrix conditioning, eigenvalue, determinant, and range diagnostics.
- Financial-return diagnostics, correlation heatmaps, and weighted networks.
- Independent Gaussian simulations and a two-explanatory-series formula check.

The central observable is a **cross-correlation coefficient indexed by scale and fluctuation order**. Its dependence on $q$ describes how fluctuation amplitudes contribute to the measured association; its dependence on $n$ describes how that association changes across temporal scales.

## Scientific method

### 1. Financial returns and integrated profiles

For positive index prices $P_t$, the preprocessing module computes unscaled log returns:

$$
r_t = \ln P_t - \ln P_{t-1}.
$$

For input series $x_t$ and $y_t$, the current profile function uses cumulative sums:

$$
X_j=\sum_{t=1}^{j}x_t,
\qquad
Y_j=\sum_{t=1}^{j}y_t.
$$

At scale $n$, the profiles are split into $M_n=\lfloor N/n\rfloor$ segments from each end, giving $2M_n$ segments. A polynomial of degree `poly_order` is fitted and removed within each segment.

Let $\epsilon_{x,v}(i)$ and $\epsilon_{y,v}(i)$ denote the residuals in segment $v$. The local detrended covariance and variances are:

$$
f_{xy}(v,n)=\frac{1}{n}\sum_{i=1}^{n}
\epsilon_{x,v}(i)\epsilon_{y,v}(i),
$$

$$
f_{xx}(v,n)=\frac{1}{n}\sum_{i=1}^{n}\epsilon_{x,v}^{2}(i),
\qquad
f_{yy}(v,n)=\frac{1}{n}\sum_{i=1}^{n}\epsilon_{y,v}^{2}(i).
$$

### 2. Pairwise coefficient: $\rho_q(n)$

For $q\ne0$, the following notation describes the implemented fluctuation functions, including their numerical floor $\varepsilon=10^{-12}$:

$$
F_{xy}^{(q)}(n)=\frac{1}{2M_n}\sum_{v=1}^{2M_n}
\mathrm{sgn}(f_{xy}(v,n))
\left(|f_{xy}(v,n)|+\varepsilon\right)^{q/2},
$$

$$
F_{xx}^{(q)}(n)=\frac{1}{2M_n}\sum_{v=1}^{2M_n}
\left[\max(f_{xx}(v,n),\varepsilon)\right]^{q/2},
$$

with the analogous expression for $F_{yy}^{(q)}(n)$. Then:

$$
\rho_q(n)=\frac{F_{xy}^{(q)}(n)}
{\sqrt{F_{xx}^{(q)}(n)F_{yy}^{(q)}(n)}}.
$$

The covariance sign is retained before averaging. Positive orders weight larger fluctuation amplitudes more strongly as $q$ increases; negative orders emphasize small amplitudes. At $q=2$, the construction corresponds to the detrended covariance/variance normalization, subject to the numerical floor.

The code also contains a separate $q=0$ branch and a reciprocal correction for negative orders. Their current handling is documented under [Implementation details](#implementation-details).

### 3. Multiple coefficient: $qDMC_X^2(n)$

For a target series $Y$ and explanatory series $X_1,\ldots,X_k$, define:

$$
\boldsymbol{\psi}=
\begin{bmatrix}
\rho_q(Y,X_1;n) & \cdots & \rho_q(Y,X_k;n)
\end{bmatrix}^{\mathsf T}.
$$

Let $R_X$ have unit diagonal and off-diagonal entries $\rho_q(X_i,X_j;n)$. The implemented quadratic form is:

$$
qDMC_X^2(n)=\boldsymbol{\psi}^{\mathsf T}
R_X^{-1}\boldsymbol{\psi}.
$$

**The output `qDMC_raw` is this squared quantity. No square root is applied.** The coefficient combines the target's pairwise correlations with the correlation structure among the explanatory series. It measures association, rather than establishing causal direction or forecasting performance.

For two explanatory series, the corresponding expression is:

$$
qDMC_{X_1,X_2}^{2}(n)=
\frac{
\rho_{YX_1}^{2}+\rho_{YX_2}^{2}
-2\rho_{YX_1}\rho_{YX_2}\rho_{X_1X_2}
}{1-\rho_{X_1X_2}^{2}},
$$

where every pairwise coefficient is evaluated at the same $q$ and $n$. The helper `compute_qdmc_step2_both_forms()` compares this expression with the matrix calculation. With nonzero matrix regularization, the matrix calculation changes, so equality with the unregularized expression is not expected.

## Repository map

| Module | Purpose |
|---|---|
| [`src/data_io.py`](src/data_io.py) | Read the configured S&P Excel-export layout. |
| [`src/preprocessing.py`](src/preprocessing.py) | Clean prices, align dates, and compute log returns. |
| [`src/validation.py`](src/validation.py) | Inspect missing values, duplicate dates, ordering, and invalid prices. |
| [`src/diagnostics.py`](src/diagnostics.py) | Descriptive statistics, time-series tests, and matrix diagnostics. |
| [`src/analysis/rho_q.py`](src/analysis/rho_q.py) | Profiles, segmentation, detrending, fluctuation functions, and $\rho_q$. |
| [`src/analysis/pairwise_rho_q.py`](src/analysis/pairwise_rho_q.py) | Evaluate all unique asset pairs across scales and orders. |
| [`src/analysis/qdmc.py`](src/analysis/qdmc.py) | Build correlation components and calculate qDMC with diagnostics. |
| [`src/analysis/phase04_check_segments.py`](src/analysis/phase04_check_segments.py) | Inspect forward/reverse segmentation. |
| [`src/analysis/phase04_validate_rho_q.py`](src/analysis/phase04_validate_rho_q.py) | Produce pairwise diagnostic reports. |
| [`src/analysis/phase04_summary.py`](src/analysis/phase04_summary.py) | Summarize pairwise results and assign descriptive strength labels. |
| [`src/plotting.py`](src/plotting.py) | Return-series plots, Q-Q plots, and conventional correlation heatmaps. |
| [`src/visualization/`](src/visualization/) | Pairwise scale curves, $q$-scale heatmaps, and correlation networks. |
| [`src/phase06_significance/compute_eq18.py`](src/phase06_significance/compute_eq18.py) | Compare the two-series and matrix qDMC expressions. |
| [`src/phase06_significance/monte_carlo.py`](src/phase06_significance/monte_carlo.py) | Generate independent Gaussian null samples. |

## Installation

Use **Python 3.10 or later**. From a terminal:

```bash
git clone https://github.com/Alikhabazian0/Multifractal-detrended-multiple-cross-correlation-coefficient-qDMC-.git qdmc-finance
cd qdmc-finance
python -m pip install numpy pandas scipy statsmodels arch matplotlib networkx xlrd
```

Run examples from the repository root so that imports beginning with `src` resolve correctly. The repository exposes Python functions; assemble the workflow in a script or notebook.

## Quick start

This example generates **synthetic correlated Gaussian series**, computes pairwise coefficients, and evaluates qDMC. It demonstrates the API and does not represent empirical financial results.

```python
from pathlib import Path
import numpy as np
import pandas as pd

from src.analysis.pairwise_rho_q import run_pairwise_rho_q
from src.analysis.qdmc import (
    build_rho_lookup,
    build_qdmc_components,
    compute_qdmc,
)

rng = np.random.default_rng(42)
common = rng.normal(size=1000)
returns = pd.DataFrame({
    "Y": 0.7 * common + rng.normal(size=1000),
    "X1": 0.5 * common + rng.normal(size=1000),
    "X2": 0.3 * common + rng.normal(size=1000),
})

output = Path("output")
output.mkdir(exist_ok=True)
returns.to_csv(output / "synthetic_returns.csv", index=False)

pairwise = run_pairwise_rho_q(
    returns_path=output / "synthetic_returns.csv",
    output_path=output / "pairwise_rho_q.csv",
    scales=[10, 20, 40, 80, 160],
    q_values=[2, 4],
    poly_order=1,
)

lookup = build_rho_lookup(pairwise)
psi, r_x = build_qdmc_components(
    lookup,
    dependent="Y",
    independents=["X1", "X2"],
    q=2,
    scale=40,
)
result = compute_qdmc(psi, r_x, regularization=0.0)

print("Squared multiple coefficient:", result["qDMC_raw"])
print("Minimum eigenvalue:", result["lambda_min"])
print("In theoretical range:", result["is_in_theoretical_range"])
```

The pairwise output contains `Asset_X`, `Asset_Y`, `q`, `scale`, and `rho_q`. All entries used to construct a multiple coefficient must share the same $q$ and scale.

## Financial-data workflow

1. **Read index prices.** `load_sp_index_excel()` uses the sheet `Performance Graph`, `header=6`, and the first two columns as date and price. Its reader explicitly uses `xlrd`, suitable for legacy `.xls` exports. For `.xlsx` inputs, adapt the reader to an appropriate engine such as `openpyxl`.
2. **Clean each series.** `clean_index()` drops missing dates/prices, sorts chronologically, and retains the last observation for duplicate dates.
3. **Align the sample.** `align_indices()` retains dates present in every input series. It performs no interpolation or forward filling.
4. **Calculate returns.** `compute_log_returns()` rejects nonpositive prices and computes natural-log differences.
5. **Run diagnostics and pairwise analysis.** Export the returns with a `date` column and one numeric column per asset; pass that CSV to `run_pairwise_rho_q()`.
6. **Build qDMC components.** Select the target and explanatory series and construct their correlation components for each chosen $q$ and scale.
7. **Inspect numerical validity and visualize the results.** Retain raw coefficients and diagnostics alongside summary tables and plots.

Input datasets, study dates, and empirical sample sizes are supplied by the user; the public source tree does not bundle a financial dataset.

### Default analysis parameters

| Parameter | Current default |
|---|---|
| Pairwise scales | `10, 20, 30, 40, 60, 80, 120, 160, 200, 250, 300` |
| Pairwise orders | `-4, -2, 0, 2, 4` |
| Polynomial degree | `1` |
| Matrix regularization | `0.0` |
| Gaussian simulations | `10_000` |
| Simulation seed | `42` |

Scales are measured in observations. Their calendar interpretation depends on the sampling frequency and alignment. A scale must leave at least two forward segments; invalid scales become `NaN` in the batch coefficient calculation.

## Diagnostics and visualization

The diagnostics module provides descriptive statistics, Jarque-Bera normality tests, Augmented Dickey-Fuller and Phillips-Perron unit-root tests, Ljung-Box tests on returns and squared returns, and ARCH-LM tests. Conventional Pearson and Spearman correlation matrices provide comparison baselines.

The visualization modules produce:

- $\rho_q(n)$ curves for individual asset pairs.
- Heatmaps indexed by scale and order.
- Weighted correlation networks at a selected positive $q$.
- Return-series plots, Q-Q plots, and conventional correlation heatmaps.

Network edges represent the **arithmetic mean of pairwise coefficients across available scales** at the selected $q$. They are undirected association summaries. The network helper also maps certain thematic index names to shorter labels: ESG, SDG, Clean Energy, and AI.

## Implementation details

### Orders retained in reports

The pairwise calculator can evaluate negative, zero, and positive orders. However, the current summary, validation, heatmap, curve, and network modules filter their inputs with **`q > 0`**. Consequently, those outputs exclude both zero and negative orders, even where comments mention only excluding $q=0$.

The separate logarithmic/geometric $q=0$ branch remains marked for audit in the reporting code. The quick-start example uses positive orders to match the current reporting workflow. The validator's negative-order checks therefore do not assess negative-order results after its positive-order filter.

### Numerical conventions

- **Profiles:** cumulative sums without explicit mean subtraction. For polynomial detrending of degree at least one, subtracting the input mean adds a linear profile term that the fit can remove; this equivalence does not extend to degree-zero detrending.
- **Negative orders:** when $q<0$ and $|\rho_q|>1$, the implementation replaces the coefficient by its reciprocal. Keep this convention in mind when comparing implementations.
- **Matrix inversion:** `compute_qdmc()` attempts a direct inverse and falls back to a pseudoinverse when NumPy reports a singular matrix. It records the method used.
- **Optional regularization:** the matrix becomes $R_X+\lambda I$. The default is $\lambda=0$; nonzero values change the computed observable.
- **Raw and reported values:** `qDMC_raw` retains the quadratic form. `qDMC_reported` becomes `NaN` when the raw value falls outside [0,1]. `qDMC_clipped_for_plot` is a separate display-oriented field.
- **Validity diagnostics:** the result includes the minimum eigenvalue, determinant, condition number, positive-definiteness flag, and range flag. Interpret the coefficient using these diagnostics and the admissibility of the full target/explanatory correlation structure.
- **Strength labels:** verbal categories in the summary and interpretation helpers are project-defined descriptive thresholds, rather than significance levels from the main paper.

### Significance helpers

`simulate_null_data()` generates independent standard-normal samples with shape `(n_sim, 3, n_obs)`. The formula-comparison helper evaluates qDMC for one target and two explanatory series.

These functions are building blocks for Monte Carlo analysis. The current source does not include an end-to-end routine that derives critical values or empirical p-values across all parameter combinations. Reproducing the paper's significance procedure requires specifying the simulation design, computing the null coefficients, and estimating the appropriate tail thresholds.

For reproducible studies, record the input sample and alignment, $q$ values, scales, detrending degree, numerical conventions, regularization, simulation seed, and dependency versions together with the raw results.

## References and citation

### Main qDMC paper

**da Silva Filho, A. M., de Castro, A. P. N., & Guedes, E. F. (2026).** Multifractal detrended multiple cross-correlation coefficient (qDMC). *Physica A: Statistical Mechanics and its Applications, 689*, 131424. https://doi.org/10.1016/j.physa.2026.131424

```bibtex
@article{daSilvaFilho2026qDMC,
  author  = {da Silva Filho, A. M. and de Castro, A. P. N. and Guedes, E. F.},
  title   = {Multifractal detrended multiple cross-correlation coefficient ({qDMC})},
  journal = {Physica A: Statistical Mechanics and its Applications},
  volume  = {689},
  pages   = {131424},
  year    = {2026},
  doi     = {10.1016/j.physa.2026.131424}
}
```

### Pairwise coefficient foundation

**Kwapień, J., Oświęcimka, P., & Drożdż, S. (2015).** Detrended fluctuation analysis made flexible to detect range of cross-correlated fluctuations. *Physical Review E, 92*, 052815. https://doi.org/10.1103/PhysRevE.92.052815

[Open-access manuscript](https://arxiv.org/abs/1506.08692).

When using this code in research, cite the relevant methodological papers and acknowledge this repository with the commit identifier used for the analysis.

## Author

Developed by **[Ali Khabazian](https://github.com/Alikhabazian0)**.
