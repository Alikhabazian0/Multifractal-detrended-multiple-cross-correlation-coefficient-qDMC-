from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron

def _clean_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def descriptive_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in returns.columns:
        x = _clean_series(returns[col])

        rows.append({
            "Asset": col,
            "N": len(x),
            "Mean": x.mean(),
            "Std": x.std(ddof=1),
            "Min": x.min(),
            "Q25": x.quantile(0.25),
            "Median": x.median(),
            "Q75": x.quantile(0.75),
            "Max": x.max(),
            "Skewness": skew(x, bias=False),
            "Excess_Kurtosis": kurtosis(x, fisher=True, bias=False),
        })

    return pd.DataFrame(rows)


def normality_tests(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in returns.columns:
        x = _clean_series(returns[col])
        jb_stat, jb_p = jarque_bera(x)

        rows.append({
            "Asset": col,
            "Jarque_Bera_Stat": jb_stat,
            "Jarque_Bera_p": jb_p,
            "Reject_Normality_5pct": jb_p < 0.05,
        })

    return pd.DataFrame(rows)


def adf_tests(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in returns.columns:
        x = _clean_series(returns[col])
        result = adfuller(x, autolag="AIC")

        rows.append({
            "Asset": col,
            "ADF_Stat": result[0],
            "ADF_p": result[1],
            "ADF_Lags": result[2],
            "ADF_Nobs": result[3],
            "ADF_CV_1pct": result[4]["1%"],
            "ADF_CV_5pct": result[4]["5%"],
            "ADF_CV_10pct": result[4]["10%"],
            "Stationary_5pct": result[1] < 0.05,
        })

    return pd.DataFrame(rows)


def pp_tests(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in returns.columns:
        x = _clean_series(returns[col])

        result = PhillipsPerron(x)

        rows.append({
            "Asset": col,
            "PP_Stat": result.stat,
            "PP_p": result.pvalue,

            # Long-run variance estimation
            "Bandwidth": result.lags,
            "Bandwidth_Method": "Automatic (Newey-West)",

            "PP_CV_1pct": result.critical_values["1%"],
            "PP_CV_5pct": result.critical_values["5%"],
            "PP_CV_10pct": result.critical_values["10%"],
            
            "Stationary_5pct": result.pvalue < 0.05,
        })

    return pd.DataFrame(rows)


def ljungbox_tests(returns: pd.DataFrame, lag: int = 20) -> pd.DataFrame:
    rows = []

    for col in returns.columns:
        x = _clean_series(returns[col])

        lb_r = acorr_ljungbox(x, lags=[lag], return_df=True)
        lb_r2 = acorr_ljungbox(x ** 2, lags=[lag], return_df=True)

        rows.append({
            "Asset": col,
            "Lag": lag,
            "LB_Returns_Stat": lb_r["lb_stat"].iloc[0],
            "LB_Returns_p": lb_r["lb_pvalue"].iloc[0],
            "LB_Squared_Returns_Stat": lb_r2["lb_stat"].iloc[0],
            "LB_Squared_Returns_p": lb_r2["lb_pvalue"].iloc[0],
            "Serial_Correlation_Returns_5pct": lb_r["lb_pvalue"].iloc[0] < 0.05,
            "Volatility_Clustering_5pct": lb_r2["lb_pvalue"].iloc[0] < 0.05,
        })

    return pd.DataFrame(rows)


def arch_lm_tests(returns: pd.DataFrame, lag: int = 20) -> pd.DataFrame:
    rows = []

    for col in returns.columns:
        x = _clean_series(returns[col])

        lm_stat, lm_p, f_stat, f_p = het_arch(x, nlags=lag)

        rows.append({
            "Asset": col,
            "Lag": lag,
            "ARCH_LM_Stat": lm_stat,
            "ARCH_LM_p": lm_p,
            "ARCH_F_Stat": f_stat,
            "ARCH_F_p": f_p,
            "ARCH_Effect_5pct": lm_p < 0.05,
        })

    return pd.DataFrame(rows)


def correlation_matrices(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    pearson = clean.corr(method="pearson")
    spearman = clean.corr(method="spearman")
    return pearson, spearman


def matrix_diagnostics(corr: pd.DataFrame) -> pd.DataFrame:
    matrix = corr.to_numpy(dtype=float)

    eigenvalues = np.linalg.eigvalsh(matrix)
    lambda_min = float(np.min(eigenvalues))
    lambda_max = float(np.max(eigenvalues))

    condition_number = np.inf if lambda_min <= 0 else float(lambda_max / lambda_min)
    determinant = float(np.linalg.det(matrix))

    inv_corr = np.linalg.pinv(matrix)
    vif_values = np.diag(inv_corr)

    return pd.DataFrame([{
        "N_Variables": matrix.shape[0],
        "Lambda_Min": lambda_min,
        "Lambda_Max": lambda_max,
        "Condition_Number": condition_number,
        "Determinant": determinant,
        "Mean_VIF": float(np.mean(vif_values)),
        "Max_VIF": float(np.max(vif_values)),
    }])