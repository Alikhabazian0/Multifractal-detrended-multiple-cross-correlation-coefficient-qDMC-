import numpy as np
import pandas as pd


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a, b]))


def build_rho_lookup(pairwise_df: pd.DataFrame) -> dict:
    """
    Converts Phase 4 pairwise rho_q output into a lookup dictionary.

    columns:
        Asset_X, Asset_Y, q, scale, rho_q
    """
    lookup = {}

    for _, row in pairwise_df.iterrows():
        a = row["Asset_X"]
        b = row["Asset_Y"]
        q = row["q"]
        scale = row["scale"]
        rho = row["rho_q"]

        lookup[(_pair_key(a, b), q, scale)] = rho

    return lookup


def get_pairwise_rho(
    lookup: dict,
    asset_a: str,
    asset_b: str,
    q: float,
    scale: int,
) -> float:
    if asset_a == asset_b:
        return 1.0

    key = (_pair_key(asset_a, asset_b), q, scale)

    if key not in lookup:
        raise KeyError(
            f"Missing rho_q for pair=({asset_a}, {asset_b}), q={q}, scale={scale}"
        )

    return float(lookup[key])


def build_qdmc_components(
    lookup: dict,
    dependent: str,
    independents: list[str],
    q: float,
    scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds:

        Psi_{Y,X}
        R_X

    for one q and one scale n.
    """

    psi = np.array(
        [
            get_pairwise_rho(lookup, dependent, x, q, scale)
            for x in independents
        ],
        dtype=float,
    )

    k = len(independents)
    r_x = np.eye(k)

    for i in range(k):
        for j in range(k):
            if i != j:
                r_x[i, j] = get_pairwise_rho(
                    lookup,
                    independents[i],
                    independents[j],
                    q,
                    scale,
                )

    return psi, r_x

#### ver 0 - clipped and pseudo-inverse
#def compute_qdmc(
#    psi: np.ndarray,
#    r_x: np.ndarray,
#    regularization: float = 1e-8,
#) -> dict:
#    """
#    Computes:
#
#        qDMC_x^2(n) = Psi.T @ inv(R_X) @ Psi
#
#    Uses pseudo-inverse for numerical stability.
#    """
#
#    r_x_reg = r_x + regularization * np.eye(r_x.shape[0])
#
#    condition_number = np.linalg.cond(r_x_reg)
#    determinant = np.linalg.det(r_x_reg)
#    lambda_min = np.min(np.linalg.eigvalsh(r_x_reg))
#
#    r_x_inv = np.linalg.pinv(r_x_reg)
#
#    qdmc_raw = float(psi.T @ r_x_inv @ psi)
#
#    qdmc_clipped = float(np.clip(qdmc_raw, 0.0, 1.0))
#
#    return {
#        "qDMC_raw": qdmc_raw,
#        "qDMC": qdmc_clipped,
#        "lambda_min": lambda_min,
#        "determinant": determinant,
#        "condition_number": condition_number,
#    }

# ver 1 - new raw mathmatics

def compute_qdmc(
    psi: np.ndarray,
    r_x: np.ndarray,
    regularization: float = 0.0,
) -> dict:
    """
    Computes raw and diagnostic qDMC:

        qDMC_x^2(n) = Psi.T @ R_X^{-1} @ Psi

    No silent clipping is applied.
    """

    r_x_reg = r_x + regularization * np.eye(r_x.shape[0])

    eigvals = np.linalg.eigvalsh(r_x_reg)
    lambda_min = float(np.min(eigvals))
    determinant = float(np.linalg.det(r_x_reg))
    condition_number = float(np.linalg.cond(r_x_reg))

    # r_x_inv = np.linalg.pinv(r_x_reg)  ####
 
    try:
        r_x_inv = np.linalg.inv(r_x_reg)
        inverse_method = "inverse"
    except np.linalg.LinAlgError:
        r_x_inv = np.linalg.pinv(r_x_reg)
        inverse_method = "pseudo_inverse"

    qdmc_raw = float(psi.T @ r_x_inv @ psi)

    is_positive_definite = bool(lambda_min > 1e-10)
    is_in_theoretical_range = bool(0.0 <= qdmc_raw <= 1.0)

    return {
        "qDMC_raw": qdmc_raw,
        "qDMC_reported": qdmc_raw if is_in_theoretical_range else np.nan,
        "qDMC_clipped_for_plot": float(np.clip(qdmc_raw, 0.0, 1.0)),
        "lambda_min": lambda_min,
        "determinant": determinant,
        "condition_number": condition_number,
        "is_positive_definite": is_positive_definite,
        "is_in_theoretical_range": is_in_theoretical_range,
        "regularization": regularization,
        "inverse_method": inverse_method,  ###
    }


def interpret_qdmc(value: float) -> str:
    if value < 0.2:
        return "Very weak"
    if value < 0.4:
        return "Weak"
    if value < 0.6:
        return "Average"
    if value < 0.8:
        return "Strong"
    return "Very strong"