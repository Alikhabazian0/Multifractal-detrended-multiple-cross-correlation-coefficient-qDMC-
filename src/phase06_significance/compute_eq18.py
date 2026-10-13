import numpy as np

from src.analysis.rho_q import rho_q_coefficient
from src.analysis.qdmc import compute_qdmc


def compute_qdmc_step2_both_forms(
    y,
    x1,
    x2,
    q,
    scale,
    poly_order=1,
    regularization=0.0,
):
    rho_yx1 = rho_q_coefficient(y, x1, scale=scale, q=q, poly_order=poly_order)
    rho_yx2 = rho_q_coefficient(y, x2, scale=scale, q=q, poly_order=poly_order)
    rho_x1x2 = rho_q_coefficient(x1, x2, scale=scale, q=q, poly_order=poly_order)

    denom = 1.0 - rho_x1x2**2

    if abs(denom) < 1e-12:
        qdmc_eq18 = np.nan
    else:
        qdmc_eq18 = (
            rho_yx1**2
            + rho_yx2**2
            - 2.0 * rho_yx1 * rho_yx2 * rho_x1x2
        ) / denom

    psi = np.array([rho_yx1, rho_yx2], dtype=float)
    r_x = np.array(
        [[1.0, rho_x1x2],
         [rho_x1x2, 1.0]],
        dtype=float,
    )

    matrix_result = compute_qdmc(
        psi=psi,
        r_x=r_x,
        regularization=regularization,
    )

    qdmc_matrix = matrix_result["qDMC_raw"]
    difference = qdmc_matrix - qdmc_eq18

    return {
        "q": q,
        "scale": scale,
        "rho_yx1": rho_yx1,
        "rho_yx2": rho_yx2,
        "rho_x1x2": rho_x1x2,
        "qDMC_eq18": qdmc_eq18,
        "qDMC_matrix": qdmc_matrix,
        "difference_matrix_minus_eq18": difference,
        **matrix_result,
    }