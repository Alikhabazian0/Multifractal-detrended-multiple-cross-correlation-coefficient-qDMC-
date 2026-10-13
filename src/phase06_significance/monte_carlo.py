import numpy as np


def simulate_null_data(
    n_obs: int,
    n_sim: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate the null hypothesis.

    Returns
    -------
    ndarray
        Shape = (n_sim, 3, n_obs)

        [:,0,:] = Y
        [:,1,:] = X1
        [:,2,:] = X2
    """
    rng = np.random.default_rng(seed)

    simulations = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_sim, 3, n_obs),
    )

    return simulations