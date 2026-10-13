import numpy as np
import pandas as pd


def _clean_pair(x, y):
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    return df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)


def _profile(series):
    series = np.asarray(series, dtype=float)
    # series = series - np.mean(series)
    return np.cumsum(series)


def _detrend_segment(segment, poly_order=1):
    t = np.arange(len(segment), dtype=float)

    if len(segment) <= poly_order + 1:
        return segment - np.mean(segment)

    coeffs = np.polyfit(t, segment, deg=poly_order)
    trend = np.polyval(coeffs, t)

    return segment - trend


def _segments_forward_reverse(profile, scale):
    n = len(profile)
    m_n = n // scale

    if m_n < 2:
        raise ValueError("Scale too large: fewer than 2 segments.")

    segments = []

    # Forward segmentation
    for v in range(m_n):
        start = v * scale
        end = start + scale
        segments.append(profile[start:end])

    # Reverse segmentation
    for v in range(m_n):
        end = n - v * scale
        start = end - scale
        segments.append(profile[start:end])

    return segments


def local_fluctuations(x, y, scale, poly_order=1):
    """
    Returns local detrended covariance and variances:
    f_xy(v,n), f_xx(v,n), f_yy(v,n)
    for 2M_n forward/reverse segments.
    """
    x, y = _clean_pair(x, y)

    n_obs = min(len(x), len(y))
    x = x[:n_obs]
    y = y[:n_obs]

    if scale >= n_obs:
        raise ValueError("scale must be smaller than the series length")

    X = _profile(x)
    Y = _profile(y)

    X_segments = _segments_forward_reverse(X, scale)
    Y_segments = _segments_forward_reverse(Y, scale)

    f_xy = []
    f_xx = []
    f_yy = []

    for xs, ys in zip(X_segments, Y_segments):
        dx = _detrend_segment(xs, poly_order=poly_order)
        dy = _detrend_segment(ys, poly_order=poly_order)

        f_xy.append(np.mean(dx * dy))
        f_xx.append(np.mean(dx ** 2))
        f_yy.append(np.mean(dy ** 2))

    return (
        np.asarray(f_xy, dtype=float),
        np.asarray(f_xx, dtype=float),
        np.asarray(f_yy, dtype=float),
    )


def generalized_fluctuation_functions(f_xy, f_xx, f_yy, q):
    """
    Implements F^q_XY(n), F^q_XX(n), F^q_YY(n).

    For q != 0:
        F^q_XX(n) = mean(f_xx^(q/2))
        F^q_YY(n) = mean(f_yy^(q/2))
        F^q_XY(n) = mean(sign(f_xy) * |f_xy|^(q/2))

    For q = 0:
        Uses log/geometric averaging.
    """
    eps = 1e-12

    f_xx = np.maximum(f_xx, eps)
    f_yy = np.maximum(f_yy, eps)

    if q == 0:
        F_xx = np.exp(0.5 * np.mean(np.log(f_xx)))
        F_yy = np.exp(0.5 * np.mean(np.log(f_yy)))
        F_xy = np.mean(np.sign(f_xy) * np.exp(0.5 * np.log(np.abs(f_xy) + eps)))
    else:
        F_xx = np.mean(f_xx ** (q / 2.0))
        F_yy = np.mean(f_yy ** (q / 2.0))
        F_xy = np.mean(np.sign(f_xy) * (np.abs(f_xy) + eps) ** (q / 2.0))

    return F_xy, F_xx, F_yy


def rho_q_coefficient(x, y, scale, q=2, poly_order=1):
    """
    Computes rho_q(n), the multifractal detrended cross-correlation coefficient.
    """
    f_xy, f_xx, f_yy = local_fluctuations(
        x=x,
        y=y,
        scale=scale,
        poly_order=poly_order,
    )

    F_xy, F_xx, F_yy = generalized_fluctuation_functions(
        f_xy=f_xy,
        f_xx=f_xx,
        f_yy=f_yy,
        q=q,
    )

    denom = np.sqrt(F_xx * F_yy)

    if denom == 0 or not np.isfinite(denom):
        return np.nan

    rho = F_xy / denom

    # Paper correction: keep rho_q inside [-1, 1].
    if q < 0 and abs(rho) > 1:
        rho = 1.0 / rho

    # return float(np.clip(rho, -1, 1)) - ver 1 - clipped rho
    return float(rho)


def rho_q_by_scales(x, y, scales, q_values, poly_order=1):
    rows = []

    for q in q_values:
        for scale in scales:
            try:
                rho = rho_q_coefficient(
                    x=x,
                    y=y,
                    scale=scale,
                    q=q,
                    poly_order=poly_order,
                )
            except ValueError:
                rho = np.nan

            rows.append({
                "q": q,
                "scale": scale,
                "rho_q": rho,
            })

    return pd.DataFrame(rows)

## debuger helpers

def debug_rho_components(x, y, scale, q=0, poly_order=1):
    """
    Debug helper for rho_q(n).
    Returns F_xy, F_xx, F_yy, denominator, raw rho, and clipped rho.
    """
    f_xy, f_xx, f_yy = local_fluctuations(
        x=x,
        y=y,
        scale=scale,
        poly_order=poly_order,
    )

    F_xy, F_xx, F_yy = generalized_fluctuation_functions(
        f_xy=f_xy,
        f_xx=f_xx,
        f_yy=f_yy,
        q=q,
    )

    denominator = np.sqrt(F_xx * F_yy)
    raw_rho = F_xy / denominator

    corrected_rho = raw_rho
    if q < 0 and abs(corrected_rho) > 1:
        corrected_rho = 1.0 / corrected_rho

    # clipped_rho = float(np.clip(corrected_rho, -1, 1))

    return {
        "scale": scale,
        "q": q,
        "F_xy": F_xy,
        "F_xx": F_xx,
        "F_yy": F_yy,
        "denominator": denominator,
        "raw_rho": raw_rho,
        "corrected_rho": corrected_rho,
        #"clipped_rho": clipped_rho,
        "rho_q": corrected_rho,
        "difference_Fxy_minus_denominator": F_xy - denominator,
    }