from pathlib import Path
import pandas as pd

from src.analysis.rho_q import _profile, _segments_forward_reverse


def check_segments(returns_path, scales, output_path):
    returns_path = Path(returns_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(returns_path)
    cols = [c for c in df.columns if c.lower() not in ["date", "datetime", "time"]]

    x = df[cols[0]].dropna().to_numpy()
    X = _profile(x)
    N = len(X)

    rows = []

    for scale in scales:
        M_n = N // scale
        segments = _segments_forward_reverse(X, scale)

        forward = segments[:M_n]
        reverse = segments[M_n:]

        duplicate_count = 0
        for fs in forward:
            for rs in reverse:
                if len(fs) == len(rs) and (fs == rs).all():
                    duplicate_count += 1

        rows.append({
            "N": N,
            "scale_n": scale,
            "M_n_floor_N_over_n": M_n,
            "Expected_2M_n": 2 * M_n,
            "Actual_N_segments": len(segments),
            "Unused_points": N - M_n * scale,
            "Forward_Reverse_Exact_Duplicates": duplicate_count,
            "Is_N_Divisible_By_n": N % scale == 0,
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out