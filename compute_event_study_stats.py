#!/usr/bin/env python3
"""
Compute Pearson correlation (r), p-value, and R^2 for the event study series
of mean injury crashes per site within 300 meters, as a function of months
relative to activation.

Input: data/data_proc/event_study_injury.csv (columns: rel_month, injury_crash, n_sites)
Output: data/data_proc/event_study_injury_stats.json
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import time


PROC = Path("./data/data_proc")


def main():
    src = PROC / "event_study_injury.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}. Run the analysis pipeline first.")

    df = pd.read_csv(src)
    # Ensure required columns
    for c in ["rel_month", "injury_crash"]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in {src}")

    # Drop rows with NaNs in either series
    m = df[["rel_month", "injury_crash"]].dropna()
    if len(m) < 3:
        raise ValueError("Not enough data points to compute correlation.")

    x = m["rel_month"].astype(float).values
    y = m["injury_crash"].astype(float).values

    # Pearson correlation (NumPy)
    r = float(np.corrcoef(x, y)[0, 1])

    # Linear regression via NumPy polyfit (degree 1)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # Empirical p-value via permutation test (SciPy-free)
    rng = np.random.default_rng(42)
    num_perm = 5000
    count_extreme = 0
    start = time.time()
    for _ in range(num_perm):
        y_perm = rng.permutation(y)
        r_perm = np.corrcoef(x, y_perm)[0, 1]
        if abs(r_perm) >= abs(r):
            count_extreme += 1
    p = (count_extreme + 1) / (num_perm + 1)  # add-one smoothing

    out = {
        "num_points": int(len(m)),
        "pearson_r": float(r),
        "pearson_p_value": float(p),
        "linear_regression_slope": float(slope),
        "linear_regression_intercept": float(intercept),
        "linear_regression_r2": float(r2),
        "p_value_method": "permutation_test",
        "permutations": num_perm,
    }

    dst = PROC / "event_study_injury_stats.json"
    with dst.open("w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"Saved stats to {dst}")


if __name__ == "__main__":
    main()


