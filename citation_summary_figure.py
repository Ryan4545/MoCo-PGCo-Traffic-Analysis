#!/usr/bin/env python3
"""
Create a publication-ready figure summarizing where and how often speed-camera
citations occur in Montgomery County, MD (2023-2024).

Outputs: data/figures/citations_summary_2023_2024.png
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FIG = Path("./data/figures")
FIG.mkdir(parents=True, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    for enc in [None, "utf-8", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            last_err = None
            continue
    # Fallback single try without encoding override
    return pd.read_csv(path, low_memory=False)


def load_site_geo(path="./data/moco_ase_sites_geo.csv") -> pd.DataFrame:
    df = read_csv(path)
    # Ensure expected columns
    required = {"site_id", "lat", "lon", "citations"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    # Clean types
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["citations"] = pd.to_numeric(df["citations"], errors="coerce").fillna(0)
    df = df.dropna(subset=["lat", "lon"])
    # Filter to plausible MoCo bounds for a clean map frame
    df = df[(df["lat"].between(38.9, 39.35)) & (df["lon"].between(-77.5, -76.8))]
    return df


def load_quarterly(path="./data/moco_ase_total_by_quarter.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["quarter_start", "citations"]).assign(quarter_start=pd.to_datetime([]))
    df = read_csv(path)
    if "quarter_start" not in df.columns or "citations" not in df.columns:
        return pd.DataFrame(columns=["quarter_start", "citations"]).assign(quarter_start=pd.to_datetime([]))
    df["quarter_start"] = pd.to_datetime(df["quarter_start"], errors="coerce")
    df["citations"] = pd.to_numeric(df["citations"], errors="coerce").fillna(0)
    # Keep only 2023-2024 if present
    mask = df["quarter_start"].dt.year.isin([2023, 2024])
    df = df[mask].sort_values("quarter_start")
    return df


def build_figure():
    sites = load_site_geo()
    q = load_quarterly()

    # Compute Q4 2024 total (if available)
    q4_total = None
    if len(q) > 0 and q["quarter_start"].notna().any():
        quarters = q["quarter_start"].dt.to_period("Q")
        mask_q4_2024 = quarters.astype(str) == "2024Q4"
        if mask_q4_2024.any():
            q4_total = int(q.loc[mask_q4_2024, "citations"].sum())

    # Single-panel map figure
    fig = plt.figure(figsize=(10, 7))
    ax_map = fig.add_subplot(1, 1, 1)
    ax_map.set_title("Speed-camera sites sized by citations (2023–2024)")
    # Size scaling: square-root to reduce skew
    sizes = np.sqrt(sites["citations"].clip(lower=0))
    if sizes.max() > 0:
        sizes = 20 + 160 * (sizes / sizes.max())
    else:
        sizes = np.full(len(sites), 20.0)
    sc = ax_map.scatter(
        sites["lon"], sites["lat"], s=sizes, c=sites["citations"], cmap="viridis",
        alpha=0.75, edgecolor="k", linewidth=0.5
    )
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("Citations per site (total)")

    # Add a minimal bounding box of points
    if not sites.empty:
        ax_map.set_xlim(sites["lon"].min() - 0.02, sites["lon"].max() + 0.02)
        ax_map.set_ylim(sites["lat"].min() - 0.02, sites["lat"].max() + 0.02)

    # Label top N by citations to avoid overcrowding
    top_n = 15
    top_sites = sites.sort_values("citations", ascending=False).head(top_n)
    # Alternate text offsets to reduce overlap visually
    offsets = [(6, 6), (6, -6), (-6, 6), (-6, -6)]
    for i, (_, row) in enumerate(top_sites.iterrows()):
        dx, dy = offsets[i % len(offsets)]
        ax_map.annotate(
            str(row["site_id"]),
            (row["lon"], row["lat"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)
        )

    # Suptitle with optional Q4 2024 total
    if q4_total is not None:
        fig.suptitle(f"Montgomery County, MD — Speed-camera citations (2023–2024)\nQ4 2024 total citations: {q4_total:,}", fontsize=14, y=0.98)
    else:
        fig.suptitle("Montgomery County, MD — Speed-camera citations (2023–2024)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = FIG / "citations_summary_2023_2024.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    build_figure()


