from __future__ import annotations

import os
import sys
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Geo / ML
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import networkx as nx

# --------- CONFIG IMPORT ----------
try:
    import config  # rename config_template.py to config.py
except ImportError:
    print("ERROR: Could not import config.py. Please rename config_template.py -> config.py and edit it.")
    sys.exit(1)

PATHS = config.PATHS
COLS = config.COLS
SETTINGS = config.SETTINGS


# ----------------------------
# Utilities
# ----------------------------

def ensure_dirs() -> None:
    PATHS.OUT_DIR.mkdir(parents=True, exist_ok=True)
    PATHS.FIG_DIR.mkdir(parents=True, exist_ok=True)
    PATHS.MODEL_DIR.mkdir(parents=True, exist_ok=True)


def safe_parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def to_geodf_from_latlon(df: pd.DataFrame, lat_col: str, lon_col: str, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Missing lat/lon columns: {lat_col}, {lon_col}")
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs
    )
    return gdf


def to_geodf_from_wkt(df: pd.DataFrame, wkt_col: str, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    if wkt_col not in df.columns or not wkt_col:
        raise ValueError("WKT geometry column not configured or missing.")
    geom = df[wkt_col].apply(lambda x: wkt.loads(x) if isinstance(x, str) and x.strip() else None)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom, crs=crs)
    return gdf


def pick_projected_crs(gdf: gpd.GeoDataFrame) -> str:
    return "EPSG:3857"


def set_injury_flag(crashes: pd.DataFrame) -> pd.Series:
    if COLS.CRASH_INJURY_FLAG and COLS.CRASH_INJURY_FLAG in crashes.columns:
        s = crashes[COLS.CRASH_INJURY_FLAG]
        if s.dtype == bool:
            return s
        # accept 1/0 or strings
        return s.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})
    if COLS.CRASH_SEVERITY and COLS.CRASH_SEVERITY in crashes.columns:
        return crashes[COLS.CRASH_SEVERITY].isin(COLS.INJURY_SEVERITY_VALUES)
    raise ValueError("No injury indicator available. Set COLS.CRASH_INJURY_FLAG or COLS.CRASH_SEVERITY in config.py.")


# ----------------------------
# Loaders
# ----------------------------

def load_crashes() -> gpd.GeoDataFrame:
    df = pd.read_csv(PATHS.CRASHES_CSV)
    if COLS.CRASH_DATE not in df.columns:
        raise ValueError(f"Crash date column '{COLS.CRASH_DATE}' not found in crashes CSV.")
    df[COLS.CRASH_DATE] = safe_parse_datetime(df[COLS.CRASH_DATE])
    df = df.dropna(subset=[COLS.CRASH_DATE])

    # Injury flag
    df["is_injury_crash"] = set_injury_flag(df).astype(int)

    # Geometry
    if COLS.CRASH_GEOM_WKT:
        gdf = to_geodf_from_wkt(df, COLS.CRASH_GEOM_WKT)
    else:
        gdf = to_geodf_from_latlon(df, COLS.CRASH_LAT, COLS.CRASH_LON)

    return gdf


def load_cameras() -> gpd.GeoDataFrame:
    df = pd.read_csv(PATHS.CAMERAS_CSV)
    for req in [COLS.CAMERA_ID, COLS.CAMERA_ACTIVATION_DATE]:
        if req not in df.columns:
            raise ValueError(f"Camera required column '{req}' not found in cameras CSV.")

    df[COLS.CAMERA_ACTIVATION_DATE] = safe_parse_datetime(df[COLS.CAMERA_ACTIVATION_DATE])
    df = df.dropna(subset=[COLS.CAMERA_ACTIVATION_DATE])

    if COLS.CAMERA_GEOM_WKT:
        gdf = to_geodf_from_wkt(df, COLS.CAMERA_GEOM_WKT)
    else:
        gdf = to_geodf_from_latlon(df, COLS.CAMERA_LAT, COLS.CAMERA_LON)

    return gdf


def load_optional_geo(path: Path) -> Optional[gpd.GeoDataFrame]:
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    return gdf


def load_optional_citations() -> Optional[pd.DataFrame]:
    if not PATHS.CITATIONS_CSV.exists():
        return None
    c = pd.read_csv(PATHS.CITATIONS_CSV)
    # optional date parsing
    if COLS.CITATION_DATE in c.columns:
        c[COLS.CITATION_DATE] = safe_parse_datetime(c[COLS.CITATION_DATE])
    return c


# ----------------------------
# Spatial joins + aggregation
# ----------------------------

def link_crashes_to_cameras(
    crashes_gdf: gpd.GeoDataFrame,
    cameras_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    proj = pick_projected_crs(cameras_gdf)

    cameras_p = cameras_gdf.to_crs(proj)
    crashes_p = crashes_gdf.to_crs(proj)

    cameras_p["camera_buffer"] = cameras_p.geometry.buffer(SETTINGS.CAMERA_BUFFER_METERS)
    buffers = cameras_p[[COLS.CAMERA_ID, COLS.CAMERA_ACTIVATION_DATE, "camera_buffer"]].set_geometry("camera_buffer")

    joined = gpd.sjoin(crashes_p, buffers, how="inner", predicate="within")
    # Keep original crash geometry as well
    joined = joined.drop(columns=["index_right"], errors="ignore")
    return joined


def aggregate_pre_post(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    d = joined.copy()
    d["days_from_activation"] = (d[COLS.CRASH_DATE] - d[COLS.CAMERA_ACTIVATION_DATE]).dt.days

    pre_mask = (d["days_from_activation"] < 0) & (d["days_from_activation"] >= -SETTINGS.PRE_WINDOW_DAYS)
    post_mask = (d["days_from_activation"] >= 0) & (d["days_from_activation"] <= SETTINGS.POST_WINDOW_DAYS)

    # Injury crashes only
    d_inj = d[d["is_injury_crash"] == 1]

    pre = (
        d_inj[pre_mask]
        .groupby(COLS.CAMERA_ID)
        .size()
        .rename("injury_crashes_pre")
    )
    post = (
        d_inj[post_mask]
        .groupby(COLS.CAMERA_ID)
        .size()
        .rename("injury_crashes_post")
    )

    out = pd.concat([pre, post], axis=1).fillna(0).reset_index()
    out["injury_crashes_pre"] = out["injury_crashes_pre"].astype(int)
    out["injury_crashes_post"] = out["injury_crashes_post"].astype(int)

    # Avoid divide by zero
    out["injury_reduction_rate"] = np.where(
        out["injury_crashes_pre"] > 0,
        (out["injury_crashes_pre"] - out["injury_crashes_post"]) / out["injury_crashes_pre"],
        np.nan
    )

    return out


def attach_camera_metadata(site_df: pd.DataFrame, cameras_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    cams = cameras_gdf.copy()
    cams = cams.drop(columns=["geometry"], errors="ignore")
    keep_cols = [c for c in cams.columns if c in {
        COLS.CAMERA_ID,
        COLS.CAMERA_ACTIVATION_DATE,
        COLS.CAMERA_TYPE,
        COLS.CAMERA_LAT,
        COLS.CAMERA_LON
    } or c == COLS.CAMERA_ID]

    cams = cams[keep_cols].drop_duplicates(subset=[COLS.CAMERA_ID])
    merged = site_df.merge(cams, on=COLS.CAMERA_ID, how="left")

    merged["months_active"] = (
        (pd.Timestamp.today() - merged[COLS.CAMERA_ACTIVATION_DATE]) / np.timedelta64(1, "M")
    ).clip(lower=0)

    merged["months_active"] = merged["months_active"].fillna(0)

    # pre injury rate per month within pre-window (approx)
    merged["pre_injury_rate"] = merged["injury_crashes_pre"] / max(1, SETTINGS.PRE_WINDOW_DAYS / 30.0)

    return merged


def attach_roads_class(site_df: pd.DataFrame, cameras_gdf: gpd.GeoDataFrame, roads_gdf: Optional[gpd.GeoDataFrame]) -> pd.DataFrame:
    if roads_gdf is None:
        site_df["road_type"] = "unknown"
        return site_df

    proj = pick_projected_crs(cameras_gdf)
    cams_p = cameras_gdf.to_crs(proj)
    roads_p = roads_gdf.to_crs(proj)

    # Use nearest join
    cams_p = cams_p[[COLS.CAMERA_ID, "geometry"]].copy()
    roads_p = roads_p[[COLS.ROAD_CLASS, "geometry"]].copy()

    nearest = gpd.sjoin_nearest(cams_p, roads_p, how="left", distance_col="dist_to_road")
    nearest = nearest.drop(columns=["index_right"], errors="ignore")
    nearest = nearest[[COLS.CAMERA_ID, COLS.ROAD_CLASS]].rename(columns={COLS.ROAD_CLASS: "road_type"})

    out = site_df.merge(nearest, on=COLS.CAMERA_ID, how="left")
    out["road_type"] = out["road_type"].fillna("unknown")
    return out


def attach_school_zone(site_df: pd.DataFrame, cameras_gdf: gpd.GeoDataFrame, school_gdf: Optional[gpd.GeoDataFrame]) -> pd.DataFrame:
    if school_gdf is None:
        site_df["school_zone"] = 0
        return site_df

    proj = pick_projected_crs(cameras_gdf)
    cams_p = cameras_gdf.to_crs(proj)
    school_p = school_gdf.to_crs(proj)

    cams_p = cams_p[[COLS.CAMERA_ID, "geometry"]].copy()
    school_p = school_p[["geometry"]].copy()

    joined = gpd.sjoin(cams_p, school_p, how="left", predicate="within")
    joined["school_zone"] = joined["index_right"].notna().astype(int)
    joined = joined.drop(columns=["index_right"], errors="ignore")
    out = site_df.merge(joined[[COLS.CAMERA_ID, "school_zone"]], on=COLS.CAMERA_ID, how="left")
    out["school_zone"] = out["school_zone"].fillna(0).astype(int)
    return out


def attach_landuse_density(site_df: pd.DataFrame, cameras_gdf: gpd.GeoDataFrame, landuse_gdf: Optional[gpd.GeoDataFrame]) -> pd.DataFrame:
    if landuse_gdf is None:
        site_df["commercial_density"] = 0.0
        return site_df

    proj = pick_projected_crs(cameras_gdf)
    cams_p = cameras_gdf.to_crs(proj)
    land_p = landuse_gdf.to_crs(proj)

    cams_p["buf"] = cams_p.geometry.buffer(300)  # 300m neighborhood context
    cams_buf = cams_p[[COLS.CAMERA_ID, "buf"]].set_geometry("buf")

    land_p = land_p[["geometry"]].copy()

    joined = gpd.sjoin(land_p, cams_buf, how="inner", predicate="intersects")
    dens = joined.groupby(COLS.CAMERA_ID).size().rename("commercial_density").reset_index()

    out = site_df.merge(dens, on=COLS.CAMERA_ID, how="left")
    out["commercial_density"] = out["commercial_density"].fillna(0.0)
    return out


def attach_citation_volume(site_df: pd.DataFrame, citations_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if citations_df is None:
        site_df["citation_volume"] = np.nan
        return site_df

    if COLS.CITATION_CAMERA_ID not in citations_df.columns or COLS.CITATION_COUNT not in citations_df.columns:
        site_df["citation_volume"] = np.nan
        return site_df

    # Total citations per site (you can change to pre/post window totals)
    agg = citations_df.groupby(COLS.CITATION_CAMERA_ID)[COLS.CITATION_COUNT].sum().rename("citation_volume").reset_index()
    agg = agg.rename(columns={COLS.CITATION_CAMERA_ID: COLS.CAMERA_ID})

    out = site_df.merge(agg, on=COLS.CAMERA_ID, how="left")
    return out


# ----------------------------
# Supervised learning + evaluation
# ----------------------------

def prepare_model_frame(site_df: pd.DataFrame) -> pd.DataFrame:
    df = site_df.copy()

    # basic filters for stability
    df = df[df["injury_crashes_pre"] >= SETTINGS.MIN_PRE_INJURY_CRASHES].copy()
    df = df.dropna(subset=["injury_reduction_rate"])

    # target
    df["successful_site"] = (df["injury_reduction_rate"] > SETTINGS.SUCCESS_DROP_THRESHOLD).astype(int)

    # encode road type
    df["road_type"] = df.get("road_type", "unknown").fillna("unknown").astype(str)
    df["road_type_encoded"] = df["road_type"].astype("category").cat.codes

    # citations: fill missing with median (or 0). keep as feature but acknowledge missingness
    if "citation_volume" in df.columns:
        med = df["citation_volume"].median()
        df["citation_volume"] = df["citation_volume"].fillna(med)
    else:
        df["citation_volume"] = 0.0

    # months_active
    df["months_active"] = pd.to_numeric(df["months_active"], errors="coerce").fillna(0)

    # ensure numeric
    for c in ["pre_injury_rate", "commercial_density"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["school_zone"] = df.get("school_zone", 0).fillna(0).astype(int)

    return df


def train_and_evaluate_models(df: pd.DataFrame) -> Dict[str, Any]:
    features = [
        "road_type_encoded",
        "school_zone",
        "citation_volume",
        "months_active",
        "pre_injury_rate",
        "commercial_density",
    ]

    X = df[features]
    y = df["successful_site"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SETTINGS.RANDOM_STATE, stratify=y
    )

    # Logistic Regression
    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(X_train, y_train)
    log_probs = log_model.predict_proba(X_test)[:, 1]
    log_auc = roc_auc_score(y_test, log_probs)

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=400,
        random_state=SETTINGS.RANDOM_STATE,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    rf_model.fit(X_train, y_train)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)

    # Save evaluation plots
    fig_path = PATHS.FIG_DIR / "roc_curves.png"
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, log_probs, name=f"LogReg (AUC={log_auc:.3f})")
    RocCurveDisplay.from_predictions(y_test, rf_probs, name=f"RandForest (AUC={rf_auc:.3f})")
    plt.title("ROC Curves — Predicting Successful Camera Sites")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Threshold-based reporting (0.5 default)
    def report(model_name: str, probs: np.ndarray) -> Dict[str, Any]:
        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, preds)
        rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
        return {"confusion_matrix": cm.tolist(), "classification_report": rep}

    log_report = report("logistic_regression", log_probs)
    rf_report = report("random_forest", rf_probs)

    # Coefficients / feature importance plots
    coef_path = PATHS.FIG_DIR / "logreg_coefficients.png"
    plt.figure()
    coefs = pd.Series(log_model.coef_[0], index=features).sort_values()
    coefs.plot(kind="barh")
    plt.title("Logistic Regression Coefficients (Direction of Effect)")
    plt.savefig(coef_path, dpi=200, bbox_inches="tight")
    plt.close()

    imp_path = PATHS.FIG_DIR / "rf_feature_importance.png"
    plt.figure()
    imps = pd.Series(rf_model.feature_importances_, index=features).sort_values()
    imps.plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.savefig(imp_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Persist model artifacts
    summary = {
        "features": features,
        "log_auc": float(log_auc),
        "rf_auc": float(rf_auc),
        "log_report": log_report,
        "rf_report": rf_report,
        "figures": {
            "roc_curves": str(fig_path),
            "logreg_coefficients": str(coef_path),
            "rf_feature_importance": str(imp_path),
        },
    }

    with open(PATHS.MODEL_DIR / "model_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ----------------------------
# Clustering
# ----------------------------

def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    cluster_features = ["pre_injury_rate", "citation_volume", "months_active", "commercial_density"]
    X = df[cluster_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=SETTINGS.N_CLUSTERS, random_state=SETTINGS.RANDOM_STATE, n_init="auto")
    df["cluster"] = km.fit_predict(Xs)

    # cluster profile table
    prof = df.groupby("cluster")[cluster_features + ["successful_site"]].mean().reset_index()
    prof_path = PATHS.OUT_DIR / "cluster_profiles.csv"
    prof.to_csv(prof_path, index=False)

    # plot cluster sizes
    plt.figure()
    df["cluster"].value_counts().sort_index().plot(kind="bar")
    plt.title("Cluster Sizes of Enforcement Sites")
    plt.savefig(PATHS.FIG_DIR / "cluster_sizes.png", dpi=200, bbox_inches="tight")
    plt.close()

    return df


# ----------------------------
# Network / spillover exploration
# ----------------------------

def build_camera_network(cameras_gdf: gpd.GeoDataFrame) -> nx.Graph:
    proj = pick_projected_crs(cameras_gdf)
    cams_p = cameras_gdf.to_crs(proj).copy()
    cams_p["x"] = cams_p.geometry.x
    cams_p["y"] = cams_p.geometry.y

    ids = cams_p[COLS.CAMERA_ID].tolist()
    coords = cams_p[["x", "y"]].to_numpy()

    G = nx.Graph()
    for cid in ids:
        G.add_node(cid)

    thresh = SETTINGS.NETWORK_EDGE_DISTANCE_METERS
    # O(n^2) for simplicity; for large N, use spatial index / KDTree
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            if dist <= thresh:
                G.add_edge(ids[i], ids[j], distance_m=dist)

    return G


def spillover_proxy(site_df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    df = site_df.copy()
    succ = df.set_index(COLS.CAMERA_ID)["successful_site"].to_dict()

    neighbor_success = {}
    degree = {}

    for node in G.nodes:
        nbrs = list(G.neighbors(node))
        degree[node] = len(nbrs)
        if len(nbrs) == 0:
            neighbor_success[node] = np.nan
        else:
            neighbor_success[node] = float(np.mean([succ.get(n, np.nan) for n in nbrs]))

    df["network_degree"] = df[COLS.CAMERA_ID].map(degree).fillna(0).astype(int)
    df["neighbor_success_rate"] = df[COLS.CAMERA_ID].map(neighbor_success)

    # quick plot: neighbor_success_rate vs injury_reduction_rate
    plt.figure()
    plt.scatter(df["neighbor_success_rate"], df["injury_reduction_rate"])
    plt.title("Exploratory Spillover Proxy: Neighbor Success vs Site Injury Reduction")
    plt.xlabel("Neighbor Success Rate (mean)")
    plt.ylabel("Site Injury Reduction Rate")
    plt.savefig(PATHS.FIG_DIR / "spillover_proxy_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()

    return df


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ensure_dirs()

    print("Loading crashes...")
    crashes = load_crashes()
    print(f"  crashes: {len(crashes):,}")

    print("Loading cameras...")
    cameras = load_cameras()
    print(f"  cameras: {len(cameras):,}")

    print("Loading optional spatial layers...")
    roads = load_optional_geo(PATHS.ROADS_FILE)
    school = load_optional_geo(PATHS.SCHOOLZONES_FILE)
    landuse = load_optional_geo(PATHS.LANDUSE_FILE)
    citations = load_optional_citations()

    print("Linking crashes to camera buffers...")
    joined = link_crashes_to_cameras(crashes, cameras)
    print(f"  crash-camera links: {len(joined):,}")

    print("Aggregating pre/post injury crashes at each camera site...")
    site = aggregate_pre_post(joined)

    print("Attaching camera metadata and engineered features...")
    site = attach_camera_metadata(site, cameras)
    site = attach_roads_class(site, cameras, roads)
    site = attach_school_zone(site, cameras, school)
    site = attach_landuse_density(site, cameras, landuse)
    site = attach_citation_volume(site, citations)

    # Prepare modeling dataset
    model_df = prepare_model_frame(site)

    # Save the modeling table
    model_table_path = PATHS.OUT_DIR / "camera_site_model_frame.csv"
    model_df.to_csv(model_table_path, index=False)
    print(f"Saved model frame: {model_table_path}")

    print("Training and evaluating models...")
    summary = train_and_evaluate_models(model_df)
    print(f"  LogReg AUC: {summary['log_auc']:.3f}")
    print(f"  RandForest AUC: {summary['rf_auc']:.3f}")

    print("Running clustering...")
    model_df = run_clustering(model_df)

    # Build network + spillover proxy
    print("Building camera proximity network...")
    G = build_camera_network(cameras)

    net_path = PATHS.OUT_DIR / "camera_network_stats.json"
    net_stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "edge_threshold_m": SETTINGS.NETWORK_EDGE_DISTANCE_METERS,
    }
    with open(net_path, "w") as f:
        json.dump(net_stats, f, indent=2)

    print("Computing spillover proxy features...")
    model_df = spillover_proxy(model_df, G)

    # Save final enriched dataset
    out_path = PATHS.OUT_DIR / "camera_sites_enriched.csv"
    model_df.to_csv(out_path, index=False)
    print(f"Saved enriched site table: {out_path}")

    # Also store your run settings for reproducibility
    settings_path = PATHS.OUT_DIR / "run_settings.json"
    with open(settings_path, "w") as f:
        json.dump(
            {
                "paths": {k: str(v) for k, v in asdict(PATHS).items()},
                "settings": asdict(SETTINGS),
                "columns": {k: (list(v) if isinstance(v, set) else v) for k, v in asdict(COLS).items()},
            },
            f,
            indent=2
        )

    print("Done. Figures saved in ./figures, outputs in ./data/processed, model summary in ./models.")


if __name__ == "__main__":
    main()
