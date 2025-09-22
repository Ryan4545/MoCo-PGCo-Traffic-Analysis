
import os, math, json
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIG = Path("./data/figures")
PROC = Path("./data/data_proc")
FIG.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

def read_csv(path: str) -> pd.DataFrame:
    for enc in [None, "utf-8", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise last_err

def to_month(series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def extract_lat_lon(df: pd.DataFrame):
    lat, lon = None, None
    for cand in df.columns:
        cl = cand.lower()
        if cl in ("lat","latitude","y","ycoord","y_coordinate"):
            lat = cand if lat is None else lat
        if cl in ("lon","long","longitude","x","xcoord","x_coordinate"):
            lon = cand if lon is None else lon
    if lat is None or lon is None:
        for c in df.columns:
            if "location" in c.lower():
                def parse_loc(val):
                    if pd.isna(val): return np.nan, np.nan
                    s = str(val)
                    if "POINT" in s.upper() and "(" in s and ")" in s:
                        inside = s[s.find("(")+1:s.find(")")].strip()
                        parts = inside.replace(","," ").split()
                        if len(parts) >= 2:
                            try:
                                lon_v = float(parts[0]); lat_v = float(parts[1])
                                return lat_v, lon_v
                            except: return np.nan, np.nan
                    if "," in s and "(" in s and ")" in s:
                        inside = s[s.find("(")+1:s.find(")")]
                        parts = [p.strip() for p in inside.split(",")]
                        if len(parts) >= 2:
                            try:
                                lat_v = float(parts[0]); lon_v = float(parts[1])
                                return lat_v, lon_v
                            except: return np.nan, np.nan
                    if "{" in s and "coordinates" in s:
                        try:
                            import json
                            obj = json.loads(s)
                            coords = obj.get("coordinates", None)
                            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                                lon_v, lat_v = float(coords[0]), float(coords[1])
                                return lat_v, lon_v
                        except: return np.nan, np.nan
                    return np.nan, np.nan
                lat_series, lon_series = zip(*df[c].map(parse_loc))
                df["_lat_from_location"] = lat_series
                df["_lon_from_location"] = lon_series
                if lat is None: lat = "_lat_from_location"
                if lon is None: lon = "_lon_from_location"
                break
    return lat, lon

def load_moco_ase(path="./data/moco_ase_sites_geo.csv") -> pd.DataFrame:
    """Load MoCo ASE data from the available CSV file"""
    df = read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    
    # Extract lat/lon from the available columns
    lat_col, lon_col = extract_lat_lon(df)
    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    
    # Use site_id from the data
    if "site_id" not in df.columns:
        df["site_id"] = df["lat"].round(4).astype(str) + "," + df["lon"].round(4).astype(str)
    
    # Get citations from the data
    if "citations" in df.columns:
        df["citations"] = pd.to_numeric(df["citations"], errors="coerce").fillna(0).astype(int)
    else:
        df["citations"] = 0
    
    # Load quarterly data if available
    quarterly_path = "./data/moco_ase_total_by_quarter.csv"
    if os.path.exists(quarterly_path):
        quarterly_df = read_csv(quarterly_path)
        quarterly_df.columns = [c.lower() for c in quarterly_df.columns]
        quarterly_df["quarter_start"] = pd.to_datetime(quarterly_df["quarter_start"])
        
        # Create a cross-product of sites and quarters for proper time series
        site_df = df[["site_id", "lat", "lon"]].copy()
        quarterly_expanded = []
        
        for _, quarter_row in quarterly_df.iterrows():
            quarter_start = quarter_row["quarter_start"]
            total_citations = quarter_row["citations"]
            
            # Distribute citations proportionally based on site citation weights
            site_weights = df["citations"] / df["citations"].sum() if df["citations"].sum() > 0 else 1/len(df)
            site_citations = (site_weights * total_citations).round().astype(int)
            
            for _, site_row in site_df.iterrows():
                quarterly_expanded.append({
                    "site_id": site_row["site_id"],
                    "lat": site_row["lat"],
                    "lon": site_row["lon"],
                    "citations": site_citations[site_row.name],
                    "quarter_start": quarter_start
                })
        
        df = pd.DataFrame(quarterly_expanded)
    else:
        # Fallback: use default date if no quarterly data
        df["quarter_start"] = pd.Timestamp("2024-01-01")
    
    # CRITICAL LIMITATION: No historical activation dates available in the data
    # This is a major limitation for before/after analysis
    # For now, we'll use a placeholder that indicates this limitation
    df["activation_quarter"] = pd.Timestamp("2020-01-01")  # PLACEHOLDER - NOT REAL DATA
    df["activation_note"] = "NO HISTORICAL ACTIVATION DATA AVAILABLE"
    
    return df

def load_moco_crashes(path_full="./data/bhju-22kf_full.csv", fallback="./data/bhju-22kf.csv") -> pd.DataFrame:
    """Load MoCo crash data from the available CSV file"""
    path = path_full if os.path.exists(path_full) else fallback
    df = read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    
    # Find the datetime column
    dt_col = "crash_date_time" if "crash_date_time" in df.columns else None
    if dt_col is None:
        for c in df.columns:
            if "date" in c and "time" in c:
                dt_col = c; break
    
    # Parse datetime and create crash_month
    df["crash_datetime"] = pd.to_datetime(df[dt_col], errors="coerce") if dt_col else pd.NaT
    df["crash_month"] = df["crash_datetime"].dt.to_period("M").dt.to_timestamp()
    
    # Extract lat/lon coordinates
    lat_col, lon_col = extract_lat_lon(df)
    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    
    # Determine injury crashes based on ACRS report type
    if "acrs_report_type" in df.columns:
        s = df["acrs_report_type"].astype(str).str.upper()
        df["injury_crash"] = s.str.contains("INJURY|FATAL", na=False)
        df["ksi_crash"] = s.str.contains("FATAL|INCAPACITATING", na=False)
    else:
        # Fallback: look for other injury indicators
        inj_col = None
        for c in ["injury_severity","severity","most_severe_injury"]:
            if c in df.columns: 
                inj_col = c
                break
        if inj_col:
            s = df[inj_col].astype(str).str.upper()
            df["injury_crash"] = s.str.contains("INJURY|FATAL", na=False)
            df["ksi_crash"] = s.str.contains("FATAL|INCAPACITATING", na=False)
        else:
            df["injury_crash"] = False
            df["ksi_crash"] = False
    
    return df

def crashes_within_radius(crashes: pd.DataFrame, cams_geo: pd.DataFrame, radius_m: float=300.0) -> pd.DataFrame:
    cams = cams_geo.dropna(subset=["lat","lon"]).copy()
    cams = cams[(cams["lat"].between(38, 40)) & (cams["lon"].between(-78, -76))]
    out_rows = []
    crash_pts = crashes.dropna(subset=["lat","lon","crash_month"]).copy()
    crash_pts = crash_pts[(crash_pts["lat"].between(38, 40)) & (crash_pts["lon"].between(-78, -76))]
    cam_lat = cams["lat"].values
    cam_lon = cams["lon"].values
    cam_site = cams["site_id"].values
    for idx, row in crash_pts.iterrows():
        d_km = haversine_km(row["lat"], row["lon"], cam_lat, cam_lon)
        j = int(np.argmin(d_km))
        dmin_m = float(d_km[j]*1000.0)
        if dmin_m <= radius_m:
            out_rows.append({
                "crash_month": row["crash_month"],
                "injury_crash": row.get("injury_crash", False),
                "ksi_crash": row.get("ksi_crash", False),
                "site_id": cam_site[j],
                "dist_m": dmin_m
            })
    return pd.DataFrame(out_rows)

def build_event_study(crash_join: pd.DataFrame, activation: pd.Series, pre: int=12, post: int=12, outcome_col="injury_crash") -> pd.DataFrame:
    df = crash_join.copy()
    x = activation.dropna().copy()
    act_m = x.apply(lambda d: pd.Timestamp(d.year, d.month, 1))
    df = df.merge(act_m.rename("act_month"), on="site_id", how="inner")
    df["rel_month"] = (df["crash_month"].dt.to_period("M").astype(int) - df["act_month"].dt.to_period("M").astype(int))
    df = df[(df["rel_month"]>=-pre) & (df["rel_month"]<=post)]
    grp = df.groupby(["site_id","rel_month"])[outcome_col].sum().reset_index()
    by_rel = grp.groupby("rel_month")[outcome_col].mean().reset_index()
    n_sites = grp.groupby("rel_month")["site_id"].nunique().reset_index().rename(columns={"site_id":"n_sites"})
    out = by_rel.merge(n_sites, on="rel_month", how="left")
    return out

def correlate_citations_with_crash_change(citations_df: pd.DataFrame, crash_join: pd.DataFrame, activation: pd.Series,
                                          pre_window=( -6, -1), post_window=(0, 6),
                                          outcome_col="injury_crash") -> pd.DataFrame:
    x = activation.dropna().copy()
    act_m = x.apply(lambda d: pd.Timestamp(d.year, d.month, 1))
    cj = crash_join.merge(act_m.rename("act_month"), on="site_id", how="inner")
    cj["rel_month"] = (cj["crash_month"].dt.to_period("M").astype(int) - cj["act_month"].dt.to_period("M").astype(int))
    
    def win_sum(lo, hi):
        m = cj[(cj["rel_month"]>=lo) & (cj["rel_month"]<=hi)].groupby("site_id")[outcome_col].sum()
        return m
    
    pre_sum = win_sum(*pre_window)
    post_sum = win_sum(*post_window)
    
    # Only calculate delta for sites that have data in both periods
    sites_with_both = pre_sum.index.intersection(post_sum.index)
    delta = (post_sum.loc[sites_with_both] - pre_sum.loc[sites_with_both]).rename("delta_"+outcome_col)
    
    cit = citations_df.copy()
    cit_site = cit.groupby(["site_id"])["citations"].sum().rename("citations_total_yr")
    
    # Merge and only include sites with valid delta calculations
    out = delta.to_frame().merge(cit_site, left_index=True, right_index=True, how="inner")
    
    return out

def run_all(moco_ase_path="./data/moco_ase_sites_geo.csv",
            moco_crash_path_full="./data/bhju-22kf_full.csv",
            moco_crash_path_fb="./data/bhju-22kf.csv",
            radius_m=300.0,
            pre=12, post=12):
    moco_ase = load_moco_ase(moco_ase_path)
    moco_crash = load_moco_crashes(moco_crash_path_full, moco_crash_path_fb)

    moco_ase.to_csv(PROC/"moco_ase_clean.csv", index=False)
    moco_crash.head(1000).to_csv(PROC/"moco_crash_head.csv", index=False)

    q_tot = moco_ase.groupby("quarter_start")["citations"].sum().reset_index()
    plt.figure(figsize=(9,5))
    plt.bar(q_tot["quarter_start"].astype(str), q_tot["citations"])
    plt.title("MoCo ASE citations by quarter")
    plt.xlabel("Quarter"); plt.ylabel("Citations")
    plt.tight_layout(); plt.savefig(FIG/"moco_ase_by_quarter_new.png"); plt.close()

    site_sum = moco_ase.groupby("site_id")["citations"].sum().reset_index()
    plt.figure(figsize=(9,5))
    plt.hist(site_sum["citations"].values, bins=25)
    plt.title("Citations per site (sum over available quarters)")
    plt.xlabel("Citations per site"); plt.ylabel("Number of sites")
    plt.tight_layout(); plt.savefig(FIG/"moco_ase_site_hist_new.png"); plt.close()

    cams_geo = moco_ase.groupby(["site_id","lat","lon"])["citations"].sum().reset_index()
    crash_join = crashes_within_radius(moco_crash, cams_geo, radius_m=radius_m)
    crash_join.to_csv(PROC/"moco_crashes_joined_to_sites.csv", index=False)

    activation = moco_ase.drop_duplicates("site_id").set_index("site_id")["activation_quarter"]
    est = build_event_study(crash_join, activation, pre=pre, post=post, outcome_col="injury_crash")
    est.to_csv(PROC/"event_study_injury.csv", index=False)
    if len(est)>=3:
        plt.figure(figsize=(10,5))
        plt.plot(est["rel_month"], est["injury_crash"], marker="o")
        plt.axvline(0, linestyle="--")
        plt.title("Event study: mean injury crashes per site within 300 m")
        plt.xlabel("Months relative to activation"); plt.ylabel("Mean injury crashes")
        plt.tight_layout(); plt.savefig(FIG/"event_study_injury.png"); plt.close()

    corr_df = correlate_citations_with_crash_change(moco_ase, crash_join, activation, pre_window=(-6,-1), post_window=(0,6), outcome_col="injury_crash")
    corr_df.to_csv(PROC/"citations_vs_injury_delta.csv", index=True)
    if len(corr_df)>=5:
        plt.figure(figsize=(6,6))
        plt.scatter(corr_df["citations_total_yr"], corr_df["delta_injury_crash"])
        plt.xlabel("Total citations (available period)")
        plt.ylabel("Δ injury crashes (0..+6 vs -6..-1 months)")
        plt.title("Site-level: Citations vs change in injury crashes")
        plt.tight_layout(); plt.savefig(FIG/"corr_citations_vs_injury.png"); plt.close()

    return {
        "moco_ase_rows": len(moco_ase),
        "moco_crash_rows": len(moco_crash),
        "joined_crashes": len(crash_join),
        "event_study_rows": len(est),
        "corr_rows": len(corr_df),
        "notes": "If event_study_rows==0, likely not enough pre/post coverage or activation dates outside crash range."
    }

def load_fars_accidents(fars_dir="./data/fars", years=range(2005,2011), state_fips=24, county_fips=None):
    rows=[]
    for y in years:
        p = Path(fars_dir)/str(y)/"ACCIDENT.csv"
        if p.exists():
            df = read_csv(str(p))
            df.columns=[c.upper() for c in df.columns]
            df = df[df["STATE"]==state_fips]
            if county_fips is not None and "COUNTY" in df.columns:
                df = df[df["COUNTY"].isin(county_fips)]
            if "MONTH" in df.columns:
                df["MONTH_DT"] = pd.to_datetime(dict(year=y, month=df["MONTH"].fillna(1).astype(int), day=1))
            else:
                df["MONTH_DT"] = pd.Timestamp(y,1,1)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
