"""
Preprocessing Pipeline for Pollinating Insects (Butterfly) Decline Analysis
============================================================================
Research Question: What caused the decrease in pollinating insects (butterfly)?

This script:
1. Loads all CSV data sources (agri-environment schemes, habitat connectivity,
   butterfly abundance, plant abundance)
2. Cleans and aligns datasets to a common year-based index
3. Engineers features relevant to explaining butterfly decline
4. Detects and flags outliers using Isolation Forest
5. Exports a single merged, preprocessed CSV ready for ML modelling

Data sources (from data_source/):
  - Agri-environment schemes (higher-level & lower-level) — land area under conservation
  - Habitat connectivity — composite butterfly connectivity index
  - Butterfly wider countryside — 7 abundance indices (target + factors)
  - Plants wider countryside — plant abundance across habitats
  - Habitat connectivity species trends — categorical species-level changes
"""

import os
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data_source")

# ──────────────────────────────────────────────
# 1. LOAD RAW DATASETS
# ──────────────────────────────────────────────

def load_agri_schemes():
    """Load and pivot agri-environment scheme data (higher & lower level).
    Returns UK-wide total area per year for each scheme level."""
    frames = {}
    for level, fname in [
        ("higher", "agri-environment-schemes-higher-level.csv"),
        ("lower", "agri-environment-schemes-lower-level.csv"),
    ]:
        path = os.path.join(DATA, "agri-environment-schemes", fname)
        df = pd.read_csv(path)
        # Sum across all countries to get UK-wide totals per year
        agg = df.groupby("Year")["Area (Million Hectares)"].sum().reset_index()
        agg.rename(columns={"Area (Million Hectares)": f"agri_area_{level}_mha"}, inplace=True)
        frames[level] = agg

    merged = frames["higher"].merge(frames["lower"], on="Year", how="outer")
    merged.sort_values("Year", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def load_habitat_connectivity():
    """Load habitat connectivity composite trends (smoothed index)."""
    path = os.path.join(DATA, "habitat-connectivity",
                        "habitat-connectivity_UK-butterflies_composite-trends.csv")
    df = pd.read_csv(path)
    # Column names have note references — normalise using partial matching
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "year" in cl:
            col_map[c] = "Year"
        elif "unsmoothed" in cl:
            col_map[c] = "habitat_connectivity_raw"
        elif "smoothed" in cl and "lower" not in cl and "upper" not in cl:
            col_map[c] = "habitat_connectivity_index"
        elif "lower" in cl:
            col_map[c] = "habitat_conn_ci_lower"
        elif "upper" in cl:
            col_map[c] = "habitat_conn_ci_upper"
    df.rename(columns=col_map, inplace=True)

    cols_keep = [c for c in ["Year", "habitat_connectivity_index",
                              "habitat_connectivity_raw",
                              "habitat_conn_ci_lower",
                              "habitat_conn_ci_upper"] if c in df.columns]
    return df[cols_keep]


def load_butterfly_abundance():
    """Load all 7 butterfly wider-countryside abundance CSVs.
    Returns smoothed indices for each category keyed on Year."""
    folder = os.path.join(DATA, "butterfly-wider-countryside")
    files = {
        "all_species": "butterfly-wider-countryside_abundance-of-all-species.csv",
        "habitat_specialist": "butterfly-wider-countryside_abundance-of-habitat-specialist-butterfly-species.csv",
        "generalist": "butterfly-wider-countryside_abundance-of-generalist-butterfly-species.csv",
        "farmland_generalist": "butterfly-wider-countryside_abundance-of-farmland-generalists.csv",
        "farmland_specialist": "butterfly-wider-countryside_abundance-of-farmland-habitat-specialists.csv",
        "woodland_generalist": "butterfly-wider-countryside_abundance-of-woodland-generalists.csv",
        "woodland_specialist": "butterfly-wider-countryside_abundance-of-woodland-habitat-specialists.csv",
    }
    merged = None
    for key, fname in files.items():
        path = os.path.join(folder, fname)
        df = pd.read_csv(path)
        df.rename(columns={
            "Smoothed index": f"butterfly_{key}_smoothed",
            "Unsmoothed index": f"butterfly_{key}_raw",
            "Lower ci": f"butterfly_{key}_ci_lower",
            "Upper ci": f"butterfly_{key}_ci_upper",
        }, inplace=True)
        cols = ["Year", f"butterfly_{key}_smoothed", f"butterfly_{key}_raw"]
        df = df[cols]
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="Year", how="outer")

    merged.sort_values("Year", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def load_plant_abundance():
    """Load plant abundance indices across habitats, pivoted to one row per year."""
    path = os.path.join(DATA, "plants-wider-countryside",
                        "plants-wider-countryside_abundance-of-species.csv")
    df = pd.read_csv(path)
    # Pivot: each habitat becomes a column
    pivot = df.pivot_table(
        index="Year",
        columns="Habitat",
        values="Unsmoothed index",
        aggfunc="mean"
    ).reset_index()
    # Clean column names
    pivot.columns = ["Year"] + [
        f"plant_{h.lower().replace(' ', '_').replace('&', 'and')}_index"
        for h in pivot.columns[1:]
    ]
    return pivot


def load_species_connectivity_summary():
    """Load individual species connectivity trends and encode as numeric features.
    Returns per-year summary metrics from the categorical species trend data."""
    path = os.path.join(DATA, "habitat-connectivity",
                        "habitat-connectivity_UK-butterflies_individual-species-trends.csv")
    df = pd.read_csv(path)
    # Columns: Assessment period, Trend, Number of species, Total, Percentage
    # Encode period → approximate midpoint year
    period_map = {
        "Early short term (1985-2000)": 1993,
        "Late short term (2000-2012)": 2006,
        "Long term (1985-2012)": 1999,
    }
    period_col = [c for c in df.columns if "period" in c.lower() or "assessment" in c.lower()][0]
    df["Year"] = df[period_col].map(period_map)

    # Pivot trend categories into columns
    pivot = df.pivot_table(
        index="Year",
        columns="Trend",
        values="Percentage of species",
        aggfunc="first"
    ).reset_index()
    pivot.columns = ["Year"] + [
        f"species_conn_pct_{c.lower().replace(' ', '_')}" for c in pivot.columns[1:]
    ]
    return pivot


# ──────────────────────────────────────────────
# 2. MERGE ALL DATASETS ON YEAR
# ──────────────────────────────────────────────

def merge_all():
    """Merge all datasets on Year using outer join to preserve maximum coverage."""
    print("Loading datasets...")
    agri = load_agri_schemes()
    print(f"  Agri-environment schemes: {agri.shape}, years {agri['Year'].min()}-{agri['Year'].max()}")

    habitat = load_habitat_connectivity()
    print(f"  Habitat connectivity:     {habitat.shape}, years {habitat['Year'].min()}-{habitat['Year'].max()}")

    butterfly = load_butterfly_abundance()
    print(f"  Butterfly abundance:       {butterfly.shape}, years {butterfly['Year'].min()}-{butterfly['Year'].max()}")

    plants = load_plant_abundance()
    print(f"  Plant abundance:           {plants.shape}, years {plants['Year'].min()}-{plants['Year'].max()}")

    species_conn = load_species_connectivity_summary()
    print(f"  Species connectivity:      {species_conn.shape}")

    # Sequential merge on Year
    df = butterfly.merge(agri, on="Year", how="outer")
    df = df.merge(habitat, on="Year", how="outer")
    df = df.merge(plants, on="Year", how="outer")
    df = df.merge(species_conn, on="Year", how="outer")

    df.sort_values("Year", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"\nMerged dataset: {df.shape[0]} rows × {df.shape[1]} columns, "
          f"years {df['Year'].min()}-{df['Year'].max()}")
    return df


# ──────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ──────────────────────────────────────────────

def impute_missing(df):
    """Impute missing values using temporal interpolation and edge filling."""
    print("\n--- Missing Value Imputation ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Year"]

    missing_before = df[numeric_cols].isnull().sum()
    total_missing = missing_before.sum()
    print(f"Total missing values before imputation: {total_missing}")

    for col in numeric_cols:
        miss = df[col].isnull().sum()
        if miss == 0:
            continue
        pct = miss / len(df) * 100
        print(f"  {col}: {miss} missing ({pct:.1f}%)")

    # Step 1: Linear interpolation along the time axis (best for time-series)
    df_imputed = df.copy()
    df_imputed.sort_values("Year", inplace=True)
    df_imputed.set_index("Year", inplace=True)

    for col in numeric_cols:
        if df_imputed[col].isnull().any():
            # Use linear interpolation, then fill edges
            df_imputed[col] = df_imputed[col].interpolate(method="linear", limit_direction="both")

    # Step 2: For remaining NaN (e.g., columns with all-NaN stretches), forward/backward fill
    df_imputed.ffill(inplace=True)
    df_imputed.bfill(inplace=True)

    df_imputed.reset_index(inplace=True)

    missing_after = df_imputed[numeric_cols].isnull().sum().sum()
    print(f"Total missing values after imputation: {missing_after}")
    return df_imputed


# ──────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def engineer_features(df):
    """Create derived features that may help explain butterfly decline."""
    print("\n--- Feature Engineering ---")

    # 4a. Total agri-environment area (higher + lower combined)
    if "agri_area_higher_mha" in df.columns and "agri_area_lower_mha" in df.columns:
        df["agri_area_total_mha"] = df["agri_area_higher_mha"] + df["agri_area_lower_mha"]

    # 4b. Year-over-year change rates for key indices
    for col in [
        "butterfly_all_species_smoothed",
        "butterfly_habitat_specialist_smoothed",
        "butterfly_generalist_smoothed",
        "habitat_connectivity_index",
    ]:
        if col in df.columns:
            df[f"{col}_yoy_change"] = df[col].pct_change() * 100

    # 4c. Ratio of specialist to generalist butterfly populations
    if ("butterfly_habitat_specialist_smoothed" in df.columns and
            "butterfly_generalist_smoothed" in df.columns):
        df["specialist_generalist_ratio"] = (
            df["butterfly_habitat_specialist_smoothed"] /
            df["butterfly_generalist_smoothed"].replace(0, np.nan)
        )

    # 4d. Ratio of farmland specialist to woodland specialist
    if ("butterfly_farmland_specialist_smoothed" in df.columns and
            "butterfly_woodland_specialist_smoothed" in df.columns):
        df["farmland_woodland_specialist_ratio"] = (
            df["butterfly_farmland_specialist_smoothed"] /
            df["butterfly_woodland_specialist_smoothed"].replace(0, np.nan)
        )

    # 4e. Decline indicator (binary: 1 if all-species smoothed index is declining YoY)
    if "butterfly_all_species_smoothed_yoy_change" in df.columns:
        df["decline_flag"] = (df["butterfly_all_species_smoothed_yoy_change"] < 0).astype(int)

    # 4f. Cumulative agri-environment effort (rolling 5-year sum)
    if "agri_area_total_mha" in df.columns:
        df["agri_area_5yr_cumulative"] = df["agri_area_total_mha"].rolling(window=5, min_periods=1).sum()

    # 4g. Lagged features — policy effects are often delayed
    for lag in [1, 2, 3]:
        if "agri_area_total_mha" in df.columns:
            df[f"agri_area_total_lag{lag}"] = df["agri_area_total_mha"].shift(lag)
        if "habitat_connectivity_index" in df.columns:
            df[f"habitat_connectivity_lag{lag}"] = df["habitat_connectivity_index"].shift(lag)

    # Fill NaN introduced by shift/pct_change at the edges
    df.bfill(inplace=True)
    df.ffill(inplace=True)

    new_cols = [c for c in df.columns if c not in ["Year"]]
    print(f"Total features after engineering: {len(new_cols)}")
    return df


# ──────────────────────────────────────────────
# 5. OUTLIER DETECTION
# ──────────────────────────────────────────────

def detect_outliers(df):
    """Detect outliers using Isolation Forest. Flag but do not remove them."""
    print("\n--- Outlier Detection (Isolation Forest) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Year"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])

    iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
    labels = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    df["outlier_flag"] = (labels == -1).astype(int)
    df["outlier_score"] = scores

    n_outliers = df["outlier_flag"].sum()
    print(f"Outliers detected: {n_outliers} / {len(df)} rows ({n_outliers/len(df)*100:.1f}%)")
    outlier_years = df.loc[df["outlier_flag"] == 1, "Year"].tolist()
    if outlier_years:
        print(f"Outlier years: {outlier_years}")
    return df


# ──────────────────────────────────────────────
# 6. RESTRICT TO ANALYSIS PERIOD & CLEAN
# ──────────────────────────────────────────────

def restrict_period(df, start_year=1990, end_year=2024):
    """Restrict dataset to the analysis period where most data overlaps."""
    print(f"\n--- Restricting to {start_year}-{end_year} ---")
    df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"Rows after restriction: {df.shape[0]}")
    return df


# ──────────────────────────────────────────────
# 7. MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PREPROCESSING PIPELINE — Butterfly Decline Analysis")
    print("=" * 60)

    # Load & merge
    df = merge_all()

    # Restrict to analysis period (1990-2024: butterfly farmland/woodland start 1990)
    df = restrict_period(df, start_year=1990, end_year=2024)

    # Impute missing values
    df = impute_missing(df)

    # Feature engineering
    df = engineer_features(df)

    # Outlier detection
    df = detect_outliers(df)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL PREPROCESSED DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Year range: {df['Year'].min()} – {df['Year'].max()}")
    print(f"\nColumns ({df.shape[1]}):")
    for i, c in enumerate(df.columns):
        null_count = df[c].isnull().sum()
        print(f"  {i+1:2d}. {c:<50s} nulls={null_count}")

    print(f"\nOutlier flagged rows: {df['outlier_flag'].sum()}")
    print(f"\nDescriptive statistics (key columns):")
    key_cols = [c for c in df.columns if "smoothed" in c and "yoy" not in c and "raw" not in c]
    if key_cols:
        print(df[["Year"] + key_cols].describe().round(2).to_string())

    # Save
    out_path = os.path.join(BASE, "data_preprocessed.csv")
    df.to_csv(out_path, index=False)
    print(f"\nPreprocessed data saved to: {out_path}")
    return df


if __name__ == "__main__":
    main()
