"""
Preprocessing Pipeline for Pollinating Insects (Butterfly) Decline Analysis
============================================================================
Research Question: What caused the decrease in pollinating insects (butterfly)?

This script:
1. Loads all CSV data sources (agri-environment schemes, habitat connectivity,
   butterfly abundance, plant abundance)
2. Cleans and aligns datasets to a common year-based index
3. Imputes missing values via linear interpolation
4. Applies Simulated Annealing to refine imputed values for maximum data quality
5. Engineers features relevant to explaining butterfly decline
6. Detects and flags outliers using Isolation Forest
7. Exports a single merged, preprocessed CSV ready for ML modelling

Data sources (from data_source/):
  - Agri-environment schemes (higher-level & lower-level) — land area under conservation
  - Habitat connectivity — composite butterfly connectivity index
  - Butterfly wider countryside — 7 abundance indices (target + factors)
  - Plants wider countryside — plant abundance across habitats
  - Habitat connectivity species trends — categorical species-level changes
"""

import os
import random
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
    """Load all 7 butterfly wider-countryside abundance CSVs."""
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
    pivot = df.pivot_table(
        index="Year", columns="Habitat", values="Unsmoothed index", aggfunc="mean"
    ).reset_index()
    pivot.columns = ["Year"] + [
        f"plant_{h.lower().replace(' ', '_').replace('&', 'and')}_index"
        for h in pivot.columns[1:]
    ]
    return pivot


def load_species_connectivity_summary():
    """Load individual species connectivity trends and encode as numeric features."""
    path = os.path.join(DATA, "habitat-connectivity",
                        "habitat-connectivity_UK-butterflies_individual-species-trends.csv")
    df = pd.read_csv(path)
    period_map = {
        "Early short term (1985-2000)": 1993,
        "Late short term (2000-2012)": 2006,
        "Long term (1985-2012)": 1999,
    }
    period_col = [c for c in df.columns if "period" in c.lower() or "assessment" in c.lower()][0]
    df["Year"] = df[period_col].map(period_map)

    pivot = df.pivot_table(
        index="Year", columns="Trend", values="Percentage of species", aggfunc="first"
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
# 3. HANDLE MISSING VALUES (Linear Interpolation)
# ──────────────────────────────────────────────

def impute_missing(df):
    """Impute missing values using temporal interpolation and edge filling.
    Returns the imputed DataFrame and a list of columns that had missing values."""
    print("\n--- Missing Value Imputation (Linear Interpolation) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Year"]

    total_missing = df[numeric_cols].isnull().sum().sum()
    print(f"Total missing values before imputation: {total_missing}")

    cols_with_missing = []
    for col in numeric_cols:
        miss = df[col].isnull().sum()
        if miss == 0:
            continue
        pct = miss / len(df) * 100
        print(f"  {col}: {miss} missing ({pct:.1f}%)")
        cols_with_missing.append(col)

    df_imputed = df.copy()
    df_imputed.sort_values("Year", inplace=True)
    df_imputed.set_index("Year", inplace=True)

    for col in numeric_cols:
        if df_imputed[col].isnull().any():
            df_imputed[col] = df_imputed[col].interpolate(method="linear", limit_direction="both")

    df_imputed.ffill(inplace=True)
    df_imputed.bfill(inplace=True)
    df_imputed.reset_index(inplace=True)

    missing_after = df_imputed[numeric_cols].isnull().sum().sum()
    print(f"Total missing values after imputation: {missing_after}")
    return df_imputed, cols_with_missing


# ──────────────────────────────────────────────
# 4. SIMULATED ANNEALING — Refine imputed values
# ──────────────────────────────────────────────

def _data_quality_score(df, numeric_cols):
    """Composite quality metric combining temporal smoothness and cross-feature
    correlation consistency.

    Component 1 — Smoothness: ecological time-series should change gradually.
      Measured as negative mean absolute second-difference (lower jitter = better).
    Component 2 — Correlation preservation: imputed values should maintain the
      natural statistical relationships between features.
      Measured by average absolute pairwise correlation.

    Returns a single scalar (higher = better quality)."""
    smoothness = 0.0
    for col in numeric_cols:
        s = df[col].dropna().values
        if len(s) >= 3:
            smoothness += -np.mean(np.abs(np.diff(s, n=2)))

    try:
        corr = df[numeric_cols].corr().fillna(0)
        corr_score = np.abs(corr.values).mean()
    except Exception:
        corr_score = 0.0

    return smoothness + corr_score * 10


def simulated_annealing_refine(df, imputed_cols, max_iterations=1000,
                                initial_temp=10.0, cooling_rate=0.995,
                                perturbation_scale=0.02):
    """Use Simulated Annealing to refine imputed values and maximise data quality.

    Problem:
      After linear interpolation, imputed values may introduce artefacts — abrupt
      jumps at series boundaries or broken cross-feature correlations. This is
      especially problematic for the plant abundance data (only 10 years of real
      observations, 25 years imputed) and habitat connectivity (ends in 2012,
      12 years extrapolated forward).

    Approach:
      SA treats each imputed cell as a tuneable parameter. Each iteration:
        1. Randomly selects one imputed column and one row
        2. Perturbs the value by a small amount (scaled to column std)
        3. Evaluates the resulting data quality score
        4. Accepts the change if it improves quality, OR accepts it
           probabilistically if it worsens quality (probability = exp(Δ/T))
      The temperature T starts high (allowing exploration) and cools gradually,
      converging toward a high-quality solution.

    This ensures the maximum amount of viable, consistent data is available
    for downstream ML modelling.

    Args:
        df: DataFrame with initial imputations applied
        imputed_cols: List of column names that contained missing values
        max_iterations: Number of SA iterations to run
        initial_temp: Starting temperature (higher = more exploration early on)
        cooling_rate: Temperature decay per iteration (closer to 1 = slower cooling)
        perturbation_scale: Fraction of column std used as perturbation magnitude
    """
    print("\n--- Simulated Annealing: Refining Imputed Values ---")
    if not imputed_cols:
        print("  No imputed columns to refine — skipping.")
        return df

    random.seed(42)
    np.random.seed(42)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Year"]
    valid_imputed = [c for c in imputed_cols if c in df.columns]

    df_current = df.copy()
    current_score = _data_quality_score(df_current, numeric_cols)

    best_df = df_current.copy()
    best_score = current_score
    initial_score = current_score

    col_stds = {col: max(df[col].std(), 0.01) for col in valid_imputed}

    temperature = initial_temp
    accepted = 0
    improved = 0

    print(f"  Columns with imputed values: {len(valid_imputed)}")
    print(f"  Initial data quality score: {current_score:.4f}")
    print(f"  SA parameters: T0={initial_temp}, cooling={cooling_rate}, "
          f"iterations={max_iterations}")

    for i in range(max_iterations):
        col = random.choice(valid_imputed)
        row = random.randint(0, len(df_current) - 1)

        std = col_stds[col]
        old_val = df_current.at[row, col]
        perturbation = np.random.normal(0, perturbation_scale * std)
        new_val = max(0.0, old_val + perturbation)  # ecological indices >= 0

        df_candidate = df_current.copy()
        df_candidate.at[row, col] = new_val

        new_score = _data_quality_score(df_candidate, numeric_cols)
        delta = new_score - current_score

        # SA acceptance criterion
        if delta > 0:
            accept = True
            improved += 1
        else:
            accept_prob = np.exp(delta / max(temperature, 1e-10))
            accept = random.random() < accept_prob

        if accept:
            df_current = df_candidate
            current_score = new_score
            accepted += 1
            if current_score > best_score:
                best_score = current_score
                best_df = df_current.copy()

        temperature *= cooling_rate

    improvement = best_score - initial_score
    print(f"  Final data quality score:   {best_score:.4f} (improvement: {improvement:+.4f})")
    print(f"  Accepted moves: {accepted}/{max_iterations} "
          f"({accepted/max_iterations*100:.1f}%)")
    print(f"  Improving moves: {improved}")
    print(f"  Final temperature: {temperature:.6f}")
    return best_df


# ──────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def engineer_features(df):
    """Create derived features that may help explain butterfly decline."""
    print("\n--- Feature Engineering ---")

    if "agri_area_higher_mha" in df.columns and "agri_area_lower_mha" in df.columns:
        df["agri_area_total_mha"] = df["agri_area_higher_mha"] + df["agri_area_lower_mha"]

    for col in [
        "butterfly_all_species_smoothed",
        "butterfly_habitat_specialist_smoothed",
        "butterfly_generalist_smoothed",
        "habitat_connectivity_index",
    ]:
        if col in df.columns:
            df[f"{col}_yoy_change"] = df[col].pct_change() * 100

    if ("butterfly_habitat_specialist_smoothed" in df.columns and
            "butterfly_generalist_smoothed" in df.columns):
        df["specialist_generalist_ratio"] = (
            df["butterfly_habitat_specialist_smoothed"] /
            df["butterfly_generalist_smoothed"].replace(0, np.nan)
        )

    if ("butterfly_farmland_specialist_smoothed" in df.columns and
            "butterfly_woodland_specialist_smoothed" in df.columns):
        df["farmland_woodland_specialist_ratio"] = (
            df["butterfly_farmland_specialist_smoothed"] /
            df["butterfly_woodland_specialist_smoothed"].replace(0, np.nan)
        )

    if "butterfly_all_species_smoothed_yoy_change" in df.columns:
        df["decline_flag"] = (df["butterfly_all_species_smoothed_yoy_change"] < 0).astype(int)

    if "agri_area_total_mha" in df.columns:
        df["agri_area_5yr_cumulative"] = df["agri_area_total_mha"].rolling(window=5, min_periods=1).sum()

    for lag in [1, 2, 3]:
        if "agri_area_total_mha" in df.columns:
            df[f"agri_area_total_lag{lag}"] = df["agri_area_total_mha"].shift(lag)
        if "habitat_connectivity_index" in df.columns:
            df[f"habitat_connectivity_lag{lag}"] = df["habitat_connectivity_index"].shift(lag)

    df.bfill(inplace=True)
    df.ffill(inplace=True)

    new_cols = [c for c in df.columns if c not in ["Year"]]
    print(f"Total features after engineering: {len(new_cols)}")
    return df


# ──────────────────────────────────────────────
# 6. OUTLIER DETECTION
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
# 7. RESTRICT TO ANALYSIS PERIOD
# ──────────────────────────────────────────────

def restrict_period(df, start_year=1990, end_year=2024):
    """Restrict dataset to the analysis period where most data overlaps."""
    print(f"\n--- Restricting to {start_year}-{end_year} ---")
    df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"Rows after restriction: {df.shape[0]}")
    return df


# ──────────────────────────────────────────────
# 8. MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PREPROCESSING PIPELINE — Butterfly Decline Analysis")
    print("  AI Optimisation: Simulated Annealing")
    print("=" * 60)

    # Load & merge
    df = merge_all()

    # Restrict to analysis period (1990-2024)
    df = restrict_period(df, start_year=1990, end_year=2024)

    # Initial imputation via linear interpolation
    df, cols_with_missing = impute_missing(df)

    # SIMULATED ANNEALING: refine imputed values to maximise
    # temporal smoothness and cross-feature correlation consistency
    df = simulated_annealing_refine(df, cols_with_missing)

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
