"""
Preprocessing Pipeline for Pollinating Insects (Butterfly) Decline Analysis
============================================================================
Research Question: What caused the decrease in pollinating insects (butterfly)?

This script:
1. Loads all CSV data sources (agri-environment schemes, habitat connectivity,
   butterfly abundance, plant abundance)
2. Cleans and aligns datasets to a common year-based index
3. Engineers features relevant to explaining butterfly decline
4. Applies AI search & optimisation techniques for data quality maximisation:
   - Simulated Annealing: optimise imputed values for temporal consistency
   - Genetic Algorithm: select optimal feature subset
   - Hill Climbing: tune outlier detection threshold
   - Tabu Search: select best imputation strategy per column
5. Detects and flags outliers using optimised Isolation Forest
6. Exports a single merged, preprocessed CSV ready for ML modelling

Data sources (from data_source/):
  - Agri-environment schemes (higher-level & lower-level) — land area under conservation
  - Habitat connectivity — composite butterfly connectivity index
  - Butterfly wider countryside — 7 abundance indices (target + factors)
  - Plants wider countryside — plant abundance across habitats
  - Habitat connectivity species trends — categorical species-level changes
"""

import os
import copy
import random
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
# 3b. TABU SEARCH — Best imputation method per column
# ──────────────────────────────────────────────

def _smoothness_score(series):
    """Measure temporal smoothness as negative mean absolute second-difference.
    Higher (less negative) = smoother."""
    s = series.dropna().values
    if len(s) < 3:
        return 0.0
    second_diff = np.diff(s, n=2)
    return -np.mean(np.abs(second_diff))


def _impute_column(series, method):
    """Apply a single imputation method to a series with missing values."""
    s = series.copy()
    if method == "linear":
        s = s.interpolate(method="linear", limit_direction="both")
    elif method == "spline":
        try:
            s = s.interpolate(method="spline", order=3, limit_direction="both")
        except Exception:
            s = s.interpolate(method="linear", limit_direction="both")
    elif method == "nearest":
        s = s.interpolate(method="nearest", limit_direction="both")
    elif method == "mean":
        s = s.fillna(s.mean())
    elif method == "median":
        s = s.fillna(s.median())
    # Edge fill for any remaining NaN
    s = s.ffill().bfill()
    return s


def tabu_search_imputation(df, max_iterations=50, tabu_tenure=10):
    """Use Tabu Search to find the best imputation method for each column.

    Tabu Search maintains a memory of recently tried (column, method) pairs
    to avoid cycling back to poor solutions. It evaluates combinations by
    measuring temporal smoothness of the imputed series.

    Args:
        df: DataFrame with missing values (indexed by Year)
        max_iterations: Number of search iterations
        tabu_tenure: How many iterations a move stays in the tabu list
    """
    print("\n--- Tabu Search: Optimal Imputation Method Selection ---")
    METHODS = ["linear", "spline", "nearest", "mean", "median"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Year"]
    cols_with_missing = [c for c in numeric_cols if df[c].isnull().any()]

    if not cols_with_missing:
        print("  No missing values — skipping Tabu Search.")
        return df, {}

    df_work = df.copy()
    df_work.set_index("Year", inplace=True)

    # Initial solution: linear interpolation for all columns
    current_solution = {col: "linear" for col in cols_with_missing}
    current_df = df_work.copy()
    for col in cols_with_missing:
        current_df[col] = _impute_column(df_work[col], "linear")

    def evaluate(solution, base_df):
        """Total smoothness score across all imputed columns."""
        temp = base_df.copy()
        for col, method in solution.items():
            temp[col] = _impute_column(base_df[col], method)
        return sum(_smoothness_score(temp[col]) for col in solution)

    best_solution = current_solution.copy()
    best_score = evaluate(current_solution, df_work)
    current_score = best_score

    # Tabu list: maps (col, method) → iteration when it expires
    tabu_list = {}

    print(f"  Columns to optimise: {len(cols_with_missing)}")
    print(f"  Methods: {METHODS}")
    print(f"  Initial score (smoothness): {best_score:.4f}")

    for iteration in range(max_iterations):
        best_neighbour = None
        best_neighbour_score = -np.inf

        # Generate neighbourhood: change one column's method
        for col in cols_with_missing:
            for method in METHODS:
                if method == current_solution[col]:
                    continue
                # Check tabu status
                if (col, method) in tabu_list and tabu_list[(col, method)] > iteration:
                    # Aspiration criterion: allow if it improves global best
                    candidate = current_solution.copy()
                    candidate[col] = method
                    score = evaluate(candidate, df_work)
                    if score > best_score:
                        # Aspiration — override tabu
                        if score > best_neighbour_score:
                            best_neighbour = candidate
                            best_neighbour_score = score
                    continue

                candidate = current_solution.copy()
                candidate[col] = method
                score = evaluate(candidate, df_work)
                if score > best_neighbour_score:
                    best_neighbour = candidate
                    best_neighbour_score = score

        if best_neighbour is None:
            break

        # Identify the move made
        for col in cols_with_missing:
            if best_neighbour[col] != current_solution[col]:
                tabu_list[(col, current_solution[col])] = iteration + tabu_tenure
                break

        current_solution = best_neighbour
        current_score = best_neighbour_score

        if current_score > best_score:
            best_score = current_score
            best_solution = current_solution.copy()

    # Apply best solution
    result_df = df_work.copy()
    for col, method in best_solution.items():
        result_df[col] = _impute_column(df_work[col], method)

    result_df.ffill(inplace=True)
    result_df.bfill(inplace=True)
    result_df.reset_index(inplace=True)

    print(f"  Final score (smoothness): {best_score:.4f}")
    print(f"  Best methods selected:")
    for col, method in best_solution.items():
        print(f"    {col:<50s} → {method}")

    return result_df, best_solution


# ──────────────────────────────────────────────
# 3c. SIMULATED ANNEALING — Refine imputed values
# ──────────────────────────────────────────────

def _data_quality_score(df, numeric_cols):
    """Composite quality metric combining smoothness and correlation consistency.
    Measures how well imputed data preserves temporal trends and inter-feature
    relationships."""
    # Component 1: Temporal smoothness (low second-derivative)
    smoothness = 0.0
    for col in numeric_cols:
        s = df[col].dropna().values
        if len(s) >= 3:
            smoothness += -np.mean(np.abs(np.diff(s, n=2)))

    # Component 2: Correlation matrix stability (Frobenius norm of correlation)
    try:
        corr = df[numeric_cols].corr()
        # Penalise NaN correlations
        corr = corr.fillna(0)
        # Higher average absolute correlation = more signal preserved
        corr_score = np.abs(corr.values).mean()
    except Exception:
        corr_score = 0.0

    return smoothness + corr_score * 10


def simulated_annealing_refine(df, imputed_cols, max_iterations=500,
                                initial_temp=10.0, cooling_rate=0.98,
                                perturbation_scale=0.02):
    """Use Simulated Annealing to refine imputed values for data quality.

    After initial imputation, some values may break temporal trends or
    cross-feature correlations. SA perturbs imputed values and accepts
    improvements (and occasionally worse solutions) to escape local optima,
    converging on values that maximise overall data quality.

    Args:
        df: DataFrame with initial imputations applied
        imputed_cols: List of columns that contained imputed values
        max_iterations: SA iteration count
        initial_temp: Starting temperature
        cooling_rate: Multiplicative cooling factor per iteration
        perturbation_scale: Fraction of column std to use as perturbation size
    """
    print("\n--- Simulated Annealing: Refining Imputed Values ---")
    if not imputed_cols:
        print("  No imputed columns to refine — skipping.")
        return df

    random.seed(42)
    np.random.seed(42)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Year"]
    df_current = df.copy()
    current_score = _data_quality_score(df_current, numeric_cols)

    best_df = df_current.copy()
    best_score = current_score

    # Pre-compute column std for perturbation sizing
    col_stds = {col: df[col].std() for col in imputed_cols if col in df.columns}

    temperature = initial_temp
    accepted = 0
    improved = 0

    print(f"  Imputed columns to refine: {len(imputed_cols)}")
    print(f"  Initial quality score: {current_score:.4f}")
    print(f"  Temperature: {initial_temp}, cooling: {cooling_rate}")

    for i in range(max_iterations):
        # Pick a random imputed column and row to perturb
        col = random.choice([c for c in imputed_cols if c in df.columns])
        row = random.randint(0, len(df_current) - 1)

        # Perturb value
        std = col_stds.get(col, 1.0)
        old_val = df_current.at[row, col]
        perturbation = np.random.normal(0, perturbation_scale * std)
        new_val = old_val + perturbation

        # Ensure non-negative (ecological indices can't be negative)
        new_val = max(0, new_val)

        df_candidate = df_current.copy()
        df_candidate.at[row, col] = new_val

        new_score = _data_quality_score(df_candidate, numeric_cols)
        delta = new_score - current_score

        # Accept or reject
        if delta > 0:
            accept = True
            improved += 1
        else:
            # Probability of accepting worse solution decreases with temperature
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

    print(f"  Final quality score: {best_score:.4f} (Δ = {best_score - _data_quality_score(df, numeric_cols):+.4f})")
    print(f"  Iterations: {max_iterations}, accepted: {accepted}, improved: {improved}")
    print(f"  Final temperature: {temperature:.6f}")
    return best_df


# ──────────────────────────────────────────────
# 3d. HILL CLIMBING — Optimise outlier detection threshold
# ──────────────────────────────────────────────

def _outlier_objective(X_scaled, contamination):
    """Evaluate an outlier contamination threshold.
    Objective: maximise separation between inlier and outlier scores
    while keeping outlier count reasonable (not too many, not too few)."""
    if contamination <= 0.01 or contamination >= 0.5:
        return -np.inf
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    labels = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    n_outliers = (labels == -1).sum()
    n_total = len(labels)

    if n_outliers == 0 or n_outliers == n_total:
        return -np.inf

    # Score separation between inliers and outliers
    inlier_scores = scores[labels == 1]
    outlier_scores = scores[labels == -1]
    separation = np.mean(inlier_scores) - np.mean(outlier_scores)

    # Penalty for too many or too few outliers (prefer 5-15% range)
    outlier_ratio = n_outliers / n_total
    ratio_penalty = -10 * abs(outlier_ratio - 0.10)

    return separation + ratio_penalty


def hill_climbing_outlier_threshold(X_scaled, initial_contamination=0.1,
                                     step_size=0.01, max_iterations=50):
    """Use Hill Climbing to find the optimal outlier detection threshold.

    Starts at an initial contamination value and iteratively moves to
    neighbouring values that improve inlier/outlier score separation.

    Args:
        X_scaled: Standardised feature matrix
        initial_contamination: Starting contamination parameter
        step_size: Size of each step in contamination space
        max_iterations: Maximum iterations before stopping
    Returns:
        Optimal contamination value
    """
    print("\n--- Hill Climbing: Optimising Outlier Threshold ---")
    current = initial_contamination
    current_score = _outlier_objective(X_scaled, current)

    print(f"  Initial contamination: {current:.3f}, score: {current_score:.4f}")

    for i in range(max_iterations):
        # Try neighbours
        neighbours = [
            current + step_size,
            current - step_size,
            current + step_size / 2,
            current - step_size / 2,
        ]
        # Filter valid range
        neighbours = [c for c in neighbours if 0.02 <= c <= 0.40]

        best_neighbour = current
        best_neighbour_score = current_score

        for candidate in neighbours:
            score = _outlier_objective(X_scaled, candidate)
            if score > best_neighbour_score:
                best_neighbour = candidate
                best_neighbour_score = score

        if best_neighbour == current:
            # No improvement — local optimum reached
            print(f"  Converged at iteration {i+1}")
            break

        current = best_neighbour
        current_score = best_neighbour_score

    print(f"  Optimal contamination: {current:.3f}, score: {current_score:.4f}")
    return current


# ──────────────────────────────────────────────
# 3e. GENETIC ALGORITHM — Feature selection
# ──────────────────────────────────────────────

def _fitness_feature_subset(X, chromosome, min_features=8):
    """Evaluate fitness of a feature subset (chromosome).
    Fitness balances information content (PCA variance) with feature count.
    
    A good feature subset should:
    - Retain enough features for meaningful ML analysis (min 8)
    - Maximise variance explained (information preserved)
    - Not be excessively large (avoid noise/redundancy)
    """
    selected = [i for i, bit in enumerate(chromosome) if bit == 1]
    if len(selected) < min_features:
        # Heavily penalise subsets that are too small for ML
        return -1.0 + 0.01 * len(selected)

    X_sub = X[:, selected]
    # Explained variance from PCA
    n_components = min(3, X_sub.shape[1], X_sub.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(X_sub)
    var_explained = sum(pca.explained_variance_ratio_)

    # Reward having more features (richer ML input), with diminishing returns
    feature_bonus = 0.01 * np.log1p(len(selected))

    # Penalty only for extreme redundancy (> 90% of all features)
    total = len(chromosome)
    if len(selected) / total > 0.9:
        redundancy_penalty = 0.05 * (len(selected) / total - 0.9)
    else:
        redundancy_penalty = 0.0

    return var_explained + feature_bonus - redundancy_penalty


def genetic_algorithm_feature_selection(feature_names, X_scaled,
                                         population_size=30,
                                         generations=60,
                                         crossover_rate=0.8,
                                         mutation_rate=0.05):
    """Use a Genetic Algorithm to select the optimal feature subset.

    Each individual is a binary chromosome where 1 = feature selected.
    Fitness is measured by PCA variance explained on the selected features,
    with a minimum feature count to ensure ML viability.
    Essential domain features are always protected from removal.

    Args:
        feature_names: List of feature column names
        X_scaled: Standardised feature matrix
        population_size: Number of individuals per generation
        generations: Number of GA generations
        crossover_rate: Probability of single-point crossover
        mutation_rate: Per-bit mutation probability
    Returns:
        List of selected feature names, fitness history
    """
    print("\n--- Genetic Algorithm: Feature Selection ---")
    random.seed(42)
    np.random.seed(42)

    n_features = len(feature_names)

    # Essential features that must always be retained for domain relevance
    essential_keywords = [
        "butterfly_all_species_smoothed", "butterfly_habitat_specialist_smoothed",
        "butterfly_generalist_smoothed", "butterfly_farmland_specialist_smoothed",
        "butterfly_woodland_specialist_smoothed",
        "agri_area_total_mha", "habitat_connectivity_index",
        "specialist_generalist_ratio", "decline_flag",
    ]
    protected_indices = set()
    for i, name in enumerate(feature_names):
        if name in essential_keywords:
            protected_indices.add(i)

    print(f"  Total features: {n_features}, protected: {len(protected_indices)}, "
          f"population: {population_size}, generations: {generations}")

    # Initialise population — random binary chromosomes with ~70% features ON
    # Protected features always ON
    population = []
    for _ in range(population_size):
        chrom = [1 if random.random() < 0.7 else 0 for _ in range(n_features)]
        for idx in protected_indices:
            chrom[idx] = 1
        population.append(chrom)

    def select_parents(pop, fitnesses):
        """Tournament selection (size 3)."""
        parents = []
        for _ in range(2):
            candidates = random.sample(range(len(pop)), min(3, len(pop)))
            winner = max(candidates, key=lambda i: fitnesses[i])
            parents.append(pop[winner])
        return parents

    best_fitness_history = []
    best_overall_chrom = None
    best_overall_fitness = -np.inf

    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [_fitness_feature_subset(X_scaled, chrom) for chrom in population]

        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]

        if gen_best_fitness > best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall_chrom = population[gen_best_idx][:]

        best_fitness_history.append(gen_best_fitness)

        # Create next generation
        new_population = []
        # Elitism: keep best individual
        new_population.append(population[gen_best_idx][:])

        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitnesses)

            # Crossover
            if random.random() < crossover_rate:
                point = random.randint(1, n_features - 1)
                child1 = parent1[:point] + parent2[point:]
                child2 = parent2[:point] + parent1[point:]
            else:
                child1 = parent1[:]
                child2 = parent2[:]

            # Mutation (skip protected features)
            for child in [child1, child2]:
                for j in range(n_features):
                    if j in protected_indices:
                        child[j] = 1  # Always keep protected
                        continue
                    if random.random() < mutation_rate:
                        child[j] = 1 - child[j]

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    selected_indices = [i for i, bit in enumerate(best_overall_chrom) if bit == 1]
    selected_features = [feature_names[i] for i in selected_indices]

    removed = [feature_names[i] for i in range(n_features)
               if best_overall_chrom[i] == 0]

    print(f"  Best fitness: {best_overall_fitness:.4f}")
    print(f"  Features selected: {len(selected_features)} / {n_features}")
    if removed:
        print(f"  Features removed ({len(removed)}): {removed}")
    print(f"  Fitness progression: gen 0 = {best_fitness_history[0]:.4f} → "
          f"gen {generations-1} = {best_fitness_history[-1]:.4f}")

    return selected_features, best_fitness_history


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
    """Detect outliers using Isolation Forest with hill-climbing-optimised threshold."""
    print("\n--- Outlier Detection (Isolation Forest + Hill Climbing) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Year"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])

    # Use Hill Climbing to find optimal contamination threshold
    optimal_contamination = hill_climbing_outlier_threshold(X_scaled)

    iso = IsolationForest(contamination=optimal_contamination, random_state=42,
                          n_estimators=200)
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
    print("  with AI Search & Optimisation Techniques")
    print("=" * 60)

    # Load & merge
    df = merge_all()

    # Restrict to analysis period (1990-2024: butterfly farmland/woodland start 1990)
    df = restrict_period(df, start_year=1990, end_year=2024)

    # ── AI TECHNIQUE 1: Tabu Search for imputation method selection ──
    # Instead of blindly using linear interpolation everywhere, Tabu Search
    # evaluates multiple methods per column and avoids cycling back to poor ones.
    df_tabu, imputation_methods = tabu_search_imputation(df)

    # Fallback: ensure basic imputation for any remaining NaN
    df_tabu = impute_missing(df_tabu)

    # ── AI TECHNIQUE 2: Simulated Annealing to refine imputed values ──
    # Fine-tunes the imputed values to maximise temporal smoothness and
    # cross-feature correlation consistency.
    imputed_cols = list(imputation_methods.keys())
    df_refined = simulated_annealing_refine(df_tabu, imputed_cols)

    # Feature engineering
    df_engineered = engineer_features(df_refined)

    # ── AI TECHNIQUE 3: Hill Climbing for outlier threshold ──
    # (Integrated into detect_outliers — finds optimal contamination parameter)
    df_outliers = detect_outliers(df_engineered)

    # ── AI TECHNIQUE 4: Genetic Algorithm for feature selection ──
    # Selects the most informative feature subset based on PCA variance explained.
    numeric_cols = [c for c in df_outliers.select_dtypes(include=[np.number]).columns
                    if c not in ("Year", "outlier_flag", "outlier_score")]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_outliers[numeric_cols])
    selected_features, ga_history = genetic_algorithm_feature_selection(
        numeric_cols, X_scaled
    )

    # Keep Year + selected features + metadata columns
    keep_cols = ["Year"] + [c for c in selected_features if c in df_outliers.columns]
    keep_cols += ["outlier_flag", "outlier_score"]
    # Ensure decline_flag is always kept (classification target)
    if "decline_flag" in df_outliers.columns and "decline_flag" not in keep_cols:
        keep_cols.append("decline_flag")
    keep_cols = list(dict.fromkeys(keep_cols))  # deduplicate preserving order
    df_final = df_outliers[keep_cols].copy()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL PREPROCESSED DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print(f"Year range: {df_final['Year'].min()} – {df_final['Year'].max()}")
    print(f"\nColumns ({df_final.shape[1]}):")
    for i, c in enumerate(df_final.columns):
        null_count = df_final[c].isnull().sum()
        print(f"  {i+1:2d}. {c:<50s} nulls={null_count}")

    print(f"\nOutlier flagged rows: {df_final['outlier_flag'].sum()}")

    print("\n--- AI Optimisation Summary ---")
    print("  1. Tabu Search:        Selected best imputation method per column")
    print("  2. Simulated Annealing: Refined imputed values for quality")
    print("  3. Hill Climbing:       Optimised outlier detection threshold")
    print(f"  4. Genetic Algorithm:   Selected {len(selected_features)}/{len(numeric_cols)} features")

    print(f"\nDescriptive statistics (key columns):")
    key_cols = [c for c in df_final.columns if "smoothed" in c and "yoy" not in c and "raw" not in c]
    if key_cols:
        print(df_final[["Year"] + key_cols].describe().round(2).to_string())

    # Save
    out_path = os.path.join(BASE, "data_preprocessed.csv")
    df_final.to_csv(out_path, index=False)
    print(f"\nPreprocessed data saved to: {out_path}")
    return df_final


if __name__ == "__main__":
    main()
