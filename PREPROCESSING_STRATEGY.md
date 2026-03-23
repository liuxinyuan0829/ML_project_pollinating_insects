# Preprocessing Strategy for Pollinating Insects (Butterfly) Decline Analysis

## 1. Research Question

**What caused the decrease in pollinating insects (butterfly) in the UK?**

The UK has experienced a steady decline in pollinating insect populations due to factors including pesticide use, habitat loss, fragmentation, and changes in land management. This preprocessing pipeline consolidates multiple UK Biodiversity indicator datasets into a single, analysis-ready dataset for machine learning modelling.

---

## 2. Data Sources

Five categories of datasets from DEFRA's UK Biodiversity Indicators were integrated:

| # | Dataset | Files | Period | Key Metric |
|---|---------|-------|--------|------------|
| 1 | **Agri-Environment Schemes (England)** | `agri-environment-schemes-higher-level-England.csv`, `agri-environment-schemes-lower-level-England.csv` | 1992–2022 | Land area under conservation schemes (million hectares), England only |
| 2 | **Habitat Connectivity** | `habitat-connectivity_UK-butterflies_composite-trends.csv` | 1985–2012 | Smoothed connectivity index (1985 = 100) |
| 3 | **Habitat Connectivity Species Trends** | `habitat-connectivity_UK-butterflies_individual-species-trends.csv` | Summary | % species increasing/decreasing/stable |
| 4 | **Butterfly Wider Countryside** | 7 CSV files (all species, habitat specialist, generalist, farmland generalist/specialist, woodland generalist/specialist) | 1976–2024 | Smoothed abundance indices |
| 5 | **Plants Wider Countryside** | `plants-wider-countryside_abundance-of-species.csv` | 2015–2024 | Plant abundance index by habitat type (Arable, Lowland Grassland, Broadleaved Woodland & Hedges) |

---

## 3. Preprocessing Pipeline

The pipeline is implemented in `preprocess.py` (Python) and consists of five sequential stages:

### 3.1 Data Loading & Merging

- **Agri-environment schemes (England only)**: Data is loaded directly for England — no cross-country aggregation needed. Higher-level and lower-level scheme areas are kept as **separate features** because, per the dataset documentation, higher-level agreements may be underpinned by an entry-level scheme, so the two areas cannot be meaningfully summed.
- **Habitat connectivity**: The smoothed composite index is extracted along with 95% confidence interval bounds.
- **Butterfly abundance**: All 7 abundance CSVs are loaded, and both smoothed and unsmoothed (raw) indices are retained per category.
- **Plant abundance**: Pivoted from long format (Habitat × Year) to wide format — one column per habitat type (Arable, Broadleaved Woodland & Hedges, Lowland Grassland). Additionally, 95% confidence interval widths are extracted per habitat as uncertainty proxy features.
- **Species connectivity trends**: Categorical trend summaries (Increased/Decreased/No Change) are encoded as percentage-of-species columns, mapped to approximate midpoint years.

All datasets are merged via **outer join on Year** to preserve maximum temporal coverage before restricting the analysis window.

### 3.2 Analysis Period Restriction

The dataset is restricted to **1990–2024** (35 years). This period:
- Covers the start of farmland and woodland butterfly indices (base year 1990).
- Captures the full arc of agri-environment scheme rollout (1992 onward).
- Includes the most recent butterfly population observations through 2024.

### 3.3 Missing Value Imputation

Different datasets have different temporal coverage, creating systematic gaps:

| Feature Group | Missing % | Reason |
|---------------|-----------|--------|
| Agri-environment (lower-level) | 42.9% | Data starts 2003 |
| Habitat connectivity | 34.3% | Data ends 2012 |
| Plant abundance | 71.4% | Data starts 2015 |
| Species connectivity summary | 91.4% | Only 3 summary observations |

**Imputation strategy (two-tier):**

1. **Linear interpolation** along the time axis — preserves temporal trends and is the most appropriate method for ecological time-series data where values change gradually.
2. **Forward-fill / backward-fill** — applied to remaining edge NaN values (start/end of series) where interpolation cannot extrapolate.

**Result:** 313 missing values reduced to 0.

### 3.4 Simulated Annealing — Imputed Value Refinement

After linear interpolation, imputed values may introduce artefacts such as abrupt jumps at series boundaries or broken cross-feature correlations. This is especially problematic for plant abundance data (only 10 years of real observations, 25 years imputed) and habitat connectivity (ends in 2012, 12 years extrapolated forward).

**Simulated Annealing (SA)** is applied as an AI optimisation technique to refine these imputed values and maximise overall data quality.

**Problem formulation:**
- Each imputed cell is treated as a tuneable parameter.
- The objective function is a composite **data quality score** combining:
  1. **Temporal smoothness** — ecological time-series should change gradually; measured as negative mean absolute second-difference (lower jitter = better).
  2. **Cross-feature correlation consistency** — imputed values should preserve the natural statistical relationships between features; measured by average absolute pairwise correlation.

**SA algorithm:**
1. Select a random imputed cell (column and row).
2. Perturb the value by a small amount (scaled to 2% of that column's standard deviation).
3. Evaluate the data quality score of the modified dataset.
4. **Accept** the change if it improves quality; otherwise accept it with probability $P = e^{\Delta / T}$ where $\Delta$ is the score change and $T$ is the current temperature.
5. Cool the temperature and repeat.

**Algorithmic parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Initial temperature ($T_0$) | 10.0 | High enough to allow broad exploration of the solution space early on |
| Cooling rate | 0.995 | Slow geometric cooling; ensures ~1000 iterations before temperature becomes negligible |
| Max iterations | 1000 | Sufficient for convergence given 15 imputed columns × 35 rows |
| Perturbation scale | 0.02 (2% of column σ) | Small perturbations preserve the original interpolation structure while allowing refinement |
| Value constraint | ≥ 0 | Ecological indices cannot be negative |
| Random seed | 42 | Reproducibility |

**Why Simulated Annealing?**
SA was chosen over other AI search/optimisation techniques (hill climbing, tabu search, genetic algorithms) for this task because:
- Unlike hill climbing, SA can **escape local optima** through its probabilistic acceptance of worse solutions at high temperatures.
- The problem is a continuous-parameter refinement task (not combinatorial), where SA's neighbourhood-based search is well-suited.
- With 15 columns × 35 rows = 525 potential parameters, SA is more computationally efficient than a genetic algorithm, which would require maintaining a population of full datasets.
- Tabu search's memory-based approach adds complexity without clear benefit for this continuous optimisation domain.

### 3.5 Feature Engineering

20 derived features were created to capture relationships relevant to butterfly decline:

| Feature | Description | Rationale |
|---------|-------------|----------|
| `butterfly_*_yoy_change` | Year-over-year % change in smoothed indices (×4) | Captures rate of decline/recovery |
| `habitat_connectivity_index_yoy_change` | YoY % change in connectivity | Habitat fragmentation dynamics |
| `specialist_generalist_ratio` | Habitat specialist / generalist abundance ratio | Specialists decline faster when habitats degrade |
| `farmland_woodland_specialist_ratio` | Farmland / woodland specialist ratio | Differential habitat pressures |
| `decline_flag` | Binary: 1 if all-species index declined YoY | Classification target variable |
| `agri_area_higher_5yr_avg` | 5-year rolling average of higher-level scheme area | Smoothed policy intensity (higher-level) |
| `agri_area_lower_5yr_avg` | 5-year rolling average of lower-level scheme area | Smoothed policy intensity (lower-level) |
| `agri_area_higher_mha_lag1/2/3` | Higher-level area shifted by 1–3 years | Delayed higher-level policy effects |
| `agri_area_lower_mha_lag1/2/3` | Lower-level area shifted by 1–3 years | Delayed lower-level policy effects |
| `habitat_connectivity_lag1/2/3` | Connectivity shifted by 1–3 years | Lagged environmental impact |

**Note:** Higher-level and lower-level agri-environment scheme areas are intentionally kept separate throughout preprocessing and feature engineering — they are **not summed** — because higher-level agreements may be underpinned by an entry-level scheme, making the areas non-additive.

### 3.6 Outlier Detection

**Method:** Isolation Forest (scikit-learn)
- 200 estimators, contamination threshold = 10%, random state = 42
- Applied to all standardised numeric features (StandardScaler)

**Results:** 4 outlier years flagged:
- **1990** — Base year, extreme values in some indices
- **1992** — Spikes in generalist butterfly abundance
- **1997** — Unusual pattern in agri-environment transitions
- **2024** — Most recent year with sharp raw abundance drops

**Decision:** Outliers are **flagged but retained** (`outlier_flag` = 1). Ecological outliers often represent real events (e.g., severe weather, policy changes) and should not be discarded. The `outlier_score` column provides a continuous anomaly metric for sensitivity analysis.

---

## 4. Output Dataset

**File:** `data_preprocessed.csv`

| Property | Value |
|----------|-------|
| Rows | 35 (one per year, 1990–2024) |
| Columns | 50 |
| Missing values | 0 |
| Outlier-flagged rows | 4 |

### Column Groups

| Group | Count | Examples |
|-------|-------|---------|
| Identifier | 1 | `Year` |
| Butterfly abundance (smoothed) | 7 | `butterfly_all_species_smoothed`, `butterfly_farmland_specialist_smoothed`, … |
| Butterfly abundance (raw) | 7 | `butterfly_all_species_raw`, … |
| Agri-environment areas (England) | 2 | `agri_area_higher_mha`, `agri_area_lower_mha` (kept separate, not summed) |
| Habitat connectivity | 4 | `habitat_connectivity_index`, `habitat_connectivity_raw`, CI bounds |
| Plant abundance | 3 | `plant_arable_index`, `plant_broadleaved_woodland_and_hedges_index`, `plant_lowland_grassland_index` |
| Plant CI width (uncertainty) | 3 | `plant_arable_ci_width`, `plant_broadleaved_woodland_and_hedges_ci_width`, `plant_lowland_grassland_ci_width` |
| Species trend summary | 3 | `species_conn_pct_decreased`, `species_conn_pct_increased`, `species_conn_pct_no_change` |
| Engineered — change rates | 4 | `butterfly_all_species_smoothed_yoy_change`, … |
| Engineered — ratios | 2 | `specialist_generalist_ratio`, `farmland_woodland_specialist_ratio` |
| Engineered — lagged | 9 | `agri_area_higher_mha_lag1/2/3`, `agri_area_lower_mha_lag1/2/3`, `habitat_connectivity_lag1/2/3` |
| Engineered — rolling averages | 2 | `agri_area_higher_5yr_avg`, `agri_area_lower_5yr_avg` |
| Engineered — classification target | 1 | `decline_flag` |
| Outlier metadata | 2 | `outlier_flag`, `outlier_score` |

---

## 5. Target Variables for ML Modelling

Two target variables are available depending on the modelling approach:

1. **Regression target:** `butterfly_all_species_smoothed` — predict the continuous abundance index.
2. **Classification target:** `decline_flag` — predict whether butterflies declined in a given year (binary: 0/1).

### Candidate Explanatory Features

| Feature | Hypothesis |
|---------|-----------|
| `agri_area_higher_mha` (+ lags, 5yr avg) | Higher-level targeted conservation → direct habitat restoration |
| `agri_area_lower_mha` (+ lags, 5yr avg) | Entry-level schemes → broad-based environmental protection |
| `habitat_connectivity_index` (+ lags) | Higher connectivity → healthier populations |
| `plant_*_index` columns | Plant abundance as food source for pollinators |
| `specialist_generalist_ratio` | Declining ratio indicates habitat degradation |

---

## 6. Tools & Libraries Used

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12 | Pipeline implementation |
| pandas | 3.0.1 | Data loading, merging, manipulation |
| NumPy | 2.4.3 | Numerical operations |
| scikit-learn | 1.8.0 | Isolation Forest outlier detection, StandardScaler |
| SciPy | 1.17.1 | Available for advanced interpolation |

---

## 7. Next Steps

The preprocessed dataset is ready for supervised/unsupervised ML modelling using techniques such as:
- **Decision Trees** — feature importance for identifying key drivers of decline
- **Linear Regression** — quantifying factor contributions
- **Support Vector Machines** — classification of decline vs. non-decline years
- **K-Means Clustering** — identifying distinct ecological regime periods
- **Naïve Bayes** — probabilistic classification of population trends
