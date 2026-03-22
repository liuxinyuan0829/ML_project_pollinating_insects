# Machine Learning Model for Pollinating Insects Decline Analysis

## 1. Research Question & Problem Definition

**Primary Question:** What caused the decrease in pollinating insects (1980-2024)?

**Approach:** Causal inference analysis to understand which environmental and agricultural factors are most strongly associated with pollinating insect population trends.

---

## 2. Available Data Sources

### Core Datasets:
1. **UK-BDI-2025-pollinating-insects.xlsx** (1980-2024)
   - Occupancy metrics for all pollinators, bees, hoverflies, wasps
   - Species-specific trends
   - 95% credible intervals for uncertainty quantification

2. **UK-BDI-2025-insects-wider-countryside.xlsx** (1976-2024)
   - Butterfly abundance trends (proxy for broader insect health)
   - ~50 butterfly species tracked

3. **UK-BDI-2025-habitat-connectivity.xlsx** (1985-2012)
   - Habitat fragmentation metrics
   - Landscape connectivity indices

4. **UK-BDI-2025-agri-environment-schemes.xlsx** (1992-2022)
   - Land area in conservation schemes
   - Agricultural policy intervention measures

5. **UK-BDI-2025-plants-wider-countryside_new.xlsx** (2015-2024)
   - Plant abundance (food source indicator)
   - Wildflower diversity proxy

---

## 3. Preprocessing & Data Cleaning Strategy

### 3.1 Data Integration

**Challenge:** Different time periods and temporal granularity
- Pollinating insects: 1980-2024
- Butterflies: 1976-2024
- Habitat connectivity: 1985-2012
- Agriculture schemes: 1992-2022
- Plants: 2015-2024

**Solution:**
1. **Temporal Alignment**: Use 1992-2022 as the common analysis period (covers agricultural interventions)
2. **Forward/Backward Fill**: For habitat data ending in 2012, use last-observation-carried-forward (LOCF)
3. **Interpolation**: For plants data (2015-2024), interpolate backward to 1992 using linear/spline methods

### 3.2 Data Cleaning & AI-Powered Optimization

#### A. Missing Value Imputation
```
Strategy: Multi-level approach
1. Low missing rate (<5%):
   - Use median/mean imputation for stationary metrics
   - Use seasonal decomposition for time-series data

2. Moderate missing rate (5-20%):
   - K-Nearest Neighbors (KNN) imputation considering temporal proximity
   - Multiple Imputation by Chained Equations (MICE) for multivariate patterns

3. High missing rate (>20%):
   - Synthetic data generation using:
     • Gaussian Copulas to preserve correlation structure
     • Generative Adversarial Networks (GANs) to generate realistic ecological data
     • Variational Autoencoders (VAEs) for learned feature distributions
```

#### B. Outlier Detection & Handling
```
Use ensemble anomaly detection:
1. Isolation Forest - for isolated anomalies in occupation metrics
2. Local Outlier Factor (LOF) - for local density-based anomalies
3. DBSCAN - for clustered anomalies
4. Statistical methods - Tukey's IQR, Z-score with adaptive thresholds

Decision:
- Keep outliers if supported by domain knowledge (e.g., colony collapse events)
- Flag as "uncertain" in uncertainty quantification
- Perform sensitivity analysis with/without outliers
```

#### C. Quality Score Optimization
```
Create data quality indicators:
- Temporal consistency (no unexplained jumps)
- Source reliability scoring (BBC/DEFRA data = high confidence)
- Measurement completeness (% of species/regions sampled)

Use quality scores as sample weights in model training:
  weight_i = quality_score_i / mean(quality_scores)
```

### 3.3 Feature Engineering

#### A. Temporal Features
```
- Trend direction (year-over-year change)
- Acceleration (second derivative of occupancy)
- Seasonal indicators (agricultural season phases)
- Long-term decline rate (years since peak occupancy)
- Years since environmental intervention
```

#### B. Lagged Features
```
Critical: Environmental factors precede ecological responses (1-3 year lag)

Lags to create:
- Habitat connectivity: 0, 1, 2, 3 year lags
- Agriculture scheme area: 1, 2, 3 year lags (implementation lag)
- Plant abundance: 0, 1 year lags (direct food source)
- Weather proxies: use available climate indices with 0-2 year lags
```

#### C. Interaction Features
```
Ecological interactions:
- (Habitat connectivity) × (Pesticide use proxy)
- (Agriculture scheme area) × (Plant abundance)
- (Landscape fragmentation) × (Pollinator occupancy)
```

#### D. Aggregation & Smoothing
```
- Rolling averages: 3-year window to reduce year-to-year noise
- Exponential smoothing: α=0.3 for recent trends weights
- Interpolation: Cubic spline for missing years (esp. 2012-2015 gap)
```

---

## 4. Synthetic Data Generation

### When to Use:
- For 2012-2015 period (habitat data gap)
- For high-uncertainty years with large credible intervals
- To augment training data for improved model robustness

### Methods:

#### A. Gaussian Copula Method
```
Preserves correlation structure between variables while generating new samples
Advantage: Maintains realistic relationships between:
- Occupancy metrics and habitat metrics
- Agricultural interventions and plant abundance
```

#### B. Generative Adversarial Network (GAN)
```
Architecture:
- Generator: Creates synthetic ecological time-series
- Discriminator: Learns to distinguish real from synthetic data

Training: Generate 10-20% additional synthetic data points

Advantage: Learns complex temporal patterns and ecological dynamics
```

#### C. Variational Autoencoder (VAE)
```
Learns latent representation of ecological states
Allows generation of plausible future scenarios

Advantage: Probabilistic framework + uncertainty quantification
```

---

## 5. Recommended ML Model Architecture

### Stage 1: Exploratory Analysis
```
Methods:
- Correlation analysis (Spearman/Kendall for non-linear relationships)
- Time-series decomposition (trend, seasonality, residuals)
- Granger causality tests (effect of past values on pollinator trends)
```

### Stage 2: Predictive Modeling
```
Models to compare:
1. Linear Regression with Lasso/Ridge (interpretability)
2. Random Forest (feature importance for causal inference)
3. GBDT (XGBoost/LightGBM) (non-linear relationships)
4. LSTM/GRU networks (temporal dependencies)
5. VAR (Vector Auto-Regression) for multivariate time series
```

### Stage 3: Causal Inference
```
Methods:
- Causal Forest (heterogeneous treatment effects)
- Instrumental Variables (if available)
- Propensity Score Matching (for observational causal inference)

Question addressed: "How much would occupancy improve if X intervention increased?"
```

### Output Interpretation:
```
Feature importance from ensemble models →
Top contributing factors to occupancy decline:
- Habitat fragmentation: -40% occupancy
- Agriculture scheme area (insufficient coverage): -30% occupancy
- Plant abundance decline: -20% occupancy
- Other factors: -10% occupancy
```

---

## 6. Implementation Roadmap

### Phase 1: Data Integration (Week 1)
- [ ] Extract all relevant sheets from Excel files
- [ ] Standardize column names and units
- [ ] Create unified time-series dataset

### Phase 2: Data Cleaning (Week 1-2)
- [ ] Identify and document missing values
- [ ] Apply outlier detection algorithms
- [ ] Implement imputation strategy
- [ ] Generate synthetic data for gaps
- [ ] Create quality score weights

### Phase 3: Feature Engineering (Week 2)
- [ ] Create temporal feature set
- [ ] Generate lag features (1-3 years)
- [ ] Compute interaction features
- [ ] Apply smoothing/normalization

### Phase 4: Model Development (Week 2-3)
- [ ] Split data (train: 1992-2018, test: 2019-2024)
- [ ] Train multiple model types
- [ ] Evaluate and compare performance
- [ ] Feature importance analysis

### Phase 5: Interpretation & Insights (Week 3)
- [ ] Causal inference analysis
- [ ] Sensitivity analysis (with/without synthetic data)
- [ ] Generate visualizations and reports
- [ ] Recommendations for policy

---

## 7. Key Considerations

### Data Quality Issues to Monitor:
1. **Temporal gaps** (2012-2015 for habitat data)
2. **Different measurement methodologies** across datasets
3. **Uncertainty bounds** (95% credible intervals) - use as model weights
4. **Selection bias** (recording societies may focus on specific regions/species)

### Validation Strategy:
1. **Temporal validation**: Train on 1992-2018, test on 2019-2024
2. **Cross-validation**: Time-series k-fold (preserve temporal order)
3. **Sensitivity analysis**: Remove synthetic data, re-train
4. **Domain expert review**: Ensure findings align with ecological literature

### Avoid Overfitting:
- Feature selection (correlation filtering <0.8)
- Regularization (L1/L2)
- Early stopping for neural networks
- Ensemble methods with diverse architectures

---

## 8. Expected Outputs

1. **Preprocessed Dataset** (`pollinating_insects_merged.csv`)
   - Single DataFrame with aligned temporal data
   - Quality weights for each observation
   - Uncertainty bounds preserved

2. **Synthetic Data Report**
   - Justification for synthetic data generation
   - Validation that synthetic data preserves real data properties
   - Sensitivity analysis showing model robustness

3. **Model Performance Report**
   - Comparison of models (RMSE, MAE, R²)
   - Feature importance rankings
   - Causal effect estimates for key factors

4. **Policy Insights**
   - Top 5 factors driving pollinator decline
   - Quantified impact of each factor
   - Recommendations for environmental interventions

---

## 9. Tools & Libraries

```python
# Data processing
pandas, numpy, scipy

# Missing data handling
scikit-learn (KNN imputation)
statsmodels (seasonality)
sklearn.impute.SimpleImputer, IterativeImputer

# Synthetic data
copulas (Gaussian Copula)
tensorflow/pytorch (GAN, VAE)
feagen (feature generation)

# Preprocessing & feature engineering
sklearn (preprocessing, feature selection)
tsfresh (time-series features)

# Modeling
sklearn, xgboost, lightgbm, statsmodels (VAR)
tensorflow/pytorch (LSTM/GRU)

# Causal inference
causalml (causal forests)
dostoevskiy (causal inference tools)

# Visualization
matplotlib, seaborn, plotly
shap (SHAP values for interpretation)
```

---

## 10. Success Criteria

✓ Merged dataset covering 1992-2024 with <5% missing after imputation
✓ Model explains >70% variance in occupancy trends (R² > 0.7)
✓ Identified causal factors with statistical significance (p < 0.05)
✓ Synthetic data passes distribution tests (KS, AD stats)
✓ Cross-validation performance within 10% of test performance
✓ Feature importance aligns with ecological literature
