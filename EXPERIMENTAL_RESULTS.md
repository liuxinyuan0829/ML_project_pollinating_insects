# Experimental Results and Analysis

## 1. Overview

This study applied AI search optimisation and supervised machine learning techniques to investigate the drivers of pollinating insect (butterfly) decline in the United Kingdom. The analysis used five UK Biodiversity Indicator datasets covering agri-environment schemes (England), habitat connectivity, butterfly abundance, plant abundance and species connectivity trends from 1990 to 2024. The preprocessing pipeline produced a unified dataset of 35 observations across 50 features, which was then used to train and evaluate two supervised regression models: Linear Regression and a Decision Tree Regressor.

## 2. Simulated Annealing in Data Preprocessing

Simulated Annealing (SA) was applied during preprocessing to refine values imputed by linear interpolation. The merged dataset contained 313 missing values (out of 35 rows × 30 raw columns) due to differing temporal coverage across datasets — notably, plant abundance data covers only 2015–2024 (71.4% missing in the analysis window) and habitat connectivity ends in 2012 (34.3% missing). After linear interpolation and edge-filling eliminated all gaps, SA was used to perturb imputed cells and optimise a composite quality score combining temporal smoothness and cross-feature correlation consistency.

The SA algorithm ran for 1,000 iterations with an initial temperature of 10.0 and a cooling rate of 0.995. It accepted 984 of 1,000 proposed moves (98.4%), of which 391 were improving. The high acceptance rate at early iterations reflects the algorithm's ability to explore broadly before converging as temperature decreases — a key advantage over pure hill climbing, which would be trapped by the first local optimum encountered. This ensured that imputed values maintained realistic ecological gradients rather than introducing artefacts at series boundaries, providing the maximum amount of viable data for downstream modelling.

## 3. Supervised Learning Results

Two regression models were trained to predict the butterfly all-species smoothed abundance index from 23 environmental and policy driver features. Both models were evaluated using Leave-One-Out Cross-Validation (LOOCV), which is appropriate for the small sample size (n=35) as it provides an almost unbiased estimate of generalisation error.

**Linear Regression** achieved a training R² of 0.999 and a cross-validated R² of 0.857, with a LOOCV RMSE of 1.44 index points and MAE of 0.74. The train–CV gap of 0.14 suggests mild overfitting, expected given 23 features for 35 samples. The most influential standardised coefficients were the 5-year rolling average of higher-level agri-environment scheme area (−2.85, negative association) and lower-level scheme area (+1.94, positive association), followed by lagged higher-level scheme area and habitat connectivity features.

The **Decision Tree Regressor** (max depth 4, minimum 3 samples per leaf) achieved a training R² of 0.985 and a cross-validated R² of 0.888, with LOOCV RMSE of 1.28. Its train–CV gap of 0.097 indicates better generalisation than Linear Regression, suggesting that non-linear relationships and interaction effects between features are important in this domain. The tree's primary split was on the 5-year average of lower-level (entry-level) scheme area: when this exceeded 1.23 million hectares, predicted butterfly abundance dropped substantially (82–87 versus 89–94). Secondary splits on habitat connectivity index and higher-level scheme area further refined predictions.

## 4. Key Findings

Both models consistently identified agri-environment scheme variables as the dominant drivers. Specifically, the 5-year rolling average of entry-level scheme area accounted for 76% of variance reduction in the Decision Tree and was the top-ranked feature in the combined importance analysis (score: 0.84). Habitat connectivity, both current and lagged by up to three years, emerged as a secondary but consistent factor across both models. Plant abundance features showed limited direct predictive power, likely because their temporal coverage (10 years) required extensive imputation.

The negative association between current higher-level scheme area and butterfly abundance, combined with a positive lagged effect (lag 2–3 years), suggests that conservation interventions take several years to produce measurable ecological benefits — a finding consistent with established ecological literature on habitat restoration timescales.

## 5. Implications for Further AI Research

These results demonstrate that AI techniques can extract meaningful, actionable patterns from fragmented biodiversity datasets. The SA-optimised preprocessing successfully bridged gaps across datasets with different temporal coverage, while both ML models explained over 85% of variance in butterfly abundance using only environmental policy variables. For Bee Positive and similar NGOs, this provides a quantitative evidence base showing that agri-environment scheme design — particularly the balance between entry-level and higher-level interventions — measurably influences pollinator populations, and that lagged effects must be accounted for when evaluating policy effectiveness. The Decision Tree's interpretable rules offer a practical tool for communicating these relationships to policymakers and farmers. Further research could extend this framework with additional data sources (e.g., pesticide usage, climate variables) and more advanced ensemble methods to improve prediction accuracy and support targeted biodiversity interventions.
