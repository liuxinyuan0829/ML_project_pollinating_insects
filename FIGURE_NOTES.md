# Figure Notes — ml_model_evaluation.png

This file describes the six panels in `ml_model_evaluation.png`, which summarise the performance of the two supervised regression models (Linear Regression and Decision Tree) trained on the preprocessed butterfly decline dataset (35 observations, 23 environmental/policy features, target: `butterfly_all_species_smoothed`).

---

## Panel 1 (Top Left): Training Fit — Actual vs Predicted

Shows the actual butterfly abundance index (black line) alongside training predictions from Linear Regression (blue dashed) and Decision Tree (red dashed) over 1990–2024. Both models closely track the declining trend from ~95 in 1990 to ~82 in 2024. Linear Regression fits almost perfectly (training R² = 0.999) while the Decision Tree shows slight step-wise approximations typical of tree-based models (training R² = 0.985). This panel illustrates model capacity but not generalisation — see Panel 2 for unbiased performance.

## Panel 2 (Top Centre): Cross-Validated Predictions (LOOCV)

Displays Leave-One-Out Cross-Validation predictions, where each point was predicted by a model trained on the other 34 observations. This gives an unbiased estimate of how well each model would perform on unseen data. The Decision Tree (red) tracks the actual values more consistently, while Linear Regression (blue) shows larger deviations in the 2005–2010 period. The DT achieves LOOCV R² = 0.888 versus LR's 0.857, indicating the Decision Tree generalises better on this dataset.

## Panel 3 (Top Right): Residual Plot (LOOCV)

Plots LOOCV residuals (actual minus predicted) against predicted values for both models. Ideally, residuals should scatter randomly around zero with no systematic pattern. Decision Tree residuals (red) are more tightly clustered around zero across the full range. Linear Regression residuals (blue) show a few large deviations (up to ±6 index points), particularly at higher predicted values. This suggests LR occasionally overfits to specific training configurations when one sample is held out.

## Panel 4 (Bottom Left): Predicted vs Actual Scatter (LOOCV)

A scatter plot of LOOCV-predicted values against actual values, with the dashed diagonal representing perfect prediction. Points closer to the diagonal indicate better predictions. Both models cluster along the line, with the Decision Tree (red) showing slightly tighter grouping. A few LR predictions (blue) deviate noticeably, particularly in the 82–87 range where the butterfly decline accelerated. This panel provides a direct visual comparison of prediction accuracy between the two models.

## Panel 5 (Bottom Centre): Linear Regression — Top 10 Standardised Coefficients

Horizontal bar chart showing the 10 largest standardised coefficients from the Linear Regression model. Green bars indicate positive associations with butterfly abundance; red bars indicate negative associations. The two largest coefficients are `agri_area_higher_5yr_avg` (−2.85, strongly negative) and `agri_area_lower_5yr_avg` (+1.94, strongly positive). This means higher-level scheme expansion is associated with lower abundance in the short term, while entry-level scheme coverage is associated with higher abundance. The lag-2 higher-level area (+1.48) suggests that higher-level interventions show positive effects after a two-year delay. Habitat connectivity (lag 3) and species connectivity trends also appear among the top 10.

## Panel 6 (Bottom Right): Decision Tree — Feature Importance

Horizontal bar chart showing features ranked by variance reduction (how much each feature reduces prediction error when used for splitting). `agri_area_lower_5yr_avg` dominates at 76.2% importance — this single feature explains the majority of the tree's predictive power. `agri_area_lower_mha` (10.4%) and `habitat_connectivity_index` (8.7%) are secondary factors. The remaining features (`agri_area_higher_mha`, `agri_area_higher_mha_lag3`, `agri_area_lower_mha_lag3`) contribute small but non-zero amounts. This concentrated importance profile reflects the Decision Tree's finding that the scale of entry-level agri-environment schemes is the single most informative predictor of butterfly population trends.
