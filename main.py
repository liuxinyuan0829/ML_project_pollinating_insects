
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import (
    LeaveOneOut, cross_val_score, cross_val_predict, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))


def load_data():
    path = os.path.join(BASE, "data_preprocessed.csv")
    df = pd.read_csv(path)
    print(f"Loaded preprocessed data: {df.shape[0]} rows × {df.shape[1]} columns")

    target_col = "butterfly_all_species_smoothed"

    feature_cols = [
        "agri_area_higher_mha",
        "agri_area_lower_mha",
        "habitat_connectivity_index",
        "plant_arable_index",
        "plant_broadleaved_woodland_and_hedges_index",
        "plant_lowland_grassland_index",
        "plant_arable_ci_width",
        "plant_broadleaved_woodland_and_hedges_ci_width",
        "plant_lowland_grassland_ci_width",
        "species_conn_pct_decreased",
        "species_conn_pct_increased",
        "species_conn_pct_no_change",
        "agri_area_higher_mha_lag1",
        "agri_area_higher_mha_lag2",
        "agri_area_higher_mha_lag3",
        "agri_area_lower_mha_lag1",
        "agri_area_lower_mha_lag2",
        "agri_area_lower_mha_lag3",
        "habitat_connectivity_index_lag1",
        "habitat_connectivity_index_lag2",
        "habitat_connectivity_index_lag3",
        "agri_area_higher_5yr_avg",
        "agri_area_lower_5yr_avg",
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df[target_col].values
    years = df["Year"].values

    print(f"Target: {target_col}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target range: {y.min():.2f} – {y.max():.2f} (mean={y.mean():.2f})")

    return df, X, y, years, feature_cols, target_col


def train_linear_regression(X, y, feature_cols):
    print("\n" + "=" * 70)
    print("MODEL 1: LINEAR REGRESSION (Ordinary Least Squares)")
    print("=" * 70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nAlgorithmic Parameters:")
    print("  fit_intercept  = True   (baseline abundance when all features are 0)")
    print("  normalize      = False  (we pre-scale with StandardScaler instead)")
    print("  Regularisation = None   (OLS — no L1/L2 penalty)")
    print("  Feature scaling: StandardScaler (μ=0, σ=1) applied to all features")

    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_scaled, y)

    y_pred_train = lr.predict(X_scaled)

    r2_train = r2_score(y, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y, y_pred_train))
    mae_train = mean_absolute_error(y, y_pred_train)

    print(f"\n--- Training Performance ---")
    print(f"  R² Score : {r2_train:.4f}  (proportion of variance explained)")
    print(f"  RMSE     : {rmse_train:.4f}  (root mean squared error)")
    print(f"  MAE      : {mae_train:.4f}  (mean absolute error)")
    print(f"  Intercept: {lr.intercept_:.4f}")

    loo = LeaveOneOut()
    cv_scores = cross_val_score(lr, X_scaled, y, cv=loo, scoring="r2")
    cv_predictions = cross_val_predict(lr, X_scaled, y, cv=loo)

    r2_cv = r2_score(y, cv_predictions)
    rmse_cv = np.sqrt(mean_squared_error(y, cv_predictions))
    mae_cv = mean_absolute_error(y, cv_predictions)

    print(f"\n--- Leave-One-Out Cross-Validation (n=35 folds) ---")
    print(f"  R² Score (CV) : {r2_cv:.4f}")
    print(f"  RMSE (CV)     : {rmse_cv:.4f}")
    print(f"  MAE (CV)      : {mae_cv:.4f}")
    print(f"  Mean fold R²  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if r2_train - r2_cv > 0.15:
        print(f"  ⚠ Overfitting detected: training R²={r2_train:.4f} vs CV R²={r2_cv:.4f}")
        print(f"    Gap = {r2_train - r2_cv:.4f}. Ridge or Lasso regression may improve"
              f" generalisation.")
    else:
        print(f"  ✓ No severe overfitting: train-CV R² gap = {r2_train - r2_cv:.4f}")

    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": lr.coef_,
        "Abs_Coefficient": np.abs(lr.coef_),
    }).sort_values("Abs_Coefficient", ascending=False)

    print(f"\n--- Standardised Coefficients (Feature Importance) ---")
    print(f"  {'Feature':<50s} {'Coeff':>10s} {'|Coeff|':>10s}")
    print(f"  {'─' * 50} {'─' * 10} {'─' * 10}")
    for _, row in coef_df.iterrows():
        direction = "↑" if row["Coefficient"] > 0 else "↓"
        print(f"  {row['Feature']:<50s} {row['Coefficient']:>+10.4f} {row['Abs_Coefficient']:>10.4f} {direction}")

    print(f"\n  Interpretation: A positive coefficient means that feature is associated")
    print(f"  with HIGHER butterfly abundance; negative means associated with DECLINE.")

    return lr, scaler, y_pred_train, cv_predictions, coef_df


def train_decision_tree(X, y, feature_cols):
    print("\n" + "=" * 70)
    print("MODEL 2: DECISION TREE REGRESSOR")
    print("=" * 70)

    params = {
        "max_depth": 4,
        "min_samples_split": 4,
        "min_samples_leaf": 3,
        "random_state": 42,
    }

    print("\nAlgorithmic Parameters:")
    for k, v in params.items():
        print(f"  {k:<22s} = {v}")
    print("\nParameter Justification:")
    print("  max_depth=4        → Limits complexity; 35 samples cannot support")
    print("                       a deep tree without severe overfitting.")
    print("  min_samples_split=4 → Requires ≥4 samples to split; prevents")
    print("                       unreliable splits from tiny groups.")
    print("  min_samples_leaf=3  → Each prediction based on ≥3 data points")
    print("                       (≈8.6% of dataset), ensuring robustness.")
    print("  criterion=squared_error → Minimises MSE at each split; standard")
    print("                       for regression tasks.")

    dt = DecisionTreeRegressor(**params)
    dt.fit(X, y)

    y_pred_train = dt.predict(X)

    r2_train = r2_score(y, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y, y_pred_train))
    mae_train = mean_absolute_error(y, y_pred_train)

    print(f"\n--- Training Performance ---")
    print(f"  R² Score : {r2_train:.4f}")
    print(f"  RMSE     : {rmse_train:.4f}")
    print(f"  MAE      : {mae_train:.4f}")
    print(f"  Tree depth: {dt.get_depth()}, leaves: {dt.get_n_leaves()}")

    loo = LeaveOneOut()
    cv_predictions = cross_val_predict(
        DecisionTreeRegressor(**params), X, y, cv=loo
    )

    r2_cv = r2_score(y, cv_predictions)
    rmse_cv = np.sqrt(mean_squared_error(y, cv_predictions))
    mae_cv = mean_absolute_error(y, cv_predictions)

    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv5_scores = cross_val_score(
        DecisionTreeRegressor(**params), X, y, cv=kf5, scoring="r2"
    )

    print(f"\n--- Leave-One-Out Cross-Validation (n=35 folds) ---")
    print(f"  R² Score (LOOCV) : {r2_cv:.4f}")
    print(f"  RMSE (LOOCV)     : {rmse_cv:.4f}")
    print(f"  MAE  (LOOCV)     : {mae_cv:.4f}")

    print(f"\n--- 5-Fold Cross-Validation ---")
    print(f"  R² Score (5-fold)  : {cv5_scores.mean():.4f} ± {cv5_scores.std():.4f}")
    print(f"  Individual folds   : {[f'{s:.4f}' for s in cv5_scores]}")

    if r2_train - r2_cv > 0.15:
        print(f"\n  ⚠ Overfitting detected: training R²={r2_train:.4f} vs CV R²={r2_cv:.4f}")
    else:
        print(f"\n  ✓ No severe overfitting: train-CV R² gap = {r2_train - r2_cv:.4f}")

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": dt.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print(f"\n--- Feature Importance (Variance Reduction) ---")
    print(f"  {'Feature':<50s} {'Importance':>12s} {'Cumulative':>12s}")
    print(f"  {'─' * 50} {'─' * 12} {'─' * 12}")
    cumulative = 0.0
    for _, row in importance_df.iterrows():
        if row["Importance"] > 0:
            cumulative += row["Importance"]
            print(f"  {row['Feature']:<50s} {row['Importance']:>12.4f} {cumulative:>12.4f}")

    print(f"\n--- Decision Tree Rules ---")
    tree_rules = export_text(dt, feature_names=feature_cols, max_depth=4)
    print(tree_rules)

    return dt, y_pred_train, cv_predictions, importance_df


def compare_models(y, lr_train_pred, lr_cv_pred, dt_train_pred, dt_cv_pred):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON — Linear Regression vs Decision Tree")
    print("=" * 70)

    metrics = {}
    for name, y_train_pred, y_cv_pred in [
        ("Linear Regression", lr_train_pred, lr_cv_pred),
        ("Decision Tree", dt_train_pred, dt_cv_pred),
    ]:
        metrics[name] = {
            "R² (train)": r2_score(y, y_train_pred),
            "R² (LOOCV)": r2_score(y, y_cv_pred),
            "RMSE (train)": np.sqrt(mean_squared_error(y, y_train_pred)),
            "RMSE (LOOCV)": np.sqrt(mean_squared_error(y, y_cv_pred)),
            "MAE (train)": mean_absolute_error(y, y_train_pred),
            "MAE (LOOCV)": mean_absolute_error(y, y_cv_pred),
        }

    print(f"\n  {'Metric':<20s} {'Linear Regression':>20s} {'Decision Tree':>20s} {'Better':>12s}")
    print(f"  {'─' * 20} {'─' * 20} {'─' * 20} {'─' * 12}")

    for metric in metrics["Linear Regression"]:
        lr_val = metrics["Linear Regression"][metric]
        dt_val = metrics["Decision Tree"][metric]

        if "R²" in metric:
            better = "LR" if lr_val >= dt_val else "DT"
        else:
            better = "LR" if lr_val <= dt_val else "DT"

        print(f"  {metric:<20s} {lr_val:>20.4f} {dt_val:>20.4f} {better:>12s}")

    lr_cv_r2 = metrics["Linear Regression"]["R² (LOOCV)"]
    dt_cv_r2 = metrics["Decision Tree"]["R² (LOOCV)"]

    print(f"\n--- Evaluation Discussion ---")
    print(f"  1. R² (coefficient of determination) measures the proportion of variance")
    print(f"     in butterfly abundance explained by the model. R²=1 is perfect; R²=0")
    print(f"     means the model is no better than predicting the mean.")
    print(f"  2. RMSE (root mean squared error) is in the same units as the target")
    print(f"     (butterfly abundance index, base=100). It penalises large errors more")
    print(f"     than small ones.")
    print(f"  3. MAE (mean absolute error) gives the average prediction error in index")
    print(f"     units. It is more robust to outliers than RMSE.")
    print(f"  4. LOOCV (Leave-One-Out Cross-Validation) gives an unbiased estimate of")
    print(f"     generalisation performance on unseen data, which is critical with only")
    print(f"     35 samples.")

    if lr_cv_r2 > dt_cv_r2:
        print(f"\n  Conclusion: Linear Regression generalises better (LOOCV R²={lr_cv_r2:.4f}")
        print(f"  vs {dt_cv_r2:.4f}), suggesting the relationship between environmental")
        print(f"  drivers and butterfly abundance is approximately linear over this period.")
    else:
        print(f"\n  Conclusion: Decision Tree generalises better (LOOCV R²={dt_cv_r2:.4f}")
        print(f"  vs {lr_cv_r2:.4f}), suggesting important non-linear relationships or")
        print(f"  interaction effects between environmental drivers.")

    return metrics


def analyse_decline_drivers(lr_coef_df, dt_importance_df, feature_cols):
    print("\n" + "=" * 70)
    print("ANALYSIS: KEY DRIVERS OF BUTTERFLY POPULATION DECLINE")
    print("=" * 70)

    lr_imp = lr_coef_df.copy()
    lr_imp["Normalised"] = lr_imp["Abs_Coefficient"] / lr_imp["Abs_Coefficient"].max()

    dt_imp = dt_importance_df.copy()
    dt_imp["Normalised"] = dt_imp["Importance"] / max(dt_imp["Importance"].max(), 1e-10)

    combined = lr_imp[["Feature", "Coefficient", "Normalised"]].rename(
    )
    combined = combined.merge(
        on="Feature", how="outer"
    ).fillna(0)

    combined["Combined_score"] = (combined["LR_importance"] + combined["DT_importance"]) / 2
    combined.sort_values("Combined_score", ascending=False, inplace=True)

    print(f"\n--- Combined Feature Importance Ranking ---")
    print(f"  {'Rank':<5s} {'Feature':<45s} {'LR':<8s} {'DT':<8s} {'Combined':<10s} {'Direction':<10s}")
    print(f"  {'─' * 5} {'─' * 45} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 10}")

    for rank, (_, row) in enumerate(combined.iterrows(), 1):
        direction = "Positive" if row["LR_coeff"] > 0 else "Negative"
        print(f"  {rank:<5d} {row['Feature']:<45s} {row['LR_importance']:<8.3f} "
              f"{row['DT_importance']:<8.3f} {row['Combined_score']:<10.3f} {direction:<10s}")

    top_n = min(5, len(combined))
    top_features = combined.head(top_n)

    print(f"\n--- Top {top_n} Drivers of Butterfly Abundance ---")
    for rank, (_, row) in enumerate(top_features.iterrows(), 1):
        direction = "INCREASE" if row["LR_coeff"] > 0 else "DECREASE"
        print(f"\n  {rank}. {row['Feature']}")
        print(f"     Combined importance: {row['Combined_score']:.3f}")
        print(f"     Linear Regression coefficient: {row['LR_coeff']:+.4f} (standardised)")
        if row["LR_coeff"] > 0:
            print(f"     → Higher values of this feature are associated with HIGHER butterfly abundance")
        else:
            print(f"     → Higher values of this feature are associated with LOWER butterfly abundance")

    return combined


def plot_results(years, y, lr_train_pred, lr_cv_pred, dt_train_pred, dt_cv_pred,
                 lr_coef_df, dt_importance_df, combined_importance):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ML Model Evaluation — Butterfly Decline Analysis",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(years, y, "ko-", label="Actual", markersize=4, linewidth=1.5)
    ax.plot(years, lr_train_pred, "b--", label="LR (train)", alpha=0.8, linewidth=1)
    ax.plot(years, dt_train_pred, "r--", label="DT (train)", alpha=0.8, linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Butterfly Abundance Index")
    ax.set_title("Training Fit: Actual vs Predicted")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(years, y, "ko-", label="Actual", markersize=4, linewidth=1.5)
    ax.plot(years, lr_cv_pred, "b^-", label="LR (LOOCV)", alpha=0.7, markersize=3)
    ax.plot(years, dt_cv_pred, "rs-", label="DT (LOOCV)", alpha=0.7, markersize=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Butterfly Abundance Index")
    ax.set_title("Cross-Validated Predictions (LOOCV)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    lr_residuals = y - lr_cv_pred
    dt_residuals = y - dt_cv_pred
    ax.scatter(lr_cv_pred, lr_residuals, c="blue", alpha=0.6, label="LR residuals", s=20)
    ax.scatter(dt_cv_pred, dt_residuals, c="red", alpha=0.6, label="DT residuals", s=20)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Predicted Abundance Index")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Plot (LOOCV)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(y, lr_cv_pred, c="blue", alpha=0.7, label="Linear Regression", s=25)
    ax.scatter(y, dt_cv_pred, c="red", alpha=0.7, label="Decision Tree", s=25)
    lims = [min(y.min(), lr_cv_pred.min(), dt_cv_pred.min()) - 1,
            max(y.max(), lr_cv_pred.max(), dt_cv_pred.max()) + 1]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect prediction")
    ax.set_xlabel("Actual Abundance Index")
    ax.set_ylabel("Predicted Abundance Index (LOOCV)")
    ax.set_title("Predicted vs Actual (LOOCV)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    top_lr = lr_coef_df.head(10)
    colors = ["green" if c > 0 else "red" for c in top_lr["Coefficient"]]
    ax.barh(range(len(top_lr)), top_lr["Coefficient"].values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_lr)))
    ax.set_yticklabels(top_lr["Feature"].values, fontsize=7)
    ax.set_xlabel("Standardised Coefficient")
    ax.set_title("Linear Regression: Top 10 Coefficients")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[1, 2]
    top_dt = dt_importance_df[dt_importance_df["Importance"] > 0].head(10)
    ax.barh(range(len(top_dt)), top_dt["Importance"].values, color="darkorange", alpha=0.7)
    ax.set_yticks(range(len(top_dt)))
    ax.set_yticklabels(top_dt["Feature"].values, fontsize=7)
    ax.set_xlabel("Importance (Variance Reduction)")
    ax.set_title("Decision Tree: Feature Importance")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    out_path = os.path.join(BASE, "ml_model_evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlots saved to: {out_path}")


def main():
    print("=" * 70)
    print("MACHINE LEARNING MODELLING — Butterfly Decline Analysis")
    print("Models: Linear Regression, Decision Tree Regressor")
    print("=" * 70)

    df, X, y, years, feature_cols, target_col = load_data()

    lr, scaler, lr_train_pred, lr_cv_pred, lr_coef_df = train_linear_regression(
        X, y, feature_cols
    )
    dt, dt_train_pred, dt_cv_pred, dt_importance_df = train_decision_tree(
        X, y, feature_cols
    )

    metrics = compare_models(y, lr_train_pred, lr_cv_pred, dt_train_pred, dt_cv_pred)

    combined_imp = analyse_decline_drivers(lr_coef_df, dt_importance_df, feature_cols)

    plot_results(years, y, lr_train_pred, lr_cv_pred, dt_train_pred, dt_cv_pred,
                 lr_coef_df, dt_importance_df, combined_imp)

    combined_imp.to_csv(os.path.join(BASE, "feature_importance.csv"), index=False)
    print(f"Feature importance saved to: feature_importance.csv")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
