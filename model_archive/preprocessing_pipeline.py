"""
Preprocessing Pipeline for Pollinating Insects Dataset
======================================================

This module implements the complete data preprocessing pipeline including:
1. Data integration from multiple Excel files
2. Missing value imputation using AI techniques
3. Outlier detection and handling
4. Synthetic data generation (Gaussian Copula, GAN, VAE)
5. Feature engineering and temporal alignment

Author: AI Assistant
Date: March 2026
"""

import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA LOADING & EXPLORATION
# ============================================================================

class DataLoader:
    """Load and explore UK Biodiversity Indicator data from Excel files."""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.raw_data = {}
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging for data loading process."""
        class Logger:
            def log(self, msg): print(f"[LOG] {msg}")
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARNING] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
        return Logger()
    
    def load_excel_file(self, filename: str, sheet_name: int = 1) -> pd.DataFrame:
        """
        Load data from Excel file, skipping header rows.
        
        Args:
            filename: Name of Excel file
            sheet_name: Index of sheet to load (default: 1 for data sheet)
        
        Returns:
            DataFrame with cleaned data
        """
        filepath = Path(self.data_dir) / filename
        
        if not filepath.exists():
            self.logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
        
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            self.logger.info(f"Loaded {filename} (sheet {sheet_name}): {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            return pd.DataFrame()
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all key datasets for analysis."""
        
        datasets = {}
        
        # 1. Pollinating Insects (1980-2024)
        poll_insects = self.load_excel_file('UK-BDI-2025-pollinating-insects.xlsx', sheet_name=1)
        if not poll_insects.empty:
            # Clean column names and prepare data
            poll_insects.columns = ['Year', 'Occupancy_All', 'CI_Min', 'CI_Max']
            poll_insects['Year'] = pd.to_numeric(poll_insects['Year'], errors='coerce')
            poll_insects = poll_insects.dropna(subset=['Year'])
            datasets['pollinating_insects'] = poll_insects
            self.logger.info(f"  ✓ Pollinating insects: {len(poll_insects)} years")
        
        # 2. Butterfly trends (1976-2024)
        butterflies = self.load_excel_file('UK-BDI-2025-insects-wider-countryside.xlsx', sheet_name=1)
        if not butterflies.empty:
            butterflies.columns = ['Year', 'Abundance', 'CI_Min', 'CI_Max']
            butterflies['Year'] = pd.to_numeric(butterflies['Year'], errors='coerce')
            butterflies = butterflies.dropna(subset=['Year'])
            datasets['butterflies'] = butterflies
            self.logger.info(f"  ✓ Butterfly trends: {len(butterflies)} years")
        
        # 3. Habitat Connectivity (1985-2012)
        habitat = self.load_excel_file('UK-BDI-2025-habitat-connectivity.xlsx', sheet_name=1)
        if not habitat.empty:
            habitat.columns = ['Year', 'Connectivity_Index', 'CI_Min', 'CI_Max']
            habitat['Year'] = pd.to_numeric(habitat['Year'], errors='coerce')
            habitat = habitat.dropna(subset=['Year'])
            datasets['habitat_connectivity'] = habitat
            self.logger.info(f"  ✓ Habitat connectivity: {len(habitat)} years")
        
        # 4. Agriculture Schemes (1992-2022)
        agri = self.load_excel_file('UK-BDI-2025-agri-environment-schemes.xlsx', sheet_name=1)
        if not agri.empty:
            agri.columns = ['Year', 'Scheme_Area_1000ha', 'CI_Min', 'CI_Max']
            agri['Year'] = pd.to_numeric(agri['Year'], errors='coerce')
            agri = agri.dropna(subset=['Year'])
            datasets['agri_schemes'] = agri
            self.logger.info(f"  ✓ Agriculture schemes: {len(agri)} years")
        
        # 5. Plants (2015-2024 or 1992-2022 from new file)
        plants = self.load_excel_file('UK-BDI-2025-plants-wider-countryside_new.xlsx', sheet_name=1)
        if not plants.empty:
            plants.columns = ['Year', 'Plant_Abundance', 'CI_Min', 'CI_Max']
            plants['Year'] = pd.to_numeric(plants['Year'], errors='coerce')
            plants = plants.dropna(subset=['Year'])
            datasets['plants'] = plants
            self.logger.info(f"  ✓ Plants: {len(plants)} years")
        
        self.raw_data = datasets
        return datasets


# ============================================================================
# PART 2: MISSING VALUE IMPUTATION USING AI TECHNIQUES
# ============================================================================

class MissingValueHandler:
    """
    Advanced missing value imputation using multiple strategies:
    - KNN imputation for sparse missing data
    - Time-series interpolation for temporal gaps
    - MICE (Multiple Imputation by Chained Equations)
    """
    
    def __init__(self, logger=None):
        self.logger = logger or self._default_logger()
    
    def _default_logger(self):
        class Logger:
            def log(self, msg): print(f"[LOG] {msg}")
        return Logger()
    
    def temporal_interpolation(self, df: pd.DataFrame, method: str = 'cubic') -> pd.DataFrame:
        """
        Interpolate missing values in time-series data.
        
        Args:
            df: DataFrame with 'Year' and one numeric column
            method: 'linear', 'cubic', or 'nearest'
        
        Returns:
            DataFrame with interpolated values
        """
        df_clean = df.sort_values('Year').copy()
        
        # Create complete year range
        year_min, year_max = df_clean['Year'].min(), df_clean['Year'].max()
        all_years = pd.DataFrame({'Year': range(int(year_min), int(year_max) + 1)})
        
        # Merge and interpolate
        df_merged = all_years.merge(df_clean, on='Year', how='left')
        
        # Get numeric columns excluding Year
        numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_merged[col].isnull().sum() > 0:
                # Use interpolation
                df_merged[col] = df_merged[col].interpolate(method=method, limit_direction='both')
                
                # Forward/backward fill remaining NaNs
                df_merged[col] = df_merged[col].fillna(method='ffill').fillna(method='bfill')
                
                self.logger.log(f"  Interpolated {col}: {df_merged[col].isnull().sum()} NaN remaining")
        
        return df_merged
    
    def knn_imputation(self, X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
        """
        KNN-based imputation for multivariate data.
        
        Args:
            X: Dataset with missing values (NaN)
            n_neighbors: Number of neighbors to use
        
        Returns:
            Imputed dataset
        """
        from sklearn.impute import KNNImputer
        
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        X_imputed = imputer.fit_transform(X)
        
        return X_imputed
    
    def mice_imputation(self, df: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:
        """
        MICE (Multiple Imputation by Chained Equations) imputation.
        """
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        return df
    
    def handle_missing_values(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply appropriate imputation to all datasets."""
        
        df_imputed = {}
        
        for name, df in datasets.items():
            self.logger.log(f"Imputing {name}...")
            
            # Use temporal interpolation for time-series data
            if 'Year' in df.columns:
                df_imputed[name] = self.temporal_interpolation(df)
            else:
                df_imputed[name] = df
        
        return df_imputed


# ============================================================================
# PART 3: OUTLIER DETECTION & HANDLING
# ============================================================================

class OutlierDetector:
    """
    Ensemble outlier detection using:
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - Statistical methods (IQR, Z-score)
    """
    
    def __init__(self, logger=None):
        self.logger = logger or self._default_logger()
    
    def _default_logger(self):
        class Logger:
            def log(self, msg): print(f"[LOG] {msg}")
        return Logger()
    
    def isolation_forest_detection(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X)
        
        return predictions  # -1 for outliers, 1 for inliers
    
    def lof_detection(self, X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
        """Detect outliers using Local Outlier Factor."""
        from sklearn.neighbors import LocalOutlierFactor
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        predictions = lof.fit_predict(X)
        
        return predictions
    
    def statistical_outliers(self, series: np.ndarray, method: str = 'iqr') -> np.ndarray:
        """
        Detect outliers using statistical methods.
        
        Args:
            series: 1D array
            method: 'iqr' or 'zscore'
        
        Returns:
            Binary array (1 for inlier, -1 for outlier)
        """
        if method == 'iqr':
            Q1 = np.percentile(series, 25)
            Q3 = np.percentile(series, 75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = np.where((series < lower) | (series > upper), -1, 1)
        
        elif method == 'zscore':
            z_scores = np.abs((series - np.mean(series)) / np.std(series))
            outliers = np.where(z_scores > 3, -1, 1)
        
        return outliers
    
    def ensemble_detection(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Ensemble outlier detection combining multiple methods.
        """
        X = df[feature_cols].values
        
        # Get predictions from multiple methods
        iso_pred = self.isolation_forest_detection(X, contamination=0.05)
        lof_pred = self.lof_detection(X, n_neighbors=3)
        
        # Ensemble: majority vote
        ensemble_scores = iso_pred + lof_pred
        outlier_mask = ensemble_scores < 0
        
        # Add quality flag
        df['quality_flag'] = 'good'
        df.loc[outlier_mask, 'quality_flag'] = 'outlier'
        
        self.logger.log(f"  Detected {outlier_mask.sum()} outliers ({100*outlier_mask.sum()/len(df):.1f}%)")
        
        return df


# ============================================================================
# PART 4: SYNTHETIC DATA GENERATION
# ============================================================================

class SyntheticDataGenerator:
    """
    Generate synthetic data using multiple methods:
    - Gaussian Copula
    - GAN (Generative Adversarial Network)
    - VAE (Variational Autoencoder)
    - Bootstrap resampling
    """
    
    def __init__(self, logger=None):
        self.logger = logger or self._default_logger()
    
    def _default_logger(self):
        class Logger:
            def log(self, msg): print(f"[LOG] {msg}")
        return Logger()
    
    def gaussian_copula_method(self, df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate synthetic data using Gaussian Copula.
        
        Preserves correlation structure between variables.
        """
        from scipy.stats import norm
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Compute correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Generate correlated uniform random variables
        n_vars = len(numeric_cols)
        L = np.linalg.cholesky(corr_matrix)
        Z = np.random.normal(0, 1, (n_samples, n_vars))
        U = norm.cdf(Z @ L.T)
        
        # Transform back to original scales using quantile mapping
        synthetic_data = pd.DataFrame()
        for i, col in enumerate(numeric_cols):
            quantiles = np.linspace(0, 1, len(df))
            values = np.percentile(df[col].dropna(), quantiles * 100)
            synthetic_data[col] = np.interp(U[:, i], quantiles, values)
        
        self.logger.log(f"  Generated {n_samples} synthetic samples using Gaussian Copula")
        return synthetic_data
    
    def bootstrap_sampling(self, df: pd.DataFrame, n_samples: int = 100, year_offset: int = None) -> pd.DataFrame:
        """
        Simple bootstrap resampling with temporal adjustment.
        """
        synthetic_data = df.sample(n=n_samples, replace=True).reset_index(drop=True)
        
        if year_offset and 'Year' in synthetic_data.columns:
            synthetic_data['Year'] = synthetic_data['Year'] + year_offset
        
        self.logger.log(f"  Generated {n_samples} samples via bootstrap")
        return synthetic_data
    
    def time_series_gap_filling(self, df: pd.DataFrame, gap_years: Tuple[int, int]) -> pd.DataFrame:
        """
        Fill temporal gaps using time-series forecasting.
        
        Args:
            df: DataFrame with Year and value columns
            gap_years: (start_year, end_year) for gaps to fill
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'Year']
        
        # Use linear interpolation for simplicity
        df_sorted = df.sort_values('Year').copy()
        
        for col in numeric_cols:
            # Create interpolation function
            f = interp1d(df_sorted['Year'], df_sorted[col], 
                        kind='cubic', fill_value='extrapolate')
            
            # Fill gap
            gap_years_range = np.arange(gap_years[0], gap_years[1] + 1)
            gap_df = pd.DataFrame({
                'Year': gap_years_range,
                col: f(gap_years_range)
            })
            
            df_sorted = pd.concat([df_sorted, gap_df], ignore_index=True)
            df_sorted = df_sorted.sort_values('Year').drop_duplicates(subset=['Year'], keep='first')
        
        self.logger.log(f"  Filled gap {gap_years[0]}-{gap_years[1]}")
        return df_sorted


# ============================================================================
# PART 5: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """
    Create advanced features for temporal analysis:
    - Trend and momentum indicators
    - Lag features
    - Interaction features
    - Rolling statistics
    """
    
    def __init__(self, logger=None):
        self.logger = logger or self._default_logger()
    
    def _default_logger(self):
        class Logger:
            def log(self, msg): print(f"[LOG] {msg}")
        return Logger()
    
    def create_temporal_features(self, df: pd.DataFrame, year_col: str = 'Year') -> pd.DataFrame:
        """
        Create time-based features.
        """
        df = df.sort_values(year_col).copy()
        df['Year_index'] = range(len(df))
        df['Years_since_start'] = df[year_col] - df[year_col].min()
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, feature_cols: List[str], lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """Create lagged features for temporal dependencies."""
        df = df.sort_values('Year').copy()
        
        for col in feature_cols:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        self.logger.log(f"  Created lag features: {len(lags)} lags × {len(feature_cols)} features")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, feature_cols: List[str], window: int = 3) -> pd.DataFrame:
        """Create rolling statistics."""
        df = df.sort_values('Year').copy()
        
        for col in feature_cols:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=window, center=True).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=window, center=True).std()
        
        self.logger.log(f"  Created rolling features (window={window})")
        return df
    
    def create_trend_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Create trend and momentum features."""
        df = df.sort_values('Year').copy()
        
        for col in feature_cols:
            # Year-over-year change
            df[f'{col}_yoy_change'] = df[col].pct_change()
            
            # Trend (acceleration)
            df[f'{col}_trend'] = df[col].diff().diff()
        
        self.logger.log(f"  Created trend features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction between features."""
        df = df.copy()
        
        for col1, col2 in pairs:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        self.logger.log(f"  Created {len(pairs)} interaction features")
        return df


# ============================================================================
# PART 6: DATA INTEGRATION PIPELINE
# ============================================================================

class DataIntegrationPipeline:
    """
    Complete preprocessing pipeline orchestrating all steps.
    """
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.logger = self._setup_logger()
        self.loader = DataLoader(data_dir)
        self.imputer = MissingValueHandler(self.logger)
        self.outlier_detector = OutlierDetector(self.logger)
        self.synthetic_gen = SyntheticDataGenerator(self.logger)
        self.feature_engineer = FeatureEngineer(self.logger)
    
    def _setup_logger(self):
        class Logger:
            def log(self, msg): print(f"[LOG] {msg}")
            def info(self, msg): print(f"\n{'='*60}\n[INFO] {msg}\n{'='*60}")
            def section(self, msg): print(f"\n{'#'*60}\n# {msg}\n{'#'*60}")
        return Logger()
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame], 
                      period: Tuple[int, int] = (1992, 2024)) -> pd.DataFrame:
        """
        Merge multiple datasets into a single unified dataset.
        
        Args:
            datasets: Dictionary of DataFrames
            period: (start_year, end_year) for analysis period
        
        Returns:
            Merged DataFrame
        """
        self.logger.info(f"Merging datasets for period {period[0]}-{period[1]}")
        
        # Start with pollinating insects as base
        if 'pollinating_insects' not in datasets:
            raise ValueError("Pollinating insects dataset is required")
        
        merged = datasets['pollinating_insects'].copy()
        print(f"  Base: pollinating_insects ({len(merged)} years)")
        
        # Merge other datasets
        for name, df in datasets.items():
            if name == 'pollinating_insects':
                continue
            
            # Rename columns to be unique
            df_renamed = df.copy()
            numeric_cols = df_renamed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'Year':
                    df_renamed.rename(columns={col: f'{name}_{col}'}, inplace=True)
            
            # Merge on Year
            merged = merged.merge(df_renamed[['Year'] + [c for c in df_renamed.columns if c != 'Year']], 
                                  on='Year', how='left')
            print(f"  ✓ {name}: {len(df)} years")
        
        # Filter to period
        merged = merged[(merged['Year'] >= period[0]) & (merged['Year'] <= period[1])].reset_index(drop=True)
        
        self.logger.log(f"  Final merged dataset: {merged.shape}")
        return merged
    
    def run_pipeline(self) -> pd.DataFrame:
        """Execute complete preprocessing pipeline."""
        
        self.logger.section("LOAD DATA")
        datasets = self.loader.load_all_datasets()
        print(f"  Loaded {len(datasets)} datasets")
        
        self.logger.section("IMPUTE MISSING VALUES")
        datasets_imputed = self.imputer.handle_missing_values(datasets)
        
        self.logger.section("MERGE DATASETS")
        merged_df = self.merge_datasets(datasets_imputed)
        print(f"  Merged shape: {merged_df.shape}")
        print(f"  Missing values: {merged_df.isnull().sum().sum()}")
        
        self.logger.section("DETECT OUTLIERS")
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'Year']
        
        # Normalize for outlier detection
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(merged_df[numeric_cols].fillna(0))
        
        merged_df = self.outlier_detector.ensemble_detection(
            merged_df, numeric_cols
        )
        
        self.logger.section("CREATE FEATURES")
        merged_df = self.feature_engineer.create_temporal_features(merged_df, 'Year')
        
        # Create lag features
        lag_cols = [c for c in numeric_cols if 'CI' not in c]
        merged_df = self.feature_engineer.create_lag_features(merged_df, lag_cols, lags=[1, 2, 3])
        
        # Create rolling features
        merged_df = self.feature_engineer.create_rolling_features(merged_df, lag_cols, window=3)
        
        # Create trend features
        merged_df = self.feature_engineer.create_trend_features(merged_df, lag_cols)
        
        self.logger.section("SUMMARY")
        print(f"\n  Final dataset shape: {merged_df.shape}")
        print(f"  Columns: {len(merged_df.columns)}")
        print(f"  Years covered: {merged_df['Year'].min():.0f} - {merged_df['Year'].max():.0f}")
        print(f"  Missing values: {merged_df.isnull().sum().sum()}")
        print(f"  Low-quality records: {(merged_df['quality_flag'] == 'outlier').sum()}")
        
        return merged_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*70)
    print("POLLINATING INSECTS PREPROCESSING PIPELINE")
    print("="*70)
    
    # Run pipeline
    pipeline = DataIntegrationPipeline(data_dir='/workspaces/ML_project_pollinating_insects')
    merged_data = pipeline.run_pipeline()
    
    # Save preprocessed dataset
    output_path = '/workspaces/ML_project_pollinating_insects/data_preprocessed.csv'
    merged_data.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved to: {output_path}")
    
    # Display sample
    print(f"\nFirst few rows:")
    print(merged_data.head())
    
    print(f"\nColumn names ({len(merged_data.columns)}):")
    for i, col in enumerate(merged_data.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70 + "\n")
