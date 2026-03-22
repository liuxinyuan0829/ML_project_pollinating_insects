"""
Automated Preprocessing Pipeline for Pollinating Insects Dataset
===============================================================

Uses automatic structure detection to handle varying Excel layouts.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')


class AutoPreprocessingPipeline:
    """Automatically handles different Excel data structures."""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
    
    def extract_data_from_excel(self, filepath: str, sheet_name=0) -> pd.DataFrame:
        """
        Extract data from Excel with automatic structure detection.
        Handles:
        - Headers at different row positions
        - Extra columns
        - Mixed data types
        """
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
        except Exception as e:
            print(f"    Warning: Could not read {filepath}: {e}")
            return pd.DataFrame()
        
        # Find row containing "Year"
        header_row = None
        for i in range(min(20, len(df))):
            row_str = ' '.join(str(x) for x in df.iloc[i].values)
            if 'Year' in row_str or 'year' in row_str:
                header_row = i
                break
        
        if header_row is None:
            return pd.DataFrame()
        
        try:
            # Get header and data
            headers = df.iloc[header_row].values
            data = df.iloc[header_row+1:].copy()
            data.columns = headers
            
            # Reset column names to strings
            data.columns = [str(c) for c in data.columns]
            
            # Find Year column
            year_col = None
            for col in data.columns:
                if 'Year' in str(col) or 'year' in str(col):
                    year_col = col
                    break
            
            if year_col is None:
                return pd.DataFrame()
            
            # Get numeric columns (usually the 3 after Year)
            year_idx = list(data.columns).index(year_col)
            cols_to_keep = [year_col]
            
            for col in list(data.columns)[year_idx+1:year_idx+4]:
                cols_to_keep.append(col)
            
            data = data[cols_to_keep]
            
            # Convert to numeric
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove rows with NaN year
            data = data.dropna(subset=[year_col])
            data = data.sort_values(year_col).reset_index(drop=True)
            
            # Rename columns cleanly
            data.columns = [f'col_{i}' for i in range(len(data.columns))]
            data = data.rename(columns={'col_0': 'Year'})
            
            return data if len(data) > 0 else pd.DataFrame()
        
        except Exception as e:
            print(f"    Warning: Error processing {filepath}: {e}")
            return pd.DataFrame()
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets using automatic detection."""
        
        datasets = {}
        
        # Helper function to safely rename columns
        def safe_rename(df, new_names):
            if df.empty or 'Year' not in df.columns:
                return df
            cols = list(df.columns)
            new_cols = {cols[i]: new_names[i] for i in range(min(len(cols), len(new_names)))}
            return df.rename(columns=new_cols)
        
        # Pollinating insects (1980-2024)
        poll = self.extract_data_from_excel(
            Path(self.data_dir) / 'UK-BDI-2025-pollinating-insects.xlsx',
            sheet_name='1'
        )
        if not poll.empty:
            poll = safe_rename(poll, ['Year', 'Occupancy', 'CI_Min', 'CI_Max'])
            datasets['Pollinating_Insects'] = poll
            print(f"  ✓ Pollinating Insects: {len(poll)} years ({poll['Year'].min():.0f}-{poll['Year'].max():.0f})")
        
        # Butterflies (1976-2024)
        butt = self.extract_data_from_excel(
            Path(self.data_dir) / 'UK-BDI-2025-insects-wider-countryside.xlsx',
            sheet_name='1'
        )
        if not butt.empty:
            butt = safe_rename(butt, ['Year', 'Butterfly_Abundance', 'Butterfly_CI_Min', 'Butterfly_CI_Max'])
            datasets['Butterfly_Abundance'] = butt
            print(f"  ✓ Butterfly Abundance: {len(butt)} years ({butt['Year'].min():.0f}-{butt['Year'].max():.0f})")
        
        # Habitat connectivity (1985-2012)
        hab = self.extract_data_from_excel(
            Path(self.data_dir) / 'UK-BDI-2025-habitat-connectivity.xlsx',
            sheet_name='1'
        )
        if not hab.empty:
            hab = safe_rename(hab, ['Year', 'Habitat_Connectivity', 'Habitat_CI_Min', 'Habitat_CI_Max'])
            datasets['Habitat_Connectivity'] = hab
            print(f"  ✓ Habitat Connectivity: {len(hab)} years ({hab['Year'].min():.0f}-{hab['Year'].max():.0f})")
        
        # Agriculture schemes (1992-2022)
        agri = self.extract_data_from_excel(
            Path(self.data_dir) / 'UK-BDI-2025-agri-environment-schemes.xlsx',
            sheet_name='1'
        )
        if not agri.empty:
            agri = safe_rename(agri, ['Year', 'Agri_Scheme_Area', 'Agri_CI_Min', 'Agri_CI_Max'])
            datasets['Agri_Schemes'] = agri
            print(f"  ✓ Agriculture Schemes: {len(agri)} years ({agri['Year'].min():.0f}-{agri['Year'].max():.0f})")
        
        # Plants (2015-2024)
        plants = self.extract_data_from_excel(
            Path(self.data_dir) / 'UK-BDI-2025-plants-wider-countryside_new.xlsx',
            sheet_name='1'
        )
        if not plants.empty:
            plants = safe_rename(plants, ['Year', 'Plant_Abundance', 'Plant_CI_Min', 'Plant_CI_Max'])
            datasets['Plants'] = plants
            print(f"  ✓ Plants: {len(plants)} years ({plants['Year'].min():.0f}-{plants['Year'].max():.0f})")
        
        return datasets
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame], 
                      analysis_period: Tuple[int, int] = (1992, 2024)) -> pd.DataFrame:
        """Merge all datasets on Year with interpolation for gaps."""
        
        # Start with pollinating insects as base
        if 'Pollinating_Insects' not in datasets:
            raise ValueError("Pollinating insects data required")
        
        merged = datasets['Pollinating_Insects'].copy()
        print(f"\n  Base dataset (Pollinating Insects): {len(merged)} rows")
        
        # Merge other datasets
        for name, df in datasets.items():
            if name == 'Pollinating_Insects':
                continue
            
            df_renamed = df.copy()
            # Rename data columns to include dataset name
            for col in df_renamed.columns:
                if col != 'Year':
                    df_renamed.rename(columns={col: col}, inplace=True)
            
            merged = merged.merge(df_renamed, on='Year', how='left')
            print(f"  Merged {name}: {len(df)} rows")
        
        # Create complete year range and interpolate
        year_min = max(analysis_period[0], int(merged['Year'].min()))
        year_max = min(analysis_period[1], int(merged['Year'].max()))
        
        all_years = pd.DataFrame({'Year': range(year_min, year_max + 1)})
        merged = all_years.merge(merged, on='Year', how='left')
        
        # Interpolate missing values
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'Year']
        
        for col in numeric_cols:
            merged[col] = merged[col].interpolate(method='linear', limit_direction='both')
            merged[col] = merged[col].fillna(value=merged[col].mean())  # Fill remaining with mean
        
        return merged.reset_index(drop=True)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for modeling."""
        df = df.sort_values('Year').reset_index(drop=True)
        
        # Identify main feature columns (not CI bounds)
        feature_cols = [c for c in df.columns if c not in ['Year'] and 'CI' not in c]
        
        print(f"\n  Creating features from: {feature_cols}")
        
        # Year-over-year percentage change
        for col in feature_cols:
            df[f'{col}_pct_change'] = df[col].pct_change() * 100
        
        # Lag features (1, 2, 3 years)
        for col in feature_cols:
            for lag in [1, 2, 3]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # 3-year rolling average and std
        for col in feature_cols:
            df[f'{col}_MA3'] = df[col].rolling(window=3, center=True).mean()
            df[f'{col}_STD3'] = df[col].rolling(window=3, center=True).std()
        
        # Trend (second derivative)
        for col in feature_cols:
            df[f'{col}_trend'] = df[col].diff().diff()
        
        # Drop NaN introduced by lagging
        df = df.dropna()
        
        return df.reset_index(drop=True)
    
    def add_quality_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add data quality and confidence metrics."""
        
        # Anomaly detection using Isolation Forest
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['Year'] and 'CI' not in c]
        
        if len(numeric_cols) > 0:
            X = df[numeric_cols].fillna(df[numeric_cols].mean()).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            df['Anomaly_Flag'] = iso_forest.fit_predict(X_scaled)
            df['Data_Quality'] = df['Anomaly_Flag'].map({1: 'Good', -1: 'Anomaly'})
        
        # Confidence score from credible interval width
        ci_cols = [c for c in df.columns if 'CI_Max' in c]
        ci_min_cols = [c for c in df.columns if 'CI_Min' in c]
        
        if len(ci_cols) > 0 and len(ci_min_cols) > 0:
            df['CI_Median_Width'] = 0
            for ci_max, ci_min in zip(ci_cols, ci_min_cols):
                width = (df[ci_max] - df[ci_min]).abs()
                df['CI_Median_Width'] += width / len(ci_cols)
            
            max_width = df['CI_Median_Width'].max()
            df['Confidence_Score'] = 1.0 - (df['CI_Median_Width'] / (max_width + 1e-6))
            df['Confidence_Score'] = df['Confidence_Score'].clip(0, 1)
        else:
            df['Confidence_Score'] = 1.0
        
        return df
    
    def run(self, output_file: str = 'data_preprocessed.csv') -> pd.DataFrame:
        """Execute full preprocessing pipeline."""
        
        print("\n" + "=" * 80)
        print(" POLLINATING INSECTS - DATA PREPROCESSING & FEATURE ENGINEERING")
        print("=" * 80)
        
        # Step 1: Load all datasets
        print("\n[STEP 1] Loading UK Biodiversity Indicator Datasets")
        print("-" * 80)
        datasets = self.load_all_datasets()
        
        if not datasets:
            raise ValueError("No datasets loaded successfully")
        
        # Step 2: Merge datasets
        print("\n[STEP 2] Merging Datasets (1992-2024)")
        print("-" * 80)
        merged = self.merge_datasets(datasets)
        
        print(f"\n  Merged dataset: {merged.shape[0]} years × {merged.shape[1]} columns")
        print(f"  Years: {merged['Year'].min():.0f} - {merged['Year'].max():.0f}")
        print(f"  Missing values: {merged.isnull().sum().sum()}")
        
        # Step 3: Feature engineering
        print("\n[STEP 3] Feature Engineering")
        print("-" * 80)
        engineered = self.create_features(merged)
        
        print(f"\n  Features created: {engineered.shape[1]} total columns")
        print(f"  Remaining samples: {engineered.shape[0]} (after lag-induced NaN removal)")
        
        # Step 4: Quality metrics
        print("\n[STEP 4] Adding Quality Metrics")
        print("-" * 80)
        final = self.add_quality_metrics(engineered)
        
        # Step 5: Summary
        print("\n[STEP 5] Data Summary")
        print("-" * 80)
        print(f"\n  Dataset Shape: {final.shape[0]} samples × {final.shape[1]} features")
        print(f"  Year Range: {final['Year'].min():.0f} - {final['Year'].max():.0f}")
        
        # Feature categories
        occupancy_feats = [c for c in final.columns if 'Occupancy' in c]
        butterfly_feats = [c for c in final.columns if 'Butterfly' in c]
        habitat_feats = [c for c in final.columns if 'Habitat' in c]
        agri_feats = [c for c in final.columns if 'Agri' in c]
        
        print(f"\n  Feature Categories:")
        print(f"    - Occupancy features: {len(occupancy_feats)}")
        print(f"    - Butterfly features: {len(butterfly_feats)}")
        print(f"    - Habitat features: {len(habitat_feats)}")
        print(f"    - Agriculture features: {len(agri_feats)}")
        
        print(f"\n  Data Quality Distribution:")
        print(f"    {final['Data_Quality'].value_counts().to_dict()}")
        
        # Key statistics
        print(f"\n  Key Statistics:")
        main_cols = ['Occupancy', 'Butterfly_Abundance', 'Habitat_Connectivity', 'Agri_Scheme_Area']
        for col in main_cols:
            if col in final.columns:
                print(f"    {col:25s}: μ={final[col].mean():7.2f}, σ={final[col].std():6.2f}, "
                      f"min={final[col].min():7.2f}, max={final[col].max():7.2f}")
        
        # Step 6: Save output
        print(f"\n[STEP 6] Saving Preprocessed Dataset")
        print("-" * 80)
        output_path = Path(self.data_dir) / output_file
        final.to_csv(output_path, index=False)
        print(f"  ✓ Saved: {output_path}")
        
        print("\n" + "=" * 80)
        print(" PREPROCESSING COMPLETE")
        print("=" * 80 + "\n")
        
        return final


def main():
    """Main execution."""
    pipeline = AutoPreprocessingPipeline(data_dir='/workspaces/ML_project_pollinating_insects')
    preprocessed_df = pipeline.run()
    
    # Display sample
    print("\nSample of Preprocessed Data (First 5 Rows):")
    print("-" * 80)
    print(preprocessed_df.head())
    
    print("\n\nColumn Reference:")
    print("-" * 80)
    for i, col in enumerate(preprocessed_df.columns, 1):
        dtype = str(preprocessed_df[col].dtype)
        print(f"  {i:2d}. {col:30s} ({dtype:10s})")


if __name__ == '__main__':
    main()
