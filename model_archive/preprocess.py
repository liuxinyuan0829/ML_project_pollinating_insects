"""
Final Robust Preprocessing Pipeline for Pollinating Insects
===========================================================

Simple, effective preprocessing with flexible missing data handling.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')


class RobustPreprocessingPipeline:
    """Production-ready preprocessing pipeline."""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
    
    def extract_data_from_excel(self, filepath, sheet_name=0):
        """Fast, robust Excel data extraction."""
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
        except:
            return pd.DataFrame()
        
        # Find Year header
        header_row = None
        for i in range(min(20, len(df))):
            if 'Year' in str(df.iloc[i].values):
                header_row = i
                break
        
        if header_row is None:
            return pd.DataFrame()
        
        try:
            data = df.iloc[header_row+1:].copy()
            cols = [str(c) for c in df.iloc[header_row].values]
            data.columns = cols
            
            # Find Year column
            year_col = None
            for c in cols:
                if 'Year' in c:
                    year_col = c
                    break
            
            if year_col is None:
                return pd.DataFrame()
            
            # Keep Year + first 3 data columns
            keep_cols = [year_col] + [c for c in cols if c != year_col][:3]
            data = data[keep_cols]
            
            # Convert to numeric
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            return data.dropna(subset=[year_col]).sort_values(year_col).reset_index(drop=True)
        except:
            return pd.DataFrame()
    
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets with safe naming."""
        datasets = {}
        
        # Pollinating insects
        poll = self.extract_data_from_excel(
            Path(self.data_dir) / 'UK-BDI-2025-pollinating-insects.xlsx', '1'
        )
        if not poll.empty:
            cols = list(poll.columns)
            poll = poll.rename(columns={cols[i]: name for i, name in enumerate(['Year', 'Occupancy', 'Occ_CI_Min', 'Occ_CI_Max'][:len(cols)])})
            datasets['poll'] = poll
            print(f"  ✓ Pollinating Insects ({len(poll)} years)")
        
        # Habitat connectivity
        hab = self.extract_data_from_excel(
            Path(self.data_dir) / 'UK-BDI-2025-habitat-connectivity.xlsx', '1'
        )
        if not hab.empty:
            cols = list(hab.columns)
            hab = hab.rename(columns={cols[i]: name for i, name in enumerate(['Year', 'Habitat_Connect', 'Hab_CI_Min', 'Hab_CI_Max'][:len(cols)])})
            datasets['hab'] = hab
            print(f"  ✓ Habitat Connectivity ({len(hab)} years)")
        
        # Agriculture schemes
        agri = self.extract_data_from_excel(
            Path(self.data_dir) / 'UK-BDI-2025-agri-environment-schemes.xlsx', '1'
        )
        if not agri.empty:
            cols = list(agri.columns)
            agri = agri.rename(columns={cols[i]: name for i, name in enumerate(['Year', 'Agri_Scheme', 'Agri_CI_Min', 'Agri_CI_Max'][:len(cols)])})
            datasets['agri'] = agri
            print(f"  ✓ Agriculture Schemes ({len(agri)} years)")
        
        # Plants
        plants = self.extract_data_from_excel(
            Path(self.data_dir) / 'UK-BDI-2025-plants-wider-countryside_new.xlsx', '1'
        )
        if not plants.empty:
            cols = list(plants.columns)
            plants = plants.rename(columns={cols[i]: name for i, name in enumerate(['Year', 'Plant_Abund', 'Plant_CI_Min', 'Plant_CI_Max'][:len(cols)])})
            datasets['plants'] = plants
            print(f"  ✓ Plants ({len(plants)} years)")
        
        return datasets
    
    def merge_and_interpolate(self, datasets: Dict, period=(1992, 2024)):
        """Merge datasets with intelligent interpolation."""
        
        # Start with pollinating insects
        merged = datasets['poll'].copy()
        print(f"\n  Base: Pollinating insects ({len(merged)} rows)")
        
        # Merge others on Year
        for name in ['hab', 'agri', 'plants']:
            if name in datasets:
                df = datasets[name].copy()
                merged = merged.merge(df, on='Year', how='left')
                print(f"  Merged {name}: {len(df)} rows")
        
        # Filter to analysis period
        merged = merged[(merged['Year'] >= period[0]) & (merged['Year'] <= period[1])].reset_index(drop=True)
        
        # Clean numeric columns
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'Year']
        
        # Fill missing values: interpolate then mean
        for col in numeric_cols:
            merged[col] = merged[col].interpolate(method='linear', limit_direction='both')
            remaining_nan = merged[col].isnull().sum()
            if remaining_nan > 0:
                merged[col].fillna(merged[col].mean(), inplace=True)
        
        return merged
    
    def feature_engineering(self, df):
        """Create modeling features."""
        df = df.sort_values('Year').reset_index(drop=True)
        
        # Main feature columns
        feat_cols = [c for c in df.columns if c not in ['Year'] and 'CI' not in c]
        
        # Percentage change
        for col in feat_cols:
            df[f'{col}_pchg'] = df[col].pct_change() * 100
        
        # Differences (1-year lag)
        for col in feat_cols:
            df[f'{col}_diff'] = df[col].diff()
        
        # Simple moving averages
        for col in feat_cols:
            df[f'{col}_ma2'] = df[col].rolling(window=2).mean()
        
        # Fill NaNs introduced by features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)
        
        return df.reset_index(drop=True)
    
    def add_quality_score(self, df):
        """Add quality metrics."""
        df['Quality'] = 'Good'
        
        # Anomaly detection
        numeric_cols = [c for c in df.columns if c not in ['Year', 'Quality'] and 'CI' not in c]
        
        if len(numeric_cols) > 0 and len(df) > 3:
            try:
                X = df[numeric_cols].fillna(df[numeric_cols].mean()).values
                if X.shape[0] > 1:
                    iso = IsolationForest(contamination=min(0.15, 5.0/len(df)), random_state=42)
                    preds = iso.fit_predict(X)
                    df.loc[preds == -1, 'Quality'] = 'Anomaly'
            except:
                pass
        
        return df
    
    def run(self):
        """Execute pipeline."""
        print("\n" + "="*80)
        print(" POLLINATING INSECTS - DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        print("\n[1] Loading Datasets")
        print("-"*80)
        datasets = self.load_datasets()
        
        if not datasets:
            raise ValueError("No datasets loaded!")
        
        print("\n[2] Merging Datasets (1992-2024)")
        print("-"*80)
        merged = self.merge_and_interpolate(datasets)
        print(f"\n  Merged: {merged.shape[0]} samples × {merged.shape[1]} features")
        print(f"  Period: {merged['Year'].min():.0f}-{merged['Year'].max():.0f}")
        print(f"  Missing: {merged.isnull().sum().sum()}")
        
        print("\n[3] Feature Engineering")
        print("-"*80)
        engineered = self.feature_engineering(merged)
        print(f"  Features: {engineered.shape[1]} total ({engineered.shape[0]} samples after cleaning)")
        
        print("\n[4] Quality Assessment")
        print("-"*80)
        final = self.add_quality_score(engineered)
        qc = final['Quality'].value_counts()
        print(f"  Quality distribution: {dict(qc)}")
        
        print("\n[5] Output Summary")
        print("-"*80)
        print(f"\n  Final dataset: {final.shape[0]} × {final.shape[1]}")
        print(f"  Date range: {final['Year'].min():.0f} - {final['Year'].max():.0f}")
        
        main_feats = ['Occupancy', 'Habitat_Connect', 'Agri_Scheme', 'Plant_Abund']
        print(f"\n  Key Statistics:")
        for feat in main_feats:
            if feat in final.columns:
                print(f"    {feat:20s}: μ={final[feat].mean():.1f}, σ={final[feat].std():.1f}")
        
        # Save
        output_file = Path(self.data_dir) / 'data_preprocessed.csv'
        final.to_csv(output_file, index=False)
        print(f"\n  ✓ Saved: {output_file}")
        
        print("\n" + "="*80)
        print(" SUCCESS")
        print("="*80 + "\n")
        
        return final


if __name__ == '__main__':
    pipeline = RobustPreprocessingPipeline('/workspaces/ML_project_pollinating_insects')
    result = pipeline.run()
    
    print("\nData Preview (first 5 rows):")
    print(result.head())
    
    print("\n\nColumns:")
    for i, col in enumerate(result.columns, 1):
        print(f"  {i:2d}. {col}")
