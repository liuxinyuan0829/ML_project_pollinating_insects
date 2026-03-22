"""
Corrected Preprocessing Pipeline for Pollinating Insects Dataset
==============================================================

Handles actual Excel file structure with proper parsing.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class PreprocessingPipeline:
    """Complete data preprocessing pipeline."""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
    
    def load_pollinating_insects(self) -> pd.DataFrame:
        """Load pollinating insects data from Excel."""
        df = pd.read_excel(
            Path(self.data_dir) / 'UK-BDI-2025-pollinating-insects.xlsx',
            sheet_name='1',
            header=None
        )
        # Data starts at row 3
        header = df.iloc[2].values
        data = df.iloc[3:].copy()
        data.columns = ['Year', 'Occupancy', 'CI_Min', 'CI_Max']
        data = data.dropna(subset=['Year'])
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
        data['Occupancy'] = pd.to_numeric(data['Occupancy'], errors='coerce')
        data['CI_Min'] = pd.to_numeric(data['CI_Min'], errors='coerce')
        data['CI_Max'] = pd.to_numeric(data['CI_Max'], errors='coerce')
        
        return data.dropna(subset=['Year', 'Occupancy']).sort_values('Year').reset_index(drop=True)
    
    def load_butterflies(self) -> pd.DataFrame:
        """Load butterfly abundance data."""
        df = pd.read_excel(
            Path(self.data_dir) / 'UK-BDI-2025-insects-wider-countryside.xlsx',
            sheet_name='1',
            header=None
        )
        # Find header row (usually row with "Year")
        header_row = None
        for i in range(min(10, len(df))):
            if 'Year' in str(df.iloc[i].values):
                header_row = i
                break
        
        if header_row is None:
            header_row = 2
        
        data = df.iloc[header_row+1:].copy()
        data.columns = df.iloc[header_row].values
        
        if 'Year' in data.columns:
            data = data[['Year', data.columns[1], data.columns[2], data.columns[3]]]
            data.columns = ['Year', 'Butterfly_Abundance', 'Butterfly_CI_Min', 'Butterfly_CI_Max']
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['Butterfly_Abundance'] = pd.to_numeric(data['Butterfly_Abundance'], errors='coerce')
            
            return data.dropna(subset=['Year', 'Butterfly_Abundance']).sort_values('Year').reset_index(drop=True)
        
        return pd.DataFrame()
    
    def load_habitat_connectivity(self) -> pd.DataFrame:
        """Load habitat connectivity data."""
        df = pd.read_excel(
            Path(self.data_dir) / 'UK-BDI-2025-habitat-connectivity.xlsx',
            sheet_name='1',
            header=None
        )
        
        # Find header row
        header_row = None
        for i in range(min(10, len(df))):
            if 'Year' in str(df.iloc[i].values):
                header_row = i
                break
        
        if header_row is None:
            header_row = 2
        
        data = df.iloc[header_row+1:].copy()
        data.columns = df.iloc[header_row].values
        
        if 'Year' in data.columns:
            data = data[['Year', data.columns[1], data.columns[2], data.columns[3]]]
            data.columns = ['Year', 'Habitat_Connectivity', 'Habitat_CI_Min', 'Habitat_CI_Max']
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['Habitat_Connectivity'] = pd.to_numeric(data['Habitat_Connectivity'], errors='coerce')
            
            return data.dropna(subset=['Year', 'Habitat_Connectivity']).sort_values('Year').reset_index(drop=True)
        
        return pd.DataFrame()
    
    def load_agri_schemes(self) -> pd.DataFrame:
        """Load agriculture environment schemes data."""
        df = pd.read_excel(
            Path(self.data_dir) / 'UK-BDI-2025-agri-environment-schemes.xlsx',
            sheet_name='1',
            header=None
        )
        
        # Find header row
        header_row = None
        for i in range(min(10, len(df))):
            if 'Year' in str(df.iloc[i].values):
                header_row = i
                break
        
        if header_row is None:
            header_row = 2
        
        data = df.iloc[header_row+1:].copy()
        data.columns = df.iloc[header_row].values
        
        if 'Year' in data.columns:
            data = data[['Year', data.columns[1], data.columns[2], data.columns[3]]]
            data.columns = ['Year', 'Agri_Scheme_Area', 'Agri_CI_Min', 'Agri_CI_Max']
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['Agri_Scheme_Area'] = pd.to_numeric(data['Agri_Scheme_Area'], errors='coerce')
            
            return data.dropna(subset=['Year', 'Agri_Scheme_Area']).sort_values('Year').reset_index(drop=True)
        
        return pd.DataFrame()
    
    def load_plants(self) -> pd.DataFrame:
        """Load plants data."""
        try:
            df = pd.read_excel(
                Path(self.data_dir) / 'UK-BDI-2025-plants-wider-countryside_new.xlsx',
                sheet_name='1',
                header=None
            )
            
            # Find header row
            header_row = None
            for i in range(min(10, len(df))):
                if 'Year' in str(df.iloc[i].values):
                    header_row = i
                    break
            
            if header_row is None:
                header_row = 2
            
            data = df.iloc[header_row+1:].copy()
            data.columns = df.iloc[header_row].values
            
            if 'Year' in data.columns:
                data = data[['Year', data.columns[1], data.columns[2], data.columns[3]]]
                data.columns = ['Year', 'Plant_Abundance', 'Plant_CI_Min', 'Plant_CI_Max']
                data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
                data['Plant_Abundance'] = pd.to_numeric(data['Plant_Abundance'], errors='coerce')
                
                return data.dropna(subset=['Year', 'Plant_Abundance']).sort_values('Year').reset_index(drop=True)
        
        except Exception as e:
            print(f"  Warning: Could not load plants data: {e}")
        
        return pd.DataFrame()
    
    def temporal_interpolation(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Interpolate missing years in time-series."""
        df = df.sort_values('Year').copy()
        
        year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
        all_years = pd.DataFrame({'Year': range(year_min, year_max + 1)})
        
        df = all_years.merge(df, on='Year', how='left')
        
        # Interpolate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].interpolate(method='cubic', limit_direction='both')
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def merge_all_datasets(self, datasets: Dict[str, pd.DataFrame], 
                          period: Tuple[int, int] = (1992, 2024)) -> pd.DataFrame:
        """Merge all datasets on Year."""
        
        # Start with pollinating insects
        merged = datasets['pollinating_insects'].copy()
        print(f"  Base: Pollinating insects ({len(merged)} years)")
        
        # Merge others
        for name in ['butterflies', 'habitat', 'agri', 'plants']:
            if name in datasets and not datasets[name].empty:
                df = datasets[name].copy()
                merged = merged.merge(df, on='Year', how='left')
                print(f"  + {name:15s}: {len(df)} years")
        
        # Filter to period
        merged = merged[(merged['Year'] >= period[0]) & (merged['Year'] <= period[1])].reset_index(drop=True)
        
        # Interpolate remaining NaNs
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'Year']
        
        for col in numeric_cols:
            if merged[col].isnull().sum() > 0:
                merged[col] = merged[col].interpolate(method='linear')
                merged[col] = merged[col].fillna(method='ffill').fillna(method='bfill')
        
        return merged
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""
        df = df.sort_values('Year').copy()
        
        # Identify main value columns
        value_cols = [c for c in df.columns if c not in ['Year'] and 'CI' not in c]
        
        # Year-over-year changes
        for col in value_cols:
            df[f'{col}_change'] = df[col].pct_change() * 100
        
        # Lag features (1, 2, 3 years)
        for col in value_cols:
            for lag in [1, 2, 3]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # Rolling averages
        for col in value_cols:
            df[f'{col}_MA3'] = df[col].rolling(window=3, center=True).mean()
        
        # Trend indicator
        for col in value_cols:
            df[f'{col}_trend'] = df[col].diff()
        
        # Drop rows with NaN from lagging
        df = df.dropna()
        
        return df
    
    def add_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add data quality indicators."""
        from sklearn.ensemble import IsolationForest
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['Year']]
        
        # Standardize for outlier detection
        scaler = StandardScaler()
        X = scaler.fit_transform(df[numeric_cols].fillna(df[numeric_cols].mean()))
        
        # Isolation Forest for outlier detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(X)
        
        df['Quality'] = 'Good'
        df.loc[predictions == -1, 'Quality'] = 'Anomaly'
        
        # Add confidence score based on CI width
        if 'CI_Max' in df.columns and 'CI_Min' in df.columns:
            df['CI_Width'] = (df['CI_Max'] - df['CI_Min']).abs()
            df['Confidence_Score'] = 1.0 / (1.0 + df['CI_Width'] / df['CI_Width'].max())
        else:
            df['Confidence_Score'] = 1.0
        
        return df
    
    def run(self, output_file: str = 'data_preprocessed.csv') -> pd.DataFrame:
        """Execute complete preprocessing pipeline."""
        
        print("\n" + "=" * 70)
        print("POLLINATING INSECTS - DATA PREPROCESSING PIPELINE")
        print("=" * 70)
        
        print("\n[1] Loading datasets...")
        datasets = {
            'pollinating_insects': self.load_pollinating_insects(),
            'butterflies': self.load_butterflies(),
            'habitat': self.load_habitat_connectivity(),
            'agri': self.load_agri_schemes(),
            'plants': self.load_plants(),
        }
        
        print("\n[2] Merging datasets...")
        merged = self.merge_all_datasets(datasets)
        print(f"\n  Merged dataset: {merged.shape[0]} years × {merged.shape[1]} features")
        print(f"  Period: {merged['Year'].min():.0f} - {merged['Year'].max():.0f}")
        print(f"  Missing values: {merged.isnull().sum().sum()}")
        
        print("\n[3] Creating features...")
        merged = self.create_features(merged)
        print(f"  Final features: {merged.shape[1]} columns")
        
        print("\n[4] Adding quality indicators...")
        merged = self.add_quality_scores(merged)
        
        print("\n[5] Summary Statistics")
        print(f"\n  Dataset shape: {merged.shape}")
        print(f"  Columns: {list(merged.columns)[:15]}...")
        print(f"\n  Data Quality Distribution:")
        print(merged['Quality'].value_counts().to_string())
        
        print(f"\n  Key Statistics:")
        for col in ['Occupancy', 'Butterfly_Abundance', 'Habitat_Connectivity']:
            if col in merged.columns:
                print(f"    {col:25s}: {merged[col].min():.2f} - {merged[col].max():.2f} (mean: {merged[col].mean():.2f})")
        
        print("\n[6] Saving preprocessed data...")
        output_path = Path(self.data_dir) / output_file
        merged.to_csv(output_path, index=False)
        print(f"  ✓ Saved to: {output_path}")
        
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE")
        print("=" * 70 + "\n")
        
        return merged


if __name__ == '__main__':
    pipeline = PreprocessingPipeline(data_dir='/workspaces/ML_project_pollinating_insects')
    preprocessed_data = pipeline.run()
    
    print("\nSample of preprocessed data (first 5 rows):")
    print(preprocessed_data.head())
    
    print("\nColumn information:")
    for i, col in enumerate(preprocessed_data.columns, 1):
        print(f"  {i:2d}. {col}")
