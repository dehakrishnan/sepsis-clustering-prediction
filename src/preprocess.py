import pandas as pd
import numpy as np

def extract_first_6_hours_features(patient_dfs: list) -> pd.DataFrame:
    """
    Calculates average, min, max of vital signs over the first 6 hours 
    and checks for sepsis label occurrence.
    """
    summaries = []
    
    for df in patient_dfs:
        # Slice first 6 hours
        df_6h = df.head(6)
        
        patient_id = df.attrs.get('patient_id', len(summaries))
        
        # Compute mean, min, max for numeric columns (excluding SepsisLabel if present)
        numeric_cols = df_6h.select_dtypes(include=[np.number]).columns.drop('SepsisLabel', errors='ignore')
        
        features = {}
        for col in numeric_cols:
            features[f'{col}_mean'] = df_6h[col].mean()
            features[f'{col}_min'] = df_6h[col].min()
            features[f'{col}_max'] = df_6h[col].max()
            features[f'{col}_missing_pct'] = df_6h[col].isna().mean() # Missingness feature
            
        # Target: Did sepsis happen within these hours or overall? 
        # (Depending on exact prediction window design, check if SepsisLabel == 1 anywhere in timeline)
        sepsis_target = 1 if (df['SepsisLabel'] == 1).any() else 0
        features['SepsisLabel'] = sepsis_target
        features['patient_id'] = patient_id
        
        summaries.append(features)
        
    summary_df = pd.DataFrame(summaries)
    return summary_df

def impute_missing_data(summary_df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handles missing values using imputation (mean or forward-fill equivalent).
    """
    df_imputed = summary_df.copy()
    
    if strategy == 'mean':
        # Fill numeric NaNs with column means
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(df_imputed[numeric_cols].mean())
        # Fallback for columns entirely NaN
        df_imputed = df_imputed.fillna(0)
    elif strategy == 'zero':
        df_imputed = df_imputed.fillna(0)
        
    return df_imputed