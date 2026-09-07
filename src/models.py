import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def identify_subtypes(summary_df: pd.DataFrame, n_clusters: int = 3) -> tuple:
    """
    Applies StandardScaler and K-Means clustering to identify patient subtypes.
    """
    # Drop target and ID columns for clustering features
    feature_cols = summary_df.select_dtypes(include=[np.number]).columns.drop(['SepsisLabel', 'patient_id'], errors='ignore')
    X = summary_df[feature_cols]
    
    # --- ADD SCALING HERE ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, cluster_labels)
    
    df_clustered = summary_df.copy()
    df_clustered['subtype'] = cluster_labels
    
    return df_clustered, kmeans, score

def train_baseline_model(train_df: pd.DataFrame):
    """
    Trains a baseline Random Forest model on the entire patient population.
    """
    feature_cols = train_df.select_dtypes(include=[np.number]).columns.drop(['SepsisLabel', 'patient_id', 'subtype'], errors='ignore')
    X = train_df[feature_cols]
    y = train_df['SepsisLabel']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, feature_cols

def train_subtype_models(train_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Trains specialized Random Forest models for each identified patient subtype.
    """
    subtype_models = {}
    subtypes = train_df['subtype'].unique()
    
    for sub in subtypes:
        sub_data = train_df[train_df['subtype'] == sub]
        
        # Lowered thresholds so smaller clusters can train during testing
        if len(sub_data) < 30 or sub_data['SepsisLabel'].sum() < 3:
            print(f"Warning: Subtype {sub} has low prevalence ({len(sub_data)} patients). Marking as unreliable.")
            subtype_models[sub] = None
            continue
            
        X_sub = sub_data[feature_cols]
        y_sub = sub_data['SepsisLabel']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_sub, y_sub)
        subtype_models[sub] = model
        
    return subtype_models

def evaluate_models(test_df: pd.DataFrame, baseline_model, subtype_models, feature_cols) -> dict:
    """
    Evaluates baseline vs subtype-specific models using AUC scores.
    """
    X_test = test_df[feature_cols]
    y_test = test_df['SepsisLabel']
    
    # Baseline AUC (evaluated on the entire test set)
    baseline_preds = baseline_model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_preds)
    
    results = {'baseline_auc': baseline_auc, 'subtype_aucs': {}}
    
    # Subtype-specific evaluation
    for sub, model in subtype_models.items():
        sub_mask = test_df['subtype'] == sub
        if model is None or sub_mask.sum() == 0:
            print(f"Note: Subtype {sub} has no test samples or no model.")
            continue
            
        X_sub_test = test_df.loc[sub_mask, feature_cols]
        y_sub_test = test_df.loc[sub_mask, 'SepsisLabel']
        
        # Check if both classes (0 and 1) exist in this subtype's test set
        if len(y_sub_test.unique()) > 1:
            sub_preds = model.predict_proba(X_sub_test)[:, 1]
            results['subtype_aucs'][sub] = roc_auc_score(y_sub_test, sub_preds)
        else:
            print(f"Note: Subtype {sub} test set only contains one class (positive or negative cases only). Cannot compute AUC.")
            
    return results