import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Import your custom modules
from data_loader import load_patient_data
from preprocess import extract_first_6_hours_features, impute_missing_data
from models import identify_subtypes, train_baseline_model, train_subtype_models, evaluate_models

def main():
   # Load from both training sets if you want more data
    dir_a = os.path.join('data', 'training_setA')
    dir_b = os.path.join('data', 'training_setB')
    
    print("--- STEP 1: Loading Patient Data ---")
    patient_dfs = load_patient_data(dir_a, max_files=250) + load_patient_data(dir_b, max_files=250)
    print(f"Loaded {len(patient_dfs)} total patient records successfully.")
    
    if len(patient_dfs) == 0:
        print("No data found! Please make sure your PhysioNet CSV/PSV files are placed in data/raw/")
        return

    print("\n--- STEP 2: Extracting First 6-Hour Features ---")
    summary_df = extract_first_6_hours_features(patient_dfs)
    
    print("Handling missing values (imputation)...")
    clean_summary_df = impute_missing_data(summary_df, strategy='mean')

    print("\n--- STEP 3: Clustering and Subtype Identification ---")
    # Let's try 3 clusters/subtypes as a starting point
    clustered_df, kmeans_model, silhouette_avg = identify_subtypes(clean_summary_df, n_clusters=3)
    print(f"Clustering complete. Silhouette Score: {silhouette_avg:.4f}")
    
    # Print distribution of patients across subtypes
    print("Patient count per subtype:")
    print(clustered_df['subtype'].value_counts())

    print("\n--- STEP 4: Splitting Data for Training and Testing ---")
    # Stratify by subtype so every group is represented in the test set
    train_data, test_data = train_test_split(
        clustered_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=clustered_df['subtype']
    )

    print("\n--- STEP 5: Training Baseline Model ---")
    baseline_model, feature_cols = train_baseline_model(train_data)
    print("Baseline Random Forest trained on all patients.")

    print("\n--- STEP 6: Training Subtype-Specific Models ---")
    subtype_models = train_subtype_models(train_data, feature_cols)
    print("Subtype-specific models trained.")

    print("\n--- STEP 7: Evaluating and Comparing Models ---")
    results = evaluate_models(test_data, baseline_model, subtype_models, feature_cols)
    
    print("\n================ RESULTS ================")
    print(f"Baseline Model AUC: {results['baseline_auc']:.4f}")
    print("-" * 42)
    for sub, auc in results['subtype_aucs'].items():
        print(f"Subtype {sub} Model AUC: {auc:.4f}")
    print("=========================================")

if __name__ == '__main__':
    main()