import os
import pandas as pd

def load_patient_data(data_dir: str, max_files: int = None) -> list:
    """
    Loads raw PhysioNet 2019 patient data files from the specified directory
    with error handling for individual files.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    all_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.psv') or f.endswith('.csv')
    ])
    
    if max_files:
        all_files = all_files[:max_files]
        
    patient_dfs = []
    print(f"Attempting to load {len(all_files)} files from {data_dir}...")
    
    for i, file_path in enumerate(all_files):
        try:
            sep = '|' if file_path.endswith('.psv') else ','
            df = pd.read_csv(file_path, sep=sep)
            
            # Store filename/id as an attribute for tracking
            df.attrs['patient_id'] = os.path.basename(file_path)
            patient_dfs.append(df)
            
            # Print progress every 50 files so you know it's working
            if (i + 1) % 50 == 0:
                print(f"Loaded {i + 1}/{len(all_files)} files...")
                
        except Exception as e:
            print(f"Warning: Could not read {file_path}. Error: {e}")
            
    return patient_dfs