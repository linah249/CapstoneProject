"""
create_raw_scaler.py (FINAL MEMORY-SAFE FIX)
Loads CSV data file-by-file and subject-by-subject, fitting the StandardScaler
incrementally using partial_fit to avoid large memory allocations.
"""
import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
# NOTE: We assume the data path structure based on your existing files:
DATA_DIR = r"C:\Users\linah\Desktop\Final Year Project Linah\sEMG dataset\Basic Movements" # <--- Must be manually verified
MODELS_DIR = r"C:\Users\linah\Desktop\Final Year Project Linah\results\ann_advanced\models"
TRAIN_SUBJECTS = list(range(1, 9))
BATCH_SIZE = 5000

def create_and_save_raw_scaler():
    
    scaler_raw = StandardScaler()
    total_samples = 0
    
    print(f"[RAW SCALER] Starting final memory-safe fitting process with BATCH_SIZE={BATCH_SIZE}...")
    
    # 1. Iterate through all training subjects
    for sid in TRAIN_SUBJECTS:
        
        # 2. Find all relevant CSV files for this subject
        sbj_pattern = f"Nearlab_sbj{sid}_WL512_S128*.csv"
        search_pattern = os.path.join(DATA_DIR, sbj_pattern)
        subject_files = glob.glob(search_pattern)
        
        if not subject_files:
             print(f"[WARNING] No files found for Subject {sid} at {search_pattern}. Skipping.")
             continue
             
        print(f"[RAW SCALER] Loading and fitting Subject {sid} (found {len(subject_files)} files)...")

        subject_samples = 0
        
        # 3. Process each file (r1, r2, r3, etc.) one at a time
        for file_path in subject_files:
            try:
                # Read the file directly into a pandas DataFrame
                df = pd.read_csv(file_path, header=None)
                
                # Get only the EMG columns (first 5120 columns)
                X_flat = df.iloc[:, :5120].values
                N_file = X_flat.shape[0]
                
                # 4. Use partial_fit to process the file's data in smaller batches
                for i in range(0, N_file, BATCH_SIZE):
                    batch_data = X_flat[i:i + BATCH_SIZE]
                    # partial_fit works incrementally, avoiding large array creation
                    scaler_raw.partial_fit(batch_data)
                
                subject_samples += N_file
                
                # Explicitly delete the DataFrame and data to free memory
                del df, X_flat
                
            except Exception as e:
                print(f"[ERROR] Failed to process file {os.path.basename(file_path)}: {e}")
                
        total_samples += subject_samples
        print(f"  Processed {subject_samples} total samples from Subject {sid}.")
        
    print(f"[RAW SCALER] Final fit complete.")
    print(f"[RAW SCALER] Total flat samples processed for fitting: {total_samples}, Dim: 5120")

    # 5. Save Scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler_raw_emg.joblib")
    joblib.dump(scaler_raw, scaler_path)
    print(f"[RAW SCALER] Saved raw EMG scaler to: {scaler_path}")
    
    return scaler_path

if __name__ == "__main__":
    # --- IMPORTANT: Ensure DATA_DIR is correct for file listing ---
    if not os.path.exists(DATA_DIR):
        print(f"\nFATAL ERROR: Data directory {DATA_DIR} not found. Please verify the path.")
    else:
        os.makedirs(MODELS_DIR, exist_ok=True)
        create_and_save_raw_scaler()