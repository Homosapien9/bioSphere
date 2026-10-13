"""
Held-out evaluation with proper train/validation/test separation.

Changes:
- Three-way split: train (60%), validation (20%), test (20%)
- Model selection using validation set only
- Test set used only for final reporting
- Proper cold-start evaluation
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import config
from src import models as M
from src import uncertainty as U
from src.features import TARGET, usable_for_training

# Constants
SPLIT_RATIOS = {
    "train": 0.6,
    "validation": 0.2,
    "test": 0.2
}

def create_three_way_split(frame: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test split with patient-level separation.
    
    Args:
        frame: Feature frame with patient_id column
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train, validation, test) DataFrames
    """
    # Get unique patients
    patients = frame["patient_id"].unique()
    n_patients = len(patients)
    
    # Calculate split sizes
    n_train = int(SPLIT_RATIOS["train"] * n_patients)
    n_val = int(SPLIT_RATIOS["validation"] * n_patients)
    n_test = n_patients - n_train - n_val
    
    # Random shuffle with seed
    rng = np.random.RandomState(random_state)
    shuffled_patients = rng.permutation(patients)
    
    # Split patients
    train_patients = shuffled_patients[:n_train]
    val_patients = shuffled_patients[n_train:n_train+n_val]
    test_patients = shuffled_patients[n_train+n_val:]
    
    # Create masks
    train_mask = frame["patient_id"].isin(train_patients)
    val_mask = frame["patient_id"].isin(val_patients)
    test_mask = frame["patient_id"].isin(test_patients)
    
    return frame[train_mask], frame[val_mask], frame[test_mask]

def evaluate_cold_start(model_artifact, all_data: pd.DataFrame) -> dict:
    """
    Evaluate model performance on completely new users.
    
    Args:
        model_artifact: Fitted model artifact
        all_data: Complete feature frame
        
    Returns:
        Dictionary with cold-start metrics
    """
    # Identify patients with minimal history
    patient_history = all_data.groupby("patient_id").size()
    new_patients = patient_history[patient_history < 3].index
    
    if len(new_patients) == 0:
        return {
            "status": "no_new_users", 
            "n_patients": 0,
            "message": "No patients with minimal history found"
        }
    
    # Filter data for new patients
    new_data = all_data[all_data["patient_id"].isin(new_patients)]
    
    # Make predictions using wearable-only features
    wearable_features = [f"{col}_lag1" for col in config.WEARABLE_COLUMNS]
    predictions = M.predict(model_artifact, new_data[wearable_features])
    actual = new_data[TARGET].values
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))
    
    return {
        "status": "evaluated",
        "n_patients": len(new_patients),
        "n_samples": len(new_data),
        "mae": mae,
        "rmse": rmse,
        "patients": list(new_patients)
    }

def run_proper_evaluation(feature_frame: pd.DataFrame, random_state: int = 42) -> dict:
    """
    Evaluation with proper separation of concerns.
    
    Args:
        feature_frame: Complete feature frame
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with evaluation results
    """
    # Create three-way split
    train, val, test = create_three_way_split(feature_frame, random_state)
    
    # Train all models on training set
    artifacts = {}
    
    # Baselines
    for kind in ["persistence", "personal_mean"]:
        artifacts[kind] = M.fit_baseline(kind, train)
    
    # ML models with different configurations
    for config_name in config.CONFIGURATIONS:
        artifacts[f"ridge_{config_name}"] = M.fit_sklearn(
            "ridge", config_name, train
        )
    
    # Optional: HGBR comparison
    try:
        artifacts["hgbr_with_clinical"] = M.fit_sklearn(
            "hgbr", "with_clinical", train
        )
    except Exception as e:
        print(f"HGBR failed to train: {e}")
    
    # Model selection using validation set
    val_scores = {}
    for name, artifact in artifacts.items():
        val_pred = M.predict(artifact, val)
        val_scores[name] = {
            "mae": np.mean(np.abs(val_pred - val[TARGET].values)),
            "rmse": np.sqrt(np.mean((val_pred - val[TARGET].values) ** 2))
        }
    
    # Select best model based on validation MAE
    best_model_name = min(val_scores, key=lambda x: val_scores[x]["mae"])
    best_model = artifacts[best_model_name]
    
    # Final evaluation on test set
    test_pred = M.predict(best_model, test)
    test_metrics = {
        "mae": np.mean(np.abs(test_pred - test[TARGET].values)),
        "rmse": np.sqrt(np.mean((test_pred - test[TARGET].values) ** 2)),
        "n_samples": len(test)
    }
    
    # Cold-start evaluation
    cold_start_results = evaluate_cold_start(best_model, feature_frame)
    
    return {
        "best_model": best_model_name,
        "validation_scores": val_scores,
        "test_metrics": test_metrics,
        "cold_start": cold_start_results,
        "split_info": {
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test)
        }
    }
