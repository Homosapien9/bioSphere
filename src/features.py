# File: src/features.py (MODIFICATIONS)
"""
Leakage-safe chronological feature engineering.

Key changes:
- Explicit handling of new users (no history)
- Clear separation of personal history vs wearable-only features
- Proper documentation of feature dependencies
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config
from src.fhir_loader import CLINICAL_FEATURES

TARGET = "target"

def wearable_feature_names() -> list[str]:
    """Features that work for new users (no history required)."""
    names = []
    for col in config.WEARABLE_COLUMNS:
        names += [
            f"{col}_lag1",           # Yesterday's value
            f"{col}_lag2",           # Day before yesterday
            f"{col}_lag1_missing",   # Missing indicator
        ]
    return names

def personal_history_feature_names() -> list[str]:
    """Features that require personal history."""
    return [
        "rating_lag1",              # Yesterday's rating
        "rating_lag2",              # Day before yesterday's rating
        "rating_expanding_mean",    # Expanding mean of past ratings
        "rating_dev_lag1",          # Deviation from personal mean
        "rating_lag1_missing",      # Missing indicator
    ]

def get_feature_names_for_scenario(scenario: str) -> list[str]:
    """Get appropriate features for different user scenarios."""
    if scenario == "new_user_no_history":
        return wearable_feature_names()
    elif scenario == "new_user_limited_history":
        return wearable_feature_names() + [
            "rating_lag1", "rating_lag1_missing"
        ]
    else:  # existing_user
        return wearable_feature_names() + personal_history_feature_names()

def build_features_for_new_user(wearable_data: pd.DataFrame) -> pd.DataFrame:
    """Build features for a completely new user (no history)."""
    features = pd.DataFrame(index=wearable_data.index)
    
    for col in config.WEARABLE_COLUMNS:
        if col in wearable_data.columns:
            # Only lag features work for new users
            features[f"{col}_lag1"] = wearable_data[col].shift(1)
            features[f"{col}_lag2"] = wearable_data[col].shift(2)
            features[f"{col}_lag1_missing"] = wearable_data[col].shift(1).isna().astype(float)
    
    return features
