"""
Leakage-safe chronological feature engineering.

Changes:
- Fixed rating_expanding_mean to exclude current day's rating
- Added separate feature sets for different user scenarios
- Improved documentation
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
        "rating_expanding_mean",    # Expanding mean of past ratings (fixed)
        "rating_dev_lag1",          # Deviation from personal mean
        "rating_lag1_missing",      # Missing indicator
    ]

def get_feature_names_for_scenario(scenario: str) -> list[str]:
    """
    Get appropriate features for different user scenarios.
    
    Args:
        scenario: One of 'new_user_no_history', 'new_user_limited_history', 'existing_user'
        
    Returns:
        List of feature names for the scenario
    """
    if scenario == "new_user_no_history":
        return wearable_feature_names()
    elif scenario == "new_user_limited_history":
        return wearable_feature_names() + [
            "rating_lag1", "rating_lag1_missing"
        ]
    else:  # existing_user
        return wearable_feature_names() + personal_history_feature_names()

def _features_for_patient(g: pd.DataFrame, with_baseline: bool) -> pd.DataFrame:
    """
    Build features for a single patient with leakage prevention.
    
    Args:
        g: Patient's data sorted by date
        with_baseline: Include personal history features
        
    Returns:
        DataFrame with engineered features
    """
    g = g.sort_values("date").copy()
    
    # Wearable features (work for all users)
    for col in config.WEARABLE_COLUMNS:
        s = g[col]
        lag1 = s.shift(1)  # Yesterday's value
        lag2 = s.shift(2)  # Day before yesterday
        
        g[f"{col}_lag1"] = lag1
        g[f"{col}_lag2"] = lag2
        g[f"{col}_lag1_missing"] = lag1.isna().astype(float)
    
    # Personal history features (only if with_baseline=True)
    if with_baseline and TARGET in g.columns:
        r = g[TARGET]
        
        # FIXED: Exclude current day's rating from expanding mean
        # Use shift(1) to only include past ratings
        past_ratings = r.shift(1)
        
        g["rating_lag1"] = past_ratings
        g["rating_lag2"] = r.shift(2)
        
        # Expanding mean of past ratings only (not including current)
        g["rating_expanding_mean"] = past_ratings.expanding(min_periods=1).mean()
        
        # Deviation from personal mean
        prior_mean = r.shift(2).expanding(min_periods=1).mean()
        g["rating_dev_lag1"] = past_ratings - prior_mean
        
        g["rating_lag1_missing"] = past_ratings.isna().astype(float)
    
    return g

def build_feature_frame(daily: pd.DataFrame, clinical: pd.DataFrame | None = None,
                        with_baseline: bool = True) -> pd.DataFrame:
    """
    Build the model-ready feature frame with leakage prevention.
    
    Args:
        daily: Merged wearable+diary frame
        clinical: Optional clinical features
        with_baseline: Include personal history features
        
    Returns:
        DataFrame with engineered features
    """
    parts = []
    for pid, g in daily.groupby("patient_id", sort=True):
        parts.append(_features_for_patient(g, with_baseline))
    
    frame = pd.concat(parts, ignore_index=True)
    
    # Merge clinical features if provided
    if clinical is not None:
        keep = ["patient_id"] + [c for c in CLINICAL_FEATURES if c in clinical.columns]
        frame = frame.merge(clinical[keep], on="patient_id", how="left")
    
    return frame

def usable_for_training(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to rows with known target and sufficient history.
    
    Args:
        frame: Feature frame
        
    Returns:
        Filtered DataFrame for training
    """
    ok = frame[TARGET].notna()
    
    # For existing users, require at least one prior rating
    if "rating_lag1" in frame.columns:
        ok &= frame["rating_lag1"].notna()
    
    return frame[ok].reset_index(drop=True)
    
