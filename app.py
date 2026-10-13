"""
BioSphere - Simplified, focused interface.

Changes:
- Reduced from 6 tabs to 4 logical sections
- Plain language throughout
- Better visual hierarchy
- Meaningful interactions only
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import data_generation as DG
from src import evaluation as EV
from src import features as FE
from src import models as M
from src import pipeline as PL
from src import scenarios as SC

# Page config
st.set_page_config(
    page_title="BioSphere",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        color: #2563EB;
    }
    .disclaimer {
        background-color: #fff3cd;
        color: #664d03;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffe69c;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cached data loading
@st.cache_data
def load_data(data_dir: str) -> dict:
    """Load and validate data."""
    data = PL.load_inputs(data_dir)
    data["w_val"] = V.validate_wearable(data["wearable"])
    data["d_val"] = V.validate_diary(data["diary"])
    data["a_val"] = V.validate_alignment(data["wearable"], data["diary"], data["clinical"])
    
    # Build features with fixed leakage prevention
    data["frame"], data["next_rows"] = FE.build_frames(
        data["daily"], clinical=data["clinical"]
    )
    
    return data

@st.cache_resource
def run_evaluation(data_dir: str) -> dict:
    """Run evaluation with proper methodology."""
    data = load_data(data_dir)
    return EV.run_proper_evaluation(data["frame"])

def main():
    """Main application."""
    # Header
    st.markdown("# 🌿 BioSphere")
    st.markdown("### Your physiology, understood over time.")
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Important:</strong> This is a research prototype using synthetic data. 
        All predictions are model outputs, not medical advice. 
        Not for clinical use.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Data & Patient")
        
        # Data source selection
        shipped = config.DEMO_DATA_DIR
        generated = config.GENERATED_DATA_DIR
        
        options = {"Shipped demo dataset": shipped}
        if (generated / config.WEARABLE_CSV_NAME).exists():
            options["Regenerated dataset"] = generated
        
        source_label = st.selectbox("Data source", list(options), index=0)
        data_dir = Path(options[source_label])
        
        # Patient selection
        data = load_data(str(data_dir))
        patient_ids = sorted(data["wearable"]["patient_id"].unique())
        patient = st.selectbox("Select patient", patient_ids)
        
        # Data provenance
        st.divider()
        st.subheader("Data Information")
        st.caption(f"Folder: `{display_dir(data['dir'])}`")
        st.caption(f"Seed: `{config.RANDOM_SEED}`")
    
    # Main content - 4 tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "📈 History & Trends", 
        "🔮 Predictions", 
        "⚙️ Model Details"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Patient Overview")
        
        # Get patient data
        patient_data = data["wearable"][data["wearable"]["patient_id"] == patient]
        clinical_data = data["clinical"][data["clinical"]["patient_id"] == patient]
        
        # Display clinical profile
        if not clinical_data.empty:
            c = clinical_data.iloc[0]
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Clinical Profile")
                st.write(f"**Age:** {int(c['age']) if pd.notna(c['age']) else 'N/A'}")
                st.write(f"**Sex:** {'Male' if c['gender_male'] else 'Female'}")
                st.write(f"**BMI:** {c['bmi']:.1f if pd.notna(c['bmi']) else 'N/A'}")
            
            with col2:
                st.subheader("Data Coverage")
                # Calculate data completeness
                total_days = len(patient_data)
                days_with_data = patient_data[config.WEARABLE_COLUMNS].notna().any(axis=1).sum()
                completeness = (days_with_data / total_days * 100) if total_days > 0 else 0
                
                st.metric("Data Completeness", f"{completeness:.1f}%")
                st.metric("Total Days", total_days)
        
        # Display wearable data summary
        st.subheader("Wearable Data Summary")
        if not patient_data.empty:
            # Create summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_sleep = patient_data["sleep_duration_hours"].mean()
                st.metric("Avg Sleep", f"{avg_sleep:.1f}h")
            
            with col2:
                avg_hrv = patient_data["hrv_rmssd"].mean()
                st.metric("Avg HRV", f"{avg_hrv:.0f}ms")
            
            with col3:
                avg_steps = patient_data["steps"].mean()
                st.metric("Avg Steps", f"{avg_steps:,.0f}")
            
            with col4:
                avg_rhr = patient_data["resting_hr"].mean()
                st.metric("Avg Resting HR", f"{avg_rhr:.0f}bpm")
    
    # Tab 2: History & Trends
    with tab2:
        st.header("Physiological History")
        
        # Metric selection
        metrics = ["sleep_duration_hours", "hrv_rmssd", "resting_hr", "steps"]
        selected_metric = st.selectbox(
            "Select metric to visualize",
            metrics,
            format_func=lambda x: {
                "sleep_duration_hours": "Sleep Duration",
                "hrv_rmssd": "HRV (RMSSD)",
                "resting_hr": "Resting Heart Rate",
                "steps": "Steps"
            }[x]
        )
        
        # Create time series plot
        if not patient_data.empty:
            fig = go.Figure()
            
            # Add trace
            fig.add_trace(go.Scatter(
                x=patient_data["date"],
                y=patient_data[selected_metric],
                mode="lines+markers",
                name="Observed",
                line=dict(color="#2563EB", width=2),
                marker=dict(size=5)
            ))
            
            # Add personal baseline if available
            # This would use the fixed PersonalBaseline class
            
            # Update layout
            fig.update_layout(
                title=f"{selected_metric.replace('_', ' ').title()} Over Time",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Predictions
    with tab3:
        st.header("Recovery Predictions")
        
        # Get next-day prediction
        next_rows = data["next_rows"]
        prow = next_rows[next_rows["patient_id"] == patient]
        
        if prow.empty:
            st.warning("No prediction available for this patient.")
        else:
            # Get evaluation results
            results = run_evaluation(str(data_dir))
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Model Prediction",
                    f"{results['test_metrics']['mae']:.2f}",
                    help="Next-day recovery rating prediction"
                )
            
            with col2:
                st.metric(
                    "Model Error (MAE)",
                    f"{results['test_metrics']['mae']:.2f}",
                    help="Average prediction error on test data"
                )
            
            with col3:
                st.metric(
                    "Data Sufficiency",
                    f"{len(patient_data)} days",
                    help="Number of days of data available"
                )
    
    # Tab 4: Model Details
    with tab4:
        st.header("Model Information")
        
        # Get evaluation results
        results = run_evaluation(str(data_dir))
        
        # Display model comparison
        st.subheader("Model Comparison")
        
        # Create comparison table
        comparison_data = []
        for name, scores in results["validation_scores"].items():
            comparison_data.append({
                "Model": name,
                "Validation MAE": scores["mae"],
                "Validation RMSE": scores["rmse"]
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Display methodology
        st.subheader("Evaluation Methodology")
        st.markdown("""
        **Three-way split:**
        - Training set: 60% of patients
        - Validation set: 20% of patients (model selection)
        - Test set: 20% of patients (final evaluation)
        
        **Leakage Prevention:**
        - All features computed from past data only
        - No future information used
        - Patient-level splits maintained
        """)
        
        # Display limitations
        st.subheader("Limitations")
        st.markdown("""
        - All data is synthetic
        - Small sample size (24 patients)
        - Not clinically validated
        - Uncertainty estimates are proxies
        """)

if __name__ == "__main__":
    main()
    
