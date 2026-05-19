"""
CyberSentinel AI - SOC Dashboard
Real-time security monitoring with UNSW-NB15 trained models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from core.real_data_loader import load_unsw_data, OptimizedPreprocessor
from models.ensemble_model import OptimizedEnsemble
from engine.evaluator import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title="CyberSentinel AI - SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-good { color: #00cc66; font-weight: bold; }
    .metric-warn { color: #ff8800; font-weight: bold; }
    .metric-bad { color: #ff4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_system():
    """Load data, train model, return all components"""
    
    with st.spinner("🔄 Loading UNSW-NB15 dataset..."):
        train_df, test_df = load_unsw_data()
    
    if train_df is None:
        st.error("❌ Dataset not found! Please download UNSW-NB15.")
        return None, None, None, None, None, None
    
    with st.spinner("⚙️ Preprocessing (encoding, scaling, feature selection)..."):
        preprocessor = OptimizedPreprocessor(n_features=75)
        X_train, X_test, y_train, y_test, selected_features = preprocessor.fit_transform(
            train_df, test_df
        )
    
    with st.spinner("🔵 Training Random Forest + XGBoost Ensemble..."):
        ensemble = OptimizedEnsemble(tune_hyperparams=True, n_jobs=-1)
        ensemble.fit(X_train, y_train)
    
    with st.spinner("📊 Evaluating model..."):
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)
        
        evaluator = ModelEvaluator()
        evaluator.evaluate(y_test, y_pred, y_proba)
    
    return train_df, test_df, preprocessor, ensemble, evaluator, selected_features


def plot_attack_distribution(train_df):
    """Plot attack type distribution"""
    if 'attack_cat' not in train_df.columns:
        return None
    
    attack_counts = train_df['attack_cat'].value_counts()
    
    fig = px.bar(
        x=attack_counts.index,
        y=attack_counts.values,
        title="Attack Type Distribution (Training Set)",
        labels={'x': 'Attack Type', 'y': 'Count'},
        color=attack_counts.values,
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=400)
    return fig


def plot_confusion_matrix(evaluator):
    """Plot confusion matrix as heatmap"""
    cm = evaluator.confusion_mat
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Normal', 'Predicted Attack'],
        y=['Actual Normal', 'Actual Attack'],
        text=[[f'TN: {cm[0][0]}', f'FP: {cm[0][1]}'],
              [f'FN: {cm[1][0]}', f'TP: {cm[1][1]}']],
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=False
    ))
    fig.update_layout(title="Confusion Matrix", height=400)
    return fig


def plot_metric_gauge(value, title, threshold_good=0.90, threshold_warn=0.80):
    """Plot a gauge chart for a single metric"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, threshold_warn * 100], 'color': "#ffcccc"},
                {'range': [threshold_warn * 100, threshold_good * 100], 'color': "#ffffcc"},
                {'range': [threshold_good * 100, 100], 'color': "#ccffcc"},
            ],
        }
    ))
    fig.update_layout(height=250)
    return fig


def main():
    st.markdown('<p class="main-header">🛡️ CyberSentinel AI - Security Operations Center</p>', 
                unsafe_allow_html=True)
    
    # Initialize system
    train_df, test_df, preprocessor, ensemble, evaluator, features = init_system()
    
    if train_df is None:
        st.stop()
    
    st.success("✅ System initialized! Model trained and evaluated on UNSW-NB15.")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Info")
        st.metric("Dataset", "UNSW-NB15")
        st.metric("Training Records", f"{len(train_df):,}")
        st.metric("Test Records", f"{len(test_df):,}")
        st.metric("Features Used", f"{len(features) if features is not None else 75}")
        
        st.divider()
        st.subheader("🎯 Model Architecture")
        st.markdown("""
        **Ensemble Model:**
        - Random Forest (tuned)
        - XGBoost (tuned)
        - Soft Voting Classifier
        
        **Preprocessing:**
        - Categorical encoding
        - Standard scaling
        - Mutual info feature selection (top 75)
        """)
        
        st.divider()
        st.subheader("ℹ️ About")
        st.markdown("""
        **CyberSentinel AI** detects network 
        intrusions using ML ensemble trained 
        on the UNSW-NB15 benchmark dataset.
        
        Detects 9 attack types:
        - Generic, Exploits, Fuzzers
        - DoS, Reconnaissance
        - Analysis, Backdoor
        - Shellcode, Worms
        """)
    
    # Main dashboard
    metrics = evaluator.get_resume_metrics()
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Accuracy",
            metrics['Accuracy'],
            delta=None
        )
    with col2:
        st.metric(
            "Precision",
            metrics['Precision'],
            delta=None
        )
    with col3:
        st.metric(
            "Recall",
            metrics['Recall'],
            delta=None
        )
    with col4:
        st.metric(
            "F1 Score",
            metrics['F1 Score'],
            delta=None
        )
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Performance", "🔥 Attack Analysis", "📋 Data Explorer", "📝 Resume Summary"
    ])
    
    with tab1:
        st.subheader("Model Performance Metrics")
        
        # Gauge charts
        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        
        m = evaluator.metrics
        with col_g1:
            st.plotly_chart(plot_metric_gauge(m['accuracy'], "Accuracy"), width='stretch')
        with col_g2:
            st.plotly_chart(plot_metric_gauge(m['precision'], "Precision"), width='stretch')
        with col_g3:
            st.plotly_chart(plot_metric_gauge(m['recall'], "Recall"), width='stretch')
        with col_g4:
            st.plotly_chart(plot_metric_gauge(m['f1'], "F1 Score"), width='stretch')
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix(evaluator), width='stretch')
        
        # Classification report
        st.subheader("Detailed Classification Report")
        st.text(evaluator.report)
    
    with tab2:
        st.subheader("Attack Type Distribution")
        
        # Attack distribution chart
        attack_fig = plot_attack_distribution(train_df)
        if attack_fig:
            st.plotly_chart(attack_fig, width='stretch')
        
        # Attack stats
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.subheader("Attack Categories")
            if 'attack_cat' in train_df.columns:
                attack_stats = train_df['attack_cat'].value_counts().reset_index()
                attack_stats.columns = ['Attack Type', 'Count']
                attack_stats['Percentage'] = (attack_stats['Count'] / len(train_df) * 100).round(1)
                st.dataframe(attack_stats, width='stretch')
        
        with col_a2:
            st.subheader("Detection Capabilities")
            cm = evaluator.confusion_mat
            detection_stats = pd.DataFrame({
                'Metric': ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
                'Count': [cm[0][0], cm[0][1], cm[1][0], cm[1][1]],
                'Description': [
                    'Normal correctly identified',
                    'Normal flagged as attack (false alarm)',
                    'Attack missed by system',
                    'Attack correctly detected'
                ]
            })
            st.dataframe(detection_stats, width='stretch')
    
    with tab3:
        st.subheader("Dataset Explorer")
        
        # Show sample data
        st.markdown("**Training Data Sample (first 100 rows)**")
        st.dataframe(train_df.head(100), width='stretch')
        
        # Feature information
        st.subheader("Selected Features (Top 15)")
        if features is not None:
            feature_df = pd.DataFrame({
                'Rank': range(1, min(16, len(features)+1)),
                'Feature': features[:15]
            })
            st.dataframe(feature_df, width='stretch')
        
        # Data statistics
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.subheader("Training Set")
            st.json({
                'Total Records': len(train_df),
                'Features': train_df.shape[1],
                'Normal Traffic': int(sum(train_df['label'] == 0)),
                'Attack Traffic': int(sum(train_df['label'] == 1)),
            })
        
        with col_d2:
            st.subheader("Test Set")
            st.json({
                'Total Records': len(test_df),
                'Features': test_df.shape[1],
                'Normal Traffic': int(sum(test_df['label'] == 0)),
                'Attack Traffic': int(sum(test_df['label'] == 1)),
            })
    
    with tab4:
        st.subheader("Resume-Ready Summary")
        
        st.markdown("""
        ### 🎯 Copy-Paste for Your Resume
    """)

    st.divider()

    st.subheader("📊 Key Metrics")
    st.json({
        'Dataset': 'UNSW-NB15 (public benchmark)',
        'Model': 'Random Forest + XGBoost Ensemble',
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1 Score': metrics['F1 Score'],
        'Features Used': 75,
        'Total Features': 194,
        'Attack Types Detected': 9
    })

    # Footer
    st.divider()
    st.markdown(
    "<center><small>CyberSentinel AI | Research-Grade Intrusion Detection System | UNSW-NB15 Benchmark</small></center>",
    unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()