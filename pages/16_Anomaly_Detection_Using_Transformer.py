import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import tempfile
from pathlib import Path
import pandas as pd
import tabpfn_client

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import streamlit as st

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMjJhMDA0ZTctYWVmOS00MWZkLWExYTAtNzczZTgzYTFjNWU4IiwiZXhwIjoxNzkwNTYxOTQyfQ.FkeXhyUZTRsqM3vr-cUa9Etq6UmIOmBPQ48NSfyNF_k"

def setup_tabpfn_for_cloud():
    """Setup TabPFN to work in cloud environments by patching file operations"""
    
    # Create a temporary directory for caching
    temp_cache_dir = tempfile.mkdtemp(prefix='tabpfn_')
    
    # Set environment variables
    os.environ['TABPFN_ACCESS_TOKEN'] = token
    os.environ['TABPFN_CACHE_DIR'] = temp_cache_dir
    
    # Patch the problematic methods
    try:
        import tabpfn_client.client as client_module
        
        # Store original methods
        original_makedirs = os.makedirs
        original_mkdir = os.mkdir
        
        def safe_makedirs(name, mode=0o777, exist_ok=False):
            """Safe version of makedirs that handles permission errors"""
            try:
                return original_makedirs(name, mode, exist_ok)
            except (PermissionError, OSError):
                # If we can't create the directory, just continue
                return None
        
        def safe_mkdir(name, mode=0o777):
            """Safe version of mkdir that handles permission errors"""
            try:
                return original_mkdir(name, mode)
            except (PermissionError, OSError):
                # If we can't create the directory, just continue
                return None
        
        # Apply patches
        os.makedirs = safe_makedirs
        os.mkdir = safe_mkdir
        
        # Patch the dataset cache manager
        if hasattr(client_module, 'DatasetUIDCacheManager'):
            original_save_cache = client_module.DatasetUIDCacheManager.save_cache
            
            def safe_save_cache(self):
                """Safe version of save_cache that doesn't create directories"""
                try:
                    return original_save_cache(self)
                except (PermissionError, OSError):
                    # If we can't save cache, just continue without caching
                    return None
            
            client_module.DatasetUIDCacheManager.save_cache = safe_save_cache
        
        st.success("✅ TabPFN patched for cloud environment")
        return True
        
    except Exception as e:
        st.warning(f"⚠️ Patching failed: {str(e)}")
        return False



# Add a new section for F1 Race Telematics
st.divider()
st.subheader("Race Telematics Data Imputation using TabPFN")

# Initialize session state
if 'telematics_data_generated' not in st.session_state:
    st.session_state.telematics_data_generated = False
if 'original_telematics_df' not in st.session_state:
    st.session_state.original_telematics_df = None
if 'incomplete_telematics_df' not in st.session_state:
    st.session_state.incomplete_telematics_df = None
if 'missing_indices' not in st.session_state:
    st.session_state.missing_indices = None

if st.button('Generate F1 Telematics Data'):
    # Create realistic F1 race telematics data
    np.random.seed(42)
    
    # Generate 100 rows of race data
    n_rows = 100
    
    # Time column (seconds into race)
    time = np.arange(0, n_rows * 0.1, 0.1)  # Every 0.1 seconds
    
    # Turbocharger speed (RPM) - varies with throttle and gear
    base_turbo = 80000
    turbo_variation = np.sin(time * 0.5) * 10000 + np.random.normal(0, 2000, n_rows)
    turbocharger_speed = np.clip(base_turbo + turbo_variation, 60000, 120000)
    
    # Engine speed (RPM) - correlates with turbo but with different pattern
    base_engine = 12000
    engine_variation = np.sin(time * 0.3) * 2000 + np.random.normal(0, 500, n_rows)
    engine_speed = np.clip(base_engine + engine_variation, 8000, 15000)
    
    # Create DataFrame
    telematics_df = pd.DataFrame({
        'Time (s)': time,
        'Turbocharger Speed (RPM)': turbocharger_speed,
        'Engine Speed (RPM)': engine_speed
    })
    
    # Round to realistic precision
    telematics_df['Time (s)'] = telematics_df['Time (s)'].round(1)
    telematics_df['Turbocharger Speed (RPM)'] = telematics_df['Turbocharger Speed (RPM)'].round(0)
    telematics_df['Engine Speed (RPM)'] = telematics_df['Engine Speed (RPM)'].round(0)
    
    # Remove some random rows to create missing data
    missing_indices = np.random.choice(n_rows, size=15, replace=False)
    incomplete_df = telematics_df.copy()
    incomplete_df.loc[missing_indices, 'Turbocharger Speed (RPM)'] = np.nan
    incomplete_df.loc[missing_indices, 'Engine Speed (RPM)'] = np.nan
    
    # Store in session state
    st.session_state.telematics_data_generated = True
    st.session_state.original_telematics_df = telematics_df
    st.session_state.incomplete_telematics_df = incomplete_df
    st.session_state.missing_indices = missing_indices
    
    st.success("✅ F1 Telematics data generated successfully!")

# Display data if it exists
if st.session_state.telematics_data_generated:
    st.write("**F1 Telematics Data Overview:**")
    
    # Create a comprehensive table showing all data
    display_df = st.session_state.original_telematics_df.copy()
    
    # Add missing data columns
    display_df['Turbo Missing'] = st.session_state.incomplete_telematics_df['Turbocharger Speed (RPM)']
    display_df['Engine Missing'] = st.session_state.incomplete_telematics_df['Engine Speed (RPM)']
    
    # Show the comprehensive table
    st.dataframe(display_df.head(15), use_container_width=True)
    
    st.info(f"📊 **Legend:** Original columns show complete data, Missing columns show NaN where data was removed ({len(st.session_state.missing_indices)} rows)")

# Separate button for filling missing data
if st.session_state.telematics_data_generated and st.button('Fill Missing Data with TabPFN'):
    try:
        # Setup TabPFN for cloud environment
        if not setup_tabpfn_for_cloud():
            st.error("❌ Failed to setup TabPFN for cloud environment")
            st.stop()
        
        # Get data from session state
        incomplete_df = st.session_state.incomplete_telematics_df
        missing_indices = st.session_state.missing_indices
        original_df = st.session_state.original_telematics_df
        
        # Prepare data for TabPFN
        # Get complete data (where both turbo and engine speeds are available)
        complete_mask = ~(incomplete_df['Turbocharger Speed (RPM)'].isna() | 
                         incomplete_df['Engine Speed (RPM)'].isna())
        
        # Training data - only complete rows
        X_train = incomplete_df.loc[complete_mask, ['Time (s)']]
        y_turbo_train = incomplete_df.loc[complete_mask, 'Turbocharger Speed (RPM)']
        y_engine_train = incomplete_df.loc[complete_mask, 'Engine Speed (RPM)']
        
        # Test data (missing rows)
        X_test = incomplete_df.loc[missing_indices, ['Time (s)']]
        
        st.info(f"Training with {len(X_train)} complete samples, predicting {len(X_test)} missing values")
        
        # Train TabPFN models
        turbo_model = tabpfn_client.TabPFNRegressor()
        engine_model = tabpfn_client.TabPFNRegressor()
        
        # Fit models
        turbo_model.fit(X_train, y_turbo_train)
        engine_model.fit(X_train, y_engine_train)
        
        # Predict missing values
        turbo_predictions = turbo_model.predict(X_test)
        engine_predictions = engine_model.predict(X_test)
        
        # Fill in the missing data
        filled_df = incomplete_df.copy()
        filled_df.loc[missing_indices, 'Turbocharger Speed (RPM)'] = turbo_predictions
        filled_df.loc[missing_indices, 'Engine Speed (RPM)'] = engine_predictions
        
        # Round predictions
        filled_df['Turbocharger Speed (RPM)'] = filled_df['Turbocharger Speed (RPM)'].round(0)
        filled_df['Engine Speed (RPM)'] = filled_df['Engine Speed (RPM)'].round(0)
        
        st.success("✅ Missing data filled successfully with TabPFN!")
        
        # Create comprehensive comparison table
        comparison_df = original_df.copy()
        
        # Add missing data columns
        comparison_df['Turbo Missing'] = incomplete_df['Turbocharger Speed (RPM)']
        comparison_df['Engine Missing'] = incomplete_df['Engine Speed (RPM)']
        
        # Add predicted data columns
        comparison_df['Turbo Predicted'] = filled_df['Turbocharger Speed (RPM)']
        comparison_df['Engine Predicted'] = filled_df['Engine Speed (RPM)']
        
        # Calculate accuracy metrics
        original_turbo = original_df.loc[missing_indices, 'Turbocharger Speed (RPM)']
        predicted_turbo = filled_df.loc[missing_indices, 'Turbocharger Speed (RPM)']
        
        original_engine = original_df.loc[missing_indices, 'Engine Speed (RPM)']
        predicted_engine = filled_df.loc[missing_indices, 'Engine Speed (RPM)']
        
        turbo_mae = np.mean(np.abs(original_turbo - predicted_turbo))
        engine_mae = np.mean(np.abs(original_engine - predicted_engine))
        
        # Show accuracy metrics
        st.write("**Prediction Accuracy:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Turbocharger MAE", f"{turbo_mae:.0f} RPM")
        with col2:
            st.metric("Engine Speed MAE", f"{engine_mae:.0f} RPM")
        
        # Show comprehensive table
        st.write("**Complete Comparison Table:**")
        st.dataframe(comparison_df.head(20), use_container_width=True)
        
        # Show only the missing/predicted rows for detailed comparison
        st.write("**Detailed View - Missing Rows Only:**")
        detailed_df = comparison_df.loc[missing_indices].copy()
        detailed_df['Turbo Error'] = np.abs(detailed_df['Turbocharger Speed (RPM)'] - detailed_df['Turbo Predicted'])
        detailed_df['Engine Error'] = np.abs(detailed_df['Engine Speed (RPM)'] - detailed_df['Engine Predicted'])
        
        st.dataframe(detailed_df, use_container_width=True)
        
        st.info("📊 **Legend:** Original = Ground Truth, Missing = NaN values, Predicted = TabPFN predictions, Error = Absolute difference")
        
        # Create comparison plots
        st.write("**📈 Visual Comparison Plots:**")
        
        # Create subplots for comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Turbocharger Speed - Time Series
        axes[0, 0].plot(comparison_df['Time (s)'], comparison_df['Turbocharger Speed (RPM)'], 
                       'b-', label='Original', linewidth=2, alpha=0.7)
        axes[0, 0].scatter(comparison_df.loc[missing_indices, 'Time (s)'], 
                          comparison_df.loc[missing_indices, 'Turbo Predicted'], 
                          color='red', s=50, label='TabPFN Predictions', zorder=5)
        axes[0, 0].set_title('Turbocharger Speed: Original vs Predicted')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Turbocharger Speed (RPM)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Engine Speed - Time Series
        axes[0, 1].plot(comparison_df['Time (s)'], comparison_df['Engine Speed (RPM)'], 
                       'g-', label='Original', linewidth=2, alpha=0.7)
        axes[0, 1].scatter(comparison_df.loc[missing_indices, 'Time (s)'], 
                          comparison_df.loc[missing_indices, 'Engine Predicted'], 
                          color='orange', s=50, label='TabPFN Predictions', zorder=5)
        axes[0, 1].set_title('Engine Speed: Original vs Predicted')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Engine Speed (RPM)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Turbocharger - Scatter Plot (Predicted vs Actual)
        axes[1, 0].scatter(original_turbo, predicted_turbo, alpha=0.7, s=60)
        axes[1, 0].plot([original_turbo.min(), original_turbo.max()], 
                       [original_turbo.min(), original_turbo.max()], 
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 0].set_title(f'Turbocharger: Predicted vs Actual\nMAE = {turbo_mae:.0f} RPM')
        axes[1, 0].set_xlabel('Actual Turbocharger Speed (RPM)')
        axes[1, 0].set_ylabel('Predicted Turbocharger Speed (RPM)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Engine Speed - Scatter Plot (Predicted vs Actual)
        axes[1, 1].scatter(original_engine, predicted_engine, alpha=0.7, s=60, color='orange')
        axes[1, 1].plot([original_engine.min(), original_engine.max()], 
                       [original_engine.min(), original_engine.max()], 
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 1].set_title(f'Engine Speed: Predicted vs Actual\nMAE = {engine_mae:.0f} RPM')
        axes[1, 1].set_xlabel('Actual Engine Speed (RPM)')
        axes[1, 1].set_ylabel('Predicted Engine Speed (RPM)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additional error analysis plot
        st.write("**📊 Error Analysis:**")
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        
        # Error distribution for Turbocharger
        turbo_errors = np.abs(original_turbo - predicted_turbo)
        axes2[0].hist(turbo_errors, bins=8, alpha=0.7, color='blue', edgecolor='black')
        axes2[0].axvline(turbo_mae, color='red', linestyle='--', linewidth=2, label=f'Mean Error: {turbo_mae:.0f} RPM')
        axes2[0].set_title('Turbocharger Prediction Errors')
        axes2[0].set_xlabel('Absolute Error (RPM)')
        axes2[0].set_ylabel('Frequency')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)
        
        # Error distribution for Engine Speed
        engine_errors = np.abs(original_engine - predicted_engine)
        axes2[1].hist(engine_errors, bins=8, alpha=0.7, color='green', edgecolor='black')
        axes2[1].axvline(engine_mae, color='red', linestyle='--', linewidth=2, label=f'Mean Error: {engine_mae:.0f} RPM')
        axes2[1].set_title('Engine Speed Prediction Errors')
        axes2[1].set_xlabel('Absolute Error (RPM)')
        axes2[1].set_ylabel('Frequency')
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Summary statistics
        st.write("**📈 Prediction Quality Summary:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Turbo MAE", f"{turbo_mae:.0f} RPM")
        with col2:
            st.metric("Engine MAE", f"{engine_mae:.0f} RPM")
        with col3:
            turbo_rmse = np.sqrt(np.mean((original_turbo - predicted_turbo)**2))
            st.metric("Turbo RMSE", f"{turbo_rmse:.0f} RPM")
        with col4:
            engine_rmse = np.sqrt(np.mean((original_engine - predicted_engine)**2))
            st.metric("Engine RMSE", f"{engine_rmse:.0f} RPM")
        
    except Exception as e:
        st.error(f"❌ TabPFN prediction failed: {str(e)}")
        st.info("Make sure TabPFN is properly initialized first.")
    