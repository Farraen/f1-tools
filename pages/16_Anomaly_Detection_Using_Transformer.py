import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import tempfile
from pathlib import Path

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

if st.button('Train model'):
    # Setup TabPFN for cloud environment
    if not setup_tabpfn_for_cloud():
        st.error("❌ Failed to setup TabPFN for cloud environment")
        st.stop()
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Use TabPFN model with comprehensive patching
    try:
        model = tabpfn_client.TabPFNClassifier()
        model.fit(X_train, y_train)
        st.success("✅ TabPFN Model trained successfully!")
    except Exception as e:
        st.error(f"❌ TabPFN training failed: {str(e)}")
        st.info("This might be due to network connectivity, authentication issues, or cloud environment restrictions.")
        st.stop()
    # Get predictions
    predictions = model.predict(X_test)
    # Get probability estimates
    probabilities = model.predict_proba(X_test)
    
    # Display probabilities in a more readable format
    st.subheader("📊 TabPFN Probability Estimates")
    
    # Convert to DataFrame for better display
    import pandas as pd
    prob_df = pd.DataFrame(probabilities, columns=['Class 0 Probability', 'Class 1 Probability'])
    prob_df['Prediction'] = predictions
    prob_df['Actual'] = y_test
    
    # Show model accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, predictions)
    st.metric("Model Accuracy", f"{accuracy:.3f}")
    
    # Show first 10 rows
    st.write("**First 10 predictions with probabilities:**")
    st.dataframe(prob_df.head(10), use_container_width=True)
    