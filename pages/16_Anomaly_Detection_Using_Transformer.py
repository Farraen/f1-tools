import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

import tabpfn_client

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import streamlit as st

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMjJhMDA0ZTctYWVmOS00MWZkLWExYTAtNzczZTgzYTFjNWU4IiwiZXhwIjoxNzkwNTYxOTQyfQ.FkeXhyUZTRsqM3vr-cUa9Etq6UmIOmBPQ48NSfyNF_k"

class TabPFNWrapper:
    """Custom wrapper for TabPFN that handles cloud deployment issues"""
    
    def __init__(self):
        self.model = None
        self.token_set = False
        
    def set_token_safely(self, token):
        """Set TabPFN token with cloud-friendly approach"""
        try:
            # Method 1: Try setting environment variable first
            os.environ['TABPFN_ACCESS_TOKEN'] = token
            
            # Method 2: Try to create a temp directory for caching
            temp_dir = tempfile.mkdtemp(prefix='tabpfn_')
            os.environ['TABPFN_CACHE_DIR'] = temp_dir
            
            # Method 3: Try the original method
            tabpfn_client.set_access_token(token)
            self.token_set = True
            return True, "Token set successfully"
            
        except Exception as e:
            # Fallback: Try to monkey patch the problematic method
            try:
                self._patch_service_wrapper()
                tabpfn_client.set_access_token(token)
                self.token_set = True
                return True, f"Token set with patching: {str(e)}"
            except Exception as e2:
                return False, f"Failed to set token: {str(e2)}"
    
    def _patch_service_wrapper(self):
        """Monkey patch the service wrapper to avoid file system issues"""
        import tabpfn_client.service_wrapper as sw
        
        # Store original method
        original_set_token = sw.UserAuthenticationClient.set_token
        
        def safe_set_token(cls, token):
            """Safe version that doesn't create directories"""
            try:
                # Try original method first
                return original_set_token(token)
            except (PermissionError, OSError) as e:
                # If it fails due to permissions, just store token in memory
                cls._token = token
                return True
        
        # Apply the patch
        sw.UserAuthenticationClient.set_token = classmethod(safe_set_token)
    
    def create_classifier(self):
        """Create TabPFN classifier with error handling"""
        if not self.token_set:
            raise Exception("Token not set. Call set_token_safely() first.")
        
        try:
            return tabpfn_client.TabPFNClassifier()
        except Exception as e:
            raise Exception(f"Failed to create TabPFN classifier: {str(e)}")
    
    def cleanup(self):
        """Clean up temporary directories"""
        try:
            if 'TABPFN_CACHE_DIR' in os.environ:
                cache_dir = os.environ['TABPFN_CACHE_DIR']
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
        except Exception:
            pass  # Ignore cleanup errors

if st.button('Train model'):
    # Initialize the custom wrapper
    tabpfn_wrapper = TabPFNWrapper()
    
    # Set token using the custom wrapper
    success, message = tabpfn_wrapper.set_token_safely(token)
    
    if success:
        st.success(f"✅ {message}")
    else:
        st.error(f"❌ {message}")
        st.stop()
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Use TabPFN model with custom wrapper
    try:
        model = tabpfn_wrapper.create_classifier()
        model.fit(X_train, y_train)
        st.success("✅ TabPFN Model trained successfully!")
    except Exception as e:
        st.error(f"❌ TabPFN training failed: {str(e)}")
        st.info("This might be due to network connectivity, authentication issues, or cloud environment restrictions.")
        st.stop()
    finally:
        # Clean up temporary files
        tabpfn_wrapper.cleanup()
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
    