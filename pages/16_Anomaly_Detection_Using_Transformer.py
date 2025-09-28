import os
import matplotlib.pyplot as plt
import torch
import numpy as np

import tabpfn_client

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import streamlit as st

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMjJhMDA0ZTctYWVmOS00MWZkLWExYTAtNzczZTgzYTFjNWU4IiwiZXhwIjoxNzkwNTYxOTQyfQ.FkeXhyUZTRsqM3vr-cUa9Etq6UmIOmBPQ48NSfyNF_k"

if st.button('Train model'):
    # Set environment variables for TabPFN (cloud-friendly approach)
    os.environ['TABPFN_ACCESS_TOKEN'] = token
    
    # Try to set cache directory to a writable location
    try:
        # Use /tmp directory which is usually writable in cloud environments
        os.environ['TABPFN_CACHE_DIR'] = '/tmp/tabpfn_cache'
        st.info("🔧 Using /tmp directory for TabPFN cache")
    except Exception as e:
        st.warning(f"⚠️ Could not set cache directory: {str(e)}")
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Use TabPFN model with environment variable approach
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
    