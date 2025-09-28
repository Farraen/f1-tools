import os
import matplotlib.pyplot as plt
import torch
import numpy as np

import tabpfn_client

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import streamlit as st

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMjJhMDA0ZTctYWVmOS00MWZkLWExYTAtNzczZTgzYTFjNWU4IiwiZXhwIjoxNzkwNTYxOTQyfQ.FkeXhyUZTRsqM3vr-cUa9Etq6UmIOmBPQ48NSfyNF_k"

tabpfn_client.set_access_token(token)

if st.button('Train model'):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Use it like any sklearn model
    model = tabpfn_client.TabPFNClassifier()
    model.fit(X_train, y_train)
    # Get predictions
    predictions = model.predict(X_test)
    # Get probability estimates
    probabilities = model.predict_proba(X_test)
    
    # Display probabilities in a more readable format
    st.subheader("📊 Probability Estimates")
    
    # Convert to DataFrame for better display
    import pandas as pd
    prob_df = pd.DataFrame(probabilities, columns=['Class 0 Probability', 'Class 1 Probability'])
    prob_df['Prediction'] = predictions
    prob_df['Actual'] = y_test
    
    # Show first 10 rows
    st.write("**First 10 predictions with probabilities:**")
    st.dataframe(prob_df.head(10), use_container_width=True)
    