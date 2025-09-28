import os
from symbol import yield_expr
import matplotlib.pyplot as plt
import torch
import numpy as np
import tempfile
from pathlib import Path
import pandas as pd
import tabpfn_client
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

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


def st_title(text):
    st.markdown(f'<p class="title_medium">{text}</p>', unsafe_allow_html=True)

def st_text(text):
    st.markdown(f'<p class="text_small">{text}</p>', unsafe_allow_html=True)






def generate_data(t=0):    
    n_rows = 100
    time = np.arange(0, n_rows * 0.1, 0.1) + t  # Every 0.1 seconds
    y = np.sin(time)
    return time, y

if "plot" not in st.session_state:
    st.session_state.plot = None

if "plot_handle" not in st.session_state:
    st.session_state.plot_handle = None

if "animation_running" not in st.session_state:
    st.session_state.animation_running = False

if "t" not in st.session_state:
    st.session_state.t = 0




if st.button('Start Animation'):
    st.session_state.animation_running = True

if st.button('Stop Animation'):
    st.session_state.animation_running = False



if st.session_state.animation_running:

    while st.session_state.animation_running:
        t = st.session_state.t + 0.1
        st.session_state.t = t
        x,y = generate_data(t)
        fig = px.line(x=x, y=y)
        st.session_state.plot = fig
        st.session_state.plot_handle.plotly_chart(fig)


        
if st.session_state.plot is None:
    x,y = generate_data()
    fig = px.line(x=x, y=y)
    st.session_state.plot_handle = st.empty()
    st.session_state.plot_handle.plotly_chart(fig)
    st.session_state.plot = fig
else:
    st.session_state.plot_handle.   plotly_chart(st.session_state.plot)

time.sleep(0.02)