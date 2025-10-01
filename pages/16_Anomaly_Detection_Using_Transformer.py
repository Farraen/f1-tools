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
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from PIL import Image


st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

# Page title and description
st.title("Anomaly Detection using Transformer")

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMjJhMDA0ZTctYWVmOS00MWZkLWExYTAtNzczZTgzYTFjNWU4IiwiZXhwIjoxNzkwNTYxOTQyfQ.FkeXhyUZTRsqM3vr-cUa9Etq6UmIOmBPQ48NSfyNF_k"


# MongoDB
@st.cache_resource 
def connect_mongo():
    mongoUser = 'farraen'
    mongoPwd = 'rI68TwqYQTSDu5Pp'
    mongoDb = 'f1_analysis_max' 
    mongoDb2 = 'f1_info'  

    uri = f"mongodb+srv://{mongoUser}:{mongoPwd}@cluster0.rjtb7gz.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri, server_api=ServerApi('1'))

    try:
        client.admin.command('ping')
        db_status = "Connected"
    except Exception as e:
        db_status = "Connection error"


    db = client[mongoDb]
    db_info = client[mongoDb2]

    return db,db_info,client,db_status


# Function 
def GetRaceInfo():
    #dbcol = db[f"race_{st.session_state.year}_{st.session_state.race}"]
    race_info = db_info[f"race_{st.session_state.year}_{st.session_state.race}"]
    cur = race_info.find_one()
    st.session_state.total_laps = cur['total_laps']
    st.session_state.race_name = cur['Name']       

def GetRaceData(year,race,lap):

    dbcol = db[f"race_{year}_{race}"]
    race_dict = dbcol.find_one({"LapNumber": lap})
    tel = race_dict['Telemetry']
    df = pd.DataFrame(tel)

    df = df[["Time","RPM","Speed","nGear","Throttle","Distance"]]

    return df


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



if "last_tabpfn_time" not in st.session_state:
    st.session_state.last_tabpfn_time = 0

if "reconstructed_signal" not in st.session_state:
    st.session_state.reconstructed_signal = None

if "processing_length" not in st.session_state:
    st.session_state.processing_length = 10.0


if 'year' not in st.session_state:
    st.session_state.year = 2023

if 'race' not in st.session_state:
    st.session_state.race = 6

if 'lap' not in st.session_state:
    st.session_state.lap = 51

if 'selected_anomalies' not in st.session_state:
    st.session_state.selected_anomalies = []

# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

# Connect to database
db,db_info,client,st.session_state.db_status = connect_mongo()


# Race info
dbcol = db[f"race_{st.session_state.year}_{st.session_state.race}"]
item_dict = dbcol.find({},{ "_id": 0, "LapNumber": 1, "LapTime": 1 , "Compound": 1 , "Position": 1 , "AirTempMean": 1 })
df_laps = pd.DataFrame(item_dict)
df_laps['Year'] = st.session_state.year
df_laps['Round'] = st.session_state.race


lap = st.session_state.lap
race_dict = dbcol.find_one({"LapNumber": lap})
tel = race_dict['Telemetry']
df = pd.DataFrame(tel)

# Get next lap data for comparison
next_lap = lap + 1
next_race_dict = dbcol.find_one({"LapNumber": next_lap})
if next_race_dict:
    next_tel = next_race_dict['Telemetry']
    df_next = pd.DataFrame(next_tel)
else:
    df_next = None
    st.warning(f"Next lap ({next_lap}) data not available")


with st.expander('Introduction',expanded=True):
    st.write("")
    
    image = read_image("images/Page4_tech.png")
    st.image(image,use_column_width=True)

with st.expander('Testing',expanded=True):
    # Training split control
    col0 , col1,col2 = st.columns([0.05,1,0.18])
    with col1:
        st.header("Training Configuration")
        training_split = st.slider(
            "Training Data Percentage", 
            min_value=10, 
            max_value=90, 
            value=75, 
            step=5,
            help="Percentage of data to use for training. Remaining data will be used for prediction."
        )


    confidence_level = 80
    

    anomaly_threshold = 3.5

    # Initial plot

    # Split data based on slider value
    split_point = int(len(df) * (training_split / 100))

    x_features = ['Speed','nGear','Throttle','RelativeDistance']
    # Training data (first X% based on slider)
    df_train = df.iloc[:split_point]
    x_train = df_train[x_features]
    y_train = df_train['RPM']

    # Test data (remaining %)
    df_test = df.iloc[split_point:]
    x_test = df_test[x_features]
    y_test = df_test['RPM']



    # Create TabPFN model
    st.session_state.tabpfn_model = tabpfn_client.TabPFNRegressor()

    # Fit the model on training data only
    st.session_state.tabpfn_model.fit(x_train, y_train)

    # Calculate quantiles for confidence interval
    alpha = (100 - confidence_level) / 100
    lower_quantile = alpha / 2
    upper_quantile = 1 - (alpha / 2)
    quantiles = [lower_quantile, upper_quantile]

    # Predict on test data with confidence intervals
    y_predicted = st.session_state.tabpfn_model.predict(x_test)
    y_confidence = st.session_state.tabpfn_model.predict(x_test, 
                                                    output_type='quantiles', 
                                                    quantiles=quantiles)

    # Store results for plotting
    st.session_state.df_test = df_test.copy()
    st.session_state.y_predicted = y_predicted
    st.session_state.y_test = y_test
    st.session_state.y_lower = y_confidence[0]
    st.session_state.y_upper = y_confidence[1]
    st.session_state.confidence_level = confidence_level
        

    # Plot comparison
    if 'y_predicted' in st.session_state:
        # Split the data for different colors
        split_point = int(len(df) * (training_split / 100))
        df_train_plot = df.iloc[:split_point]
        df_test_plot = df.iloc[split_point:]
        
        # Create plot with training data (darker blue)
        fig = px.line(df_train_plot, x='Time', y='RPM', 
                    title=f'RPM Data with TabPFN Prediction Overlay (Last {100-training_split}%)',
                    labels={'RPM': 'RPM'},
                    color_discrete_sequence=['#1f77b4'])  # Darker blue
        
        # Add test data (lighter blue)
        fig.add_scatter(x=df_test_plot['Time'], 
                    y=df_test_plot['RPM'],
                    mode='lines',
                    name=f'Ground Truth (Last {100-training_split}%)',
                    line=dict(color='#87ceeb', width=2))  # Light blue
        
        # Add smooth confidence interval (continuous shaded area)
        # Combine upper and lower bounds into a single continuous area
        confidence_x = list(st.session_state.df_test['Time']) + list(st.session_state.df_test['Time'][::-1])
        confidence_y = list(st.session_state.y_upper) + list(st.session_state.y_lower[::-1])
        
        fig.add_scatter(x=confidence_x,
                    y=confidence_y,
                    mode='lines',
                    name=f'{st.session_state.confidence_level}% Confidence Interval',
                    line=dict(
                        color='rgba(255,0,0,0.1)',
                        width=0
                    ),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    showlegend=True)
        
        # Add predicted values only for the test portion (remaining %)
        fig.add_scatter(x=st.session_state.df_test['Time'], 
                    y=st.session_state.y_predicted,
                    mode='lines',
                    name='TabPFN Prediction',
                    line=dict(
                        color='red', 
                        dash='dash', 
                        width=2,
                        smoothing=1.3,
                        shape='spline'
                    ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="RPM",
            showlegend=True,
            title=f'RPM Data with TabPFN Prediction and {st.session_state.confidence_level}% Confidence Interval',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            )
        )
        
        st.plotly_chart(fig)
        
        
    else:
        # Show original plot if no predictions yet
        fig_original = px.line(df, x='Time', y='RPM', title='Original RPM Data')
        fig_original.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            )
        )
        st.plotly_chart(fig_original)

    # Next Lap Analysis
    if df_next is not None and 'y_predicted' in st.session_state:
        st.header(f"Next Lap Analysis (Lap {next_lap})")
        
        # Predict for the entire next lap
        x_next_full = df_next[x_features]
        y_next_full = df_next['RPM']
        
        # Use the same trained model to predict entire next lap
        y_next_predicted = st.session_state.tabpfn_model.predict(x_next_full)
        
        # Anomaly Detection for Next Lap
        prediction_errors = np.abs(y_next_full - y_next_predicted)
        error_mean = np.mean(prediction_errors)
        error_std = np.std(prediction_errors)
        
        # Calculate anomaly scores (z-scores)
        anomaly_scores = (prediction_errors - error_mean) / error_std
        
        # Identify anomalies
        anomalies = anomaly_scores > anomaly_threshold
        anomaly_indices = np.where(anomalies)[0]
        
        # Create next lap plot - show entire lap
        fig_next = px.line(df_next, x='Time', y='RPM', 
                        title=f'Next Lap {next_lap} RPM Data with TabPFN Prediction and Anomaly Detection',
                        labels={'RPM': 'RPM'},
                        color_discrete_sequence=['#1f77b4'])  # Darker blue for ground truth
        
        # Add predicted values for entire lap
        fig_next.add_scatter(x=df_next['Time'], 
                            y=y_next_predicted,
                            mode='lines',
                            name='TabPFN Prediction',
                            line=dict(
                                color='red', 
                                dash='dash', 
                                width=2,
                                smoothing=1.3,
                                shape='spline'
                            ))
        
        # Add anomaly points with clickable functionality
        if len(anomaly_indices) > 0:
            anomaly_times = df_next.iloc[anomaly_indices]['Time']
            anomaly_rpm = df_next.iloc[anomaly_indices]['RPM']
             
            # Create customdata for each anomaly point to store index
            customdata = [[idx] for idx in anomaly_indices]
            
            fig_next.add_scatter(x=anomaly_times,
                                y=anomaly_rpm,
                                mode='markers',
                                name=f'Anomalies (>{anomaly_threshold}σ)',
                                customdata=customdata,
                                marker=dict(
                                    color='orange',
                                    size=8,
                                    symbol='diamond',
                                    line=dict(color='darkorange', width=2)
                                ))
        
        fig_next.update_layout(
            xaxis_title="Time",
            yaxis_title="RPM",
            showlegend=True,
            title=f'Next Lap {next_lap} RPM Data with TabPFN Prediction and Anomaly Detection',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            )
        )
        
        # Display the plot and handle click events
        selected_points = st.plotly_chart(fig_next, on_select="rerun", selection_mode="points")
        
        # Handle anomaly selection
        if selected_points and 'selection' in selected_points:
            # Get selected point indices
            selected_indices = []
            for point in selected_points['selection']['points']:
                if 'customdata' in point and point['customdata']:
                    selected_indices.append(point['customdata'][0])
            
            # Update selected anomalies
            st.session_state.selected_anomalies = selected_indices
        
        # Display selection controls
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Clear Selection"):
                st.session_state.selected_anomalies = []
                st.rerun()
        
        with col2:
            if st.button("Select All Anomalies"):
                st.session_state.selected_anomalies = list(anomaly_indices)
                st.rerun()
        
        # Display selected anomalies info
        if st.session_state.selected_anomalies:
            st.subheader(f"Selected Anomalies ({len(st.session_state.selected_anomalies)} selected)")
            
            # Create dataframe for selected anomalies
            selected_df = df_next.iloc[st.session_state.selected_anomalies].copy()
            selected_df['Prediction_Error'] = prediction_errors[st.session_state.selected_anomalies]
            selected_df['Anomaly_Score'] = anomaly_scores[st.session_state.selected_anomalies]
            selected_df['Predicted_RPM'] = y_next_predicted[st.session_state.selected_anomalies]
            
            # Display key columns
            display_cols = ['Time', 'RPM', 'Predicted_RPM', 'Prediction_Error', 'Anomaly_Score', 'Speed', 'nGear', 'Throttle']
            st.dataframe(selected_df[display_cols].round(2), use_container_width=True)
            
            # Summary statistics for selected anomalies
            if len(st.session_state.selected_anomalies) > 1:
                st.subheader("Selected Anomalies Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Prediction Error", f"{np.mean(prediction_errors[st.session_state.selected_anomalies]):.2f}")
                with col2:
                    st.metric("Max Anomaly Score", f"{np.max(anomaly_scores[st.session_state.selected_anomalies]):.2f}")
                with col3:
                    st.metric("Time Range", f"{np.min(selected_df['Time']):.1f}s - {np.max(selected_df['Time']):.1f}s")
        
        # Display all anomalies summary
        st.subheader("All Anomalies Summary")
        total_points = len(df_next)
        anomaly_count = len(anomaly_indices)
        anomaly_percentage = (anomaly_count / total_points) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Points", f"{total_points}")
        with col2:
            st.metric("Anomalies Detected", f"{anomaly_count}")
        with col3:
            st.metric("Anomaly Rate", f"{anomaly_percentage:.1f}%")
        with col4:
            st.metric("Threshold", f"{anomaly_threshold}σ")
        
    

    elif df_next is not None:
        st.header(f"Next Lap Data Available (Lap {next_lap})")
        st.info("Train the model on current lap first to see next lap predictions")
        
        # Show next lap data without predictions
        fig_next_original = px.line(df_next, x='Time', y='RPM', title=f'Next Lap {next_lap} Original RPM Data')
        fig_next_original.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            )
        )
        st.plotly_chart(fig_next_original)


