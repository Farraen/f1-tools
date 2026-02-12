import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.title('Interactive Report Generator')

st.caption(
    "An app that shows how interactive reports can be generated using Streamlit and Plotly. This example uses CatBoost as the model candidate."
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None

if 'trained' not in st.session_state:
    st.session_state.trained = False

# Step 1: Load Data
st.subheader('Step 1: Load Data')

try:
    df = pd.read_excel("doe.xlsx")
    
    # Filter out diesel-specific columns to simulate gasoline engine
    # Remove diesel-specific measurements like EGR, soot, NOx, etc.
    diesel_keywords = ['egr', 'soot', 'nox', 'dpf', 'scr', 'urea', 'diesel', 'bsfc_ind']
    
    # Keep only columns that don't contain diesel keywords (case insensitive)
    cols_to_keep = [col for col in df.columns 
                    if not any(keyword in col.lower() for keyword in diesel_keywords)]
    
    df = df[cols_to_keep]
    
    st.success(f'DOE data loaded: {df.shape[0]} rows, {df.shape[1]} columns (gasoline engine data)')
    st.dataframe(df.head(), use_container_width=True)
except Exception as e:
    st.error(f"Error loading doe.xlsx: {e}")
    df = None

st.write("**Data shape:**", df.shape)

# Define default inputs and outputs (gasoline engine parameters)
inputs = ["SPEED_A", "TORQUE_R", "BobOV_XPC_VGT"]
outputs = ["Air_Fuel_Ratio", "EXH_TEMP_MANI_OUT", "pMAX", "TC1_TURBO_SPEED"]

# Step 2: Configure Training
if df is not None:
    st.subheader('Step 2: Configure Model')
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        input_cols = st.multiselect('Select input features', 
                                     df.columns.tolist(), 
                                     default=inputs)
    
    with col2:
        output_cols = st.multiselect('Select target variables', 
                                     [col for col in df.columns if col not in input_cols],
                                     default=outputs)
    
    with col3:
        test_size = st.slider('Test split ratio', 0.1, 0.5, 0.33, 0.05)

# Step 3: Set Hyperparameters
if df is not None and input_cols and output_cols:
    st.subheader('Step 3: Set Hyperparameters')
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        iterations = st.number_input('Iterations', 50, 1000, 150, 50)
        learning_rate = st.number_input('Learning rate', 0.01, 0.5, 0.1, 0.01)
    
    with col2:
        depth = st.number_input('Depth', 1, 10, 6, 1)
        l2_leaf_reg = st.number_input('L2 leaf regularization', 1, 10, 3, 1)
    
    with col3:
        # Multi-output or single output
        is_multi = len(output_cols) > 1
        loss_function = 'MultiRMSE' if is_multi else 'RMSE'
        st.write(f'**Loss function:** {loss_function}')
        st.write(f'**Target type:** {"Multi-output" if is_multi else "Single output"}')

    # Training button
    train_button = st.button('Train Model', type='primary')
    
    if train_button:
        with st.spinner('Training model...'):
            # Prepare data
            X = df[input_cols]
            y = df[output_cols]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Define parameters
            params = {
                'learning_rate': learning_rate,
                'depth': depth,
                'l2_leaf_reg': l2_leaf_reg,
                'loss_function': loss_function,
                'eval_metric': loss_function,
                'task_type': 'CPU',
                'iterations': iterations,
                'verbose': False
            }
            
            # Train model
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.output_cols = output_cols
            st.session_state.trained = True
            
        st.success('Model trained successfully!')

# Step 4: Display Results
if st.session_state.trained:
    st.subheader('Step 4: Model Performance')
    
    model = st.session_state.model
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    output_cols = st.session_state.output_cols
    
    # Calculate scores
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric('Training R²', f'{score_train:.4f}')
    with col2:
        st.metric('Validation R²', f'{score_test:.4f}')
    
    # Select target to visualize
    selected_output = st.selectbox('Select target to visualize', output_cols)
    
    col1, col2 = st.columns([1, 1])
    
    # Training plot
    with col1:
        st.write('**Training Set Performance**')
        y_pred_train = model.predict(X_train)
        
        if len(output_cols) > 1:
            y_pred_train_df = pd.DataFrame(y_pred_train, columns=output_cols)
            y_actual = y_train[selected_output].values
            y_predicted = y_pred_train_df[selected_output].values
        else:
            y_actual = y_train.values.flatten()
            y_predicted = y_pred_train.flatten()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_actual, 
            y=y_predicted, 
            mode='markers',
            name='Predictions',
            marker=dict(size=6, opacity=0.6)
        ))
        
        # Add perfect prediction line
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], 
            y=[min_val, max_val], 
            mode='lines',
            name='Perfect Fit',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Actual',
            yaxis_title='Predicted',
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Validation plot
    with col2:
        st.write('**Validation Set Performance**')
        y_pred_test = model.predict(X_test)
        
        if len(output_cols) > 1:
            y_pred_test_df = pd.DataFrame(y_pred_test, columns=output_cols)
            y_actual = y_test[selected_output].values
            y_predicted = y_pred_test_df[selected_output].values
        else:
            y_actual = y_test.values.flatten()
            y_predicted = y_pred_test.flatten()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_actual, 
            y=y_predicted, 
            mode='markers',
            name='Predictions',
            marker=dict(size=6, opacity=0.6)
        ))
        
        # Add perfect prediction line
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], 
            y=[min_val, max_val], 
            mode='lines',
            name='Perfect Fit',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Actual',
            yaxis_title='Predicted',
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time Series Prediction Section
    st.subheader('Time Series Prediction')
    st.info('Model predictions over time series data showing Exhaust Temperature and Peak Cylinder Pressure')
    
    # Use test data as time series (sorted by index to simulate time progression)
    X_timeseries = X_test.sort_index()
    y_timeseries_pred = model.predict(X_timeseries)
    
    # Create time index (simulating time progression)
    time_index = np.arange(len(X_timeseries))
    
    # Get predictions for exhaust temp and pmax
    if len(output_cols) > 1:
        y_pred_df = pd.DataFrame(y_timeseries_pred, columns=output_cols)
        
        # Check if these columns exist in the output
        has_exh_temp = 'EXH_TEMP_MANI_OUT' in output_cols
        has_pmax = 'pMAX' in output_cols
        
        if has_exh_temp and has_pmax:
            exh_temp_pred = y_pred_df['EXH_TEMP_MANI_OUT'].values
            pmax_pred = y_pred_df['pMAX'].values
            
            # Create subplots with shared x-axis
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Exhaust Temperature (°C)', 'Peak Cylinder Pressure (bar)')
            )
            
            # Add Exhaust Temperature trace
            fig.add_trace(
                go.Scatter(
                    x=time_index,
                    y=exh_temp_pred,
                    mode='lines',
                    name='Exhaust Temp',
                    line=dict(color='#ff6b35', width=2),
                    hovertemplate='Time: %{x}<br>Temp: %{y:.2f}°C<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add pMAX trace
            fig.add_trace(
                go.Scatter(
                    x=time_index,
                    y=pmax_pred,
                    mode='lines',
                    name='pMAX',
                    line=dict(color='#64b5f6', width=2),
                    hovertemplate='Time: %{x}<br>Pressure: %{y:.2f} bar<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Update layout with range slider
            fig.update_xaxes(
                title_text="Time Step",
                row=2, col=1,
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=50, label="50pts", step="all"),
                        dict(count=100, label="100pts", step="all"),
                        dict(step="all", label="All")
                    ])
                )
            )
            
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_yaxes(title_text="Pressure (bar)", row=2, col=1)
            
            fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning('Time series plot requires both EXH_TEMP_MANI_OUT and pMAX in the output variables.')
    else:
        st.warning('Time series plot requires multiple output variables.')
    
     
 
    # Generate HTML Report section
    st.subheader('Step 5: Generate Interactive HTML Report')
    
    
    report_name = st.text_input('Report filename', 'catboost_model_report.html')
    
    # Get data from session state
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    model = st.session_state.model
    output_cols = st.session_state.output_cols
    
    # Generate predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate R² scores
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    
    # Get model parameters
    model_params = model.get_params()
    
    # Create plots for each output
    train_plots_html = []
    test_plots_html = []
    
    for output in output_cols:
        # Training plot
        if len(output_cols) > 1:
            y_pred_train_df = pd.DataFrame(y_pred_train, columns=output_cols)
            y_actual_train = y_train[output].values
            y_predicted_train = y_pred_train_df[output].values
        else:
            y_actual_train = y_train.values.flatten()
            y_predicted_train = y_pred_train.flatten()
        
        fig_train = go.Figure()
        fig_train.add_trace(go.Scatter(
            x=y_actual_train, 
            y=y_predicted_train, 
            mode='markers',
            name='Predictions',
            marker=dict(color='#ff6b35', opacity=0.7, size=6),
            hovertemplate=f'Actual: %{{x:.2f}}<br>Predicted: %{{y:.2f}}<br>R² = {score_train:.3f}<extra></extra>'
        ))
        
        # Add perfect prediction line
        min_val = min(y_actual_train.min(), y_predicted_train.min())
        max_val = max(y_actual_train.max(), y_predicted_train.max())
        fig_train.add_trace(go.Scatter(
            x=[min_val, max_val], 
            y=[min_val, max_val], 
            mode='lines',
            name='Perfect Fit',
            line=dict(color='white', dash='dash', width=2),
            showlegend=False
        ))
        
        fig_train.update_layout(
            title=dict(text=f'Training Data: {output} (R² = {score_train:.3f})', font=dict(size=14)),
            xaxis_title=f'Actual {output}',
            yaxis_title=f'Predicted {output}',
            height=350,
            margin=dict(l=60, r=20, t=50, b=50),
            paper_bgcolor='#1a2332',
            plot_bgcolor='#2a3441',
            font=dict(color='white'),
            autosize=True
        )
        
        train_plots_html.append(fig_train.to_html(include_plotlyjs=False, div_id=f'train_plot_{output}'))
        
        # Test plot
        if len(output_cols) > 1:
            y_pred_test_df = pd.DataFrame(y_pred_test, columns=output_cols)
            y_actual_test = y_test[output].values
            y_predicted_test = y_pred_test_df[output].values
        else:
            y_actual_test = y_test.values.flatten()
            y_predicted_test = y_pred_test.flatten()
        
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(
            x=y_actual_test, 
            y=y_predicted_test, 
            mode='markers',
            name='Predictions',
            marker=dict(color='#64b5f6', opacity=0.7, size=6),
            hovertemplate=f'Actual: %{{x:.2f}}<br>Predicted: %{{y:.2f}}<br>R² = {score_test:.3f}<extra></extra>'
        ))
        
        # Add perfect prediction line
        min_val = min(y_actual_test.min(), y_predicted_test.min())
        max_val = max(y_actual_test.max(), y_predicted_test.max())
        fig_test.add_trace(go.Scatter(
            x=[min_val, max_val], 
            y=[min_val, max_val], 
            mode='lines',
            name='Perfect Fit',
            line=dict(color='white', dash='dash', width=2),
            showlegend=False
        ))
        
        fig_test.update_layout(
            title=dict(text=f'Testing Data: {output} (R² = {score_test:.3f})', font=dict(size=14)),
            xaxis_title=f'Actual {output}',
            yaxis_title=f'Predicted {output}',
            height=350,
            margin=dict(l=60, r=20, t=50, b=50),
            paper_bgcolor='#1a2332',
            plot_bgcolor='#2a3441',
            font=dict(color='white'),
            autosize=True
        )
        
        test_plots_html.append(fig_test.to_html(include_plotlyjs=False, div_id=f'test_plot_{output}'))
    
    # Generate the HTML report
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create plot containers HTML
    plots_html = ""
    for i in range(len(output_cols)):
        plots_html += f'''
            <div class="plot-container">
                {train_plots_html[i]}
            </div>
            <div class="plot-container">
                {test_plots_html[i]}
            </div>
        '''
    
    # Generate time series plot for HTML report
    timeseries_plot_html = ""
    has_exh_temp = 'EXH_TEMP_MANI_OUT' in output_cols
    has_pmax = 'pMAX' in output_cols
    
    if len(output_cols) > 1 and has_exh_temp and has_pmax:
        # Use test data as time series
        X_timeseries = X_test.sort_index()
        y_timeseries_pred = model.predict(X_timeseries)
        time_index = np.arange(len(X_timeseries))
        
        y_pred_df = pd.DataFrame(y_timeseries_pred, columns=output_cols)
        exh_temp_pred = y_pred_df['EXH_TEMP_MANI_OUT'].values
        pmax_pred = y_pred_df['pMAX'].values
        
        # Create time series plot
        fig_timeseries = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Exhaust Temperature (°C)', 'Peak Cylinder Pressure (bar)')
        )
        
        fig_timeseries.add_trace(
            go.Scatter(
                x=time_index,
                y=exh_temp_pred,
                mode='lines',
                name='Exhaust Temp',
                line=dict(color='#ff6b35', width=2),
                hovertemplate='Time: %{x}<br>Temp: %{y:.2f}°C<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig_timeseries.add_trace(
            go.Scatter(
                x=time_index,
                y=pmax_pred,
                mode='lines',
                name='pMAX',
                line=dict(color='#64b5f6', width=2),
                hovertemplate='Time: %{x}<br>Pressure: %{y:.2f} bar<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig_timeseries.update_xaxes(
            title_text="Time Step",
            row=2, col=1,
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=50, label="50pts", step="all"),
                    dict(count=100, label="100pts", step="all"),
                    dict(step="all", label="All")
                ])
            )
        )
        
        fig_timeseries.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig_timeseries.update_yaxes(title_text="Pressure (bar)", row=2, col=1)
        
        fig_timeseries.update_layout(
            height=600,
            showlegend=False,
            margin=dict(l=60, r=30, t=50, b=100),
            hovermode='x unified',
            paper_bgcolor='#1a2332',
            plot_bgcolor='#2a3441',
            font=dict(color='white')
        )
        
        timeseries_plot_html = fig_timeseries.to_html(
            include_plotlyjs=False, 
            div_id='timeseries_plot', 
            config={'displayModeBar': True, 'staticPlot': False}
        )
    
    # Create input/output columns lists
    input_cols_html = "".join([f'<div class="list-item">{col}</div>' for col in input_cols])
    output_cols_html = "".join([f'<div class="list-item">{col}</div>' for col in output_cols])
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gasoline Engine Model Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Arial', 'Helvetica', sans-serif; margin: 0; padding: 0; background-color: #0d1421; color: #ffffff; min-height: 100vh; line-height: 1.4; overflow-x: hidden; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: #1a2332; padding: 20px; box-shadow: 0 0 15px rgba(0, 0, 0, 0.5); box-sizing: border-box; width: 100%; }}
        .header {{ display: flex; align-items: center; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 2px solid #ff6b35; }}
        .logo {{ margin-right: 20px; }}
        .logo img {{ height: 50px; width: auto; }}
        .header-content {{ flex: 1; }}
        .header h1 {{ color: #ffffff; font-size: 2.0em; margin: 0; font-weight: 300; letter-spacing: 0.5px; }}
        .header p {{ color: #b8c5d1; font-size: 1.0em; margin: 5px 0 0 0; font-weight: 300; }}
        .two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .collapsible-section {{ margin: 15px 0; background-color: #2a3441; border-left: 3px solid #ff6b35; box-shadow: 0 1px 5px rgba(0, 0, 0, 0.3); overflow: hidden; }}
        .collapsible-header {{ padding: 8px 15px; background-color: #3a4552; cursor: pointer; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #4a5568; transition: background-color 0.3s ease; }}
        .collapsible-header:hover {{ background-color: #4a5568; }}
        .collapsible-header h2 {{ color: #ffffff; margin: 0; font-size: 1.1em; font-weight: 400; text-transform: uppercase; letter-spacing: 0.3px; }}
        .collapsible-content {{ padding: 15px; display: none; box-sizing: border-box; width: 100%; }}
        .collapsible-content.active {{ display: block; }}
        .toggle-icon {{ color: #ff6b35; font-size: 1.0em; font-weight: bold; transition: transform 0.3s ease; }}
        .toggle-icon.rotated {{ transform: rotate(180deg); }}
        .stat-item {{ margin: 8px 0; padding: 8px 12px; background-color: #3a4552; display: flex; justify-content: space-between; align-items: center; border: 1px solid #4a5568; }}
        .stat-label {{ font-weight: 500; color: #e2e8f0; font-size: 0.9em; }}
        .stat-value {{ color: #ff6b35; font-size: 1.1em; font-weight: 600; }}
        .list-item {{ margin: 4px 0; padding: 6px 10px; background-color: #3a4552; color: #e2e8f0; font-size: 0.8em; border: 1px solid #4a5568; font-family: 'Courier New', monospace; }}
        .plots-container {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 15px; margin-top: 20px; width: 100%; box-sizing: border-box; }}
        .plot-container {{ background-color: #1a2332; padding: 10px; border: 1px solid #4a5568; width: 100%; min-width: 0; box-sizing: border-box; overflow: hidden; }}
        .plot-container .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
        .plot-container-full {{ background-color: #1a2332; padding: 10px; border: 1px solid #4a5568; width: 100%; box-sizing: border-box; overflow: hidden; display: block; }}
        .plot-container-full .plotly-graph-div {{ width: 100% !important; height: 100% !important; display: block !important; }}
        .plot-container-full .js-plotly-plot {{ width: 100% !important; }}
        .footer {{ text-align: center; margin-top: 25px; padding-top: 15px; border-top: 1px solid #4a5568; color: #8a9ba8; font-size: 0.8em; }}
        .section-title {{ color: #ff6b35; font-size: 0.9em; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.3px; }}
        @media (max-width: 768px) {{ .two-column, .plots-container {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">


        <div class="header">
            <div class="logo">
                <img src="https://img.redbull.com/image/upload/e_trim:1:transparent/w_250/bo_5px_solid_rgb:00000000/q_auto:best,f_auto/redbullcom/static/powertrains-logo-white-v1.png" alt="Logo">
            </div>
            <div class="header-content">
                <h1>Engine ML Model Report</h1>
                <p>Model Training and Performance Analysis</p>
                <p style="font-size: 0.85em; color: #ff6b35; margin-top: 10px; font-style: normal;">Disclaimer: This generated report is just for fun and not intended for real engineering application. Using the Red Bull logo because they are my favourite F1 team 😆.</p>
            </div>
        </div>
        
        <div class="collapsible-section">
            <div class="collapsible-header" onclick="toggleSection(this)">
                <h2>Dataset Overview</h2>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="collapsible-content active">
                <div class="two-column">
                    <div>
                        <div class="stat-item">
                            <span class="stat-label">Total Dataset Rows</span>
                            <span class="stat-value">{len(df):,}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Training Samples</span>
                            <span class="stat-value">{len(X_train):,}</span>
                        </div>
                    </div>
                    <div>
                        <div class="stat-item">
                            <span class="stat-label">Test Samples</span>
                            <span class="stat-value">{len(X_test):,}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Total Features</span>
                            <span class="stat-value">{len(input_cols)}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="collapsible-section">
            <div class="collapsible-header" onclick="toggleSection(this)">
                <h2>Feature Information</h2>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="collapsible-content">
                <div class="two-column">
                    <div>
                        <h3 style="color: #ff6b35; margin-bottom: 10px;">Input Features ({len(input_cols)})</h3>
                        {input_cols_html}
                    </div>
                    <div>
                        <h3 style="color: #ff6b35; margin-bottom: 10px;">Target Variables ({len(output_cols)})</h3>
                        {output_cols_html}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="collapsible-section">
            <div class="collapsible-header" onclick="toggleSection(this)">
                <h2>Model Information</h2>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="collapsible-content">
                <div class="two-column">
                    <div>
                        <div class="stat-item">
                            <span class="stat-label">Model Type</span>
                            <span class="stat-value">CatBoostRegressor</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Loss Function</span>
                            <span class="stat-value">{model_params.get('loss_function', 'N/A')}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Learning Rate</span>
                            <span class="stat-value">{model_params.get('learning_rate', 'N/A')}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Iterations</span>
                            <span class="stat-value">{model_params.get('iterations', 'N/A')}</span>
                        </div>
                    </div>
                    <div>
                        <div class="stat-item">
                            <span class="stat-label">Max Depth</span>
                            <span class="stat-value">{model_params.get('depth', 'N/A')}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">L2 Leaf Regularization</span>
                            <span class="stat-value">{model_params.get('l2_leaf_reg', 'N/A')}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Training R²</span>
                            <span class="stat-value">{score_train:.4f}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Test R²</span>
                            <span class="stat-value">{score_test:.4f}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="collapsible-section">
            <div class="collapsible-header" onclick="toggleSection(this)">
                <h2>Model Performance Analysis</h2>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="collapsible-content">
                <div class="plots-container">
                    {plots_html}
                </div>
            </div>
        </div>
        
        {'<div class="collapsible-section"><div class="collapsible-header" onclick="toggleSection(this)"><h2>Time Series Prediction</h2><span class="toggle-icon">▼</span></div><div class="collapsible-content"><div class="plot-container-full">' + timeseries_plot_html + '</div></div></div>' if timeseries_plot_html else ''}
        
        <div class="footer">
            <p>Generated on {current_time} | Generated by Farraen's F1 Dashboard</p>
        </div>
    </div>
    
    <script>
        function toggleSection(header) {{
            const content = header.nextElementSibling;
            const icon = header.querySelector('.toggle-icon');
            
            if (content.classList.contains('active')) {{
                content.classList.remove('active');
                icon.classList.add('rotated');
            }} else {{
                content.classList.add('active');
                icon.classList.remove('rotated');
            }}
        }}
        
        // Track if time series has been initialized
        let timeseriesInitialized = false;
        
        // Override toggleSection to resize plots after opening
        const originalToggleSection = toggleSection;
        toggleSection = function(header) {{
            const content = header.nextElementSibling;
            const wasActive = content.classList.contains('active');
            
            originalToggleSection(header);
            
            // Resize plots when section is expanded
            setTimeout(function() {{
                if (content.classList.contains('active') && !wasActive) {{
                    const plots = content.querySelectorAll('.plotly-graph-div');
                    plots.forEach(function(plot) {{
                        const containerWidth = plot.parentElement.offsetWidth - 20;
                        const update = {{ width: containerWidth }};
                        Plotly.relayout(plot, update);
                    }});
                    
                    // Mark time series as initialized
                    if (header.textContent.includes('Time Series')) {{
                        timeseriesInitialized = true;
                    }}
                }}
            }}, 100);
        }};

        document.addEventListener('DOMContentLoaded', function() {{
            const sections = document.querySelectorAll('.collapsible-section');
            const totalSections = sections.length;
            
            sections.forEach((section, index) => {{
                const content = section.querySelector('.collapsible-content');
                const icon = section.querySelector('.toggle-icon');
                
                // Keep first section and last two sections expanded
                if (index === 0 || index >= totalSections - 2) {{
                    content.classList.add('active');
                    icon.classList.remove('rotated');
                }} else {{
                    content.classList.remove('active');
                    icon.classList.add('rotated');
                }}
            }});
            
            // Force all plots to correct width on load
            setTimeout(function() {{
                const plotDivs = document.querySelectorAll('.plotly-graph-div');
                plotDivs.forEach(function(div) {{
                    const update = {{
                        width: div.parentElement.offsetWidth - 20
                    }};
                    Plotly.relayout(div, update);
                }});
            }}, 200);
            
            // Make Plotly plots responsive on window resize only
            let resizeTimeout;
            window.addEventListener('resize', function() {{
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(function() {{
                    const plotDivs = document.querySelectorAll('.plotly-graph-div');
                    plotDivs.forEach(function(div) {{
                        const update = {{
                            width: div.parentElement.offsetWidth - 20
                        }};
                        Plotly.relayout(div, update);
                    }});
                }}, 250);
            }});
        }});
    </script>
</body>
</html>
'''
    
    # Single download button that generates and downloads the report
    st.download_button(
        label="📥 Generate Report",
        data=html_content,
        file_name=report_name,
        mime="text/html",
        type="primary"
    )

st.write('---')
st.caption('Copyright © 2025 Farraen. All rights reserved.')
