import numpy as np
import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score

# Page configuration
st.set_page_config(
    page_title="F1 Tools - Stochastic Variational Inference",
    layout="wide"
)

st.title("Engine Air Flow Parameter Estimation")
st.markdown("### Stochastic Variational Inference (SVI) for Engine Calibration")

dev = 'cpu'

st.markdown("---")

# Engine Air Flow Model Parameters
# ================================

# Operating conditions (Engine Speed, Throttle Position, etc.)
engine_speeds = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000])  # RPM
throttle_positions = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])    # 0-1 scale
ambient_temp = 25.0  # °C
ambient_pressure = 101.325  # kPa

# Ground Truth Calibratable Parameters (what we want to estimate)
true_params = {
    'flow_coefficient': 0.85,      # Flow coefficient through throttle body
    'pressure_loss_coeff': 0.12,   # Pressure loss coefficient in intake manifold
    'thermal_conductivity': 0.15,  # Heat transfer coefficient
    'manifold_volume': 2.5,        # Intake manifold volume (liters)
    'noise_std': 0.01              # Measurement noise standard deviation
}

# Display parameters
col1, col2 = st.columns(2)
with col1:
    st.subheader("🔧 Ground Truth Parameters")
    for param, value in true_params.items():
        st.write(f"**{param}:** {value}")

# Create operating condition combinations
n_points = len(engine_speeds) * len(throttle_positions)
engine_speed_data = np.repeat(engine_speeds, len(throttle_positions))
throttle_data = np.tile(throttle_positions, len(engine_speeds))

with col2:
    st.subheader("📊 Operating Conditions")
    st.write(f"**Data points:** {n_points}")
    st.write(f"**Engine Speeds:** {engine_speeds} RPM")
    st.write(f"**Throttle Positions:** {throttle_positions}")

# Physics-Based Air Flow Model
# =============================

def airflow_model(engine_speed, throttle_pos, flow_coeff, pressure_loss_coeff, 
                 thermal_cond, manifold_vol, ambient_temp=25.0, ambient_pressure=101.325):
    """
    Simplified physics-based air flow model for intake manifold
    
    Parameters:
    - engine_speed: Engine speed in RPM
    - throttle_pos: Throttle position (0-1)
    - flow_coeff: Flow coefficient through throttle body
    - pressure_loss_coeff: Pressure loss coefficient
    - thermal_cond: Thermal conductivity coefficient
    - manifold_vol: Intake manifold volume (liters)
    """
    
    # Convert to numpy arrays if needed
    if torch.is_tensor(engine_speed):
        engine_speed = engine_speed.detach().numpy()
    if torch.is_tensor(throttle_pos):
        throttle_pos = throttle_pos.detach().numpy()
    if torch.is_tensor(flow_coeff):
        flow_coeff = flow_coeff.detach().numpy()
    if torch.is_tensor(pressure_loss_coeff):
        pressure_loss_coeff = pressure_loss_coeff.detach().numpy()
    if torch.is_tensor(thermal_cond):
        thermal_cond = thermal_cond.detach().numpy()
    if torch.is_tensor(manifold_vol):
        manifold_vol = manifold_vol.detach().numpy()
    
    # Air density calculation (simplified)
    air_density = 1.225 * (ambient_pressure / 101.325) * (288.15 / (ambient_temp + 273.15))
    
    # Mass flow rate through throttle body (simplified orifice equation)
    # Q = Cd * A * sqrt(2 * ΔP / ρ)
    throttle_area = throttle_pos * 0.02  # Simplified throttle area (m²)
    pressure_drop = 10.0 * throttle_pos**2  # Simplified pressure drop (kPa)
    
    mass_flow_throttle = flow_coeff * throttle_area * np.sqrt(2 * pressure_drop * 1000 / air_density)
    
    # Pressure loss in intake manifold (simplified)
    manifold_pressure_loss = pressure_loss_coeff * mass_flow_throttle**2
    
    # Heat transfer effects (simplified)
    temp_rise = thermal_cond * mass_flow_throttle * 0.1  # Simplified temperature rise
    
    # Final manifold pressure
    manifold_pressure = ambient_pressure - manifold_pressure_loss
    
    # Air mass in manifold (ideal gas law approximation)
    manifold_temp = ambient_temp + temp_rise
    manifold_density = air_density * (manifold_pressure / ambient_pressure) * (ambient_temp + 273.15) / (manifold_temp + 273.15)
    manifold_mass = manifold_density * manifold_vol / 1000  # Convert liters to m³
    
    # Volumetric efficiency (simplified)
    volumetric_efficiency = 0.8 + 0.2 * throttle_pos - 0.1 * (engine_speed / 4000)
    volumetric_efficiency = np.clip(volumetric_efficiency, 0.3, 0.95)
    
    # Final air flow rate
    air_flow_rate = mass_flow_throttle * volumetric_efficiency
    
    return torch.tensor(air_flow_rate, dtype=torch.float32, device=dev)

# Generate synthetic data with ground truth parameters
st.subheader("📈 Data Generation")
with st.spinner("Generating synthetic air flow data..."):
    # Generate true air flow rates
    true_air_flow = []
    for i in range(len(engine_speed_data)):
        flow = airflow_model(
            engine_speed_data[i], 
            throttle_data[i],
            true_params['flow_coefficient'],
            true_params['pressure_loss_coeff'],
            true_params['thermal_conductivity'],
            true_params['manifold_volume']
        )
        true_air_flow.append(flow.item())

    true_air_flow = np.array(true_air_flow)

    # Add measurement noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, true_params['noise_std'], len(true_air_flow))
    observed_air_flow = true_air_flow + noise

    st.success(f"✅ Generated {len(observed_air_flow)} data points")
    st.write(f"**Air flow range:** {observed_air_flow.min():.3f} - {observed_air_flow.max():.3f} kg/s")
    st.write(f"**Noise level:** {true_params['noise_std']:.3f} kg/s")

# Bayesian Model for Parameter Estimation
# =======================================

def airflow_bayesian_model(engine_speed_data, throttle_data, observed_flow):
    """
    Bayesian model for estimating air flow parameters
    """
    # Prior distributions for calibratable parameters
    flow_coeff = pyro.sample("flow_coefficient", dist.Normal(torch.tensor(0.8, device=dev), torch.tensor(0.2, device=dev)))
    pressure_loss_coeff = pyro.sample("pressure_loss_coeff", dist.Normal(torch.tensor(0.1, device=dev), torch.tensor(0.05, device=dev)))
    thermal_cond = pyro.sample("thermal_conductivity", dist.Normal(torch.tensor(0.2, device=dev), torch.tensor(0.1, device=dev)))
    manifold_vol = pyro.sample("manifold_volume", dist.Normal(torch.tensor(2.0, device=dev), torch.tensor(0.5, device=dev)))
    
    # Noise parameter
    noise_std = pyro.sample("noise_std", dist.LogNormal(torch.tensor(-2.0, device=dev), torch.tensor(0.5, device=dev)))
    
    # Convert inputs to tensors
    engine_speed_tensor = torch.tensor(engine_speed_data, dtype=torch.float32, device=dev)
    throttle_tensor = torch.tensor(throttle_data, dtype=torch.float32, device=dev)
    observed_tensor = torch.tensor(observed_flow, dtype=torch.float32, device=dev)
    
    # Compute predicted air flow for each data point
    predicted_flow = []
    for i in range(len(engine_speed_data)):
        pred = airflow_model(
            engine_speed_tensor[i], 
            throttle_tensor[i],
            flow_coeff,
            pressure_loss_coeff,
            thermal_cond,
            manifold_vol
        )
        predicted_flow.append(pred)
    
    predicted_flow_tensor = torch.stack(predicted_flow)
    
    # Likelihood
    with pyro.plate("data", len(observed_flow)):
        pyro.sample("obs", dist.Normal(predicted_flow_tensor, noise_std), obs=observed_tensor)

# Create guide
guide = AutoMultivariateNormal(airflow_bayesian_model, init_loc_fn=init_to_mean)

st.subheader("🧠 Bayesian Model")
st.write("**Parameters to estimate:**")
st.write("- **flow_coefficient:** Flow coefficient through throttle body")
st.write("- **pressure_loss_coeff:** Pressure loss coefficient in intake manifold") 
st.write("- **thermal_conductivity:** Heat transfer coefficient")
st.write("- **manifold_volume:** Intake manifold volume (liters)")
st.write("- **noise_std:** Measurement noise standard deviation")

st.markdown("---")

# SVI Training Section
st.subheader("SVI Training")

# Training parameters
col1, col2, col3 = st.columns(3)
with col1:
    num_steps = st.slider("Number of Steps", min_value=50, max_value=1000, value=200, step=50)
with col2:
    learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.01, value=0.005, step=0.001)
with col3:
    num_particles = st.slider("Number of Particles", min_value=50, max_value=200, value=100, step=25)

if st.button("Run SVI Training", type="primary"):
    
    # Initialize SVI
    svi = SVI(airflow_bayesian_model, guide, optim.Adam({"lr": learning_rate}), 
            loss=Trace_ELBO(num_particles=num_particles, vectorize_particles=True))

    # Clear parameter store
    pyro.clear_param_store()

    # Training loop
    losses = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    st.write("Starting SVI training for air flow parameter estimation...")
    
    for step in range(num_steps):
        elbo = svi.step(engine_speed_data, throttle_data, observed_air_flow)
        losses.append(elbo)
        
        # Update progress
        progress = (step + 1) / num_steps
        progress_bar.progress(progress)
        status_text.text(f"Step {step + 1}/{num_steps}: ELBO = {elbo:.4f}")
        
        if step % 20 == 0:
            st.write(f"Step {step}: ELBO = {elbo:.4f}")

    progress_bar.empty()
    status_text.empty()
    
    st.success("SVI Training Completed!")
    
    # Display final results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final ELBO", f"{losses[-1]:.4f}")
    with col2:
        st.metric("ELBO Improvement", f"{losses[0] - losses[-1]:.4f}")
    
    # Create Plotly ELBO convergence plots
    st.subheader("ELBO Convergence Analysis")
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Full ELBO Convergence", "Last 100 Steps Detail"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Full convergence plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(losses))),
            y=losses,
            mode='lines',
            name='ELBO',
            line=dict(color='blue', width=2),
            hovertemplate='Step: %{x}<br>ELBO: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Last 100 steps detail
    last_steps = min(100, len(losses))
    fig.add_trace(
        go.Scatter(
            x=list(range(len(losses) - last_steps, len(losses))),
            y=losses[-last_steps:],
            mode='lines',
            name='ELBO (Last 100)',
            line=dict(color='red', width=2),
            hovertemplate='Step: %{x}<br>ELBO: %{y:.4f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        title_text="ELBO Convergence During SVI Training",
        title_x=0.5,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_yaxes(title_text="ELBO", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_yaxes(title_text="ELBO", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional analysis plots
    st.subheader("Training Analysis")
    
    # ELBO improvement over time
    elbo_improvement = [losses[0] - loss for loss in losses]
    
    fig_improvement = go.Figure()
    fig_improvement.add_trace(
        go.Scatter(
            x=list(range(len(elbo_improvement))),
            y=elbo_improvement,
            mode='lines',
            name='ELBO Improvement',
            line=dict(color='green', width=2),
            hovertemplate='Step: %{x}<br>Improvement: %{y:.4f}<extra></extra>'
        )
    )
    
    fig_improvement.update_layout(
        title="ELBO Improvement Over Time",
        xaxis_title="Step",
        yaxis_title="ELBO Improvement",
        height=400
    )
    
    st.plotly_chart(fig_improvement, use_container_width=True)
    
    # Training statistics
    st.subheader("Training Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial ELBO", f"{losses[0]:.4f}")
    with col2:
        st.metric("Final ELBO", f"{losses[-1]:.4f}")
    with col3:
        st.metric("Total Improvement", f"{losses[0] - losses[-1]:.4f}")
    with col4:
        st.metric("Convergence Rate", f"{(losses[0] - losses[-1])/len(losses):.6f}")
    
    # Parameter estimation results
    st.subheader("Parameter Estimation Results")
    
    # Get estimated parameters
    estimated_params = {}
    st.write("**Extracted Parameters:**")
    for name, value in pyro.get_param_store().items():
        if isinstance(value, torch.Tensor):
            # Handle different tensor shapes
            if value.numel() == 1:
                # Single element tensor - convert to scalar
                estimated_params[name] = value.item()
                st.write(f"- {name}: {value.item():.4f} (scalar)")
            elif value.numel() > 1:
                # Multi-element tensor - take the mean or first element
                estimated_params[name] = value.mean().item()
                st.write(f"- {name}: {value.mean().item():.4f} (mean of {value.numel()} elements)")
            else:
                # Empty tensor
                estimated_params[name] = 0.0
                st.write(f"- {name}: 0.0 (empty tensor)")
        else:
            estimated_params[name] = value
            st.write(f"- {name}: {value} (non-tensor)")
    
    # Show what we're looking for vs what we found
    st.write("**Parameter Matching:**")
    st.write(f"Looking for: {list(true_params.keys())}")
    st.write(f"Found in Pyro store: {list(estimated_params.keys())}")
    
    # Check for missing parameters
    missing_params = set(true_params.keys()) - set(estimated_params.keys())
    if missing_params:
        st.warning(f"⚠️ Missing parameters: {missing_params}")
    else:
        st.success("✅ All parameters found!")
    
    # Create comparison table
    comparison_data = []
    for param_name in true_params.keys():
        true_val = true_params[param_name]
        est_val = estimated_params.get(param_name, None)
        if est_val is not None and est_val != "N/A":
            error = abs(true_val - est_val) / true_val * 100
            comparison_data.append({
                "Parameter": param_name,
                "True Value": true_val,
                "Estimated Value": est_val,
                "Error %": f"{error:.2f}%"
            })
        else:
            comparison_data.append({
                "Parameter": param_name,
                "True Value": true_val,
                "Estimated Value": "Not Found",
                "Error %": "N/A"
            })
    
    if comparison_data:
        import pandas as pd
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Parameter comparison visualization
        fig_params = go.Figure()
        
        params = [row["Parameter"] for row in comparison_data]
        true_values = [row["True Value"] for row in comparison_data]
        est_values = [row["Estimated Value"] for row in comparison_data]
        
        fig_params.add_trace(go.Bar(
            name='True Values',
            x=params,
            y=true_values,
            marker_color='blue',
            opacity=0.7
        ))
        
        fig_params.add_trace(go.Bar(
            name='Estimated Values',
            x=params,
            y=est_values,
            marker_color='red',
            opacity=0.7
        ))
        
        fig_params.update_layout(
            title="Parameter Estimation Comparison",
            xaxis_title="Parameters",
            yaxis_title="Values",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_params, use_container_width=True)




    # Visualization of Results using Plotly
    # =====================================
    
    st.subheader("📊 Model Performance Visualization")
    
    # Generate predictions using estimated parameters
    predicted_flow_estimated = []
    for i in range(len(engine_speed_data)):
        pred = airflow_model(
            engine_speed_data[i], 
            throttle_data[i],
            estimated_params.get('flow_coefficient', 0.8),
            estimated_params.get('pressure_loss_coeff', 0.1),
            estimated_params.get('thermal_conductivity', 0.2),
            estimated_params.get('manifold_volume', 2.0)
        )
        predicted_flow_estimated.append(pred.item())

    predicted_flow_estimated = np.array(predicted_flow_estimated)

    # Create comprehensive visualization using Plotly subplots
    fig_results = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Air Flow vs Engine Speed", 
            "Air Flow vs Throttle Position",
            "Parameter Comparison", 
            "Residuals Plot"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # 1. Air flow vs Engine Speed
    fig_results.add_trace(
        go.Scatter(
            x=engine_speed_data,
            y=observed_air_flow,
            mode='markers',
            name='Observed',
            marker=dict(color='blue', size=8, opacity=0.6),
            hovertemplate='Engine Speed: %{x} RPM<br>Air Flow: %{y:.3f} kg/s<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig_results.add_trace(
        go.Scatter(
            x=engine_speed_data,
            y=true_air_flow,
            mode='markers',
            name='True',
            marker=dict(color='green', size=8, opacity=0.6),
            hovertemplate='Engine Speed: %{x} RPM<br>Air Flow: %{y:.3f} kg/s<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig_results.add_trace(
        go.Scatter(
            x=engine_speed_data,
            y=predicted_flow_estimated,
            mode='markers',
            name='Estimated',
            marker=dict(color='red', size=8, opacity=0.6),
            hovertemplate='Engine Speed: %{x} RPM<br>Air Flow: %{y:.3f} kg/s<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. Air flow vs Throttle Position
    fig_results.add_trace(
        go.Scatter(
            x=throttle_data,
            y=observed_air_flow,
            mode='markers',
            name='Observed',
            marker=dict(color='blue', size=8, opacity=0.6),
            hovertemplate='Throttle: %{x:.2f}<br>Air Flow: %{y:.3f} kg/s<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig_results.add_trace(
        go.Scatter(
            x=throttle_data,
            y=true_air_flow,
            mode='markers',
            name='True',
            marker=dict(color='green', size=8, opacity=0.6),
            hovertemplate='Throttle: %{x:.2f}<br>Air Flow: %{y:.3f} kg/s<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig_results.add_trace(
        go.Scatter(
            x=throttle_data,
            y=predicted_flow_estimated,
            mode='markers',
            name='Estimated',
            marker=dict(color='red', size=8, opacity=0.6),
            hovertemplate='Throttle: %{x:.2f}<br>Air Flow: %{y:.3f} kg/s<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )

    # 3. Parameter comparison
    param_names = list(true_params.keys())
    true_values = [true_params[p] for p in param_names]
    est_values = [estimated_params.get(p, 0.0) for p in param_names]

    fig_results.add_trace(
        go.Bar(
            name='True',
            x=param_names,
            y=true_values,
            marker_color='green',
            opacity=0.7,
            hovertemplate='Parameter: %{x}<br>True Value: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig_results.add_trace(
        go.Bar(
            name='Estimated',
            x=param_names,
            y=est_values,
            marker_color='red',
            opacity=0.7,
            hovertemplate='Parameter: %{x}<br>Estimated Value: %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )

    # 4. Residuals
    residuals_true = observed_air_flow - true_air_flow
    residuals_estimated = observed_air_flow - predicted_flow_estimated

    fig_results.add_trace(
        go.Scatter(
            x=true_air_flow,
            y=residuals_true,
            mode='markers',
            name='True Model',
            marker=dict(color='green', size=8, opacity=0.6),
            hovertemplate='Predicted: %{x:.3f} kg/s<br>Residual: %{y:.3f} kg/s<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig_results.add_trace(
        go.Scatter(
            x=predicted_flow_estimated,
            y=residuals_estimated,
            mode='markers',
            name='Estimated Model',
            marker=dict(color='red', size=8, opacity=0.6),
            hovertemplate='Predicted: %{x:.3f} kg/s<br>Residual: %{y:.3f} kg/s<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )

    # Add horizontal line at y=0 for residuals plot
    fig_results.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=2)

    # Update layout
    fig_results.update_layout(
        height=800,
        title_text="Comprehensive Model Performance Analysis",
        title_x=0.5,
        showlegend=True
    )

    # Update axes labels
    fig_results.update_xaxes(title_text="Engine Speed (RPM)", row=1, col=1)
    fig_results.update_yaxes(title_text="Air Flow Rate (kg/s)", row=1, col=1)
    fig_results.update_xaxes(title_text="Throttle Position", row=1, col=2)
    fig_results.update_yaxes(title_text="Air Flow Rate (kg/s)", row=1, col=2)
    fig_results.update_xaxes(title_text="Parameters", row=2, col=1)
    fig_results.update_yaxes(title_text="Parameter Value", row=2, col=1)
    fig_results.update_xaxes(title_text="Predicted Air Flow (kg/s)", row=2, col=2)
    fig_results.update_yaxes(title_text="Residuals (kg/s)", row=2, col=2)

    st.plotly_chart(fig_results, use_container_width=True)

    # Calculate and display performance metrics
    st.subheader("📈 Model Performance Metrics")
    
    r2_true = r2_score(observed_air_flow, true_air_flow)
    r2_estimated = r2_score(observed_air_flow, predicted_flow_estimated)
    rmse_true = np.sqrt(np.mean((observed_air_flow - true_air_flow)**2))
    rmse_estimated = np.sqrt(np.mean((observed_air_flow - predicted_flow_estimated)**2))

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score (True Model)", f"{r2_true:.4f}")
    with col2:
        st.metric("R² Score (Estimated Model)", f"{r2_estimated:.4f}")
    with col3:
        st.metric("RMSE (True Model)", f"{rmse_true:.4f}")
    with col4:
        st.metric("RMSE (Estimated Model)", f"{rmse_estimated:.4f}")

    # Performance comparison chart
    fig_performance = go.Figure()
    
    models = ['True Model', 'Estimated Model']
    r2_scores = [r2_true, r2_estimated]
    rmse_scores = [rmse_true, rmse_estimated]
    
    fig_performance.add_trace(go.Bar(
        name='R² Score',
        x=models,
        y=r2_scores,
        marker_color=['green', 'red'],
        opacity=0.7,
        hovertemplate='Model: %{x}<br>R² Score: %{y:.4f}<extra></extra>'
    ))
    
    fig_performance.add_trace(go.Bar(
        name='RMSE',
        x=models,
        y=rmse_scores,
        marker_color=['darkgreen', 'darkred'],
        opacity=0.7,
        hovertemplate='Model: %{x}<br>RMSE: %{y:.4f}<extra></extra>',
        yaxis='y2'
    ))
    
    fig_performance.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis=dict(title="R² Score", side="left"),
        yaxis2=dict(title="RMSE", side="right", overlaying="y"),
        height=400,
        barmode='group'
    )
    
    st.plotly_chart(fig_performance, use_container_width=True)