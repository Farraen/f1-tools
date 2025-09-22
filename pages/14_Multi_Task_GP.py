import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
import warnings
from PIL import Image


warnings.filterwarnings("ignore")

class SharedLatentMultiTaskGP:
    """
    Multi-Task GP with shared latent function using scikit-learn
    Implements the core concept of shared latent functions without GPyTorch
    """
    
    def __init__(self, latent_dim=2):
        self.latent_dim = latent_dim
        self.shared_latent_gp = None  # GP for shared latent function
        self.task_transforms = {}     # Task-specific transformations
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_latent = StandardScaler()
        self.trained = False
        self.latent_representations = {}
        self.task_correlations = {}
        
    def create_kernel(self, length_scale=1.0, noise_level=0.1, kernel_type='rbf'):
        """Create a composite kernel for the GP"""
        if kernel_type == 'rbf':
            base_kernel = RBF(length_scale=length_scale)
        elif kernel_type == 'matern':
            base_kernel = Matern(length_scale=length_scale, nu=2.5)
        else:
            base_kernel = RBF(length_scale=length_scale)
            
        kernel = (ConstantKernel(1.0) * base_kernel + 
                 WhiteKernel(noise_level=noise_level))
        return kernel
    
    def learn_shared_latent_representation(self, X_a, y_a, X_b, y_b):
        """
        Learn a shared latent representation using PCA on combined outputs
        This simulates the shared latent function concept
        """
        # Combine outputs from both tasks
        y_combined = np.vstack([y_a, y_b])
        
        # Use PCA to find shared latent space
        pca = PCA(n_components=self.latent_dim)
        latent_combined = pca.fit_transform(y_combined)
        
        # Split back to individual tasks
        n_a = len(y_a)
        latent_a = latent_combined[:n_a]
        latent_b = latent_combined[n_a:]
        
        # Store latent representations
        self.latent_representations = {
            'A': latent_a,
            'B': latent_b,
            'pca': pca
        }
        
        # Compute task correlation in latent space
        # Use a different approach since arrays have different lengths
        if latent_a.shape[1] > 0 and latent_b.shape[1] > 0:
            # Compute correlation using mean values of latent dimensions
            mean_latent_a = np.mean(latent_a, axis=0)
            mean_latent_b = np.mean(latent_b, axis=0)
            correlation = np.corrcoef(mean_latent_a, mean_latent_b)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        self.task_correlations['latent'] = correlation
        
        return latent_a, latent_b, pca
    
    def learn_task_transforms(self, X_a, y_a, X_b, y_b, latent_a, latent_b):
        """
        Learn task-specific transformations from latent space to output space
        """
        # Scale latent representations
        latent_a_scaled = self.scaler_latent.fit_transform(latent_a)
        latent_b_scaled = self.scaler_latent.transform(latent_b)
        
        # Learn transformation for Task A (Engine A)
        transform_a = GaussianProcessRegressor(
            kernel=self.create_kernel(length_scale=1.0, noise_level=0.01),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3
        )
        transform_a.fit(latent_a_scaled, y_a)
        
        # Learn transformation for Task B (Engine B)
        transform_b = GaussianProcessRegressor(
            kernel=self.create_kernel(length_scale=1.0, noise_level=0.02),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3
        )
        transform_b.fit(latent_b_scaled, y_b)
        
        self.task_transforms = {
            'A': transform_a,
            'B': transform_b
        }
        
        return transform_a, transform_b
    
    def train_shared_latent_gp(self, X_a, y_a, X_b, y_b):
        """
        Train the shared latent function GP
        This learns the mapping from input space to shared latent space
        """
        # Combine input data
        X_combined = np.vstack([X_a, X_b])
        
        # Learn shared latent representation
        latent_a, latent_b, pca = self.learn_shared_latent_representation(X_a, y_a, X_b, y_b)
        
        # Scale inputs
        X_combined_scaled = self.scaler_X.fit_transform(X_combined)
        
        # Scale latent representations
        latent_combined = np.vstack([latent_a, latent_b])
        latent_combined_scaled = self.scaler_latent.fit_transform(latent_combined)
        
        # Train shared latent GP
        self.shared_latent_gp = GaussianProcessRegressor(
            kernel=self.create_kernel(length_scale=1.0, noise_level=0.01),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        self.shared_latent_gp.fit(X_combined_scaled, latent_combined_scaled)
        
        # Learn task-specific transformations
        self.learn_task_transforms(X_a, y_a, X_b, y_b, latent_a, latent_b)
        
        return self.shared_latent_gp
    
    def predict_shared_latent(self, X_new):
        """Predict shared latent representation for new inputs"""
        if self.shared_latent_gp is None:
            raise ValueError("Shared latent GP not trained yet")
        
        X_new_scaled = self.scaler_X.transform(X_new)
        latent_pred_scaled, latent_std_scaled = self.shared_latent_gp.predict(X_new_scaled, return_std=True)
        
        # Transform back to original latent space
        latent_pred = self.scaler_latent.inverse_transform(latent_pred_scaled)
        latent_std = latent_std_scaled * self.scaler_latent.scale_
        
        return latent_pred, latent_std
    
    def predict_task(self, X_new, task_name):
        """Predict output for a specific task using shared latent function"""
        if task_name not in self.task_transforms:
            raise ValueError(f"Task {task_name} not found")
        
        # Get shared latent representation
        latent_pred, latent_std = self.predict_shared_latent(X_new)
        
        # Scale latent representation
        latent_pred_scaled = self.scaler_latent.transform(latent_pred)
        
        # Transform through task-specific GP
        task_transform = self.task_transforms[task_name]
        y_pred, y_std = task_transform.predict(latent_pred_scaled, return_std=True)
        
        # Adjust uncertainty based on latent uncertainty
        y_std = y_std + latent_std.mean(axis=1, keepdims=True) * 0.1
        
        return y_pred, y_std
    
    def predict_engine_a(self, X_new):
        """Predict using Engine A (Task A)"""
        return self.predict_task(X_new, 'A')
    
    def predict_engine_b(self, X_new):
        """Predict using Engine B (Task B)"""
        return self.predict_task(X_new, 'B')
    
    def create_pmax_sweep(self, X_base, variable_idx, sweep_range, n_points=50):
        """Create a 2D sweep of pMAX responses for both engines"""
        # Create sweep values
        sweep_values = np.linspace(sweep_range[0], sweep_range[1], n_points)
        
        # Create input matrix for sweep
        X_sweep = np.tile(X_base, (n_points, 1))
        X_sweep[:, variable_idx] = sweep_values
        
        # Get predictions from both engines
        y_pred_a, y_std_a = self.predict_engine_a(X_sweep)
        y_pred_b, y_std_b = self.predict_engine_b(X_sweep)
        
        # Extract pMAX predictions (assuming it's the 4th output, index 3)
        pmax_a = y_pred_a[:, 3]  # pMAX is at index 3
        pmax_std_a = y_std_a[:, 3]
        pmax_b = y_pred_b[:, 3]
        pmax_std_b = y_std_b[:, 3]
        
        return {
            'sweep_values': sweep_values,
            'pmax_a': pmax_a,
            'pmax_std_a': pmax_std_a,
            'pmax_b': pmax_b,
            'pmax_std_b': pmax_std_b
        }
    
    def get_training_points_for_variable(self, X_train, y_train, variable_idx):
        """Get training points for a specific variable sweep"""
        var_values = X_train[:, variable_idx]
        pmax_values = y_train[:, 3]  # pMAX is at index 3
        return var_values, pmax_values
    
    def get_latent_visualization(self, X_a, y_a, X_b, y_b):
        """Get latent space visualization for both tasks"""
        if 'A' not in self.latent_representations:
            return None
        
        latent_a = self.latent_representations['A']
        latent_b = self.latent_representations['B']
        
        # Ensure we have at least 2 dimensions for visualization
        if latent_a.shape[1] < 2:
            # If only 1 dimension, duplicate it for 2D visualization
            latent_a_viz = np.column_stack([latent_a[:, 0], latent_a[:, 0]])
            latent_b_viz = np.column_stack([latent_b[:, 0], latent_b[:, 0]])
        else:
            latent_a_viz = latent_a[:, :2]
            latent_b_viz = latent_b[:, :2]
        
        return {
            'latent_a': latent_a_viz,
            'latent_b': latent_b_viz,
            'correlation': self.task_correlations.get('latent', 0)
        }


# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

@st.cache_data
def load_images_once_1():
    image1 = read_image("images/mtgp.png")

    return image1

@st.cache_data
def load_images_once_2():
    image1 = read_image("images/mtpg_2.png")

    return image1

def create_shared_latent_gp_demo():
    """Create the shared latent multi-task GP demonstration interface"""
    
    st.title("Multi-Task Gaussian Process Engine Modeling")

    st.write('Multi-task GP is a framework to allow GPs to share covariance matrix. This enables transfer learning from ' \
    'one GP to another GP model. The application would be to train a GP model with sparse training data from another GP model '\
    'which has higher fidelity. This example would use this framework to train a GP model from another engine GP model. ' \
    'The problem that we want to solve is how we can built a statistical model out of sparse training data and use historical engine data from a similar engine.')
    
    col1, col2 = st.columns(2)
    image1 = load_images_once_1()
    col1.image(image1)

    image1 = load_images_once_2()
    col2.image(image1)

    st.write("Ref: https://www.researchgate.net/publication/257618558_Focused_multi-task_learning_in_a_Gaussian_process_framework")

    st.markdown("---")
    st.info("🧠 **True Multi-Task Learning**: Shared latent function with task-specific transformations!")
    


    # Load the virtual engine model
    try:
        model_virtual_engine = CatBoostRegressor()
        model_virtual_engine.load_model("virtual_engine")
        st.success("✅ Virtual engine model loaded successfully")
    except:
        st.error("❌ Could not load virtual engine model. Please ensure 'virtual_engine' file exists.")
        return
    
    # Initialize shared latent multi-task GP
    if 'shared_latent_gp' not in st.session_state:
        st.session_state.shared_latent_gp = SharedLatentMultiTaskGP(latent_dim=2)
    
    # Main controls
    col1, col2, col3 = st.columns([1, 1, 1], gap='large')
    
    with col1:
        st.write("**GP Training Controls**")
        n_full_samples = st.slider("Full dataset size (Engine A)", 50, 500, 200, key="sl_full")
        n_sparse_samples = st.slider("Sparse dataset size (Engine B)", 10, 100, 30, key="sl_sparse")
        
        # Latent dimension control
        latent_dim = st.slider("Latent Dimension", 1, 5, 2, key="sl_latent")
        if latent_dim != st.session_state.shared_latent_gp.latent_dim:
            st.session_state.shared_latent_gp = SharedLatentMultiTaskGP(latent_dim=latent_dim)
        
        if st.button("Generate Training Data"):
            with st.spinner("Generating training data..."):
                # Generate full dataset
                np.random.seed(42)
                X_full = np.random.uniform(
                    low=[1700, 54, 0, 0.05],
                    high=[1890, 990, 84, 0.34],
                    size=(n_full_samples, 4)
                )
                
                # Generate sparse dataset (subset of full)
                sparse_indices = np.random.choice(n_full_samples, n_sparse_samples, replace=False)
                X_sparse = X_full[sparse_indices]
                
                # Get predictions from virtual engine
                y_full = model_virtual_engine.predict(X_full)
                y_sparse = y_full[sparse_indices]
                
                # Store in session state
                st.session_state.X_full_sl = X_full
                st.session_state.y_full_sl = y_full
                st.session_state.X_sparse_sl = X_sparse
                st.session_state.y_sparse_sl = y_sparse
                
                st.success(f"✅ Generated {n_full_samples} full samples and {n_sparse_samples} sparse samples")
    
    with col2:
        st.write("**Train Shared Latent Multi-Task GP**")
        
        # Engine B scaling factor
        scale_factor = st.slider(
            "Engine B pMAX Scale Factor", 
            0.5, 2.0, 1.2, 0.1,
            help="Engine B pMAX values will be multiplied by this factor",
            key="sl_scale"
        )
        
        if st.button("Train Shared Latent Multi-Task GP") and 'X_full_sl' in st.session_state:
            with st.spinner("Training Shared Latent Multi-Task GP..."):
                # Scale Engine B data
                y_sparse_scaled = st.session_state.y_sparse_sl.copy()
                y_sparse_scaled[:, 3] = y_sparse_scaled[:, 3] * scale_factor
                
                # Train shared latent multi-task GP
                st.session_state.shared_latent_gp.train_shared_latent_gp(
                    st.session_state.X_full_sl, 
                    st.session_state.y_full_sl,
                    st.session_state.X_sparse_sl, 
                    y_sparse_scaled
                )
                
                st.session_state.shared_latent_gp.trained = True
                
                # Display latent space information
                if hasattr(st.session_state.shared_latent_gp, 'task_correlations'):
                    correlation = st.session_state.shared_latent_gp.task_correlations.get('latent', 0)
                    st.info(f"🔗 **Latent Space Correlation**: {correlation:.3f}")
                
                st.success("✅ Shared Latent Multi-Task GP trained successfully!")
                st.info(f"📈 Engine B pMAX scaled by {scale_factor:.1f}x ({(scale_factor-1)*100:.0f}% {'higher' if scale_factor > 1 else 'lower'})")
    
    with col3:
        st.write("**Analysis Controls**")
        if st.session_state.shared_latent_gp.trained:
            st.write("**Base Input Values:**")
            base_speed = st.slider("SPEED_A", 1700, 1890, 1800, key="sl_speed")
            base_torque = st.slider("TORQUE_R", 54, 990, 500, key="sl_torque")
            base_vgt = st.slider("BobOV_XPC_VGT", 0, 84, 40, key="sl_vgt")
            base_egr = st.slider("egr_vlv_position", 0.05, 0.34, 0.15, key="sl_egr")
            
            base_inputs = np.array([base_speed, base_torque, base_vgt, base_egr])
            
            # Variable to sweep
            variable_names = ["SPEED_A", "TORQUE_R", "BobOV_XPC_VGT", "egr_vlv_position"]
            sweep_variable = st.selectbox("Variable to sweep:", variable_names, key="sl_sweep_var")
            variable_idx = variable_names.index(sweep_variable)
            
            # Sweep range
            var_ranges = [(1700, 1890), (54, 990), (0, 84), (0.05, 0.34)]
            var_range = var_ranges[variable_idx]
            sweep_min = st.slider(f"Min {sweep_variable}", var_range[0], var_range[1], var_range[0], key="sl_min")
            sweep_max = st.slider(f"Max {sweep_variable}", var_range[0], var_range[1], var_range[1], key="sl_max")
            
            if st.button("Generate pMAX Sweep"):
                with st.spinner("Generating pMAX response sweep..."):
                    # Create sweep data
                    sweep_data = st.session_state.shared_latent_gp.create_pmax_sweep(
                        base_inputs, 
                        variable_idx, 
                        (sweep_min, sweep_max),
                        n_points=100
                    )
                    
                    # Get training points for the selected variable
                    train_var_a, train_pmax_a = st.session_state.shared_latent_gp.get_training_points_for_variable(
                        st.session_state.X_full_sl, 
                        st.session_state.y_full_sl, 
                        variable_idx
                    )
                    train_var_b, train_pmax_b = st.session_state.shared_latent_gp.get_training_points_for_variable(
                        st.session_state.X_sparse_sl, 
                        st.session_state.y_sparse_sl, 
                        variable_idx
                    )
                    
                    # Store sweep data in session state for plotting
                    st.session_state.sl_sweep_data = sweep_data
                    st.session_state.sl_base_inputs = base_inputs
                    st.session_state.sl_sweep_variable = sweep_variable
                    st.session_state.sl_train_var_a = train_var_a
                    st.session_state.sl_train_pmax_a = train_pmax_a
                    st.session_state.sl_train_var_b = train_var_b
                    st.session_state.sl_train_pmax_b = train_pmax_b
                    
                    st.success("✅ Sweep data generated! View plots below.")
        else:
            st.info("👆 Please train the Shared Latent Multi-Task GP first.")
    
    # Visualization section
    if st.session_state.shared_latent_gp.trained and 'sl_sweep_data' in st.session_state:
        st.markdown("---")
        st.subheader("Shared Latent Multi-Task GP pMAX Response Analysis")
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Engine A pMAX Range", 
                     f"{st.session_state.sl_sweep_data['pmax_a'].min():.1f} - {st.session_state.sl_sweep_data['pmax_a'].max():.1f}")
        with col2:
            st.metric("Engine B pMAX Range", 
                     f"{st.session_state.sl_sweep_data['pmax_b'].min():.1f} - {st.session_state.sl_sweep_data['pmax_b'].max():.1f}")
        with col3:
            rmse = np.sqrt(np.mean((st.session_state.sl_sweep_data['pmax_a'] - st.session_state.sl_sweep_data['pmax_b'])**2))
            st.metric("RMSE between engines", f"{rmse:.1f}")
        with col4:
            if hasattr(st.session_state.shared_latent_gp, 'task_correlations'):
                correlation = st.session_state.shared_latent_gp.task_correlations.get('latent', 0)
                st.metric("Latent Correlation", f"{correlation:.3f}")
        
        # Two separate plots side by side
        col1, col2 = st.columns(2, gap='large')
        
        with col1:
            st.write("**Engine A (Full Data)**")
            fig_a = go.Figure()
            
            # Engine A predictions
            fig_a.add_trace(go.Scatter(
                x=st.session_state.sl_sweep_data['sweep_values'],
                y=st.session_state.sl_sweep_data['pmax_a'],
                mode='lines',
                name='Engine A Prediction',
                line=dict(color='blue', width=3)
            ))
            
            # Engine A uncertainty
            fig_a.add_trace(go.Scatter(
                x=st.session_state.sl_sweep_data['sweep_values'],
                y=st.session_state.sl_sweep_data['pmax_a'] + 2*st.session_state.sl_sweep_data['pmax_std_a'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig_a.add_trace(go.Scatter(
                x=st.session_state.sl_sweep_data['sweep_values'],
                y=st.session_state.sl_sweep_data['pmax_a'] - 2*st.session_state.sl_sweep_data['pmax_std_a'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,255,0.3)',
                name='±2σ Uncertainty',
                hoverinfo='skip'
            ))
            
            # Engine A training points
            fig_a.add_trace(go.Scatter(
                x=st.session_state.sl_train_var_a,
                y=st.session_state.sl_train_pmax_a,
                mode='markers',
                name='Training Points',
                marker=dict(color='darkblue', size=8, symbol='circle'),
                opacity=0.7
            ))
            
            fig_a.update_layout(
                title=f'Engine A: pMAX vs {st.session_state.sl_sweep_variable}',
                xaxis_title=st.session_state.sl_sweep_variable,
                yaxis_title='pMAX',
                height=500
            )
            
            st.plotly_chart(fig_a, use_container_width=True)
        
        with col2:
            st.write("**Engine B (Sparse Data)**")
            fig_b = go.Figure()
            
            # Engine B predictions
            fig_b.add_trace(go.Scatter(
                x=st.session_state.sl_sweep_data['sweep_values'],
                y=st.session_state.sl_sweep_data['pmax_b'],
                mode='lines',
                name='Engine B Prediction',
                line=dict(color='red', width=3)
            ))
            
            # Engine B uncertainty
            fig_b.add_trace(go.Scatter(
                x=st.session_state.sl_sweep_data['sweep_values'],
                y=st.session_state.sl_sweep_data['pmax_b'] + 2*st.session_state.sl_sweep_data['pmax_std_b'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig_b.add_trace(go.Scatter(
                x=st.session_state.sl_sweep_data['sweep_values'],
                y=st.session_state.sl_sweep_data['pmax_b'] - 2*st.session_state.sl_sweep_data['pmax_std_b'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.3)',
                name='±2σ Uncertainty',
                hoverinfo='skip'
            ))
            
            # Engine B training points
            fig_b.add_trace(go.Scatter(
                x=st.session_state.sl_train_var_b,
                y=st.session_state.sl_train_pmax_b,
                mode='markers',
                name='Training Points',
                marker=dict(color='darkred', size=8, symbol='circle'),
                opacity=0.7
            ))
            
            fig_b.update_layout(
                title=f'Engine B: pMAX vs {st.session_state.sl_sweep_variable}',
                xaxis_title=st.session_state.sl_sweep_variable,
                yaxis_title='pMAX',
                height=500
            )
            
            st.plotly_chart(fig_b, use_container_width=True)
        
        # Combined comparison plot
        st.write("**Combined Comparison**")
        fig_combined = go.Figure()
        
        # Engine A predictions
        fig_combined.add_trace(go.Scatter(
            x=st.session_state.sl_sweep_data['sweep_values'],
            y=st.session_state.sl_sweep_data['pmax_a'],
            mode='lines',
            name='Engine A (Full Data)',
            line=dict(color='blue', width=2)
        ))
        
        # Engine B predictions
        fig_combined.add_trace(go.Scatter(
            x=st.session_state.sl_sweep_data['sweep_values'],
            y=st.session_state.sl_sweep_data['pmax_b'],
            mode='lines',
            name='Engine B (Sparse Data)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_combined.update_layout(
            title=f'Shared Latent Multi-Task GP: pMAX Response Comparison',
            xaxis_title=st.session_state.sl_sweep_variable,
            yaxis_title='pMAX',
            height=500
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # Latent space visualization
        st.write("**Latent Space Visualization**")
        latent_viz = st.session_state.shared_latent_gp.get_latent_visualization(
            st.session_state.X_full_sl, 
            st.session_state.y_full_sl,
            st.session_state.X_sparse_sl, 
            st.session_state.y_sparse_sl
        )
        
        if latent_viz is not None:
            fig_latent = go.Figure()
            
            # Engine A latent points
            fig_latent.add_trace(go.Scatter(
                x=latent_viz['latent_a'][:, 0],
                y=latent_viz['latent_a'][:, 1],
                mode='markers',
                name='Engine A Latent',
                marker=dict(color='blue', size=8, symbol='circle'),
                opacity=0.7
            ))
            
            # Engine B latent points
            fig_latent.add_trace(go.Scatter(
                x=latent_viz['latent_b'][:, 0],
                y=latent_viz['latent_b'][:, 1],
                mode='markers',
                name='Engine B Latent',
                marker=dict(color='red', size=8, symbol='square'),
                opacity=0.7
            ))
            
            fig_latent.update_layout(
                title=f'Shared Latent Space (Correlation: {latent_viz["correlation"]:.3f})',
                xaxis_title='Latent Dimension 1',
                yaxis_title='Latent Dimension 2',
                height=400
            )
            
            st.plotly_chart(fig_latent, use_container_width=True)
    
    else:
        st.info("👆 Please generate training data and train the Shared Latent Multi-Task GP first.")

if __name__ == "__main__":
    create_shared_latent_gp_demo()
