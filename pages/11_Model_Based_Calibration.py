import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import plotly.express as px
from PIL import Image
import plotly
from plotly.subplots import make_subplots
import datetime
import pickle
import statsmodels.api as sm
from doepy import build
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import time
import warnings
from scipy import interpolate
from scipy.optimize import minimize, LinearConstraint

warnings.filterwarnings("ignore")


st.set_page_config(layout="wide",initial_sidebar_state="collapsed")



st.markdown("""
<style>
.title_medium {
    font-size:20px !important;
}
 .text_small {
    font-size:12px !important;
}           

</style>

""", unsafe_allow_html=True)

def st_title(text):
    st.markdown(f'<p class="title_medium">{text}</p>', unsafe_allow_html=True)

def st_text(text):
    st.markdown(f'<p class="text_small">{text}</p>', unsafe_allow_html=True)


cols2 = plotly.colors.DEFAULT_PLOTLY_COLORS
cols = px.colors.qualitative.Light24

if 'doe' not in st.session_state:
    st.session_state.doe = pd.DataFrame([],columns=["SPEED_A","TORQUE_R","BobOV_XPC_VGT","egr_vlv_position"])

if 'test_result' not in st.session_state:
    st.session_state.test_result = []
    
if 'engine_model' not in st.session_state:
    engine_model = CatBoostRegressor()
    engine_model.load_model("engine_model")
    st.session_state.engine_model = engine_model



if 'doe_figure' not in st.session_state:
    st.session_state.doe_figure = []

if 'testing_figure' not in st.session_state:
    st.session_state.testing_figure = []
 
if 'sanity_figure' not in st.session_state:
    st.session_state.sanity_figure = []   
    
if 'model_mbc' not in st.session_state:
    st.session_state.model_mbc = []   
    
if 'accuracy_figure_1' not in st.session_state:
    st.session_state.accuracy_figure_1 = []   
    
if 'accuracy_figure_2' not in st.session_state:
    st.session_state.accuracy_figure_2 = []   
       
if 'optimise_figure_1' not in st.session_state:
    st.session_state.optimise_figure_1 = []   
    
if 'optimise_figure_2' not in st.session_state:
    st.session_state.optimise_figure_2 = []   
    
if 'variables' not in st.session_state:

    st.session_state.df_variables = pd.DataFrame(
        [
            {"Name": "SPEED_A", "Lower": 1700.0, "Upper": 1890, "Step":5},
            {"Name": "TORQUE_R", "Lower": 54.0, "Upper": 990.0, "Step":5},
            {"Name": "BobOV_XPC_VGT", "Lower": 0, "Upper": 84, "Step":5},
            {"Name": "egr_vlv_position", "Lower": 0.05, "Upper": 0.34, "Step":5},
        ]
    )

if 'bpx' not in st.session_state:
    nbpx = 5
    nbpy = 5
    df_variables = st.session_state.df_variables
    bpx = np.linspace(df_variables.iloc[0,1],df_variables.iloc[0,2],nbpx)
    bpy = np.linspace(df_variables.iloc[1,1],df_variables.iloc[1,2],nbpy)

    xm, ym = np.meshgrid(bpx, bpy) 
    xl = xm.flatten()
    yl = ym.flatten()

    st.session_state.bpx = bpx
    st.session_state.bpy = bpy

    st.session_state.xl = xl
    st.session_state.yl = yl
    
    st.session_state.z_vgt = np.full(np.shape(xl), 40)
    st.session_state.z_egr = np.full(np.shape(xl), 0.15)
    
    
    
    
df_outputs = pd.DataFrame(
        [
            {"Name": "BSFC_ind", "Lower": 0.0, "Upper": 1000.0, "Step":5},
            {"Name": "Air_Fuel_Ratio", "Lower": 0.0, "Upper": 1000.0, "Step":5},
            {"Name": "EXH_TEMP_MANI_OUT", "Lower": 0.0, "Upper": 1000.0, "Step":5},
            {"Name": "pMAX", "Lower": 0.0, "Upper": 1000.0, "Step":5},
            {"Name": "TC1_TURBO_SPEED", "Lower": 0.0, "Upper": 1000.0, "Step":5},
        ]
    )

outputs = ["BSFC_ind","Air_Fuel_Ratio","EXH_TEMP_MANI_OUT","pMAX","TC1_TURBO_SPEED"]


# --------------  Functions -----------------

# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image




# Initialisation 

# Load virtual engine
model_virtual_enging = CatBoostRegressor()
model_virtual_enging.load_model("virtual_engine")


    
# Dashboard

st.subheader("Model-based Calibration Methodology")

st.caption("Optimized for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")

with st.expander('Introduction',expanded=True):

    col1, col2 = st.columns([1,1],gap='large')
    
    with col1:
        longstr = '''
        
            This dashboard is designed to showcase the model-based calibration process, 
            providing a comprehensive view of its various stages. The process begins with 
            the creation of an experimental sequence, followed by the development of a 
            machine learning model, specifically a regression model, and concludes with 
            the optimization of calibration maps. The dashboard employs a virtual engine 
            to execute design of experiment test sequences, generating engine testing results 
            in the process. These results are then utilized to build a predictive model. 
            Subsequently, this model is employed by the calibration optimizer to determine 
            the precise parameters that ensure optimal engine performance. By integrating these 
            elements, the dashboard offers a detailed and interactive representation of the 
            calibration workflow, highlighting the interplay between experimental data, 
            machine learning, and optimization techniques in refining engine parameters.
            '''
        st.write(longstr)
        
        image = read_image("images/dyno.png")
        st.image(image,use_column_width=True)
        st.write("* Picture showing Farraen's PhD experiment setup for self-calibrating EMS using model-based methodologies at Loughborough University.")
    
        
    with col2:
        image = read_image("images/mbc.png")
        st.image(image,use_column_width=True)
        
        
with st.expander('Step 1: Generate DoE',expanded=True):

    st.subheader('Experiment design control panel')
    col1, col2 = st.columns([1,1],gap='large')  
    with col1:
        st.write('Set variables:')
        edited_df = st.data_editor(st.session_state.df_variables, num_rows="dynamic",use_container_width=True)
        
        param_dict = {}
        for row in edited_df.iterrows():
            param_dict[row[1]['Name']] = list(np.linspace(start=row[1]['Lower'], stop=row[1]['Upper'], num=row[1]['Step']))
        generate_doe = st.button('Generate design')
        
    with col2:
        design = st.radio(
            "Select a design",
            ["Full factorial", "Box design", "Latin hypercube sampling (LHS)","Sobol sequence"],
            index=3)
        
        num_samples = st.number_input("Number of samples (for LHS):",min_value=2, max_value=None, value=100)


    st.subheader('Generated design')


    col1, col2 = st.columns([1,1],gap='large')  

    with col1:
            
        st.dataframe(st.session_state.doe,use_container_width=True)


    with col2:
        
        st.write("View design")
        v_list = list(param_dict.keys())
         
        options = st.multiselect(
            "Choose up to three variables to display:",
            v_list,default = [v_list[0],v_list[1]],max_selections=3)
        
        doe = st.session_state.doe
        
        st_doe_plot = st.empty()
        if not st.session_state.doe_figure:
            fig = go.Figure()
            st_doe_plot.plotly_chart(fig, theme="streamlit",use_container_width=True)
        else:    
            st_doe_plot.plotly_chart(st.session_state.doe_figure, theme="streamlit",use_container_width=True)
            

col1, col2 = st.columns([1,1],gap='small')  
              
with col1.expander('Step 2: Sanity check',expanded=True):
    st.write("")
    col11, col22, col33 = st.columns([0.5,0.8,0.8],gap='small')  
    with col11:
        st.text("Inputs")
        for var in edited_df.iterrows(): 
            st.slider("-", var[1]['Lower'], var[1]['Upper'],label_visibility="collapsed",key=var[1]['Name'])
        
        
    with col22:
        image = read_image("images/engine.jpg")
        st.image(image,use_column_width=True)
        
    with col33:        
        st.text("Engine output")
        sanity_handle = st.empty()
        if not st.session_state.sanity_figure:
            fig = go.Figure()
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=300)
            sanity_handle.plotly_chart(fig, theme="streamlit",use_container_width=True)
        else:    
            sanity_handle.plotly_chart(st.session_state.sanity_figure, theme="streamlit",use_container_width=True)
        
        
        
with col2.expander('Step 3: Run testing and data collection',expanded=True):
    st.text("")
    col11, col22 = st.columns([0.5,1],gap='large')  

    with col11:
        testing = st.button('Run testing')
        loadingbar_handle = st.empty()    
        msg_handle = st.empty()
        
        
    with col22:    
        st.text("Engine measurements")

        plot_handle = st.empty()
        
        st_testing_handle = st.empty()
        if not st.session_state.testing_figure:
            fig = go.Figure()
            fig.update_layout(xaxis_title='DoE Sequence', yaxis_title="BSFC")
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=300)
            st_testing_handle.plotly_chart(fig, theme="streamlit",use_container_width=True)
        else:    
            st_testing_handle.plotly_chart(st.session_state.testing_figure, theme="streamlit",use_container_width=True)
        
        
with st.expander('Step 4: Create engine ML model',expanded=True):
    st.text("")
    col11, col22, col33 = st.columns([1,1,1],gap='large')  

    with col11:
        training = st.button('Train model')
        model_save = st.button('Save model')
        response = st.selectbox("Select responses to plot",outputs)     
        metric_1 = st.empty()
        metric_2 = st.empty()
        metric_3 = st.empty()
        metric_1.write(f"Training R2: ")
        metric_2.write(f"Validation R2: ")    

       
    with col22:    
        
        
        
        st.text("Modeling performance (Training)")

        plot_handle = st.empty()
        
        st_model_handle_1 = st.empty()
        if not st.session_state.accuracy_figure_1:
            fig = go.Figure()
            fig.update_layout(xaxis_title='Actual', yaxis_title="Prediction")
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=300)
            st_model_handle_1.plotly_chart(fig, theme="streamlit",use_container_width=True)
        else:    
            st_model_handle_1.plotly_chart(st.session_state.accuracy_figure_1, theme="streamlit",use_container_width=True)
        
    with col33:    
        st.text("Modeling performance (Validation)")

        plot_handle = st.empty()
        
        st_model_handle_2 = st.empty()
        if not st.session_state.accuracy_figure_2:
            fig = go.Figure()
            fig.update_layout(xaxis_title='Actual', yaxis_title="Prediction")
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=300)
            st_model_handle_2.plotly_chart(fig, theme="streamlit",use_container_width=True)
        else:    
            st_model_handle_2.plotly_chart(st.session_state.accuracy_figure_2, theme="streamlit",use_container_width=True)
        
                
         
with st.expander('Step 5: Optimise calibration maps',expanded=True):
    st.text("")
    
    st.text("")
    col11, col22 = st.columns([1,1],gap='large')  

    with col11:
        nbpx = st.number_input("Number of x breakpoints",value=5)
        nbpy = st.number_input("Number of y breakpoints",value=5)

    with col22:
        st.write("Optimise maps:")
        initialise_map = st.button('Initialise')
        optimise = st.button('Optimise')


    col11, col22 = st.columns([1,1],gap='large')  

    with col11:  
        
        st.text("VGT Map")

        plot_handle = st.empty()
        
        st_optimise_handle_1 = st.empty()
        if not st.session_state.optimise_figure_1:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode='markers'))
            fig.update_scenes(xaxis_title_text="Speed",  
                            yaxis_title_text="Torque",  
                            zaxis_title_text="VGT Position")
            fig.update_layout(margin=dict(l=20, r=20, t=10, b=10),height=400)
            st_optimise_handle_1.plotly_chart(fig, theme="streamlit",use_container_width=True)
        else:    
            st_optimise_handle_1.plotly_chart(st.session_state.optimise_figure_1, theme="streamlit",use_container_width=True)


        
    with col22:    
        st.text("EGR Map")

        plot_handle = st.empty()
        
        st_optimise_handle_2 = st.empty()
        if not st.session_state.optimise_figure_2:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode='markers'))
            fig.update_scenes(xaxis_title_text="Speed",  
                            yaxis_title_text="Torque",  
                            zaxis_title_text="EGR Position")
            fig.update_layout(margin=dict(l=20, r=20, t=10, b=10),height=400)
            st_optimise_handle_2.plotly_chart(fig, theme="streamlit",use_container_width=True)
        else:    
            st_optimise_handle_2.plotly_chart(st.session_state.optimise_figure_2, theme="streamlit",use_container_width=True)
   


if generate_doe:
    
    if design == "Full factorial":
        doe = build.full_fact(param_dict)
    elif design ==  "Latin hypercube sampling (LHS)":
        doe = build.space_filling_lhs(param_dict,num_samples = num_samples)
    elif design ==  "Box design":
        doe = build.box_behnken(param_dict)
    elif design == "Sobol sequence":
        param_name = list(param_dict.keys())
        
        qrng = qmc.Sobol(d=len(param_name),scramble=False)
        doe = []
        for i in range(0,num_samples):
            a = qrng.random()
            doe.append(list(a[0]))
            
        lb = edited_df['Lower'].values.tolist()
        ub = edited_df['Upper'].values.tolist()
        doe = qmc.scale(doe, lb, ub)
                
        doe = pd.DataFrame(doe,columns=param_name)
        doe.reset_index(drop=True, inplace=True)
    st.session_state.doe= doe     
    



if st.session_state.doe.size != 0:
    doe = st.session_state.doe
    fig = go.Figure()
    if len(options)==2:
        fig.add_trace(go.Scatter(x=doe[options[0]],y=doe[options[1]],mode='markers'))
        fig.update_layout(xaxis_title=options[0], yaxis_title=options[1])
    elif len(options)==3:
        fig.add_trace(go.Scatter3d(x=doe[options[0]],y=doe[options[1]],z=doe[options[2]],mode='markers'))
        fig.update_scenes(xaxis_title_text=options[0],  
                        yaxis_title_text=options[1],  
                        zaxis_title_text=options[2])
    #fig.update_traces(opacity=0.75)
    st.session_state.doe_figure = fig.update_layout(margin=dict(l=10, r=10, t=30, b=100),height=500)
    st_doe_plot.plotly_chart(st.session_state.doe_figure, theme="streamlit",use_container_width=True)   
     

xhat = []
for var in edited_df.iterrows():    
    val = st.session_state[var[1]['Name']]
    xhat.append(val)
yhat = model_virtual_enging.predict(xhat)
df_test = pd.DataFrame({'Name':df_outputs['Name'],'Value':list(yhat)}) 
fig = px.bar(df_test, x="Value", y="Name", orientation='h')
fig.update_layout(margin=dict(l=10, r=10, t=0, b=10),height=300)
sanity_handle.plotly_chart(fig, theme="streamlit",use_container_width=True)   



if testing:
    doe = st.session_state.doe
    res = []
    for row in doe.iterrows():
        yhat = model_virtual_enging.predict(row[1])
        res.append(list(yhat))
        df_res = pd.DataFrame(res,columns=outputs)        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_res.index,y=df_res["BSFC_ind"],mode='markers'))
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=300)
        fig.update_layout(xaxis_title='DoE Sequence', yaxis_title="BSFC")
        st_testing_handle.plotly_chart(fig, theme="streamlit",use_container_width=True)   
        time.sleep(0.1)
        st.session_state.test_result = df_res
        st.session_state.testing_figure = fig
    msg_handle.success('Done!')
    
    
if training:
    doe = st.session_state.doe
    res = st.session_state.test_result
    
    
    X = doe
    y = res
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    params = {'learning_rate': 0.1, 
          'l2_leaf_reg': 3, 
          'loss_function': 'MultiRMSE', 
          'eval_metric': 'MultiRMSE', 
          'task_type': 'CPU', 
          'iterations': 150,
         }
    model_mbc = CatBoostRegressor(**params)
    

    # Fit model
    model_mbc.fit(X_train, y_train)
    score_train = model_mbc.score(X_train, y_train)
    score_test = model_mbc.score(X_test, y_test)
    st.session_state.model_mbc = model_mbc
    
    
    metric_1.write(f"Training R2: {score_train:.3f}")
    metric_2.write(f"Validation R2: {score_test:.3f}")    
    


    
    yout = model_mbc.predict(X_train)
    y_pred = pd.DataFrame(yout,columns=outputs) 
    a = y_pred[response]
    b = y_train[response]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b,y=a,mode="markers",name="-"))
    #fig.update_traces(opacity=0.75)
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=20),height=300)
    fig.update_layout(xaxis_title='Actual', yaxis_title="Prediction")
    st_model_handle_1.plotly_chart(fig, theme="streamlit",use_container_width=True)
    st.session_state.accuracy_figure_1 = fig  
    
    
    yout = model_mbc.predict(X_test)
    y_pred = pd.DataFrame(yout,columns=outputs) 
    a = y_pred[response]
    b = y_test[response]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b,y=a,mode="markers",name="-"))
    #fig.update_traces(opacity=0.75)
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=20),height=300)
    fig.update_layout(xaxis_title='Actual', yaxis_title="Actual")
    st_model_handle_2.plotly_chart(fig, theme="streamlit",use_container_width=True)
    st.session_state.accuracy_figure_2 = fig
    
    
    metric_3.success('Done!')


if model_save:
    st.session_state.model_mbc.save_model("engine_model")


   
if initialise_map:
    df_variables = st.session_state.df_variables
    bpx = np.linspace(df_variables.iloc[0,1],df_variables.iloc[0,2],nbpx)
    bpy = np.linspace(df_variables.iloc[1,1],df_variables.iloc[1,2],nbpy)

    xm, ym = np.meshgrid(bpx, bpy) 
    xl = xm.flatten()
    yl = ym.flatten()

    st.session_state.bpx = bpx
    st.session_state.bpy = bpy

    st.session_state.xl = xl
    st.session_state.yl = yl
    
    st.session_state.z_vgt = np.full(np.shape(xl), 40)
    st.session_state.z_egr = np.full(np.shape(xl), 0.15)
    
    a = np.array(st.session_state.z_vgt).reshape(nbpx,nbpy).tolist()
    fig = go.Figure()
    fig.add_trace(go.Surface(z=a,colorscale ='blues'))
    fig.update_layout(autosize=True,margin=dict(l=65, r=50, b=65, t=90))
    fig.update_scenes(xaxis_title_text="Speed (RPM)", 
                      yaxis_title_text="Torque (Nm)", 
                      zaxis_title_text="VGT Position (%)")
    st.session_state.optimise_figure_1 = fig
    st_optimise_handle_1.plotly_chart(st.session_state.optimise_figure_1, theme="streamlit",use_container_width=True)


    a = np.array(st.session_state.z_egr).reshape(nbpx,nbpy).tolist()
    fig = go.Figure()
    fig.add_trace(go.Surface(z=a,colorscale ='reds'))
    fig.update_layout(autosize=True,margin=dict(l=65, r=50, b=65, t=90))
    fig.update_scenes(xaxis_title_text="Speed (RPM)", 
                      yaxis_title_text="Torque (Nm)", 
                      zaxis_title_text="EGR Position")
    st.session_state.optimise_figure_2 = fig
    st_optimise_handle_2.plotly_chart(st.session_state.optimise_figure_2, theme="streamlit",use_container_width=True)


def rosen(x):
    """Cost function for VGT optimization"""
    import random
    egr = np.full(np.shape(st.session_state.xl), 0.1)
    
    x = np.full(np.shape(st.session_state.xl), random.randrange(1, 80))
    xhat = pd.DataFrame(
        {'SPEED_A':st.session_state.xl,'TORQUE_R':st.session_state.yl,'BobOV_XPC_VGT':x,'egr_vlv_position':egr}
    )
    res = st.session_state.engine_model.predict(xhat)
    df_res_opt = pd.DataFrame(res,columns=outputs)     
    
    J = df_res_opt['BSFC_ind'].sum()
    
    
    return J

def egr_cost(x):
    """Cost function for EGR optimization"""
    import random
    vgt = np.full(np.shape(st.session_state.xl), 40)
    
    x = np.full(np.shape(st.session_state.xl), random.uniform(0.05, 0.34))
    xhat = pd.DataFrame(
        {'SPEED_A':st.session_state.xl,'TORQUE_R':st.session_state.yl,'BobOV_XPC_VGT':vgt,'egr_vlv_position':x}
    )
    res = st.session_state.engine_model.predict(xhat)
    df_res_opt = pd.DataFrame(res,columns=outputs)     
    
    J = df_res_opt['BSFC_ind'].sum()
    
    
    return J

if optimise:
    
    # Optimize VGT map
    x0 = st.session_state.z_vgt
    res = minimize(rosen, x0, method='trust-constr',
               options={'maxiter': 10, 'disp': True})
    
    st.session_state.z_vgt = res.x

    a = np.array(st.session_state.z_vgt).reshape(nbpx,nbpy).tolist()
    fig = go.Figure()
    fig.add_trace(go.Surface(z=a,colorscale ='blues'))
    fig.update_layout(autosize=True,margin=dict(l=65, r=50, b=65, t=20),height=400)
    fig.update_scenes(xaxis_title_text="Speed (RPM)", 
                      yaxis_title_text="Torque (Nm)", 
                      zaxis_title_text="VGT Position (%)")
    st.session_state.optimise_figure_1 = fig
    st_optimise_handle_1.plotly_chart(st.session_state.optimise_figure_1, theme="streamlit",use_container_width=True)

    # Optimize EGR map
    x0_egr = st.session_state.z_egr
    res_egr = minimize(egr_cost, x0_egr, method='trust-constr',
               options={'maxiter': 10, 'disp': True})
    
    st.session_state.z_egr = res_egr.x

    a = np.array(st.session_state.z_egr).reshape(nbpx,nbpy).tolist()
    fig = go.Figure()
    fig.add_trace(go.Surface(z=a,colorscale ='reds'))
    fig.update_layout(autosize=True,margin=dict(l=65, r=50, b=65, t=20),height=400)
    fig.update_scenes(xaxis_title_text="Speed (RPM)", 
                      yaxis_title_text="Torque (Nm)", 
                      zaxis_title_text="EGR Position")
    st.session_state.optimise_figure_2 = fig
    st_optimise_handle_2.plotly_chart(st.session_state.optimise_figure_2, theme="streamlit",use_container_width=True)


if None:

    
    df_doe = pd.read_excel("doe.xlsx")

    inputs = ["SPEED_A","TORQUE_R","BobOV_XPC_VGT","egr_vlv_position"]

    
        
    
    X = df_doe[inputs]
    y = df_doe[outputs]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    
    X_train
    
    params = {'learning_rate': 0.3, 
          'depth': 6, 
          'l2_leaf_reg': 3, 
          'loss_function': 'MultiRMSE', 
          'eval_metric': 'MultiRMSE', 
          'task_type': 'CPU', 
          'iterations': 150,
          'od_type': 'Iter', 
          'boosting_type': 'Plain', 
          'bootstrap_type': 'Bernoulli', 
          'allow_const_label': True, 
         }
    model = CatBoostRegressor(**params)
    

    # Fit model
    model.fit(X_train, y_train)
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    model.save_model("virtual_engine")
    score_train
    score_test

st.write('Copyright © 2026 Dr Farraen. All rights reserved.')

