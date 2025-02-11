import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image

import time
import pygad
from sklearn.metrics import r2_score
import scipy as sp
from catboost import CatBoostRegressor

import os

from fmpy import simulate_fmu, read_model_description, extract
import fmpy as fm

st.set_page_config(layout="wide")

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

if "comb_model" not in st.session_state:
    st.session_state.comb_model = []

if "iter_comb" not in st.session_state:
    st.session_state.iter_comb = []

if "fitness_comb" not in st.session_state:
    st.session_state.fitness_comb = []

if "pressure_plot_handle" not in st.session_state:
    st.session_state.pressure_plot_handle = []

if "solution_best" not in st.session_state:
    st.session_state.solution_target = {"AFRCorr":0, "CombDur":50, "IgaCorr":0}
    st.session_state.solution_best = {"AFRCorr":0.05, "CombDur":45, "IgaCorr":-2}
    st.session_state.solution_initial = {"AFRCorr":0.05, "CombDur":45, "IgaCorr":-2}
    st.session_state.optimised = False

# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

# For loading images
@st.cache_resource
def load_model():
    model = CatBoostRegressor()      # parameters not required.
    model = model.load_model('data/Page3_ml_model.pkl')
    return model



st.session_state.comb_model = load_model()

parameters = ["AFRCorr","CombDur","IgaCorr"]
output = ["Pressure","CrkAngle"]


model_name = "data/Page3_CombustionStrategy.fmu"

def update_solution():
    solution_best = st.session_state.solution_best
    for index, updates in st.session_state["df_editor"].items():
        if "edited_rows" in index:
            keys = list(st.session_state.solution_best.keys())
            for key, value in updates.items():
                row_name = keys[key]
                value = value['value']
                solution_best[row_name] = value
      
    st.session_state.solution_best = solution_best


def simulate_ml(input_data):

    input_data = list(input_data.values())

    x = [-149.2,-140.4,-100.4,-80.4,-60.4,-40.4,-30.0,-25.2,-20.4,-14.8,-10.0,-5.2,-2.0,0.4,2.0,5.2,10.0,14.8,20.4,40.4,50.0,60.4,80.4,100.4,200.4,300.4,550.0]
    y = st.session_state.comb_model.predict(input_data)

    new_x = [-149.2,-140.4,-100.4,-80.4,-60.4,-40.4,-30.0,-25.2,-20.4,-14.8,-10.0,-5.2,-2.0,0.4,2.0,5.2,10.0,14.8,20.4,40.4,50.0,60.4,80.4,100.4,200.4,300.4,550.0]
    new_y = sp.interpolate.interp1d(x, y, kind='cubic')(new_x)

    new_length = 1000
    new_xx = np.linspace(np.min(new_x), np.max(new_x), new_length)
    new_yy = sp.interpolate.interp1d(new_x, new_y, kind='slinear')(new_xx)

    df = pd.DataFrame(
        {'CrkAngle': new_xx,
        'Pressure': new_yy,
    })

    return df

def plot_result():

    df = simulate_ml(st.session_state.solution_target)
    df1 = simulate_ml(st.session_state.solution_best)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(
            x=df['CrkAngle'],
            y=df['Pressure'],
            mode='lines',
            line = dict(width = 4, color = "white"),
            marker = dict(color = "cyan", size = 15, opacity = 0.8),
            name = 'Target',
        ),row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df1['CrkAngle'],
        y=df1['Pressure'],
        mode='lines',
        line = dict(width = 4, color = "blue"),
        marker = dict(color = "cyan", size = 15, opacity = 0.8),
        name = 'Optimised',
    ),row=1, col=1)

    error = df['Pressure']-df1['Pressure']
    fig.add_trace(go.Scatter(
        x=df['CrkAngle'],
        y=error,
        opacity=0.5,
        name="Error",
    ),row=2, col=1)

    fig.update_xaxes(rangeslider= {'visible':True}, row=2, col=1,rangeslider_thickness = 0.1)

    fig.update_yaxes(title_text='Pressure (bar)', row=1, col=1)
    fig.update_yaxes(title_text='Error (bar)', row=2, col=1)
    fig.update_layout(
        xaxis_title=' Crank Angle (deg ATDC)',
        showlegend=True)   
    fig.update(layout_yaxis_range = [0,120])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(height=400)
    fig.update_layout(legend=dict(
    orientation="v",
    yanchor="auto",
    y=1,
    xanchor="right",  # changed
    x=1
    ))

    st.session_state.pressure_plot_handle = fig
    pressure_plot_placeholder.plotly_chart(fig, theme="streamlit", use_container_width=True)  




st.subheader('Combustion Optimiser')

with st.expander('Introduction', expanded=False):
    st_text('This is a prototype dashboard for parameterising a combustion strategy. \
                The combustion strategy is a subsystem for sensorless combustion pressure prediction. \
                These strategies are then embedded in a control system to predict engine variables, \
                in this case predicting the cylinder pressure and its combustion characteristics.')

    st_text('The tool uses a mathematical conbustion model and was converted in a working Simulink model. \
                The Simulink model is then converted into an FMU (a generic control system format that can run in Python). \
                The dashboard currently supports single point optimisation of three variables. In the future, it is possible \
                to use map-based corrections for more accurate combustion strategy tuning. The dashboard uses genetic algorithm to \
                tune the correction factors so that it can accurately emulate a physical sensor measurements. \
                The optimisation objective is to minimise the differences between prediciton and measured values.')
    
    st_text('Press Optimise to run the optimisation process or use the tables for manual tuning.')


    image = read_image("images/Page3_sensor.png")
    st.image(image,width=400)


with st.expander('Combustion algorithm', expanded=False):
    st_text("The image below is a screenshot of the combustion model written in Simulink and the converted into an FMU and a Python script. It can predict cylinder combustion but is lacking in knock characteristics. Knock events prediction can be added as an add-on to the model if sufficient engine cylinder measurement is available.")
    image = read_image("images/Page3_model.png")
    st.image(image,use_column_width=True)



with st.expander('FMU sensorless strategy', expanded=False):
    gap, col1, col2, gap, col3, gap, col4, gap  = st.columns([0.1,1,1,0.1,1,0.1,1,0.1])
    with col1:
        st_title('Inputs')
        model_description = read_model_description(model_name)
        vrs = {}
        for variable in model_description.modelVariables:
            if variable.start is not None and variable.name not in parameters:
                vrs[variable.name] = np.double(variable.start)
        df = pd.DataFrame([vrs], columns=vrs.keys())
        dft = df.transpose() 
        dft.columns = ['Value']
        st.data_editor(dft)
        
        
    with col2:
        st_title('Parameters')
        st.data_editor(st.session_state.solution_best)

    with col3:
        st_title('Strategy')

        image = read_image("images/Page3_fmu.png")
        st.image(image)

        st.write(f'Model name: {model_name}')

with col4:
    st_title('Outputs')
    st.data_editor(output)

with st.expander('Cylinder pressure trace and optimisation', expanded=True):

    fig = px.scatter(x=[0], y=[0])
    fig.update_layout(
        yaxis_title='Pressure (bar)',
        xaxis_title=' Crank Angle (deg ATDC)',
        showlegend=True)        
    pressure_plot_placeholder = st.plotly_chart(fig, theme="streamlit", use_container_width=True)  

        

    col1, col2 = st.columns([0.8,1],gap="Small")

    with col1:
        optimise = st.button('Optimise combustion')
        st.session_state.gen_number = st.number_input("Number of generation", value=10, placeholder="Type a number...")
        my_bar = st.progress(0)
        reset = st.button('Reset')
    with col2:  
        st_title('Tunable parameters')
        st.data_editor(st.session_state.solution_best,on_change=update_solution,key='df_editor')
        #st.data_editor(st.session_state.solution_best,key='df_editor')
        st.write('\* Changing the values will change the combustion pressure prediction')


      
    st_title('Function Objective')
    optim_plot_placeholder = st.empty()

 
plot_result()


def plot_fitness():

    if not st.session_state.fitness_comb:

        fig = px.scatter(x=[0], y=[0])
        fig.update_layout(
            yaxis_title='Cost Value',
            xaxis_title='Iteration',
            yaxis_range=[0.95,1],
            xaxis_range=[0,10],
            showlegend=True)        
    else:
        x = list(range(0,len(st.session_state.fitness_comb)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=st.session_state.fitness_comb,
            mode='lines',
            line = dict(width = 4, color = "lightblue"),
            marker = dict(color = "cyan", size = 15, opacity = 0.8),
        ))
        fig.update_layout(
        yaxis_title='Cost Value',
        xaxis_title='Iteration')     
    fig.update_layout(height=300,margin=dict(l=20, r=20, t=20, b=20))
    optim_plot_placeholder.plotly_chart(fig, theme="streamlit", use_container_width=True,height=300)  

plot_fitness()



def fitness_func(ga_instance, solution, solution_idx):
    
    df = simulate_ml(st.session_state.solution_target)


    df_param = pd.DataFrame([solution],columns=["AFRCorr","CombDur","IgaCorr"])
    param_dict = df_param.to_dict('records')

    input_data = param_dict[0]

    df1 = simulate_ml(input_data)

    fitness = r2_score(df['Pressure'],df1['Pressure'])

    return fitness

def on_generation(ga_instance):

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    st.session_state.solution_best["AFRCorr"] = solution[0]
    st.session_state.solution_best["CombDur"] = solution[1]
    st.session_state.solution_best["IgaCorr"] = solution[2]

    plot_result()

    st.session_state.iter_comb = st.session_state.iter_comb + 1
    my_bar.progress(st.session_state.iter_comb/st.session_state.gen_number)

    st.session_state.fitness_comb.append(solution_fitness)

    plot_fitness()


if optimise:

    st.session_state.optimised = False
    st.session_state.fitness_comb = []

    fitness_function = fitness_func
    num_parents_mating = 4
    sol_per_pop = 20
    num_genes = 3
    st.session_state.iter_comb = 0
    ga_instance = pygad.GA(num_generations=st.session_state.gen_number,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=[-10,40,-10],
                       init_range_high=[10,60,10],
                       gene_type=int,
                       on_generation=on_generation,
                       mutation_type='Random')
    
    ga_instance.run()

    st.session_state.optimised = True
    st.session_state.iter_comb = []

    my_bar.empty()
    st.rerun()
    

if reset:

    st.session_state.solution_best = st.session_state.solution_initial
    st.session_state.optimised = False
    st.session_state.iter_comb = []
    st.rerun()

plot_fitness()



st.write('Copyright © 2024 Farraen. All rights reserved.')
