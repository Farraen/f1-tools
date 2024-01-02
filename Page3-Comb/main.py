import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
st.title('Farraen\'s Combustion Optimiser')

if "model" not in st.session_state:
    st.session_state.model = []

if "iter" not in st.session_state:
    st.session_state.iter = []

if "fitness" not in st.session_state:
    st.session_state.fitness = []

if "pressure_plot_handle" not in st.session_state:
    st.session_state.pressure_plot_handle = []

if "solution" not in st.session_state:
    st.session_state.solution_target = []
    st.session_state.solution_best = []
    st.session_state.solution_initial = []
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
    model = model.load_model('combustion_ml_model.pkl')
    return model


st.session_state.model = load_model()

parameters = ["AFRCorr","CombDur","IgaCorr"]
output = ["Pressure","CrkAngle"]



st.session_state.solution_target = {"AFRCorr":0, "CombDur":50, "IgaCorr":0}
st.session_state.solution_best = {"AFRCorr":0.05, "CombDur":45, "IgaCorr":-2}
st.session_state.solution_initial = {"AFRCorr":0.05, "CombDur":45, "IgaCorr":-2}


model_name = "CombustionStrategy.fmu"


def simulate_ml(input_data):

    input_data = list(input_data.values())

    x = [-149.2,-140.4,-100.4,-80.4,-60.4,-40.4,-30.0,-25.2,-20.4,-14.8,-10.0,-5.2,-2.0,0.4,2.0,5.2,10.0,14.8,20.4,40.4,50.0,60.4,80.4,100.4,200.4,300.4,550.0]
    y = st.session_state.model.predict(input_data)

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


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['CrkAngle'],
        y=df['Pressure'],
        mode='lines',
        line = dict(width = 4, color = "red"),
        marker = dict(color = "cyan", size = 15, opacity = 0.8),
        name = 'Target',
    ))
    fig.add_trace(go.Scatter(
        x=df1['CrkAngle'],
        y=df1['Pressure'],
        mode='lines',
        line = dict(width = 4, color = "lightblue"),
        marker = dict(color = "cyan", size = 15, opacity = 0.8),
        name = 'Optimised strategy',
    ))

    fig.update_layout(
        yaxis_title='Pressure (bar)',
        xaxis_title=' Crank Angle (deg ATDC)',
        showlegend=True)   
    fig.update(layout_yaxis_range = [0,120])

    st.session_state.pressure_plot_handle = fig
    pressure_plot_placeholder.plotly_chart(fig, theme="streamlit", use_container_width=True)  


    error = df['Pressure']-df1['Pressure']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['CrkAngle'],
        y=error,
        mode='markers',
        opacity=0.5,
        marker = dict(color = "LightSkyBlue", size = 15, opacity = 0.8),
    ))

    fig.update_layout(
        yaxis_title='Error (bar)',
        xaxis_title=' Crank Angle (deg ATDC)',
        showlegend=True)   
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    error_plot_placeholder.plotly_chart(fig, theme="streamlit", use_container_width=True)  


with st.expander('Introduction', expanded=False):
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.write('This is a prototype dashboard for parameterising a combustion strategy. \
                 The combustion strategy is a subsystem for sensorless combustion pressure prediction. \
                 These strategies are then embedded in a control system to predict engine variables, \
                 in this case predicting the cylinder pressure and its combustion characteristics.')

        st.write('The tool uses a mathematical conbustion model and was converted in a working Simulink model. \
                 The Simulink model is then converted into an FMU (a generic control system format that can run in Python). \
                 The dashboard currently supports single point optimisation of three variables. In the future, it is possible \
                 to use map-based corrections for more accurate combustion strategy tuning. The dashboard uses genetic algorithm to \
                 tune the correction factors so that it can accurately emulate a physical sensor measurements. \
                 The optimisation objective is to minimise the differences between prediciton and measured values.')
        
        st.write('Press Optimise to run the optimisation process or use the tables for manual tuning.')



    with col2:
        image = read_image("sensor.png")
        st.image(image)


with st.expander('Combustion algorithm', expanded=False):
    image = read_image("model.png")
    st.image(image,use_column_width=True)



with st.expander('FMU sensorless strategy', expanded=False):
    st.subheader('FMU Parameters')
    gap, col1, col2, gap, col3, gap, col4, gap  = st.columns([0.1,1,1,0.1,1,0.1,1,0.1])
    with col1:
        st.subheader('Inputs')
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
        st.subheader('Tunable parameters')
        st.write('Changing the values will change the combustion pressure prediction')
        st.session_state.solution_best = st.data_editor(st.session_state.solution_best)

        st.subheader('Optimised parameters')
        datatable_handle = st.dataframe(st.session_state.solution_best)


    with col3:
        st.subheader('Strategy')

        image = read_image("fmu.png")
        st.image(image)

        st.write(f'Model name: {model_name}')



with col4:
    st.subheader('Outputs')
    st.write(output)


col1, gap, col2, gap, col3 = st.columns([0.3,0.05,0.5,0.05,1])

with col1:
    st.subheader('Controls')   
    optimise = st.button('Optimise')
    reset = st.button('Reset')
    st.session_state.gen_number = st.number_input("Number of generation", value=10, placeholder="Type a number...")

    my_bar = st.progress(0)

with col2:    
    st.subheader('Function Objective')
    optim_plot_placeholder = st.empty()

with col3:

    if not st.session_state.optimised:

        st.subheader('Pressure Trace')
        fig = px.scatter(x=[0], y=[0])
        fig.update_layout(
            yaxis_title='Pressure (bar)',
            xaxis_title=' Crank Angle (deg ATDC)',
            showlegend=True)        
        pressure_plot_placeholder = st.plotly_chart(fig, theme="streamlit", use_container_width=True)  

        st.subheader('Error plot')
        fig = px.scatter(x=[0], y=[0])
        fig.update_layout(
            yaxis_title='Error (bar)',
            xaxis_title=' Crank Angle (deg ATDC)',
            showlegend=True)        
        error_plot_placeholder = st.plotly_chart(fig, theme="streamlit", use_container_width=True)  

        plot_result()
        


def plot_fitness():

    if not st.session_state.fitness:

        fig = px.scatter(x=[0], y=[0])
        fig.update_layout(
            yaxis_title='Cost Value',
            xaxis_title='Iteration',
            yaxis_range=[0.95,1],
            xaxis_range=[0,10],
            showlegend=True)        
    else:
        x = list(range(0,len(st.session_state.fitness)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=st.session_state.fitness,
            mode='lines',
            line = dict(width = 4, color = "lightblue"),
            marker = dict(color = "cyan", size = 15, opacity = 0.8),
        ))
        fig.update_layout(
        yaxis_title='Cost Value',
        xaxis_title='Iteration')     

    optim_plot_placeholder.plotly_chart(fig, theme="streamlit", use_container_width=True)  

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

    st.session_state.iter = st.session_state.iter + 1
    my_bar.progress(st.session_state.iter/st.session_state.gen_number)

    st.session_state.fitness.append(solution_fitness)

    plot_fitness()


if optimise:

    st.session_state.optimised = False
    st.session_state.fitness = []

    fitness_function = fitness_func
    num_parents_mating = 4
    sol_per_pop = 20
    num_genes = 3
    st.session_state.iter = 0
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
    st.session_state.iter = []

    my_bar.empty()

    datatable_handle.dataframe(st.session_state.solution_best)


if reset:
    st.session_state.solution_best = st.session_state.solution_initial
    st.session_state.optimised = False
    st.session_state.iter = []
    plot_result()
    datatable_handle.dataframe(st.session_state.solution_best)

plot_fitness()
