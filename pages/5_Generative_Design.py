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
from scipy.interpolate import make_smoothing_spline
import os
import scipy.io
from sklearn.preprocessing import MinMaxScaler


clear = lambda: os.system('cls')
clear()

# Dashboard layout
st.set_page_config(layout="wide")

# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

# For range
@st.cache_resource
def load_range(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df


# For loading images
@st.cache_resource
def load_model():
    model = CatBoostRegressor()      # parameters not required.
    model = model.load_model('data/Page5_VehicleModel.pkl')
    return model
st.session_state.model = load_model()


def handle_range(start_target, end_target):

    if start_target == 'Durability':
        start_target = 0.0
    elif start_target == 'Performance':
        start_target = 20.0
    else:
        start_target = np.double(start_target)

    if end_target == 'Durability':
        end_target = 0.0
    elif end_target == 'Performance':
        end_target = 20.0
    else:
        end_target = np.double(end_target)
    
    range_list = list(range(int(start_target), int(end_target+1)))

    return range_list

# Init
if "gen_number" not in st.session_state:
    st.session_state.gen_number = []

if "solution" not in st.session_state:
    st.session_state.solution_target = []
    st.session_state.solution_best = []
    st.session_state.solution_initial = []
    st.session_state.columns = []
    st.session_state.optimised = False

if "results" not in st.session_state:
    st.session_state.results = []

if "scaler" not in st.session_state:
    st.session_state.scaler = []

if "fitness" not in st.session_state:
    st.session_state.fitness = []

df = load_range('data/Page5_range.csv')
st.session_state.solution_baseline = df.iloc[0,:].to_dict()
st.session_state.solution_best = df.iloc[3,:].to_dict()
st.session_state.columns = df.columns
st.session_state.df = df

arr = df.loc[['Min','Max'],:]


# Build dashboard
st.title('Racing Vehicle Optimiser')
st.subheader('A hobby project to optimise racing vehicle performance using generative design')

with st.expander('Introduction', expanded=False):
    col1,col2 = st.columns([1,1])
    with col1:
        st.write('A prototype decision making algorithm for determining best vehicle designs for maximum vehicle performance. \
                The optimiser automatically finds best solutions while constraining to maximum vehicle durability. The dashboard uses a \
                decision tree regression model for predicting vehicle performance particularly the vehicle acceleration. ')
        st.write('The first half of the dashboard is for manually tuning the parameters. Observe the vehicle speed trace while  \
                moving the sliders. The next section is to generate several vehicle designs using an optimiser. Hover the mouse on top \
                of the scatter plot to inspec the designs.')
    with col2:
        image = read_image("images/Page5_model.png")
        st.image(image)

col1,gap,col2,gap,col3 = st.columns([0.7,0.1,0.3,0.05,1])


with col1:
    st.subheader('Vehicle parameters')
    param_dict = {}
    for name in df.columns:
        init = st.session_state.solution_best[name]
        param_dict[name] = st.slider(name,np.double(df[name].Min), np.double(df[name].Max), np.double(init),key=name)
    st.session_state.solution_best = param_dict

with col2:

    st.subheader('Controls')
    genre = st.radio(
        "Choose test mode",
        ["0-100kmh", "0-200kmh", "Max"],index=1)

    if genre == "0-100kmh":
        mode = 100.0
    elif genre == "0-200kmh":
        mode = 200.0
    else:
        mode = 500.0


    st.subheader('Performance')
    metric1 = st.empty()
    metric2 = st.empty()
    metric3 = st.empty()
    metric4 = st.empty()


with col3:

    st.subheader('Vehicle speed plot')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        opacity=0.5,
        marker = dict(color = "LightSkyBlue", size = 15, opacity = 0.8),
    ))

    fig.update_layout(
        yaxis_title='Vehicle speed (kmh)',
        xaxis_title='Time',
        showlegend=True)   
    plot_placeholder = st.plotly_chart(fig, theme="streamlit", use_container_width=True)  




col1,gap,col2 = st.columns([0.8,0.05,1])
with col1:


    st.subheader('Optimiser')
    st.subheader('')
    start_target, end_target = st.select_slider(
        'Select optimisation target',
        options=['Durability', '1', '2', '3', '4', '5', '6', '7', '8', '9','10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 'Performance'],
        value=('Durability', 'Performance'))
    range_list = handle_range(start_target, end_target)
    st.session_state.gen_number = st.number_input("Number of generation", value=10, placeholder="Type a number...")

    generate = st.button('Generate designs')

    my_bar = st.empty()

    st.subheader('Function Objective')
    optim_plot_placeholder = st.empty()


with col2:
    st.subheader('Vehicle designs')
    st.subheader('')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        opacity=0.5,
        marker = dict(color = "LightSkyBlue", size = 15, opacity = 0.8),
    ))

    fig.update_layout(
        yaxis_title='Durability',
        xaxis_title='Performance',
        showlegend=True)   
    fig.update_layout(xaxis_range=[0,1])
    fig.update_layout(yaxis_range=[0,1])
    fig.update_layout(height=800)
    pareto_placeholder = st.plotly_chart(fig, theme="streamlit")  
    pareto_placeholder.plotly_chart(fig,height=800, use_container_width=True)


def simulate(inputs):
    t = np.arange(0, 12, 0.01)
    #fmu = 'VehicleModel.fmu'
    #result = simulate_fmu(fmu,start_values=inputs)  
    #df_base = pd.DataFrame(result)
    y_ml = st.session_state.model.predict(list(inputs.values()))
    vel = y_ml[2:]

    t_ml = [0.24,1.20,2.16,3.12,4.08,5.04,6.00,6.96,7.92,8.88,9.84,10.80,11.76]
    df_base = pd.DataFrame({'Time':t_ml,'Velocity':vel})

    mdl_base = make_smoothing_spline(df_base['Time'], df_base['Velocity'])
    v_base = mdl_base(t)
    t_base_fil = t[np.where(v_base<mode)]
    v_base_fil = v_base[np.where(v_base<mode)]
    return t_base_fil, v_base_fil, t, v_base


def simulate_optimise(inputs):
    t = np.arange(0, 12, 0.1)
    #fmu = 'VehicleModel.fmu'
    #result = simulate_fmu(fmu,start_values=inputs)  
    #df_base = pd.DataFrame(result)
    y_ml = st.session_state.model.predict(list(inputs.values()))
    vel = y_ml[2:]

    t_ml = [0.24,1.20,2.16,3.12,4.08,5.04,6.00,6.96,7.92,8.88,9.84,10.80,11.76]
    df_base = pd.DataFrame({'Time':t_ml,'Velocity':vel})

    mdl_base = make_smoothing_spline(df_base['Time'], df_base['Velocity'])
    v_base = mdl_base(t)
    t_base_fil = t[np.where(v_base<mode)]
    v_base_fil = v_base[np.where(v_base<mode)]

    return t_base_fil, v_base_fil, t, v_base

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


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def DamageModel(s0,vmax,tacc):

    x1 = -10*s0['VehicleMass']/2550
    x2 = -np.power(s0['TorqueMultiplier'],5)
    x3 = -5*s0['Wheelbase']/10
    durability = np.round(100+x1+x2+x3,2)

    x1 = -10*s0['VehicleMass']/2550
    x2 = np.power(s0['TorqueMultiplier'],5)/3
    x3 = vmax/500
    x4 = -10*s0['DragCoeff']
    x5 = -10*s0['FrontSurface']/2
    performance = np.round(100+x1+x2+x3+x4+x5,2)


    return durability, performance

def plot():

    t_base_fil, v_base_fil, t_base_full, v_base_full = simulate(st.session_state.solution_baseline)
    t_mod_fil, v_mod_fil, t_mod_full, v_mod_full = simulate(st.session_state.solution_best)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_base_fil,
        y=v_base_fil,
        mode='lines',
        name='Baseline',
        opacity=0.5,
        line = dict(color = "LightSkyBlue", width = 5),
    ))
    fig.add_trace(go.Scatter(
        x=t_mod_fil,
        y=v_mod_fil,
        mode='lines',
        name='Optimised',
        opacity=0.5,
        line = dict(color = "Cyan", width = 5),
    ))
    if mode == 100:
        fig.update_layout(xaxis_range=[0,6])
        fig.update_layout(yaxis_range=[0,250])
    elif mode == 200:
        fig.update_layout(xaxis_range=[0,6])
        fig.update_layout(yaxis_range=[0,250])
    else:
        fig.update_layout(xaxis_range=[0,12])
        fig.update_layout(yaxis_range=[0,350])
    
    fig.update_layout(
        yaxis_title='Vehicle speed (km/h)',
        xaxis_title='Time',
        showlegend=True)   
    fig.update_layout(height=600)
    plot_placeholder.plotly_chart(fig,height=600)
    plot_placeholder.plotly_chart(fig, theme="streamlit", use_container_width=True) 


    t_100_base = t_base_fil[find_nearest(v_base_fil,100)]
    t_100_mod = t_mod_fil[find_nearest(v_mod_fil,100)]
    dt_100 = np.round(t_100_mod-t_100_base,4)

    v_max_base = v_base_full[find_nearest(t_base_full,12)]
    v_max_mod = np.round(v_mod_full[find_nearest(t_mod_full,12)],1)
    dmax = np.round(v_max_mod-v_max_base,2)



    durab_base, perf_base = DamageModel(st.session_state.solution_baseline,v_max_base,t_100_base)
    durab_mod, perf_mod = DamageModel(st.session_state.solution_best,v_max_mod,t_100_mod)
    ddurab = np.round(durab_mod-durab_base,2)
    dperf = np.round(perf_mod-perf_base,2)

    metric1.metric("0-100kmh", f"{t_100_mod} s", f"{dt_100} s")
    metric2.metric("Top speed", f"{v_max_mod} km/h", f"{dmax} km/h")
    metric3.metric("Durability", f"{durab_mod} %", f"{ddurab} %")
    metric4.metric("Performance", f"{perf_mod} %", f"{dperf} %")


plot()

def rescale_solution(solution):
    solution = np.double(solution)
    df = st.session_state.df
    d = {}
    for index, var in enumerate(st.session_state.columns):
        dff = df[var]
        out= solution[index]*(dff['Max']-dff['Min'])/10 + dff['Min']
        d[var] =out
    return d


def fitness_func(bias):

    def fitness_func(ga_instance, solution, solution_idx):
        
        solution = rescale_solution(solution)
        t_mod_fil, v_mod_fil, t_mod_full, v_mod_full = simulate_optimise(solution)

        t_100_mod = t_mod_fil[find_nearest(v_mod_fil,100)]
        v_max_mod = np.round(v_mod_full[find_nearest(t_mod_full,12)],1)

        durab_mod, perf_mod  = DamageModel(solution,v_max_mod,t_100_mod)
        
        fitness = ((20-bias)/20)*durab_mod + (bias/20)*perf_mod


        return fitness

    return fitness_func



def on_generation(ga_instance):

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    st.session_state.fitness.append(solution_fitness)

    plot_fitness()


if generate:

    progress_text = "Operation in progress. Please wait."

    result = {}
    for index, bias in enumerate(range_list):
        time.sleep(0.01)
        my_bar.progress((index+1)/len(range_list), text=progress_text)


        st.session_state.optimised = False
        st.session_state.fitness = []

        fitness_function = fitness_func(bias)
        num_parents_mating = 4
        sol_per_pop = 10
        num_genes = 5
        st.session_state.iter = 0
        ga_instance = pygad.GA(num_generations=st.session_state.gen_number,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        gene_space=[0,1,2,3,4,5,6,7,8,9,10],
                        num_genes=num_genes,
                        init_range_low =0,
                        init_range_high=10,
                        gene_type=int,
                        on_generation=on_generation,
                        mutation_type='Random')


        ga_instance.run()

        # Gather results
        solution = rescale_solution(ga_instance.best_solution()[0])
        t_mod_fil, v_mod_fil,t_mod_full, v_mod_full = simulate(solution)
        MaxSpeed = np.round(v_mod_full[find_nearest(t_mod_full,12)],1)

        t_100_mod = t_mod_fil[find_nearest(v_mod_fil,100)]

        durab_mod, perf_mod = DamageModel(solution,MaxSpeed,t_100_mod)


        r1 = {'Bias':bias, 'Durability': durab_mod, 'Performance': perf_mod, 'MaxSpeed': MaxSpeed, '0-100km/h': t_100_mod}
        r1.update(solution)
        result[index] = r1


    time.sleep(1)
    my_bar.empty()    

    st.session_state.results =  pd.DataFrame.from_dict(result).T
    
    st.write(st.session_state.results)


df = st.session_state.results
if isinstance(df,pd.DataFrame):
    print(1)
    fig = px.scatter(df,
                    x="Performance",
                    y="Durability",
                    color="0-100km/h",
                    hover_data=['Bias','MaxSpeed','DragCoeff','FrontSurface','TorqueMultiplier','VehicleMass','Wheelbase']
                    )
    fig.update_traces(marker={'size': 20})
    fig.update_layout(height=800)
    pareto_placeholder.plotly_chart(fig,height=800)
    pareto_placeholder.plotly_chart(fig, theme="streamlit", use_container_width=True)  



st.write('Copyright © 2023 Farraen. All rights reserved.')





st.write('Copyright © 2024 Farraen. All rights reserved.')
