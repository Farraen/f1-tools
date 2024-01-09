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



# Build dashboard
st.subheader('Racing Vehicle Optimiser')

with st.expander('Introduction', expanded=True):

    col1,col2 = st.columns([1,1])
    with col1:
        st_text('A prototype decision making algorithm for determining best vehicle designs for maximum vehicle performance. \
                The optimiser automatically finds best solutions while constraining to maximum vehicle durability. The dashboard uses a \
                decision tree regression model for predicting vehicle performance particularly the vehicle acceleration. ')
        st_text('The first half of the dashboard is for manually tuning the parameters. Observe the vehicle speed trace while  \
                moving the sliders. The next section is to generate several vehicle designs using an optimiser. Hover the mouse on top \
                of the scatter plot to inspec the designs.')
    with col2:
        image = read_image("images/Page5_model.png")
        st.image(image)

with st.expander('Settings', expanded=False):
    genre = st.radio(
        "Choose test mode",
        ["0-100kmh", "0-200kmh", "Max"],index=1)

    if genre == "0-100kmh":
        mode = 100.0
    elif genre == "0-200kmh":
        mode = 200.0
    else:
        mode = 500.0

with st.expander('Performance visualisation', expanded=True):

    col1,col2 = st.columns([0.4,1])
    with col1:
        st_title('Vehicle parameters')
        param_dict = {}
        for name in df.columns:
            init = st.session_state.solution_best[name]
            param_dict[name] = st.slider(name,np.double(df[name].Min), np.double(df[name].Max), np.double(init),key=name)
        st.session_state.solution_best = param_dict

    with col2:
        st_title('Performance metrics')
        col11,col22,col33,col44 = st.columns([1,1,1,1])
        metric1 = col11.empty()
        metric2 = col22.empty()
        metric3 = col33.empty()
        metric4 = col44.empty()


        st_title('Vehicle speed plot')
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


with st.expander('Performance optimisation', expanded=True):


    col1,gap,col2 = st.columns([0.8,0.05,1])
    with col1:

        st_title('Objectives')
        start_target, end_target = st.select_slider(
            'Select optimisation target',
            options=['Durability', '1', '2', '3', '4', '5', '6', '7', '8', '9','10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 'Performance'],
            value=('Durability', 'Performance'))
        range_list = handle_range(start_target, end_target)
        st.session_state.gen_number = st.number_input("Number of generation", value=10, placeholder="Type a number...")

        generate = st.button('Generate designs')
        my_bar = st.empty()

    with col2:
        st_title('Function Objective')
        optim_plot_placeholder = st.empty()

    st.write("")
    st_title('Vehicle designs')
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
    fig.update_layout(height=400,margin=dict(l=20, r=20, t=20, b=20))
    pareto_placeholder = st.plotly_chart(fig, theme="streamlit", use_container_width=True,height=400)  

    st_title("Baseline performance and design")
    baseline_table_placeholder = st.empty()

    st_title("Optimised performance and design")
    optimised_table_placeholder = st.empty()


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

    fig.update_layout(height=250,margin=dict(l=20, r=20, t=20, b=20))
    optim_plot_placeholder.plotly_chart(fig, theme="streamlit", use_container_width=True,height=250)  

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
        line = dict(color = "white", width = 5),
    ))
    fig.add_trace(go.Scatter(
        x=t_mod_fil,
        y=v_mod_fil,
        mode='lines',
        name='Optimised',
        opacity=0.5,
        line = dict(color = "blue", width = 5),
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
    fig.update_layout(height=400,margin=dict(l=20, r=20, t=20, b=20))
    plot_placeholder.plotly_chart(fig, theme="streamlit", use_container_width=True,height=400) 


    t_100_base = t_base_fil[find_nearest(v_base_fil,100)]
    t_100_mod = t_mod_fil[find_nearest(v_mod_fil,100)]
    dt_100 = np.round(t_100_mod-t_100_base,4)

    v_max_base = v_base_full[find_nearest(t_base_full,12)]
    v_max_mod = np.round(v_mod_full[find_nearest(t_mod_full,12)],1)
    dmax = np.round(v_max_mod-v_max_base,2)

    durab_base, perf_base = DamageModel(st.session_state.solution_baseline,v_max_base,t_100_base)
    durab_mod, perf_mod = DamageModel(st.session_state.solution_best,v_max_mod,t_100_mod)
    ddurab = np.round(durab_mod-durab_base,1)
    dperf = np.round(perf_mod-perf_base,1)

    metric1.metric("0-100kmh (s)", "%.2f" % t_100_mod, "%.2f" % dt_100)
    metric2.metric("Top speed (km/h)", "%.1f" % v_max_mod, "%.1f" % dmax)
    metric3.metric("Durability (%)", "%.1f" % durab_mod,"%.1f" % ddurab)
    metric4.metric("Performance (%)", "%.1f" % perf_mod, "%.1f" % dperf)

def plot_pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    
    return pf_X, pf_Y


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
    

df = st.session_state.results



# Prep baseline results
t_base_fil, v_base_fil, t_base_full, v_base_full = simulate(st.session_state.solution_baseline)
t_100_base = t_base_fil[find_nearest(v_base_fil,100)]
v_max_base = v_base_full[find_nearest(t_base_full,12)]
durab_base, perf_base = DamageModel(st.session_state.solution_baseline,v_max_base,t_100_base)

r0 = {'Bias':0, 'Durability': durab_base, 'Performance': perf_base, 'MaxSpeed': v_max_base, '0-100km/h': t_100_base}
r0.update(st.session_state.solution_baseline)
df_base =  pd.DataFrame([r0])

baseline_table_placeholder.write(df_base)

if isinstance(df,pd.DataFrame):
    optimised_table_placeholder.write(df)
else:
    df_empty = pd.DataFrame([],columns=df_base.columns)
    optimised_table_placeholder.write(df_empty)



if isinstance(df,pd.DataFrame):

    pf_X, pf_Y = plot_pareto_frontier(df["Performance"], df["Durability"])

    fig = px.scatter(df,
                    x="Performance",
                    y="Durability",
                    color="0-100km/h",
                    hover_data={'Durability':':.1f%','Performance':':.1f%','Bias':True,'MaxSpeed':True,'DragCoeff':':.2f','FrontSurface':True,'TorqueMultiplier':':.2f','VehicleMass':True,'Wheelbase':True}
                    )
    fig.update_layout(coloraxis_colorbar_title_text = '0-100km/h')

    hover_select=['Bias','MaxSpeed','DragCoeff','FrontSurface','TorqueMultiplier','VehicleMass','Wheelbase','0-100km/h']

    fig.add_trace(go.Scatter(
                x=pf_X,
                y=pf_Y,
                mode='lines',
                name="Pareto front",
                line=dict(width=2, color='red'),
                ))  

    fig.add_trace(go.Scatter(
                x=df_base["Performance"],
                y=df_base["Durability"],
                mode='markers+text',
                line=dict(width=4, color='red'),
                text= ["Baseline"],
                name="Baseline",
                textposition="top right",
                customdata=df_base[hover_select].values.tolist(),
                hovertemplate =
                    'Performance: %{y:.1f}%'+
                    '<br>Durability</b>: %{x:.1f}%'+
                    '<br>Bias</b>: %{customdata[0]}'+
                    '<br>MaxSpeed</b>: %{customdata[1]:.1f}'+
                    '<br>DragCoeff</b>: %{customdata[2]}'+
                    '<br>FrontSurface</b>: %{customdata[3]}'+
                    '<br>TorqueMultiplier</b>: %{customdata[4]}'+
                    '<br>VehicleMass</b>: %{customdata[5]}'+
                    '<br>Wheelbase</b>: %{customdata[6]}'+
                    '<br>0-100km/h</b>: %{customdata[7]}',
                ))




    fig.update_traces(marker={'size': 20})
    fig.update_layout(height=400,margin=dict(l=20, r=20, t=20, b=20),showlegend=True)

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.11,
        xanchor="left",
        x=1.01
    ))

    pareto_placeholder.plotly_chart(fig, theme="streamlit", use_container_width=True,height=400)  


st.write('Copyright © 2024 Farraen. All rights reserved.')
