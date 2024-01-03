# Developed by Farraen
# Date 2018
# Migrated to python 2023

import os, sys
import time
import streamlit as st
import pandas as pd
import numpy as np
import pygad
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata
from PIL import Image


# System cache
if 'fitness_trace' not in st.session_state:
    st.session_state.fitness_trace = []

if 'results' not in st.session_state:
    st.session_state.results = []

if 'track' not in st.session_state:
    st.session_state.track = []

st.session_state.bias = []
st.session_state.track_placeholder = []
st.session_state.trackfinal_placeholder = []
st.session_state.iter_placeholder = []
st.session_state.iterCompare_placeholder = []
st.session_state.PUCompare_placeholder = []


# Commong functions

def plot_results():

    # Load last result
    if isinstance(st.session_state.results, dict):
        key = list(st.session_state.results)[-1]
        dt_track = st.session_state.results[key]
        dt_track = dt_track.drop(['No','DamageThisRace'], axis=1)

        dt_track_styled = dt_track.style.set_properties(subset=['PU Allocation'], **{'background-color': 'darkblue'})

        st.session_state.trackfinal_placeholder.dataframe(dt_track_styled,height=800)

        fig = go.Figure()

        for i in range(1,key):
            if i == key-1:
                flag = True
            else:
                flag = False

            dts = st.session_state.results[i]
            fig.add_trace(go.Scatter(
                y=dts['RUL'],
                mode='lines',
                name='Previous solutions',
                showlegend = flag,
                line=dict(
                width=1,
                color='grey')
            ))

        fig.add_trace(go.Scatter(
            y=dt_track['RUL'],
            mode='lines',
            name='Best PU allocation',
            line=dict(
            width=4,
            color='blue')
        ))
        fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="red", annotation_text="RUL threshold",annotation_position="bottom left")

        fig.update_yaxes(title_text='PU RUL (Remaining Useful Life)')
        fig.update_xaxes(title_text='Race List for the season',tickmode='linear')
        st.session_state.iterCompare_placeholder.plotly_chart(fig,use_container_width=True)


        fig2 = go.Figure()
        for i in range(1,key+1):
            
            width = 1
            cc = 'grey'
            flag = True
            nameLegend='Previous solutions'
            flagRange = False
            if i == key:
                cc = 'blue'
                width = 4
                nameLegend='Best PU Allocation'
                flagRange = True
            elif i == key-1:
                flag = True
            else:
                flag = False

            dts = st.session_state.results[i]

            MaxPowerReduced1 = np.max(dts.loc[np.where(dts['PU Allocation'] == 1)[0],'PowerReduced'].to_numpy())
            MaxPowerReduced2 = np.max(dts.loc[np.where(dts['PU Allocation'] == 2)[0],'PowerReduced'].to_numpy())
            MaxPowerReduced3 = np.max(dts.loc[np.where(dts['PU Allocation'] == 3)[0],'PowerReduced'].to_numpy())

            fig2.add_trace(go.Scatter(
                    x=[1,2,3],
                    y=[MaxPowerReduced1,MaxPowerReduced2,MaxPowerReduced3],
                    mode='lines',
                    name=nameLegend,
                    showlegend = flag,
                    line=dict(
                    width=width,
                    color=cc)
                ))
            
            if flagRange:
                ymax = max([MaxPowerReduced1,MaxPowerReduced2,MaxPowerReduced3])
                ymin = min([MaxPowerReduced1,MaxPowerReduced2,MaxPowerReduced3])

                fig2.add_hline(y=ymax, line_width=3, line_dash="dash", line_color="red", annotation_text="Max of best solution",annotation_position="bottom left")
                fig2.add_hline(y=ymin, line_width=3, line_dash="dash", line_color="red", annotation_text="Min of best solution",annotation_position="bottom left")
                fig2.update_yaxes(title_text='Power Reduction End Season (kW)')
                fig2.update_xaxes(title_text='Power Unit',tickmode='linear')



        st.session_state.PUCompare_placeholder.plotly_chart(fig2,use_container_width=True)


# Setup the page layout

st.set_page_config(layout="wide")

# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image


with st.sidebar:
    st.header('Damage Model')
    st.write('The optimiser uses an artificial damage model made solely for demonstration purposes. The data does not represent true PU values.')
    image = read_image("images/pu_damage_model.PNG")
    st.image(image)

st.title('Farraen\'s PU Decision Engine Playground')
st.write('A virtual environment to test Genetic Algorithm for optimising PU selection')
st.write('Adapted from Farraen\'s 2018 Matlab GA PU script into Python. Results may vary due to to the GA library behaviour.')

col1, dcol, col2 = st.columns([1,0.1,1])
col1.header('2018 Track information')
st.session_state.track_placeholder = col1.empty()


col2.header('Genetic Algorithm Panel')
start_button = col2.button('Optimise')
gen_number = col2.number_input("Number of generation", value=20, placeholder="Type a number...")

st.session_state.bias = col2.select_slider(
    'Select bias:',
    options=['High Performance','2','3','4','5','6','7','8','9','Longer RUL'],
    value=('2'),
    key='GA select slider'
)

if st.session_state.bias == 'High Performance':
    st.session_state.bias = 1
elif st.session_state.bias == 'Longer RUL':
    st.session_state.bias = 10
else:
    st.session_state.bias = int(st.session_state.bias)

st.session_state.iter_placeholder = col2.empty()

st.header('Results')
col11, dcol, col22 = st.columns([1,0.1,0.9])

col11.write('Final results with PU allocation for all races. RUL is the Remaining Useful Life in %.')
col11.write('The \'PU Allocation\' column is the index of the PU recommneded to be used for the race.')
st.session_state.trackfinal_placeholder = col11.empty()

col22.write('Plot shows the prediction of PU degradation for the race season using optimised PU allocation.')
st.session_state.iterCompare_placeholder = col22.empty()

col22.write('Plot shows the power loss due to degradation for all power units')
st.session_state.PUCompare_placeholder = col22.empty()


st.write('Copyright © 2023 Farraen. All rights reserved.')



# Reload data if available

# Load in track information
if not isinstance(st.session_state.track,pd.DataFrame):
    st.session_state.track = pd.read_excel('Page1/track.xlsx')

st.session_state.track = st.session_state.track_placeholder.data_editor(st.session_state.track,use_container_width=True)
plot_results()






dta = pd.DataFrame(st.session_state.fitness_trace,columns=["value"])

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=dta['value'],
    mode='markers+lines',
))
st.session_state.iter_placeholder.plotly_chart(fig)
fig.update_yaxes(title_text='Fitness Value')
fig.update_xaxes(title_text='Generation')
st.session_state.iter_placeholder.plotly_chart(fig,use_container_width=True)


def interp2(X,Y,Z,Xv,Yv):
    length_values = len(X) * len(Y)
    x_grid, y_grid = np.meshgrid(X, Y)   
    points = np.empty((length_values, 2))
    values = Z.flatten()
    points[:, 0] = x_grid.flatten()
    points[:, 1] = y_grid.flatten()
    grid_z1 = griddata(points, values, (Xv,Yv), method='cubic')

    return grid_z1


def DamageModel(x):

    dtt  = st.session_state.track

    ICE_RUL  = [80, 70, 95]
    ICE_KW  = [425, 400, 450]

    # Piston clearance/km temp
    DamageModel_X = [0.1, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 41.0]
    DamageModel_Z = [2.0, 3.0, 4.0, 6.0, 9.0, 14.0, 19.0, 25.0, 32.0]
    DamageModel_Z = [item * 0.000001 for item in DamageModel_Z]

    # Power degradation due to piston clearance
    PerfModel_X = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.015]
    PerfModel_Z = [0.1,   1,     2,     3,     5,     7,     8,     10,    15,    30]

    # RUL loss based on ICE power loss and mileage per race
    RULModel_X = np.array([150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
    RULModel_Y = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 25.0])
    RULModel_Z = np.array([[7.5, 7.0, 9.5, 11.5, 19.0, 32.0],
                  [6.5, 7.5, 9.0, 11.5, 15.5, 24.5],
                  [6.0, 6.5, 7.0, 10.0, 14.0, 17.0],
                  [4.5, 6.5, 7.0, 10.0, 11.5, 13.0],
                  [3.5, 5.0, 6.5, 8.50, 9.00, 10.0],
                  [1.5, 3.5, 5.5, 7.00, 8.00, 7.50]])
    RULModel_XX, RULModel_YY = np.meshgrid(RULModel_X, RULModel_Y) 

    # Average temperature in the race month
    AmbTemp = dtt["MinTemp"] + dtt["MaxTemp"]/2
    
    # Calculate ICE damage for each race
    interp_func = interp1d(DamageModel_X, DamageModel_Z)
    DamagePerKM = interp_func(AmbTemp)
    dt_DamagePerKM = pd.DataFrame(DamagePerKM,columns=["DamagePerKM"])
    dtt['DamageThisRace'] = dtt['Distance'] * dt_DamagePerKM['DamagePerKM']
    
    # Calculate the ICE power reduction
    interp_func2 = interp1d(PerfModel_X, PerfModel_Z)
    PowerReducedThisRace = interp_func2(dtt['DamageThisRace'])

    # Get individual power reduction for each ICE
    PowerRedICE1 = PowerReducedThisRace[np.where(x == 1)[0]]
    PowerRedICE2 = PowerReducedThisRace[np.where(x == 2)[0]]
    PowerRedICE3 = PowerReducedThisRace[np.where(x == 3)[0]]

    # Get distance ran for each ICE
    DistanceICE1 = dtt.loc[np.where(x == 1)[0],'Distance'].to_numpy()
    DistanceICE2 = dtt.loc[np.where(x == 2)[0],'Distance'].to_numpy()
    DistanceICE3 = dtt.loc[np.where(x == 3)[0],'Distance'].to_numpy()

    # RUL reduction for each ICE based on damage model
    RULReducedICE1 = interp2(RULModel_X,RULModel_Y,RULModel_Z,DistanceICE1,PowerRedICE1)
    RULReducedICE2 = interp2(RULModel_X,RULModel_Y,RULModel_Z,DistanceICE2,PowerRedICE2)
    RULReducedICE3 = interp2(RULModel_X,RULModel_Y,RULModel_Z,DistanceICE3,PowerRedICE3)

    # Calculate cummulative RUL reduction
    RULARR1 = pd.DataFrame({"Index":np.where(x == 1)[0],"CumSum":np.cumsum(RULReducedICE1)})
    RULARR2 = pd.DataFrame({"Index":np.where(x == 2)[0],"CumSum":np.cumsum(RULReducedICE2)})
    RULARR3 = pd.DataFrame({"Index":np.where(x == 3)[0],"CumSum":np.cumsum(RULReducedICE3)})
    RULARR = pd.concat([RULARR1, RULARR2, RULARR3], ignore_index=True)
    RUL_Reduced = RULARR.sort_values('Index')
    RUL_Reduced.reset_index(drop=True, inplace=True)

    # Calculate cummulative Power reduction
    RULARR1 = pd.DataFrame({"Index":np.where(x == 1)[0],"PowerReduced":np.cumsum(PowerRedICE1)})
    RULARR2 = pd.DataFrame({"Index":np.where(x == 2)[0],"PowerReduced":np.cumsum(PowerRedICE2)})
    RULARR3 = pd.DataFrame({"Index":np.where(x == 3)[0],"PowerReduced":np.cumsum(PowerRedICE3)})
    RULARR = pd.concat([RULARR1, RULARR2, RULARR3], ignore_index=True)
    PowerReduced = RULARR.sort_values('Index')
    PowerReduced.reset_index(drop=True, inplace=True)

    # Calculate RUL left for each ICE and store in data table
    RULARR1 = pd.DataFrame({"Index":np.where(x == 1)[0],"RUL":ICE_RUL[0]-np.cumsum(RULReducedICE1)})
    RULARR2 = pd.DataFrame({"Index":np.where(x == 2)[0],"RUL":ICE_RUL[1]-np.cumsum(RULReducedICE2)})
    RULARR3 = pd.DataFrame({"Index":np.where(x == 3)[0],"RUL":ICE_RUL[2]-np.cumsum(RULReducedICE3)})
    RULARR = pd.concat([RULARR1, RULARR2, RULARR3], ignore_index=True)
    RUL = RULARR.sort_values('Index')
    RUL.reset_index(drop=True, inplace=True)

    # Calculate Power left for each ICE and store in data table
    RULARR1 = pd.DataFrame({"Index":np.where(x == 1)[0],"PowerLeft":ICE_KW[0]-np.cumsum(PowerRedICE1)})
    RULARR2 = pd.DataFrame({"Index":np.where(x == 2)[0],"PowerLeft":ICE_KW[1]-np.cumsum(PowerRedICE2)})
    RULARR3 = pd.DataFrame({"Index":np.where(x == 3)[0],"PowerLeft":ICE_KW[2]-np.cumsum(PowerRedICE3)})
    RULARR = pd.concat([RULARR1, RULARR2, RULARR3], ignore_index=True)
    PowerLeft = RULARR.sort_values('Index')
    PowerLeft.reset_index(drop=True, inplace=True)

    # Calculate fitness function (inverse of penalty function)
    PowerLoss = -(np.sum(PowerRedICE1) + np.sum(PowerRedICE2) + np.sum(PowerRedICE3))

    failedPU = RUL.loc[np.where(RUL["RUL"] < 0)[0],'Index'].to_numpy()

    bias = (st.session_state.bias-1)/2
    fitness_value = PowerLoss - (bias)*np.sum(failedPU)


    return fitness_value, PowerLoss, PowerLeft, RUL, PowerReduced
        

def fitness_func(ga_instance, solution, solution_idx):
    
    fitness_value, PowerLoss, PowerLeft, RUL, PowerReduced = DamageModel(solution)

    return fitness_value


def on_start(ga_instance):
    st.session_state.fitness_trace = []
    st.session_state.results = {data: [] for data in range(1,ga_instance.num_generations)}
        
def on_generation(ga_instance):

    dtt = st.session_state.track

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    Fitness, PowerLoss, PowerLeft, RUL, PowerReduced = DamageModel(solution)
    dfsol = pd.DataFrame(solution,columns=["PU Allocation"])

    dts = pd.concat([dfsol,dtt,PowerLeft["PowerLeft"],PowerReduced["PowerReduced"],RUL["RUL"]], axis=1)
    
    index = ga_instance.generations_completed
    st.session_state.results[index] = dts

    st.session_state.fitness_trace.append(solution_fitness)
    dta = pd.DataFrame(st.session_state.fitness_trace,columns=["value"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=dta['value'],
        mode='markers+lines',
    ))
    fig.update_yaxes(title_text='Fitness Value')
    fig.update_xaxes(title_text='Generation')
    st.session_state.iter_placeholder.plotly_chart(fig,use_container_width=True)




if start_button:

    fitness_function = fitness_func

    num_parents_mating = 4

    sol_per_pop = 5
    num_genes = 21

    init_range_low = 1
    init_range_high = 3

    ga_instance = pygad.GA(num_generations=gen_number,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=[1,2,3],
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       gene_type=int,
                       on_start=on_start,
                       on_generation=on_generation,
                       mutation_type='random')
    
    ga_instance.run()

    # Latest results
    plot_results()
 
