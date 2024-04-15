import streamlit as st

import pandas as pd
import numpy as np
from numpy.matlib import repmat
import pygad
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(layout='wide')

st.subheader('Tire strategy decision engine')
st.text('A prototype decision engine for tire strategy')

if "fitness" not in st.session_state:
    #st.session_state.fitness = np.load('fitness.npy',allow_pickle='TRUE').item()
    st.session_state.fitness = []
    
if "strategy" not in st.session_state:
    st.session_state.strategy = [0,1,1,9,30]  

if "figure_1_placeholder" not in st.session_state:
    st.session_state.figure_1_placeholder = None
    
if "figure_2_placeholder" not in st.session_state:
    st.session_state.figure_2_placeholder = None

if "figure_3_placeholder" not in st.session_state:
    st.session_state.figure_3_placeholder = None
    
if "figure_4_placeholder" not in st.session_state:
    st.session_state.figure_4_placeholder = None    
    
if "plotly_1_placeholder" not in st.session_state:
    st.session_state.plotly_1_placeholder = []
    
if "plotly_2_placeholder" not in st.session_state:
    st.session_state.plotly_2_placeholder = [] 
    
if "plotly_3_placeholder" not in st.session_state:
    st.session_state.plotly_3_placeholder = [] 
    
if "plotly_4_placeholder" not in st.session_state:
    st.session_state.plotly_4_placeholder = []     
    
if "stop_strategy" not in st.session_state:
    st.session_state.stop_strategy = 2     
    
    
    
if "solutions" not in st.session_state:
    st.session_state.solutions = np.load('solutions.npy',allow_pickle='TRUE').item()
    #st.session_state.solutions = [] 
    

if "tire_dict" not in st.session_state:
    st.session_state.tire_dict ={
    1: {'Compound':'Soft','init_lap_time':90,'degradation':0.3},
    2: {'Compound':'Medium','init_lap_time':91,'degradation':0.08},
    3: {'Compound':'Hard','init_lap_time':92,'degradation':0.09},
    }

if "n_laps" not in st.session_state:
    st.session_state.n_laps = 50
    




def simulate_strategy(strategy):

    # Parse selections
    if st.session_state.stop_strategy == 1:
        
        compounds_selection = strategy[0:2]
        stop_selection = [strategy[2]]
    else:
        compounds_selection = strategy[0:3]
        stop_selection = strategy[3:5]

    # Initial strategy
    current_compound_index = 0

    # Initial performance
    current_compound = compounds_selection[current_compound_index]
    
    prop = st.session_state.tire_dict[str(current_compound+1)]
    current_max_lap_time = float(prop['init_lap_time'])
    current_degradation = float(prop['degradation'])
    
    if strategy[2] == 0:
        n_strategy = 1
    else:
        n_strategy = 2

    lap_times_arr = []
    compound_arr = []
    degradation_arr = []
    lap_on_tire = 0
    
    for lap in range(1,st.session_state.n_laps+1):
        
        if lap in stop_selection:

            # Counter
            current_compound_index = current_compound_index+1
            
            # Get compund type
            current_compound = compounds_selection[current_compound_index]
            
            # Get tire properties
            prop = st.session_state.tire_dict[str(current_compound+1)]
            current_max_lap_time = float(prop['init_lap_time'])
            current_degradation = float(prop['degradation'])
            lap_on_tire = 0
            
        lap_on_tire = lap_on_tire+1

        lap_time = current_max_lap_time + current_degradation*lap_on_tire
            
        degradation_arr.append(current_degradation)
        lap_times_arr.append(lap_time)        
        compound_arr.append(current_compound)

    t_total_race = np.sum(lap_times_arr)/60
    total_degradation = np.sum(degradation_arr)
    
    return t_total_race,n_strategy,lap_times_arr,compound_arr,total_degradation


# Plot handles

st.subheader('Decision engine')
my_bar = st.progress(0, text="Tool is ready.")    
col1, col2,col3 = st.columns([0.5,1,1],gap='large')
with col1:
    st.write('Settings:')
    optimise = st.button('Optimise')
    number = st.number_input('Number of laps',value=st.session_state.n_laps,min_value=1,max_value=50,step=1)
    st.write('Tire settings')
    st.session_state.tire_dict = st.data_editor(st.session_state.tire_dict,width=300)
    
    n_sol = st.slider('Number of solutions?', 0, 100, 25)
        
with col2:
    st.write('Optimisation:')
    
    if st.session_state.figure_3_placeholder is not None:
        st.session_state.plotly_3_placeholder.plotly_chart(st.session_state.figure_3_placeholder,use_container_width=True)
    else:
        df = pd.DataFrame({'Iterations':[0],'Fitness': [0]})
        st.session_state.figure_3_placeholder = px.line(df,x="Iterations",y="Fitness")
        st.session_state.figure_3_placeholder.update_layout(height=300,margin=dict(r=20,b=10,l=10,t=10))
        st.session_state.plotly_3_placeholder = st.plotly_chart(st.session_state.figure_3_placeholder,use_container_width=True)    

with col3:
    st.write('Pareto plot:')
    if st.session_state.figure_4_placeholder is not None:
        st.session_state.plotly_4_placeholder.plotly_chart(st.session_state.figure_4_placeholder,use_container_width=True)
    else:
        df = pd.DataFrame({'Lap time':[0],'Total race time': [0]})
        st.session_state.figure_4_placeholder = px.line(df,x="Lap time",y="Total race time")
        st.session_state.figure_4_placeholder.update_layout(height=300,margin=dict(r=20,b=10,l=10,t=10))
        st.session_state.plotly_4_placeholder = st.plotly_chart(st.session_state.figure_4_placeholder,use_container_width=True)    
    

    

st.subheader('Performance comparison')

sol_select = st.multiselect(
    'Select solution',
    range(1,2*(n_sol+1)-1),
    [1,2,3])



col1, col2 = st.columns([0.8,1],gap='small')
with col1:
    st.write('Selected strategy:')
    strategy_placeholder = st.empty()
with col2:
    if st.session_state.figure_1_placeholder is not None:
        st.session_state.plotly_1_placeholder.plotly_chart(st.session_state.figure_1_placeholder,use_container_width=True)
        st.session_state.plotly_2_placeholder.plotly_chart(st.session_state.figure_2_placeholder,use_container_width=True)
    else:
        
        t_total_race,n_strategy,lap_times_arr,compound_arr,total_degradation = simulate_strategy(st.session_state.strategy)
        df = pd.DataFrame({'Lap':list(range(1,st.session_state.n_laps+1)),'LapTime': lap_times_arr})
        st.session_state.figure_1_placeholder = px.line(df,x="Lap",y="LapTime")
        st.session_state.plotly_1_placeholder = st.plotly_chart(st.session_state.figure_1_placeholder,use_container_width=True)

        df = pd.DataFrame({'Lap':list(range(1,st.session_state.n_laps+1)),'Compound': compound_arr})
        st.session_state.figure_2_placeholder = px.line(df,x="Lap",y="Compound")
        st.session_state.plotly_2_placeholder = st.plotly_chart(st.session_state.figure_2_placeholder,use_container_width=True)


if optimise:

    def fitness_func(ga_instance, solution, solution_idx):
        
        t_total_race,n_strategy,lap_times_arr,compound_arr,total_degradation = simulate_strategy(solution)
        fitness = 100-t_total_race
        return fitness

    def on_generation(ga_instance):
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        t_total_race,n_strategy,lap_times_arr,compound_arr,total_degradation = simulate_strategy(solution)
        fitness = 100-t_total_race
        
        st.session_state.fitness.append(t_total_race)

        
    fitness_function = fitness_func

    num_generations = 10
    num_parents_mating = 5

    sol_per_pop = 5
    num_genes = 5


    sol_dict = {}



    total_run = 2*n_sol
    start_run = 0
    for stop_strategy in range(1,3):
        st.session_state.stop_strategy = stop_strategy
        if stop_strategy == 1:
            init_range_low =  [0,0,0]
            init_range_high = [3,3,st.session_state.n_laps]
            num_genes = 3
        else:
            init_range_low =  [0,0,0,0,31]
            init_range_high = [3,3,3,30,st.session_state.n_laps]
            num_genes = 5
            
            
        for sol in range(n_sol):
            
            my_bar.progress(start_run/total_run, text="Finding solutions. Please wait...")
            st.session_state.fitness = []    
            ga_instance = pygad.GA(num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                fitness_func=fitness_function,
                                sol_per_pop=sol_per_pop,
                                num_genes=num_genes,
                                init_range_low=init_range_low,
                                init_range_high=init_range_high,
                                gene_type=int,
                                mutation_num_genes=1,
                                mutation_type="random",
                                on_generation=on_generation)
            ga_instance.run()

            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            
            st.session_state.strategy = solution

            df = pd.DataFrame({'Iterations':list(range(len(st.session_state.fitness))),'Fitness': st.session_state.fitness})
            st.session_state.figure_3_placeholder = px.line(df,x="Iterations",y="Fitness")
            st.session_state.figure_3_placeholder.update_layout(height=300,margin=dict(r=20,b=10,l=10,t=10))
            st.session_state.plotly_3_placeholder.plotly_chart(st.session_state.figure_3_placeholder,use_container_width=True)
            
            t_total_race,xx,lap_times_arr,compound_arr,total_degradation = simulate_strategy(solution)
            
            if stop_strategy == 1:
                s = solution[0:2]
                c_str = []
                for i in s:
                    cmp = st.session_state.tire_dict[str(i+1)]['Compound']
                    c_str.append(cmp)
                c_str = '- '.join([str(i) for i in c_str])
                
                stop_lap = str(st.session_state.strategy[2])

            else:
                
                s = solution[0:3]
                c_str = []
                for i in s:
                    cmp = st.session_state.tire_dict[str(i+1)]['Compound']
                    c_str.append(cmp)
                c_str = '- '.join([str(i) for i in c_str])
                
                s = st.session_state.strategy[3:5]
                stop_lap = ', '.join([str(i) for i in s])

            
            sol_dict[start_run] = {'Strategy_arr':solution,'Solution number':start_run,'Strategy':c_str,'Stop lap':stop_lap,'Total time':t_total_race,'Stop strategy':stop_strategy,'Time loss':total_degradation}
            start_run = start_run+1
        
        
        
    st.session_state.solutions = sol_dict  
    #np.save('solutions.npy', sol_dict) 
    #np.save('fitness.npy', st.session_state.fitness) 
    
    my_bar.empty()



# Update dashboard
sol = st.session_state.solutions
if sol:
    df = pd.DataFrame(sol).T

    st.session_state.figure_4_placeholder = px.scatter(df,x="Total time",y="Time loss",color="Stop strategy",
                                                       hover_data={'Solution number':True,
                                                                   'Strategy':True,
                                                                   'Stop lap':True,
                                                    
                                                       })
    st.session_state.figure_4_placeholder.update_layout(height=400,margin=dict(r=20,b=10,l=10,t=10))
    st.session_state.plotly_4_placeholder.plotly_chart(st.session_state.figure_4_placeholder,use_container_width=True)


df = pd.DataFrame({'Iterations':list(range(len(st.session_state.fitness))),'Fitness': st.session_state.fitness})
st.session_state.figure_3_placeholder = px.line(df,x="Iterations",y="Fitness")
st.session_state.figure_3_placeholder.update_layout(height=400,margin=dict(r=20,b=10,l=10,t=10))
st.session_state.plotly_3_placeholder.plotly_chart(st.session_state.figure_3_placeholder,use_container_width=True)



# Update perf comparison

if st.session_state.solutions:
    
    disp_dict = {}
    fig1 = go.Figure()
    fig2 = go.Figure()
    for sol_index in sol_select:
        solution = st.session_state.solutions[sol_index]
        strategy = solution['Strategy_arr']
        t_total_race,n_strategy,lap_times_arr,compound_arr,total_degradation = simulate_strategy(strategy)

        disp_dict[sol_index] = {'Solution':sol_index,'Strategy':solution['Strategy'],'Stop lap':solution['Stop lap'],'Total time':solution['Total time'],'Stop strategy':solution['Stop strategy'],'Time loss':solution['Time loss']}
    
        trace1= go.Scatter(x=list(range(1,st.session_state.n_laps+1)), y=lap_times_arr, name='Sol ' + str(sol_index) + ': ' +solution['Strategy'])
        fig1.add_trace(trace1)
        
        trace2= go.Scatter(x=list(range(1,st.session_state.n_laps+1)), y=compound_arr, name='Sol ' + str(sol_index) + ': ' +solution['Strategy'])
        fig2.add_trace(trace2)
        
        
    strategy_placeholder.dataframe(pd.DataFrame(disp_dict).T,height=500)
    
    st.session_state.figure_1_placeholder = fig1
    st.session_state.figure_1_placeholder.update_layout(
        height=300,
        margin=dict(r=20,b=10,l=10,t=10),
        showlegend=True, 
        xaxis_title='Lap number',
        yaxis_title='Lap time',
        )
    st.session_state.plotly_1_placeholder.plotly_chart(st.session_state.figure_1_placeholder,use_container_width=True)
        
    st.session_state.figure_2_placeholder = fig2
    st.session_state.figure_2_placeholder.update_layout(
        height=300,
        margin=dict(r=20,b=10,l=10,t=10),
        showlegend=True, 
        xaxis_title='Lap number',
        yaxis_title='Compound',
        )
    st.session_state.plotly_2_placeholder.plotly_chart(st.session_state.figure_2_placeholder,use_container_width=True)







#s = st.session_state.strategy[0:3]
#c_str = []
#for i in s:
#    cmp = st.session_state.tire_dict[str(i+1)]['Compound']
#    c_str.append(cmp)
#c_str = '- '.join([str(i) for i in c_str])
#strategy_placeholder.write(c_str)

#s = st.session_state.strategy[3:5]
#s = ', '.join([str(i) for i in s])
#stop_placeholder.write("- " + s)

#time_placeholder.write(f'- {t_total_race:.2f} minutes')



