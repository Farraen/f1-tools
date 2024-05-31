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


cols2 = plotly.colors.DEFAULT_PLOTLY_COLORS
cols = px.colors.qualitative.Light24

if 'db_status' not in st.session_state:
    st.session_state.db_status = ""

if 'df_laps' not in st.session_state:
    st.session_state.df_laps = []

if 'selected' not in st.session_state:
    st.session_state.selected = []

if 'year' not in st.session_state:
    st.session_state.year = []

if 'race' not in st.session_state:
    st.session_state.race = []

if 'lap' not in st.session_state:
    st.session_state.lap = []

if 'total_laps' not in st.session_state:
    st.session_state.total_laps = []
    st.session_state.race_name = []
    
    
if 'window' not in st.session_state:
    st.session_state.window = [11600,12000]
       

if 'select' not in st.session_state:
    #st.session_state.select = pd.DataFrame([],columns=['Lap','Lap time','Position','Compound','Training','Testing'])
    df = pd.read_pickle("data/Page4_init.pkl")  
    #df.columns = ["Lap", "Lap time", "Position", "Compound", "Training","Remove"]
    st.session_state.select = df
    #df.to_pickle("./Page4_init.pkl")

if 'analysis_figure' not in st.session_state:
    st.session_state.analysis_figure = []
    
if 'analysis_figure_2' not in st.session_state:
    st.session_state.analysis_figure_2 = []
    
if 'analysis_figure_3' not in st.session_state:
    st.session_state.analysis_figure_3 = []

if 'model' not in st.session_state:
    filename = 'data/Page4_model.sav'
    model = pickle.load(open(filename, 'rb'))
    st.session_state.model = model    
    
    filename = 'data/Page4_scaler.sav'
    scaler = pickle.load(open(filename, 'rb'))
    st.session_state.scaler = scaler    

if 'select_normal' not in st.session_state:
    st.session_state.select_normal = [27,28,29]
    
if 'select_failure' not in st.session_state:
    st.session_state.select_failure = [38]
    
    
Driver = "Max"


# Support functions -------------------------------------

# MongoDB
@st.cache_resource 
def connect_mongo():
    mongoUser = 'farraen'
    mongoPwd = 'rI68TwqYQTSDu5Pp'
    mongoDb = 'f1_analysis_max' 
    mongoDb2 = 'f1_info'  

    uri = f"mongodb+srv://{mongoUser}:{mongoPwd}@cluster0.rjtb7gz.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri, server_api=ServerApi('1'))

    try:
        client.admin.command('ping')
        db_status = "Connected"
    except Exception as e:
        db_status = "Connection error"


    db = client[mongoDb]
    db_info = client[mongoDb2]

    return db,db_info,client,db_status

# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

# For loading model
@st.cache_resource
def load_model():
    filename = 'data/Page4_model.sav'
    model = pickle.load(open(filename, 'rb'))
    st.session_state.model = model

    filename = 'data/Page4_scaler.sav'
    scaler = pickle.load(open(filename, 'rb'))
    st.session_state.scaler = scaler


    return model, scaler


# Initialise

# Load anomaly detection model
st.session_state.model, st.session_state.scaler = load_model()
st.session_state.features = ["Distance","RPM","Speed","Throttle"]


# Plot settings
st.session_state.PlotHeight = 1000

# Calculate legend offset
n_legend = 3
y0 = 0.234*st.session_state.PlotHeight - 57.67
y1 = 0.044*st.session_state.PlotHeight - 9.67 -40
legend_y = -18.88 * n_legend + y0 - y1

if n_legend < 3:
    cc = cols2
else:
    cc = cols

# Connect to database
db,db_info,client,st.session_state.db_status = connect_mongo()

# Function 
def GetRaceInfo(year,race):
    race_info = db_info[f"race_{year}_{race}"]
    cur = race_info.find_one()
    total_laps = cur['total_laps']
    race_name = cur['Name']       
    
    return total_laps, race_name

def GetRaceData(year,race,lap):

    dbcol = db[f"race_{year}_{race}"]
    race_dict = dbcol.find_one({"LapNumber": lap})
    tel = race_dict['Telemetry']
    df = pd.DataFrame(tel)

    df = df[["Time","RPM","Speed","nGear","Throttle","Distance"]]

    return df

def AddAdditionalTelemetryData(df):
    x=df['Time']
    acc =(df['Speed'].diff()>0) | (df['Throttle']>0)
    brk =(df['Speed'].diff()<=0) & (df['Brake']>0)
    over = (df['Throttle'].diff()<=0) & (df['Brake']==True) & (df['RPM'].diff()>0) 
    y = ['Cruise'] * len(x)
    df['Modes'] = y
    df.loc[acc,'Modes'] = 'Accel'
    df.loc[brk,'Modes'] = 'Brake'
    df.loc[over,'Modes'] = 'Overrun'
    df['Accel'] = acc

    return df

def GetLapData(n):

    t_lap = df_laps.loc[df_laps["LapNumber"] == n,"LapTime"].to_numpy()[0]
    c_lap = df_laps.loc[df_laps["LapNumber"] == n,"Compound"].to_numpy()[0]
    p_lap = df_laps.loc[df_laps["LapNumber"] == n,"Position"].to_numpy()[0]
    a_lap = df_laps.loc[df_laps["LapNumber"] == n,"AirTempMean"].to_numpy()[0]

    if n>1:
        t_lap_0 = df_laps.loc[df_laps["LapNumber"] == n-1,"LapTime"].to_numpy()[0]
        p_lap_0 = df_laps.loc[df_laps["LapNumber"] == n-1,"Position"].to_numpy()[0]
        dt_lap_0 = t_lap - t_lap_0
        dp_lap_0 = p_lap - p_lap_0
    else:
        dt_lap_0 = 0
        dp_lap_0 = 0   

    t_lap_str = datetime.datetime.fromtimestamp(t_lap).strftime(' %M:%S:%f').replace(" 0","")[:-3]


    race_dict = dbcol.find_one({"LapNumber": n})
    tel = race_dict['Telemetry']
    df = pd.DataFrame(tel)
    df = AddAdditionalTelemetryData(df)     
    
    
    return df, t_lap_str, c_lap, p_lap, a_lap

def is_monotonic(arr):
    increasing = all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
    decreasing = all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))

    if increasing:
        return('positive')
    elif decreasing:
        return('negative')
    else:
        return('not monotonic')



# Dashboard

st.subheader("Prognostics")

with st.expander('Data visualisation',expanded=True):

    col1, col2 = st.columns([1,1],gap='large')
    
    with col1:
        longstr = '''
        
        A dashboard designed to explore the feasibility of predicting early power unit (PU) failures
        by analyzing a limited number of channels. The default race is the 2022 Australian Grand Prix, 
        where Max experienced an early PU failure. This dashboard employs basic statistical methods to establish 
        a monitoring window that alerts the race engineer or driver to potential faults.

        With additional data, more sophisticated monitoring systems incorporating machine learning models 
        can be developed. Some underlying patterns may be challenging to identify and comprehend.
            
            '''
        st.write(longstr)
        
    with col2:
        image = read_image("images/max.jpg")
        st.image(image,use_column_width=True)
    

with st.expander('Data visualisation',expanded=True):

    col1,col2 = st.columns([0.5,1],gap="medium")
    with col1:
        
        st_title("Race selection")
        st.write("Select season and race using the sliders below. The selected race telemetry data will be plotted as blue in the analysis plot.")
        year = st.slider(
            'Select season',2022,2023,2022)
        race = st.slider(
            f'Select races:',1,22,3)
        
        total_laps, race_name = GetRaceInfo(year,race) 

    with col2:
        st_title("Race info")
        st.write("Selected race metrics. The time delta is between previous and current lap time.")

        st.text(f'Selected race: {year} {race_name}')
        st.text(f'Driver: {Driver}')
        



        dbcol = db[f"race_{year}_{race}"]
        item_dict = dbcol.find({},{ "_id": 0, "LapNumber": 1, "LapTime": 1 , "Compound": 1 , "Position": 1 , "AirTempMean": 1 })
        df_laps = pd.DataFrame(item_dict)
        df_laps['Year'] = year
        df_laps['Round'] = race

        if df_laps['LapNumber'].iloc[-1] == total_laps:
            st.text(f'Status: Completed race')
            complete_flag = 1
        else:
            lp_number = np.round(df_laps['LapNumber'].iloc[-1])
            st.text('Status: Retired at lap ' + str(lp_number))
            complete_flag = 0
        last_lap = df_laps['LapNumber'].iloc[-1]



        column_to_move = df_laps.pop("Round")
        df_laps.insert(0, "Round", column_to_move)

        column_to_move = df_laps.pop("Year")
        df_laps.insert(0, "Year", column_to_move)
        
        y = [False] * len(df_laps.index)
        df_laps['Normal'] = y
        df_laps['Failure'] = y

        df_laps.loc[st.session_state.select_normal,'Normal'] = True
        df_laps.loc[st.session_state.select_failure,'Failure'] = True

        df_laps = st.data_editor(df_laps)


#laps = st.slider(f'Select lap:',1,total_laps,51)

st.session_state.select_normal = np.where(df_laps['Normal'].values)[0]
st.session_state.select_failure = np.where(df_laps['Failure'].values)[0]



with st.expander('Analysis',expanded=True):

    col1, col2 = st.columns([1,1],gap='large')
    with col1:
        st.subheader("Normal lap RPM distributions")
        st.write('Normal laps should have similar distribution pattern particukary at high engine speeds. Area of interest is between 11KRPM and 12KRPM.')
        fig = go.Figure()
        for n in st.session_state.select_normal.tolist():
            df_1, t_lap_str_1, c_lap_1, p_lap_1, a_lap_1 = GetLapData(n)
            fig.add_trace(go.Histogram(x=df_1['RPM'],name="Lap " + str(n+1)+ " (Normal)"))
        fig.add_vrect(x0=st.session_state.window[0], x1=st.session_state.window[1], line_width=4, line_color="red", fillcolor="pink", opacity=0.2,annotation_text="Monitoring window")
        fig.update_traces(opacity=0.75)
        fig.update_layout(barmode='overlay')
        fig.update_layout(margin=dict(l=20, r=20, t=10, b=20),height=300)
        fig.update_layout(xaxis_title='RPM', yaxis_title="Frequency")
        st.plotly_chart(fig, theme="streamlit")

    with col2:
        st.subheader("Normal+Failed lap RPM distributions")
        st.write('When PU degradation is happening, the RPM distribution trends seems to change for some reason.')

        fig = go.Figure()
        for n in st.session_state.select_normal.tolist():
            df_1, t_lap_str_1, c_lap_1, p_lap_1, a_lap_1 = GetLapData(n)
            fig.add_trace(go.Histogram(x=df_1['RPM'],name="Lap " + str(n+1)+ " (Normal)"))
        
        for n in st.session_state.select_failure.tolist():
            df_1, t_lap_str_1, c_lap_1, p_lap_1, a_lap_1 = GetLapData(n)
            fig.add_trace(go.Histogram(x=df_1['RPM'],name="Lap " + str(n+1)+ " (Failure)"))
        fig.add_vrect(x0=st.session_state.window[0], x1=st.session_state.window[1], line_width=4, line_color="red", fillcolor="pink", opacity=0.2,annotation_text="Monitoring window")
        fig.update_traces(opacity=0.75)
        fig.update_layout(barmode='overlay')
        fig.update_layout(margin=dict(l=20, r=20, t=10, b=20),height=300)
        fig.update_layout(xaxis_title='RPM', yaxis_title="Frequency")
        st.plotly_chart(fig, theme="streamlit")






    st.subheader("PU health pattern until failure")
    a = []
    for n in range(1,int(last_lap)):
        race_dict = dbcol.find_one({"LapNumber": n})
        tel = race_dict['Telemetry']
        df = pd.DataFrame(tel)
        s = sum(df.RPM.between(st.session_state.window[0], st.session_state.window[1]))
        a.append(s)
        

    df = pd.DataFrame(a, columns=["RPM count"])
    df['Lap'] = df.index+1

    x = list(range(1,int(last_lap)))
    y = df['RPM count']
    fitted_x = np.linspace(min(x), max(x), 20)
    fitted_y = sm.nonparametric.lowess(y, x, xvals=fitted_x)
    
    fitted_y_rs = np.interp(fitted_y, [0,100], [0,1])
    dy = np.gradient(fitted_y_rs)
    thresh = 0.01
    idx_thresh = np.argmax(dy > thresh)
    
    
        

    fitted_x_mono = np.linspace(min(x), max(x),5)
    fitted_y_mono = sm.nonparametric.lowess(y, x, xvals=fitted_x_mono)
    a = is_monotonic(fitted_y_mono)
    
    
    
    npm = np.median(fitted_y[idx_thresh:])

    fig = go.Figure()

    if a == "positive" and npm>40:
        fig.add_vrect(x0=fitted_x[idx_thresh], x1=last_lap, line_width=4, line_color="red", fillcolor="pink", opacity=0.1,annotation_text="Monitoring window")
    
    st.write('Reduction in engine rpm limit in monitoring window is necessary to prevent further damage to the PU.')
    st.write('RPM trendline monotonicity: ' + a)
    fig.add_trace(go.Scatter(x=df['Lap'],y=df['RPM count'],name="RPM in window count",mode='markers',
                    line=dict(color='rgba(0, 185, 255, 1)',width=2)))
    fig.add_trace(go.Scatter(x=fitted_x,y=fitted_y,name="Trendline",mode='lines',
                    line=dict(color='rgba(255, 50, 50, 1)',width=2)))

    fig.update_layout(yaxis_range=[0,80])

    fig.update_layout(margin=dict(l=20, r=20, t=10, b=20),height=300)
    fig.update_layout(xaxis_title='Lap number', yaxis_title="RPM window count")

    st.plotly_chart(fig, theme="streamlit")
    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fitted_x,y=dy,name="Trendline",mode='markers',
                    line=dict(color='rgba(250, 250, 0, 1)',width=2)))
    fig.add_hline(y=thresh, line_width=3, line_dash="dash", line_color="red",annotation_text="Turning point threshold")

    fig.update_layout(margin=dict(l=20, r=20, t=10, b=20),height=300)
    fig.update_layout(xaxis_title='Lap number', yaxis_title="RPM first derivative")

    st.plotly_chart(fig, theme="streamlit")
    
    
    

    

    
st.write('Copyright © 2024 Farraen. All rights reserved.')
