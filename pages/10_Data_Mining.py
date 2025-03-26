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

if 'year' not in st.session_state:
    st.session_state.year = []

if 'race' not in st.session_state:
    st.session_state.race = []

if 'lap' not in st.session_state:
    st.session_state.lap = []

if 'total_laps' not in st.session_state:
    st.session_state.total_laps = []
    st.session_state.race_name = []

    #df.to_pickle("./Page4_init.pkl")

if 'analysis_figure' not in st.session_state:
    st.session_state.analysis_figure = []
    
if 'analysis_figure_2' not in st.session_state:
    st.session_state.analysis_figure_2 = []
    
if 'analysis_figure_3' not in st.session_state:
    st.session_state.analysis_figure_3 = []


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



# Initialise

# Load anomaly detection model
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
def GetRaceInfo():
    #dbcol = db[f"race_{st.session_state.year}_{st.session_state.race}"]
    race_info = db_info[f"race_{st.session_state.year}_{st.session_state.race}"]
    cur = race_info.find_one()
    st.session_state.total_laps = cur['total_laps']
    st.session_state.race_name = cur['Name']       
    

def GetRaceData(year,race,lap):

    dbcol = db[f"race_{year}_{race}"]
    race_dict = dbcol.find_one({"LapNumber": lap})
    tel = race_dict['Telemetry']
    df = pd.DataFrame(tel)

    df = df[["Time","RPM","Speed","nGear","Throttle","Distance"]]

    return df


def PlotTelemetry(fig2,df,ToPlot,cc,trans,name):

    # Time series plot
    for index,channel in enumerate(ToPlot):

        x=df['Time']
        y=df[channel]

        if legend_y<20:
            if index==0:
                legend_flag = True
            else:
                legend_flag = False
        else:
            legend_flag = True
        fig2.add_trace(go.Scatter(
            x=x,
            y=y,
            line=dict(width=4, color=cc),
            mode='lines',
            yaxis="y",
            name= f"{name}  [{channel}]",
            opacity=trans,
            legendgroup=str(index+1),
            showlegend = legend_flag,
        ),row=index+1, col=1)

        fig2.update_yaxes(title_text=channel, row=index+1, col=1)

        if index+1==len(ToPlot):

            fig2.update_xaxes(rangeslider= {'visible':True}, row=index+1, col=1,rangeslider_thickness = 0.05)



        if legend_y>20:
            fig2.update_layout(legend_tracegroupgap=legend_y)

    return fig2



st.subheader("Data Mining and Exploration Tool (Still under development)")

st.caption("Optimized for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")

with st.expander('Data selection',expanded=True):

    col1,col2 = st.columns([0.5,1],gap="medium")
    with col1:
        
        st_title("Race selection")
        st.write("Select season and race using the sliders below. The selected race telemetry data will be plotted as blue in the analysis plot.")
        st.session_state.year = st.slider(
            'Select season',2022,2023,2023)
        st.session_state.race = st.slider(
            f'Select races:',1,22,6)
        GetRaceInfo() 

        st.session_state.lap = st.slider(
                    f'Select lap:',1,st.session_state.total_laps,51)
        
    with col2:
        st_title("Race info")
        st.write("Selected race metrics. The time delta is between previous and current lap time.")

        st.text(f'Selected race: {st.session_state.race_name}')
        row1,row2,row3 = st.columns([1,1,1],gap="medium")
        with row1:
            metric1_placeholder = st.metric("Driver", Driver)        
        with row2:
            metric2_placeholder = st.metric("Lap time", "")
        with row3:
            metric3_placeholder = st.metric("Position", "")
        row1,row2,row3 = st.columns([1,1,1],gap="medium")
        with row1:
            metric4_placeholder = st.metric("Compound", "")        
        with row2:
            metric5_placeholder = st.metric("Ambient temp", "")

with st.expander('Visualisation canvas',expanded=False):

    if not st.session_state.analysis_figure:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='lines',
            yaxis="y",
        ))
        analysis_placeholder = st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
    else:
        analysis_placeholder = st.plotly_chart(st.session_state.analysis_figure, theme="streamlit", use_container_width=True)


    
with st.expander('Visualisation canvas',expanded=True):

    col1, col2 = st.columns([1,1],gap='Medium')

    with col1:

        options = st.multiselect(
            "Choose two channels to display:",
            ["Accel","Brake","Modes","nGear","RPM","Speed","Throttle","X","Y","Z"],
            ["RPM", "Speed"],max_selections=2)
        
        modes = st.multiselect(
            "Add modes to the scatter plot?",
            ["Accel","Brake","Cruise","Overrun"])
        
        show_current = st.checkbox('Show current lap')
        
        analysis_2_placeholder = st.empty()
        
    
    with col2:
        
        
        analysis_3_placeholder = st.empty()
        
        
        
        
        







# Race info
dbcol = db[f"race_{st.session_state.year}_{st.session_state.race}"]
item_dict = dbcol.find({},{ "_id": 0, "LapNumber": 1, "LapTime": 1 , "Compound": 1 , "Position": 1 , "AirTempMean": 1 })
df_laps = pd.DataFrame(item_dict)
df_laps['Year'] = st.session_state.year
df_laps['Round'] = st.session_state.race

column_to_move = df_laps.pop("Round")
df_laps.insert(0, "Round", column_to_move)

column_to_move = df_laps.pop("Year")
df_laps.insert(0, "Year", column_to_move)

# Set metrics
n = st.session_state.lap
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

metric2_placeholder.metric("Lap time", f"{t_lap_str}", "%.3f" % dt_lap_0)
metric3_placeholder.metric("Position", "%1.f" % p_lap, "%1.f" % dp_lap_0)
metric4_placeholder.metric("Compound", f"{c_lap}")
metric5_placeholder.metric("Ambient temp", "%.1f" % a_lap + u" \u00b0C")




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
    

def UpdatePlots():

    # Subplots
    ToPlot = ['RPM','Brake','Throttle','Modes','Speed']

    # Initialise figures
    n = int(len(ToPlot))
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, row_heights=[0.2]*len(ToPlot))


    # Read from mongo
    lap = st.session_state.lap
    race_dict = dbcol.find_one({"LapNumber": lap})
    tel = race_dict['Telemetry']
    df = pd.DataFrame(tel)
    df = AddAdditionalTelemetryData(df)
    #df.to_pickle('telemetry_71.pkl')

    fig = PlotTelemetry(fig,df,ToPlot,"deepskyblue",1,f"Lap {lap}")

    # Hide legend for now
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(height=st.session_state.PlotHeight)


    return fig


fig = UpdatePlots()
st.session_state.analysis_figure = fig
analysis_placeholder.plotly_chart(fig, theme="streamlit",height=1300)





def UpdateScatter():

    fig = go.Figure()

    if len(options)>1:
        
        data = []

        #n = range(1,st.session_state.total_laps+1)
        n = range(10,20)
        for lap in n:
            
            race_dict = dbcol.find_one({"LapNumber": lap})
            tel = race_dict['Telemetry']
            df = pd.DataFrame(tel)
            df = AddAdditionalTelemetryData(df)     
            
            color_dict = {
                "Overrun": "red",
                "Brake": "yellow",
                "Cruise": "lime"
            }               
            
            fig.add_trace(go.Scatter(x=df[options[0]], y=df[options[1]],
                            mode='lines',
                            line=dict(color='rgba(135, 206, 250, 0.03)',width=7),
                            name='markers'))
            
            color_dict = {
                "Overrun": 'rgba(255, 255, 0, 0)',
                "Brake": 'rgba(0, 0, 255, 0)',
                "Cruise": 'rgba(0, 255, 255, 0)',
                "Accel":'rgba(255, 0, 0, 0)',
            }    
                       
            for x in color_dict:
                if x in modes:
                    a = color_dict[x]
                    color_dict[x] = a.replace("0)", "0.1)")

            color = df['Modes'].map(color_dict)
            
            
            fig.add_trace(go.Scatter(x=df[options[0]], y=df[options[1]],
                mode='markers',
                marker_color=color,
                name='markers'))
             
    if show_current:
        lap = st.session_state.lap   
        race_dict = dbcol.find_one({"LapNumber": lap})
        tel = race_dict['Telemetry']
        df = pd.DataFrame(tel)
        df = AddAdditionalTelemetryData(df)     
        
        fig.add_trace(go.Scatter(x=df[options[0]], y=df[options[1]],
                    mode='lines',
                    line=dict(color='rgba(135, 206, 100, 0.2)',width=2),
                    name='markers'))



    
    fig.update_layout(xaxis_title=options[0], yaxis_title=options[1])

    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(height=500)


    return fig





fig2 = UpdateScatter()
st.session_state.analysis_2_figure = fig2
analysis_2_placeholder.plotly_chart(fig2, theme="streamlit",height=500)







st.write('Copyright © 2024 Farraen. All rights reserved.')
