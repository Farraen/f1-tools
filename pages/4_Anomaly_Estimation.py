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
from sklearn.neighbors import LocalOutlierFactor 
import datetime
from sklearn.neighbors import LocalOutlierFactor
from numpy import where
from sklearn.preprocessing import StandardScaler
import pickle


st.set_page_config(layout="wide")

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

if 'select' not in st.session_state:
    #st.session_state.select = pd.DataFrame([],columns=['Lap','Lap time','Position','Compound','Training','Testing'])
    st.session_state.select = pd.read_pickle("data/Page4_init.pkl")  

if 'analysis_figure' not in st.session_state:
    st.session_state.analysis_figure = []

if 'model' not in st.session_state:
    filename = 'data/Page4_model.sav'
    model = pickle.load(open(filename, 'rb'))
    st.session_state.model = model    
    
    filename = 'data/Page4_scaler.sav'
    scaler = pickle.load(open(filename, 'rb'))
    st.session_state.scaler = scaler    

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
st.session_state.PlotHeight = 1300

# Calculate legend offset
n_legend = 4
y0 = 0.234*st.session_state.PlotHeight - 57.67
y1 = 0.044*st.session_state.PlotHeight - 9.67 - 2 + 10
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
            fig2.update_xaxes(rangeslider= {'visible':True}, row=index+1, col=1)

        if legend_y>20:
            fig2.update_layout(legend_tracegroupgap=legend_y)

    return fig2

def PlotAnomaly(fig2,df,ToPlot,name,t_outliers):

    # Time series plot
    for index,channel in enumerate(ToPlot):

        x=df['Time']
        y=df[channel]

        if any(t_outliers):

            x=df.loc[df['Time'].isin(t_outliers),'Time']
            y=df.loc[df['Time'].isin(t_outliers),channel]

            fig2.add_trace(go.Scatter(
                x=x,
                y=y,
                line=dict(width=10, color='yellow'),
                mode='markers',
                yaxis="y",
            name= f"{name} anomalies  [{channel}]",
                opacity=1,
                legendgroup=str(index+1),
                showlegend = True,
            ),row=index+1, col=1)

    return fig2         


st.header("F1 Anomaly Detection")
st.subheader("Farraen's experimental racing tools")
st.write("(For hobby and recreational purposes only)")
st.header(' ')


col1,col2 = st.columns([1,0.3])
with col1:
    with st.expander('Introduction',expanded=True):
        col11,col22 = st.columns([0.25,1])
        with col11:
            str3 = "This is a dashboard for extracting Max's race telemetry data and detecting anomalies to identify faults early.\
                 It involves a relatively basic anomaly detection process due to the limited amount of telemetry data available."
            st.write(str3)
            str3 = "A MongoDB database was set up to store telemetry data for all 2022 and 2023 races. \
                MongoDB is a document-oriented database, and data is stored as JSON documents. \
                    The database is connected via an API over the internet to the dashboard."
            st.write(str3)
            str3 = "For detecting anomalies, the LOF (Local Outlier Factor) algorithm is used. \
                It is trained using several historical telemetry data points and is employed to predict new, unseen data."
            st.write(str3)

        with col22:
            image = read_image("images/Page4_tech.png")
            st.image(image,use_column_width=True)
with col2:
    with st.expander('MongoDB database',expanded=True):
        st.info('Status: ' + st.session_state.db_status, icon="ℹ️")
        col11,col22 = st.columns([0.5,1])
        with col11:
            ping_button = st.button("Ping database")
        with col22:
            db_stat_placeholder = st.empty()



with st.expander('Data visualisation',expanded=True):

    col1,col2 = st.columns([0.5,1],gap="large")
    with col1:
        
        st.subheader("Race selection")
        st.write("Select season and race using the sliders below. The selected race telemetry data will be plotted as blue in the analysis plot.")
        st.session_state.year = st.slider(
            'Select season',2022,2023,2023)
        st.session_state.race = st.slider(
            f'Select races: {st.session_state.race_name}',1,22,6)
        GetRaceInfo() 

        st.session_state.lap = st.slider(
                    f'Select lap:',1,st.session_state.total_laps,51)
        
        st.write("")
        st.subheader("Race info")
        st.write("Selected race metrics. The time delta is between previous and current lap time.")

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

        st.write("")
        st.subheader("Anomaly detection")

        str2 = "Press Add button to add the current race into the training list. \
            Remember to use the check box to confirm the selection. Once selection has been made, \
            press Start training button to initiate anomaly detector training process."
        st.write(str2)
        row1,row2,row3,row4 = st.columns([0.23,0.4,0.5,1])
        with row1:
            select_button = st.button('Add',key="add")
        with row2:
            remove_button = st.button('Remove',key="remove")
        with row3:
            train_button = st.button('Start training',key="train")
        
        test_placeholder = st.empty()
        table_placeholder = st.empty()
        progress_placeholder = st.empty()

        st.write('Estimator specifications:')
        gap,col111 = st.columns([0.01,1])
        with col111:
            estimator1_placeholder = st.empty()
            estimator2_placeholder = st.empty() 
            estimator3_placeholder = st.empty()
            estimator4_placeholder = st.empty()
            estimator5_placeholder = st.empty() 
            estimator6_placeholder = st.empty()
            estimator7_placeholder = st.empty()
        st.write('')


    with col2:
        st.subheader("Visualisation canvas")
        str2 = "This the telemetry plot. The blue lines are the current race telemety. The grey lines \
            are the selected training data. The dark orange lines are the testing set. At the moment the testing set is not used.  \
            The yellow markers are the detected anomalies. Use the range slider at the bottom of the plot to zoom or pan the telemetry plots."
        st.write(str2)
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


if ping_button:
    try:
        client.admin.command('ping')
        db_stat_placeholder.success("Pinged the telemetry. You successfully connected to MongoDB!", icon="✅")
    except Exception as e:
        db_stat_placeholder.error(e, icon="🚨")

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

t_lap_str = datetime.datetime.utcfromtimestamp(t_lap).strftime(' %M:%S:%f').replace(" 0","")[:-3]

metric2_placeholder.metric("Lap time", f"{t_lap_str}", "%.3f" % dt_lap_0)
metric3_placeholder.metric("Position", "%1.f" % p_lap, "%1.f" % dp_lap_0)
metric4_placeholder.metric("Compound", f"{c_lap}")
metric5_placeholder.metric("Ambient temp", "%.1f" % a_lap + u" \u00b0C")


if select_button:
    row0 = st.session_state.select
    
    if not n in row0["Lap"].unique():

        row = pd.DataFrame({'Lap':n,'Lap time':t_lap,'Position':p_lap,'Compound':c_lap,'Training':False,'Testing':False}, index=[0])
        
        if isinstance(row0,pd.DataFrame):
            st.session_state.select = pd.concat([row0, row], ignore_index=True)
        else:
            st.session_state.select = row
        
df_select = table_placeholder.data_editor(st.session_state.select)


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
    
def GetAnomalies(df):

    model = st.session_state.model
    scaler = st.session_state.scaler

    t_test = df['Time'].values
    X_test = df[st.session_state.features].values
    X_test_scaled = scaler.fit_transform(X_test)
    ypred = model.predict(X_test_scaled)
    outlier_index = where(ypred==-1)
    t_outliers = t_test[outlier_index]

    return t_outliers

def UpdatePlots(df_select):

    # Subplots
    ToPlot = ['RPM','Brake','Throttle','Modes','Speed']

    # Initialise figures
    n = int(len(ToPlot))
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, row_heights=[0.2]*len(ToPlot))


    for index, row in df_select.iterrows():

        # Skip to prevent accessing mongo
        if (not row['Training']) and (not row['Testing']):
            continue
            
        # Read from mongo
        race_dict = dbcol.find_one({"LapNumber": row['Lap']})
        tel = race_dict['Telemetry']
        df = pd.DataFrame(tel)
        df = AddAdditionalTelemetryData(df)

        if row['Training']:
            fig = PlotTelemetry(fig,df,ToPlot,"lightgrey",0.3,f"Training")

        if row['Testing']:
            fig = PlotTelemetry(fig,df,ToPlot,"orange",0.3,f"Testing")



    # Read from mongo
    lap = st.session_state.lap
    race_dict = dbcol.find_one({"LapNumber": lap})
    tel = race_dict['Telemetry']
    df = pd.DataFrame(tel)
    df = AddAdditionalTelemetryData(df)
    t_outliers = GetAnomalies(df)
    
    fig = PlotTelemetry(fig,df,ToPlot,"deepskyblue",1,f"Lap {lap}")
    fig = PlotAnomaly(fig,df,ToPlot,f"Lap {lap}",t_outliers)

    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))

    fig.update_layout(height=st.session_state.PlotHeight)

    return fig


fig = UpdatePlots(df_select)
st.session_state.analysis_figure = fig
analysis_placeholder.plotly_chart(fig, theme="streamlit",height=1300)





if train_button:

    with progress_placeholder, st.spinner('Training in progress'):

        dff=[]
        offset = 0
        for index,row in df_select.iterrows():
            df = GetRaceData(st.session_state.year,st.session_state.race,row['Lap'])
            df["Time"] = df["Time"] + offset
            #df["Time"] = df["Time"]
            offset = df.loc[df.index[-1],"Time"]
            dff.append(df)
        df_train = pd.concat(dff, axis=0).reset_index()

        X_train = df_train[st.session_state.features].values
        t_train = df_train['Time'].values 


        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LocalOutlierFactor(novelty=True)
        model.fit(X_train_scaled)

        #filename = 'model.sav'
        #pickle.dump(model, open(filename, 'wb'))
        st.session_state.model = model

        #filename = 'scaler.sav'
        #pickle.dump(scaler, open(filename, 'wb'))
        st.session_state.scaler = scaler

        fig = UpdatePlots(df_select)
        st.session_state.analysis_figure = fig
        analysis_placeholder.plotly_chart(fig, theme="streamlit",height=1300)


    progress_placeholder.success('Model updated!', icon="✅")


# Diplay estimator specs
estimator1_placeholder.text(f'1. Detector type: {st.session_state.model.__module__}')
estimator2_placeholder.text(f'2. Learning mode: Unsupervised learning')        
estimator3_placeholder.text(f'3. Novelty: ' + str(st.session_state.model.__dict__['novelty']))
estimator4_placeholder.text(f'4. Number of neighbors: ' + str(st.session_state.model.__dict__['n_neighbors']))
estimator5_placeholder.text(f'5. Number of samples: ' + str(st.session_state.model.__dict__['n_samples_fit_']))
estimator6_placeholder.text(f'6. Transformation: {st.session_state.scaler.__class__.__name__}')
estimator7_placeholder.text(f'7. Features: ' + ", ".join(st.session_state.features))









st.write('Copyright © 2024 Farraen. All rights reserved.')
