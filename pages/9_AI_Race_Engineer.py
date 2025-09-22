import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from openai import OpenAI
import streamlit as st
import time
import plotly.express as px

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


# --------  For page layout  ---------------
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


# ----------- UI Section -------------------------

st_title('AI Race Engineer')

st.caption("Optimized for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")

diag_placeholder = st.empty()




# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

# OpenAI 
@st.cache_resource 
def connect_openAI():
    openai_api_key = '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'
    openai_client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
    )
    return openai_client

client = connect_openAI()

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []


prompt = st.chat_input("What is up?")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        #if "assistant" in message["role"]:
        #    st.plotly_chart(message["Payload2"], theme="streamlit", use_container_width=True)


def send_message(messages):

    persona = [{"role":"system", "content":"You are a racing car engineer"}]
    persona.extend(messages)
    openai_response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": m["role"], "content": m["content"]} for m in persona],
        stream=True)
    
    return openai_response

def send_message_secondary(prompt):

    persona = [{"role":"system", "content":"You are a racing car engineer"}]
    user_messages = [{"role":"user", "content":prompt}]
    user_messages.extend(persona)
    openai_response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": m["role"], "content": m["content"]} for m in user_messages])
    
    return openai_response

def prompt_engine(messages):

    m = messages[-1]["content"]


    openai_response = send_message_secondary(f"For this prompt '{m}', do you need access to the database? Answer yes if you need access to get the data or no if no data is needed.")
    internal_response = openai_response.choices[0].message.content
    #print(response.lower().startswith("yes"))


    #dbcol = db[f"race_{st.session_state.year}_{st.session_state.race}"]
    #race_dict = dbcol.find_one({"LapNumber": lap})
    #tel = race_dict['Telemetry']
    #df = pd.DataFrame(tel)
    #df = AddAdditionalTelemetryData(df)

    response = send_message(messages)

    return response, internal_response

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "Payload1":[], "Payload2":[]})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        #openai_response = send_message(st.session_state.messages)
        openai_response, internal_response = prompt_engine(st.session_state.messages)

        message_placeholder = st.empty()
        full_response = ""
        for response in openai_response:
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        #st.write(internal_response)

        df = px.data.iris()
        fig = px.scatter(
            df,
            x="sepal_width",
            y="sepal_length",
            color="sepal_length",
            color_continuous_scale="reds",
        )
        fig.update_layout(height=200,margin=dict(l=20, r=20, t=20, b=20))
        #st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        st.session_state.messages.append({"role": "assistant", "content": full_response,"Payload1": [],"Payload2": fig})


#st.write('Copyright © 2024 Farraen. All rights reserved.')

