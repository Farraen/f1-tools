import streamlit as st
from PIL import Image
from streamlit_d3graph import d3graph
import numpy as np
import pandas as pd
import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from openai import OpenAI
import time


st.set_page_config(layout="wide")


if "openai_model" not in st.session_state:
    #st.session_state["openai_model"] = "gpt-3.5-turbo"
    st.session_state["openai_model"] = "gpt-4o-mini"

    # ---------- Open AI ----------------------

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


# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

@st.cache_data
def load_race_data():

    df_1 = pd.read_pickle('telemetry_9.pkl')
    df_1 = df_1.loc[10:200,:].reset_index()

    df_2 = pd.read_pickle('telemetry_71.pkl')
    df_2 = df_2.loc[10:200,:].reset_index()

    return df_1,df_2


def data_json(df):

    modes = ["idle","accel","cruising","decel","coasting","overrun","tip-in","tip-out","WOT"]
    df["Mode"] = "DECEL"

    df.loc[df['Throttle']>=100,"Mode"] = "WOT"
    df.loc[(df['Throttle']>0) & (df['Throttle']<100),"Mode"] = "ACCEL"

    df['dThrottle'] = df['Throttle'].diff()
    df.loc[df['dThrottle'] < 0, 'Mode'] = 'DECEL'


    df['Event'] = (df['Mode'] != df['Mode'].shift()).cumsum()

    # Group by events and aggregate
    result = df.groupby('Event').agg({
        'Time': ['min', 'max'],               # Start and end times of each mode
        'Mode': 'first',              # Driving mode for the event
    }).reset_index(drop=True)
    result.columns = ['Start Time', 'End Time', 'Driving Mode']

    start_indices = []
    end_indices = []

    for _, row_a in result.iterrows():
        start_time = row_a['Start Time']
        end_time = row_a['End Time']

        # Find the indices in DataFrame B where the time falls within the range
        start_index = df[df['Time'] >= start_time].index.min()
        end_index = df[df['Time'] <= end_time].index.max() + 1

        # Append the indices to the respective lists
        start_indices.append(start_index)
        end_indices.append(end_index)

    result['Start Index'] = start_indices
    result['End Index'] = end_indices

    presult = result[["Start Time","End Time","Driving Mode"]]
    json_dict = presult.to_dict(orient='records')

    json_string = json.dumps(json_dict)
    return json_string, df

def plot_race_data(df_1):

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df_1['Time'],y=df_1['Throttle'],mode='lines',yaxis="y",name="Throttle"),row=1, col=1)

    fig.add_trace(go.Scatter(x=df_1['Time'],y=df_1['Brake'],mode='lines',yaxis="y",name="Brake"),row=2, col=1)
    fig.add_trace(go.Scatter(x=df_1['Time'],y=df_1['Speed'],mode='lines',yaxis="y",name="Speed"),row=3, col=1)
    fig.add_trace(go.Scatter(x=df_1['Time'],y=df_1['Mode'],mode='lines',yaxis="y",name="Mode"),row=4, col=1)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_yaxes(autorange="reversed",row=4, col=1)

    return fig


def send_message(messages):

    persona = [{"role":"system", "content":"You are a racing car engineer"}]
    persona.extend(messages)
    openai_response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": m["role"], "content": m["content"]} for m in persona],
        stream=False)
    
    return openai_response


def openai_send_message(messages_dict):

    openai_response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages_dict,
        stream=True)
    
    full_response = ""
    for response in openai_response:
        full_response += (response.choices[0].delta.content or "")
    
    json_str = full_response.strip("```json").strip()

    return json_str


def analyse(json_1,add_info, bar):
    bar.progress(0, text="Processing...")

    str0 = "modes = ['BRAKE', 'ACCEL', 'WOT', 'DECEL'], observations = ['Frequent Braking', 'Short Straights', 'Controlled Deceleration', 'Acceleration Zones'], nodes = modes + observations, adj_matrix = [[0, 2, 0, 0, 1, 0, 0, 0],[0, 0, 1, 0, 0, 1, 0, 0],[0, 0, 0, 1, 0, 0, 1, 0],[1, 0, 0, 0, 0, 0, 0, 1]]"


    str1 = f"Using this data '{json_1}', I would like to analyse the lap data, include any observations about external factor that might affect the performance such as straights and corners, temperature, wear and tear, etc. And then convert the analysis and observations into a square matrix to generate a d3graph graph knowledge. The square matrix should show the analysis of the factors affecting the performance. Add also this extra factors {add_info}. The only output i need from you is just the square matrix of your analysis that is compatible with streamlit_d3graph in json format, nothing else. This is the example output i want {str0}"
    #response = send_message({"role": "user", "content": "hello"})
    #response
    messages = [{"role": "user", "content": str1}]
    persona = [{"role":"system", "content":"You are a racing car engineer"}]
    persona.extend(messages)
    messages_dict = [{"role": m["role"], "content": m["content"]} for m in persona]
    

    bar.progress(5, text="Processing...")

    # Example loop to check JSON validity
    attempt = 0
    max_attempts = 5
    

    valid = False
    n_error = 0
    while attempt < max_attempts:
        attempt += 1
        
        json_string = openai_send_message(messages_dict)
        
        # Check if it's valid
        if is_valid_json(json_string):
            valid = True
            break  # Exit loop if valid
        else:
            n_error = n_error+1
    if n_error>0:
        st.write(f'Repeated {n_error} time(s) due to error.')

    bar.progress(70, text="Processing...")
    if valid:
        json_dict = json.loads(json_string)
        modes = json_dict["modes"]
        observations = json_dict["observations"]
        nodes = json_dict["nodes"]
        adj_matrix = json_dict["adj_matrix"]
    else:
        modes = []
        observations = []
        nodes = []
        adj_matrix = []


    return modes, observations, nodes, adj_matrix 


def is_valid_json(json_string):
    try:
        json.loads(json_string)  # Attempt to parse the JSON
        return True
    except json.JSONDecodeError:
        return False


@st.cache_data
def load_images_once():
    image1 = read_image("images/knowledge.png")

    return image1

image1 = load_images_once()


st.subheader("Advanced Data Visualisation")

st.caption("Optimized for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")

with st.expander('Introduction',expanded=True):

    str1 = "A page devoted to exploring various approaches and perspectives on analyzing large or complex datasets, and uncovering the valuable insights they can provide."
    st.write(str1)

    #image = read_image("images/Page4_tech.png")
    #st.image(image,use_column_width=True)

with st.expander('Race data to Knowledge Graph',expanded=True):

    col1, col2 = st.columns([0.5,1])

    col1.write("This innovative approach to interpreting race telemetry provides intuitive insights, enabling race engineers to make faster, more informed decisions. While the process may seem straightforward, it involves several intricate steps, as outlined below. This method of data analysis can be applied across various domains that demand deep insights. By leveraging prompt engineering and language models, it can also be seamlessly integrated with optimizers. In my perspective, this data insights system does more than deliver numerical results. It explains the rationale behind new engineering designs and thought processes, enriching understanding and decision-making.")

    col2.image(image1)


    st.divider()

    df_1, df_2 = load_race_data()
    json_1, df_1 = data_json(df_1)
    json_2, df_2 = data_json(df_2)

    col1,col2 = st.columns([1,1])

    with col1:
        
        col1.write("Max at Monaco 2023 lap 9")

        fig_1 = plot_race_data(df_1)
        st.plotly_chart(fig_1, theme="streamlit",height=400)

        add_info_1 = st.text_area("Add extra factors to be included in data insights","Excessive tire wear, long straights, dry",key="add_info_1")
        bar1 = st.empty()


    with col2:

        st.write("Max at Monaco 2023 lap 71")

        fig_2 = plot_race_data(df_2)
        st.plotly_chart(fig_2, theme="streamlit",height=400)

        add_info_2 = st.text_area("Add extra factors to be included in data insights","Light tire wear, long straights, wet",key="add_info_2")
        bar2 = st.empty()

    st.caption("If it stopped due to error, click Start again.")
    if st.button("Start analysis"):

        bar1.progress(0, text="Processing...")
        bar2.progress(0, text="Waiting...")

        modes_1, observations_1, nodes_1, adj_matrix_1 = analyse(json_1,add_info_1,bar1)
        modes_2, observations_2, nodes_2, adj_matrix_2 = analyse(json_2,add_info_2,bar2)

        col1,col2 = st.columns([1,1])
        with col1:
            if adj_matrix_1 != []:
                bar1.progress(80, text="Plotting...")
                df_matrix = pd.DataFrame(adj_matrix_1, columns=nodes_1, index=nodes_1)
                d3 = d3graph()
                d3.graph(df_matrix)
                d3.show(figsize=(400, 500))
                bar1.progress(100, text="Done for analysis 1")
                time.sleep(1)

            bar1.empty()


        with col2:
            if adj_matrix_2 != []:
                bar2.progress(80, text="Plotting...")
                df_matrix = pd.DataFrame(adj_matrix_2, columns=nodes_2, index=nodes_2)
                d3 = d3graph()
                d3.graph(df_matrix)
                d3.show(figsize=(400, 500))
                bar2.progress(100, text="Done for analysis 2")
                time.sleep(1)
            
            bar2.empty()




