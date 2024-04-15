import streamlit as st
import numpy as np
from PIL import Image
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(layout="wide")

# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image


st.subheader("Farraen's experimental racing tools")
st.header(' ')

gap_size = 0.05
gap,col11,gap,col21,gap,col31,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])


with col11:

    if st.button('Open tool', use_container_width=True,key=1):
        switch_page("PU Optimisation")
    image = read_image("images/Image_1.png")
    st.image(image)
    st.subheader('PU Selection Decision Engine')
    st.write('A decision engine for optimising PU selection for the whole season.\
                  A decison engine made up of genetic algorithm as optimiser and using an \
             aritificial damage model to calculate vehicle performance degradation.')
    
with col21:
    if st.button('Open tool', use_container_width=True,key=2):
        switch_page("PU Optimisation LLM")
    image = read_image("images/Image_2.png")
    st.image(image)
    st.subheader('PU Decision Engine + AI Race Engineer')
    st.write('Same PU decision engine dashboard but powered by LLM (Large Language Model). It can also make recommendations based on current PU strategy situation. The LLM persona has been tuned to mimic a junior race engineer.')

with col31:
    if st.button('Open tool', use_container_width=True,key=3):
        switch_page("AI Race Engineer")
    image = read_image("images/Image_3.png")
    st.image(image)
    st.subheader('AI Junior Race Engineer')
    st.write('A tuned LLM (Large Language Model) persona to mimic a junior race engineer. \
             Useful to ask anything within the racing domain. For future development, it can connect to the MongoDB database to extract and process race telemetery data. Powered by OpenAI language model.')
    

st.header(' ')
st.header(' ')

gap,col12,gap,col22,gap,col32,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])

with col12:
    if st.button('Open tool', use_container_width=True,key=4):
        switch_page("Combustion Strategy")
    image = read_image("images/Image_4.png")
    st.image(image)
    st.subheader('Model Parameter Tuner')
    st.write('A tool for parameterising models (such as combustion or exhaust temperature model) \
             for sensorless system. It is a simple tool for filling in model parameters to \
             match as measured data from physical sensor.')

with col22:
    if st.button('Open tool', use_container_width=True,key=5):
        switch_page("Anomaly Estimation")
    image = read_image("images/Image_5.png")
    st.image(image)
    st.subheader('Anomaly Detection Testing Tool')
    st.write('A tool for training a ML classifier for detecting anomalies in \
             vehicle measurement data.')
    
with col32:
    if st.button('Open tool', use_container_width=True,key=6):
        switch_page("Generative Design")
    image = read_image("images/Image_6.png")
    st.image(image)
    st.subheader('AI Designer (Generative Design)')
    st.write('A decision engine demonstrator for assisting engineers in powertrain design selection. \
             Using AI and machine learning models to make fast decisions based on user design criteria.')

st.header(' ')
st.header(' ')

gap,col12,gap,col22,gap,col32,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])

with col12:
    if st.button('Open tool', use_container_width=True,key=7):
        switch_page("Tire Strategy")
    image = read_image("images/Image_7.png")
    st.image(image)
    st.subheader('Tire Strategy')
    st.write('A decision engine platform for tire change strategy. It can help race engineers to predict optimal lap for tire change and compound selection. Genetic algorithm is used as the decision engine to generate tire change solutions.')
    
