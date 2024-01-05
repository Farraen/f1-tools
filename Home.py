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
        switch_page("AI Race Engineer")
    image = read_image("images/Image_2.png")
    st.image(image)
    st.subheader('AI Race Engineer')
    st.write('A race telemetry analysis tool using LLM (Large Language Model) to \
             extract insights from vehicle measurement data. A new concept that harnesses the \
             power of LLM to extract instructions from user prompts.')

with col31:
    if st.button('Open tool', use_container_width=True,key=3):
        switch_page("Combustion Strategy")
    image = read_image("images/Image_3.png")
    st.image(image)
    st.subheader('Model Parameter Tuner')
    st.write('A tool for parameterising models (such as combustion or exhaust temperature model) \
             for sensorless system. It is a simple tool for filling in model parameters to \
             match as measured data from physical sensor.')
    

st.header(' ')
st.header(' ')

gap,col12,gap,col22,gap,col32,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])

with col12:
    if st.button('Open tool', use_container_width=True,key=4):
        switch_page("Anomaly Estimation")
    image = read_image("images/Image_4.png")
    st.image(image)
    st.subheader('')
    st.subheader('Anomaly Detection Testing Tool')
    st.write('A tool for training a ML classifier for detecting anomalies in \
             vehicle measurement data.')
    
with col22:
    if st.button('Open tool', use_container_width=True,key=5):
        switch_page("Generative Design")
    image = read_image("images/Image_5.png")
    st.image(image)
    st.subheader('')
    st.subheader('AI Designer (Generative Design)')
    st.write('A decision engine demonstrator for assisting engineers in powertrain design selection. \
             Using AI and machine learning models to make fast decisions based on user design criteria.')
    