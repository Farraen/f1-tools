import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import time

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

    link = 'https://farraen-pu-ga-index-ptnqiv.streamlit.app/'
    button_1 = st.link_button('Open tool', link, use_container_width=True)
    image = read_image("Page0/Image_1.png")
    st.image(image)
    st.subheader('PU Selection Decision Engine')
    st.write('A decision engine for optimising PU selection for the whole season.\
                  A decison engine made up of genetic algorithm as optimiser and using an \
             aritificial damage model to calculate vehicle performance degradation.')
    
with col21:
    link = 'https://farraen-pu-ga-index-ptnqiv.streamlit.app/'
    button_2 = st.link_button('Open tool', link, use_container_width=True)
    image = read_image("Page0/Image_2.png")
    st.image(image)
    st.subheader('AI Race Engineer')
    st.write('A race telemetry analysis tool using LLM (Large Language Model) to \
             extract insights from vehicle measurement data. A new concept that is harnessing the \
             power of LLM to extract instructions from a user prompt.')

with col31:
    link = 'https://farraen-combustion-optimiser-main-rkriwj.streamlit.app/'
    button_3 = st.link_button('Open tool', link, use_container_width=True)
    image = read_image("Page0/Image_3.png")
    st.image(image)
    st.subheader('Model Parameter Tuner')
    st.write('A tool for parameterising models (such as combustion or exhaust temperature model) \
             for sensorless system. It is a simple tool for filling in model parameters to \
             match as measured data from physical sensor.')
    

st.header(' ')
st.header(' ')

gap,col12,gap,col22,gap,col32,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])

with col12:
    link = 'https://farraen-pu-ga-index-ptnqiv.streamlit.app/'
    button_4 = st.link_button('Open tool', link, use_container_width=True)
    image = read_image("Page0/Image_4.png")
    st.image(image)
    st.subheader('')
    st.subheader('Anomaly Detection Testing Tool')
    st.write('A tool for training a ML classifier for detecting anomalies in \
             vehicle measurement data.')
    
with col22:
    link = 'https://farraen-pu-ga-index-ptnqiv.streamlit.app/'
    button_5 = st.link_button('Open tool', link, use_container_width=True)
    image = read_image("Page0/Image_5.png")
    st.image(image)
    st.subheader('')
    st.subheader('AI Designer (Generative Design)')
    st.write('A decision engine demonstrator for assisting engineers in powertrain design selection. \
             Using AI and machine learning models to make fast decisions based on user design criteria.')
    