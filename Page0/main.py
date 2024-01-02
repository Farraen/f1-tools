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



# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image



st.set_page_config(layout="wide")

st.header("F1 Tools Landing Page")
st.subheader("Farraen's experimental racing tools")
st.header(' ')

gap_size = 0.05
gap,col11,gap,col21,gap,col31,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])

with col11:

    link = 'https://farraen-pu-ga-index-ptnqiv.streamlit.app/'
    button_1 = st.link_button('Open tool', link, use_container_width=True)
    image = read_image("Image_1.png")
    st.image(image)
    st.subheader('PU Selection Decision Engine')
    st.write('A decision engine for optimising PU selection for the whole season.\
                  A decison engine made up of genetic algorithm as optimiser and using an \
             aritificial damage model to calculate vehicle performance degradation.')
    
with col21:
    link = 'https://farraen-pu-ga-index-ptnqiv.streamlit.app/'
    button_2 = st.link_button('Open tool', link, use_container_width=True)
    image = read_image("Image_2.png")
    st.image(image)
    st.subheader('AI Race Engineer')
    st.write('A race telemetry analysis tool using LLM (Large Language Model) to \
             extract insights from vehicle measurement data. A new concept that is harnessing the \
             power of LLM to extract instructions from a user prompt.')
    
with col31:
    link = 'https://farraen-combustion-optimiser-main-rkriwj.streamlit.app/'
    button_3 = st.link_button('Open tool', link, use_container_width=True)
    image = read_image("Image_3.png")
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
    image = read_image("Image_4.png")
    st.image(image)
    st.subheader('')
    st.subheader('Anomaly Detection Testing Tool')
    st.write('A tool for training a ML classifier for detecting anomalies in \
             vehicle measurement data.')
    
with col22:
    link = 'https://farraen-pu-ga-index-ptnqiv.streamlit.app/'
    button_5 = st.link_button('Open tool', link, use_container_width=True)
    image = read_image("Image_5.png")
    st.image(image)
    st.subheader('')
    st.subheader('AI Designer (Generative Design)')
    st.write('A decision engine demonstrator for assisting engineers in powertrain design selection. \
             Using AI and machine learning models to make fast decisions based on user design criteria.')
    