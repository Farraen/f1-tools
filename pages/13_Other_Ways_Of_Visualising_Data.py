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

st.subheader("Data Visualisation (under development)")

st.caption("Optimized for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")

with st.expander('Introduction',expanded=True):

    str1 = "A page devoted to exploring various approaches and perspectives on analyzing large or complex datasets, and uncovering the valuable insights they can provide."
    st_text(str1)

    #image = read_image("images/Page4_tech.png")
    #st.image(image,use_column_width=True)
