import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import time

st.set_page_config(layout="wide")
st.title('Anomaly Estimation')

# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

