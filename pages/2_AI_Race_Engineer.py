import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import time

# --------  For page layout  ---------------
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


# ----------- UI Section -------------------------

st_title('AI Race Engineer')




# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image






st.write('Copyright © 2024 Farraen. All rights reserved.')
