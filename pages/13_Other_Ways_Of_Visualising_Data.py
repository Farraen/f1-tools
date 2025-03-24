import streamlit as st
from PIL import Image
from streamlit_d3graph import d3graph
import numpy as np

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

with st.expander('Knowledge graph',expanded=True):
    # Initialize
    d3 = d3graph()
    # Load karate example
    adjmat, df = d3.import_example('karate')

    label = df['label'].values
    node_size = df['degree'].values

    d3.graph(adjmat)
    d3.set_node_properties(color=df['label'].values)
    d3.show()

    d3.set_node_properties(label=label, color=label, cmap='Set1')
    d3.show()