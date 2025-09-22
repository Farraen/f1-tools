import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import time
import plotly.express as px
import math
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from matplotlib import pyplot as mplt
from matplotlib.collections import LineCollection


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


# ----------- Functions --------------------------
    
# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

# ----------- UI Section -------------------------

st.subheader('Lap Time Prediction')

st.caption("Optimized for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")

diag_placeholder = st.empty()

@st.cache_resource
def load_model():
    model = CatBoostRegressor(iterations=50,
                          learning_rate=1,
                          depth=2)
    model.load_model('lap_time_model')
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('car_spec_extend.csv', encoding='unicode_escape')
    df = df.replace(',', '', regex=True)

    results = pd.read_pickle(r'lap_time_model_results')
    tel = pd.read_pickle('telemetry')
    lap_color = pd.read_pickle('lap_color')

    return df,results,tel,lap_color

def format_seconds(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{minutes}:{seconds}:{milliseconds:03.0f}"

model = load_model()
df,results,tel,lap_color = load_data()
tracks = list(df['Track'].unique())


cols = ["Weight_kg","HP","Height_mm","Drag_coefficient","Frontal area_m2","Corners","Straights","Distance"]
resp = ["Time_s"]

with st.expander('Introduction',expanded=True):

    st.write('A simple dahsboard that can predict a lap time using vehicle parameters as inputs and a machine learning model trained using a racing car database. The prediction quality at the moment is not good as I have only train the model using about 200 vehicles with various configurations. Also, the model used is only a decision tree type model. I will improve the model later this year.')

    image = read_image("images/lap_time.png")
    st.image(image)

with st.expander('Model accuracy',expanded=False):
    col1, col2 = st.columns([1,1],gap="Medium")

    # Plot scatter plot of Actual vs Predicted values
    fig = px.scatter(results, x='Actual', y='Predicted',
                    title='Actual vs Predicted Values',
                    labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
                    trendline='ols')
    col1.plotly_chart(fig)

    mse = mean_squared_error(results["Actual"], results["Predicted"])
    r2 = r2_score(results["Actual"], results["Predicted"])
    explained_variance = explained_variance_score(results["Actual"], results["Predicted"])
    rmse = math.sqrt(mse)

    st.write(f"Validation Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"Validation Root Mean Squared Error (RMSE): {rmse:.4f}")
    st.write(f"Validation R-squared (R^2): {r2:.4f}")
    st.write(f"Validation Explained Variance Score: {explained_variance:.4f}")






with st.expander('Lap time prediction',expanded=True):
    col1, col2 = st.columns([1,1],gap="Medium")
    with col1:
        st.write('Predictors')
        track = st.selectbox("Select race track",tracks)

        weight = st.slider('Weight', 500.0,2300.0,1000.0)
        hp = st.slider('Horsepower', 150.0,1100.0,400.0)
        height = st.slider('Vehicle height (Will replace with CG)', 1000.0,1800.0,1100.0)
        Cd = st.slider('Drag coefficient', 0.25,0.4,0.3)
        Af = st.slider('Frontal area', 1.05,2.5,1.3)



    with col2:
        st.write('Track information')
        Corners   = df.loc[df['Track']==track,'Corners'].iloc[0]
        Straights = df.loc[df['Track']==track,'Straights'].iloc[0]
        Distance  = df.loc[df['Track']==track,'Distance'].iloc[0]

        col11,col22 = st.columns([1,1])
        col11.metric('Number of corners',Corners)
        col22.metric('Number of straights',Straights)
        col11.metric('Lap distance',Distance)
        
        st.write('Lap time prediction')
        xhat = [weight,hp,height,Cd,Af,Corners,Straights,Distance]
        lap_time = model.predict(xhat)
        #lap_time_str = format_seconds(lap_time)
        st.metric('Lap time',f"{lap_time:.3f} s")

    st.empty()
    st.subheader('Track plot still under development')
    st.write('Austrian Grand Prix')
    x = np.array(tel['X'].values)
    y = np.array(tel['Y'].values)
    color = lap_color
    colormap = mplt.cm.plasma

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = mplt.Normalize(color.min(),color.max())
    lc_comp = LineCollection(segments, norm=norm, cmap=colormap)
    lc_comp.set_array(color)
    lc_comp.set_linewidth(4)

    mplt.gca().add_collection(lc_comp)
    mplt.axis('equal')
    st.pyplot(mplt)


#st.write('Copyright © 2024 Farraen. All rights reserved.')

