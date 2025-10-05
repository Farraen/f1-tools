import streamlit as st
import numpy as np
from PIL import Image
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(layout="wide",initial_sidebar_state="collapsed")


# Add custom animated background with shape-shifting blocks
page_bg_img = """
<style>
/* Animated gradient background */
@keyframes gradientShift {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e, #1a1a2e, #302b63);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}

/* Ensure main content is scrollable and positioned correctly */
[data-testid="stAppViewContainer"] > .main {
    position: relative;
    z-index: 1;
}

/* Blocks container */
.blocks-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
}

/* Individual block styling */
.shape-block {
    position: absolute;
    pointer-events: none;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
    backdrop-filter: blur(5px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

/* Shape-shifting and morphing animations */
@keyframes morph1 {
    0%, 100% { 
        border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
        transform: translate(0, 0) rotate(0deg) scale(1);
        opacity: 0.3;
    }
    25% { 
        border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%;
        transform: translate(50px, -30px) rotate(90deg) scale(1.1);
        opacity: 0.5;
    }
    50% { 
        border-radius: 40% 40% 50% 60% / 30% 50% 70% 50%;
        transform: translate(-30px, -60px) rotate(180deg) scale(0.9);
        opacity: 0.4;
    }
    75% { 
        border-radius: 70% 30% 50% 50% / 60% 40% 60% 40%;
        transform: translate(-50px, 20px) rotate(270deg) scale(1.2);
        opacity: 0.6;
    }
}

@keyframes morph2 {
    0%, 100% { 
        border-radius: 40% 60% 60% 40% / 40% 40% 60% 60%;
        transform: translate(0, 0) rotate(0deg) scaleX(1) scaleY(1);
        opacity: 0.4;
    }
    33% { 
        border-radius: 60% 40% 30% 70% / 60% 60% 40% 40%;
        transform: translate(-40px, 40px) rotate(120deg) scaleX(1.2) scaleY(0.8);
        opacity: 0.6;
    }
    66% { 
        border-radius: 50% 50% 40% 60% / 50% 50% 50% 50%;
        transform: translate(60px, -40px) rotate(240deg) scaleX(0.8) scaleY(1.2);
        opacity: 0.5;
    }
}

@keyframes morph3 {
    0%, 100% { 
        border-radius: 50% 50% 50% 50% / 50% 50% 50% 50%;
        transform: translate(0, 0) rotate(0deg) scale(1);
        opacity: 0.5;
    }
    20% { 
        border-radius: 70% 30% 70% 30% / 30% 70% 30% 70%;
        transform: translate(30px, -50px) rotate(72deg) scale(1.3);
        opacity: 0.7;
    }
    40% { 
        border-radius: 30% 70% 30% 70% / 70% 30% 70% 30%;
        transform: translate(-50px, -30px) rotate(144deg) scale(0.8);
        opacity: 0.3;
    }
    60% { 
        border-radius: 60% 40% 40% 60% / 40% 60% 60% 40%;
        transform: translate(-30px, 60px) rotate(216deg) scale(1.1);
        opacity: 0.6;
    }
    80% { 
        border-radius: 40% 60% 60% 40% / 60% 40% 40% 60%;
        transform: translate(50px, 40px) rotate(288deg) scale(0.9);
        opacity: 0.4;
    }
}

@keyframes morph4 {
    0%, 100% { 
        border-radius: 35% 65% 65% 35% / 35% 65% 35% 65%;
        transform: translate(0, 0) rotate(0deg) scale(1) skew(0deg);
        opacity: 0.35;
    }
    25% { 
        border-radius: 65% 35% 35% 65% / 65% 35% 65% 35%;
        transform: translate(-60px, -40px) rotate(90deg) scale(1.2) skew(5deg);
        opacity: 0.55;
    }
    50% { 
        border-radius: 50% 50% 30% 70% / 30% 70% 50% 50%;
        transform: translate(40px, -70px) rotate(180deg) scale(0.85) skew(-5deg);
        opacity: 0.45;
    }
    75% { 
        border-radius: 70% 30% 50% 50% / 50% 50% 30% 70%;
        transform: translate(60px, 30px) rotate(270deg) scale(1.15) skew(3deg);
        opacity: 0.6;
    }
}

@keyframes morph5 {
    0%, 100% { 
        border-radius: 45% 55% 60% 40% / 55% 45% 40% 60%;
        transform: translate(0, 0) rotate(0deg) scale(1);
        opacity: 0.4;
    }
    16% { 
        border-radius: 55% 45% 40% 60% / 45% 55% 60% 40%;
        transform: translate(35px, 40px) rotate(60deg) scale(1.1);
        opacity: 0.6;
    }
    33% { 
        border-radius: 60% 40% 45% 55% / 40% 60% 55% 45%;
        transform: translate(-45px, 60px) rotate(120deg) scale(0.9);
        opacity: 0.5;
    }
    50% { 
        border-radius: 40% 60% 55% 45% / 60% 40% 45% 55%;
        transform: translate(-55px, -50px) rotate(180deg) scale(1.25);
        opacity: 0.7;
    }
    66% { 
        border-radius: 50% 50% 50% 50% / 50% 50% 50% 50%;
        transform: translate(45px, -35px) rotate(240deg) scale(0.85);
        opacity: 0.35;
    }
    83% { 
        border-radius: 65% 35% 65% 35% / 35% 65% 35% 65%;
        transform: translate(25px, 55px) rotate(300deg) scale(1.05);
        opacity: 0.55;
    }
}

/* Particle overlay for extra effect */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.03) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(255, 255, 255, 0.02) 0%, transparent 50%);
    pointer-events: none;
    z-index: -1;
}
</style>

<script>
// Create shape-shifting blocks
(function() {
    setTimeout(function() {
        let container = document.getElementById('blocks-container');
        
        if (!container) {
            container = document.createElement('div');
            container.id = 'blocks-container';
            container.className = 'blocks-container';
            
            const appView = document.querySelector('[data-testid="stAppViewContainer"]');
            if (appView) {
                appView.insertBefore(container, appView.firstChild);
            } else {
                document.body.appendChild(container);
            }
        }
        
        container.innerHTML = '';
        
        const blockCount = 20; // Number of shape-shifting blocks
        const animations = ['morph1', 'morph2', 'morph3', 'morph4', 'morph5'];
        
        for (let i = 0; i < blockCount; i++) {
            const block = document.createElement('div');
            block.className = 'shape-block';
            
            // Random size between 60-150px
            const size = Math.random() * 90 + 60;
            block.style.width = size + 'px';
            block.style.height = size + 'px';
            
            // Random position across the screen
            block.style.left = Math.random() * 100 + '%';
            block.style.top = Math.random() * 100 + '%';
            
            // Random animation
            const randomAnim = animations[Math.floor(Math.random() * animations.length)];
            const duration = Math.random() * 10 + 8; // 8-18 seconds
            const delay = Math.random() * 5; // 0-5 seconds delay
            
            block.style.animation = randomAnim + ' ' + duration + 's ease-in-out infinite';
            block.style.animationDelay = delay + 's';
            
            // Random initial opacity
            block.style.opacity = Math.random() * 0.3 + 0.3;
            
            container.appendChild(block);
        }
        
        console.log('Shape-shifting blocks created:', blockCount);
    }, 100);
})();
</script>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image


st.subheader("Farraen's experimental racing tools")
st.caption("Optimized for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")
st.write(' ')

gap_size = 0.05
gap,col11,gap,col21,gap,col31,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])


with col11:

    if st.button('Open tool', use_container_width=True,key=1):
        switch_page("PU Optimisation")
    image = read_image("images/Image_1.png")
    st.image(image)
    st.subheader('PU Selection Decision Engine')
    st.write('A decision engine designed to optimize F1 power unit (PU) selection for the entire season.\
            This engine utilizes a genetic algorithm as the optimizer and incorporates an \
            artificial damage model to assess vehicle performance degradation.')
    
with col21:
    if st.button('Open tool', use_container_width=True,key=2):
        switch_page("PU Optimisation LLM")
    image = read_image("images/Image_2.png")
    st.image(image)
    st.subheader('PU Decision Engine + AI Race Engineer')
    st.write('The same PU decision engine dashboard is now powered by a Large Language Model (LLM). \
            It can provide recommendations based on the current PU strategy situation. \
            The LLM persona is tuned to mimic a junior race engineer.')

with col31:
    
    if st.button('Open tool', use_container_width=True,key=3):
        switch_page("EV Race Car Optimiser")
    image = read_image("images/Image_12.png")
    st.image(image)
    st.subheader('EV Race Car Optimiser')
    st.write('A tool optimising EV race car parameters to get most efficient lap times.')

st.header(' ')
st.header(' ')

gap,col12,gap,col22,gap,col32,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])

with col12:
    if st.button('Open tool', use_container_width=True,key=4):
        switch_page("Lap Time Prediction")
    image = read_image("images/Image_11.jpg")
    st.image(image)
    st.subheader('Lap Time Predictor')
    st.write('A tool for predicting vehicle lap time over a circuit using machine learning.')

with col22:
    if st.button('Open tool', use_container_width=True,key=5):
        switch_page("Combustion Strategy")
    image = read_image("images/Image_4.png")
    st.image(image)
    st.subheader('Model Parameter Tuner')
    st.write('A tool for parameterising models (such as combustion or exhaust temperature model) \
             for sensorless systems. It is a simple tool for filling in model parameters to \
             match as measured data from a physical sensor.')
    
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

gap,col13,gap,col23,gap,col33,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])

with col13:
    if st.button('Open tool', use_container_width=True,key=7):
        switch_page("Tire Strategy")
    image = read_image("images/Image_7.png")
    st.image(image)
    st.subheader('Tire Strategy')
    st.write('A decision engine platform for tyre change strategy. It can help race engineers predict the optimal lap for tyre change and compound selection. Genetic algorithm is used as the decision engine to generate tyre change solutions.')
    
with col23:
    if st.button('Open tool', use_container_width=True,key=8):
        switch_page("Prognostics")
    image = read_image("images/Image_8.png")
    st.image(image)
    st.subheader('Prognostics')
    st.write('A simple dashboard to investigate early failures from race telemetry data.')
    
    
with col33:

    if st.button('Open tool', use_container_width=True,key=9):
        switch_page("AI Race Engineer")
    image = read_image("images/Image_9.png")
    st.image(image)
    st.subheader('AI Junior Race Engineer')
    st.write('A tuned LLM (Large Language Model) persona to mimic a junior race engineer. \
             Please feel free to ask anything within the racing domain. The next version of the tool can connect to a MongoDB database to extract and process race telemetery data. Powered by OpenAI language model.')
    

    
st.header(' ')
st.header(' ')

gap,col14,gap,col24,gap,col34,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])

with col14:
    if st.button('Open tool', use_container_width=True,key=10):
        switch_page("Data Mining")
    image = read_image("images/Image_10.png")
    st.image(image)
    st.subheader('Data Mining (Under development)')
    st.write('A data exploration tool for finding underlying patterns and trends.')


with col24:

    
    if st.button('Open tool', use_container_width=True,key=11):
        switch_page("Model Based Calibration")
    image = read_image("images/Image_3.png")
    st.image(image)
    st.subheader('Model Based Calibration Methodology')
    st.write('(Under development) Model-based Calibration is a process for optimally tuning system parameters.\
        It involves creating a Design of Experiments (DoE), developing models, and performing optimization.') 
    

with col34:

    if st.button('Open tool', use_container_width=True,key=12):
        switch_page("Anomaly Estimation")
    image = read_image("images/Image_5.png")
    st.image(image)
    st.subheader('Anomaly Detection Testing Tool')
    st.write('A tool for training a ML classifier for detecting anomalies in \
             vehicle measurement data.')
    


gap,col14,gap,col24,gap,col34,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])


with col14:
    if st.button('Open tool', use_container_width=True,key=13):
        switch_page("Other Ways of Visualising Data")
    image = read_image("images/big_data.png")
    st.image(image)
    st.subheader('Data Visualisation')
    st.write('A page dedicated for new ways of exploring complex and large data, and getting useful insights.It helps to find hidden patterns and relationships in more intuitive manner. For a start, I have combined time-series segmentation, prompt engineerng and knowledge graph to quickly visualise factors affecting a lap time. It is still work in progress but enough as proof of concept.')


with col24:
    if st.button('Open tool', use_container_width=True,key=14):
        switch_page("Multi Task GP")
    image = read_image("images/mtgp_1.png")
    st.image(image)
    st.subheader('Multi Task GP Model')
    st.write(' This an app example to use Multi-task GP framework to train a GP model from another GP model. ' \
    'The problem that we want to solve is how we can built a statistical model out of sparse training data and use historical engine data from a similar engine.')

with col34:
    if st.button('Open tool', use_container_width=True,key=15):
        switch_page("Stochastic Variational Inference")
    image = read_image("images/SVI.png")
    st.image(image)
    st.subheader('Stochastic Variational Inference')
    st.write('SVI (Stochastic Variational Inference) is a powerful technique to train any model or for parameterisation of engineering systems.' \
    ' It uses variational distribution to approximate the posterior distribution of the model parameters.' \
    ' It is useful when the training data is too sparse or too complex for parameterisation'
    ' The output of the process is the optimised parameters as well as the uncertainty of the parameters.')

gap,col14,gap,col24,gap,col34,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])


with col14:
    if st.button('Open tool', use_container_width=True,key=16):
        switch_page("Anomaly Detection Using Transformer")
    image = read_image("images/big_data.png")
    st.image(image)
    st.subheader('Anomaly Detection Using Transformer')
    st.write('A page dedicated for new ways of exploring complex and large data, and getting useful insights.It helps to find hidden patterns and relationships in more intuitive manner. For a start, I have combined time-series segmentation, prompt engineerng and knowledge graph to quickly visualise factors affecting a lap time. It is still work in progress but enough as proof of concept.')



with col24:
    if st.button('Open tool', use_container_width=True,key=17):
        switch_page("Anomaly Detection And Insights")
    image = read_image("images/Image_17.png")
    st.image(image)
    st.subheader('Anomaly Detection and Insights')
    st.write('A page to demonstrate the ability of using foundation models to analyse data and provide insights and root cause analysis for an engine component. It uses fuel pump failure as an example and use AI to determine failure modes and factors contributing to the failure.')
