import streamlit as st
import numpy as np
from PIL import Image
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(layout="wide",initial_sidebar_state="collapsed")




# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image


st.subheader("Farraen's experimental racing tools")
st.caption("Optimised for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")
st.write(' ')

# Initialize session state for view mode
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'all'

# Add Sort and All buttons
col_spacer, col_btn1, col_btn2 = st.columns([5, 1, 1])
with col_btn1:
    if st.button('Theme', use_container_width=True):
        st.session_state.view_mode = 'sort'
        st.rerun()
with col_btn2:
    if st.button('All', use_container_width=True):
        st.session_state.view_mode = 'all'
        st.rerun()


gap_size = 0.05

# Define all tools with their metadata
tools_data = [
    {
        'page': 'PU Optimisation',
        'image': 'images/Image_1.png',
        'title': 'PU Selection Decision Engine',
        'description': 'A decision engine designed to optimize F1 power unit (PU) selection for the entire season. This engine utilizes a genetic algorithm as the optimizer and incorporates an artificial damage model to assess vehicle performance degradation.',
        'category': 'Strategy & Optimization'
    },
    {
        'page': 'PU Optimisation LLM',
        'image': 'images/Image_2.png',
        'title': 'PU Decision Engine + AI Race Engineer',
        'description': 'The same PU decision engine dashboard is now powered by a Large Language Model (LLM). It can provide recommendations based on the current PU strategy situation. The LLM persona is tuned to mimic a junior race engineer.',
        'category': 'Strategy & Optimization'
    },
    {
        'page': 'EV Race Car Optimiser',
        'image': 'images/Image_12.png',
        'title': 'EV Race Car Optimiser',
        'description': 'A tool optimising EV race car parameters to get most efficient lap times.',
        'category': 'Optimisation'
    },
    {
        'page': 'Lap Time Prediction',
        'image': 'images/Image_11.jpg',
        'title': 'Lap Time Predictor',
        'description': 'A tool for predicting vehicle lap time over a circuit using machine learning.',
        'category': 'Strategy & Optimization'
    },
    {
        'page': 'Combustion Strategy',
        'image': 'images/Image_4.png',
        'title': 'Model Parameter Tuner',
        'description': 'A tool for parameterising models (such as combustion or exhaust temperature model) for sensorless systems. It is a simple tool for filling in model parameters to match as measured data from a physical sensor.',
        'category': 'Optimisation'
    },
    {
        'page': 'Generative Design',
        'image': 'images/Image_6.png',
        'title': 'Generative Design',
        'description': 'A decision engine demonstrator for assisting engineers in powertrain design selection. Using AI and machine learning models to make fast decisions based on user design criteria.',
        'category': 'Generative Design'
    },
    {
        'page': 'Tire Strategy',
        'image': 'images/Image_7.png',
        'title': 'Tire Strategy',
        'description': 'A decision engine platform for tyre change strategy. It can help race engineers predict the optimal lap for tyre change and compound selection. Genetic algorithm is used as the decision engine to generate tyre change solutions.',
        'category': 'Strategy & Optimization'
    },
    {
        'page': 'Prognostics',
        'image': 'images/Image_8.png',
        'title': 'Prognostics',
        'description': 'A simple dashboard to investigate early failures from race telemetry data.',
        'category': 'Anomaly Detection'
    },
    {
        'page': 'Data Mining',
        'image': 'images/Image_10.png',
        'title': 'Data Mining (Under development)',
        'description': 'A data exploration tool for finding underlying patterns and trends.',
        'category': 'Data Analysis'
    },
    {
        'page': 'Model Based Calibration',
        'image': 'images/Image_3.png',
        'title': 'Model Based Calibration Methodology',
        'description': '(Under development) Model-based Calibration is a process for optimally tuning system parameters. It involves creating a Design of Experiments (DoE), developing models, and performing optimization.',
        'category': 'Optimisation'
    },
    {
        'page': 'Anomaly Estimation',
        'image': 'images/Image_5.png',
        'title': 'Anomaly Detection Testing Tool',
        'description': 'A tool for training a ML classifier for detecting anomalies in vehicle measurement data.',
        'category': 'Anomaly Detection'
    },
    {
        'page': 'Other Ways of Visualising Data',
        'image': 'images/big_data.png',
        'title': 'Data Visualisation',
        'description': 'A page dedicated for new ways of exploring complex and large data, and getting useful insights.It helps to find hidden patterns and relationships in more intuitive manner. For a start, I have combined time-series segmentation, prompt engineerng and knowledge graph to quickly visualise factors affecting a lap time. It is still work in progress but enough as proof of concept.',
        'category': 'Data Analysis'
    },
    {
        'page': 'Multi Task GP',
        'image': 'images/mtgp_1.png',
        'title': 'Multi Task GP Model',
        'description': 'This an app example to use Multi-task GP framework to train a GP model from another GP model. The problem that we want to solve is how we can built a statistical model out of sparse training data and use historical engine data from a similar engine.',
        'category': 'Optimisation'
    },
    {
        'page': 'Stochastic Variational Inference',
        'image': 'images/SVI.png',
        'title': 'Stochastic Variational Inference',
        'description': 'SVI (Stochastic Variational Inference) is a powerful technique to train any model or for parameterisation of engineering systems. It uses variational distribution to approximate the posterior distribution of the model parameters. It is useful when the training data is too sparse or too complex for parameterisation The output of the process is the optimised parameters as well as the uncertainty of the parameters.',
        'category': 'Optimisation'
    },
    {
        'page': 'Anomaly Detection Using Transformer',
        'image': 'images/big_data.png',
        'title': 'Anomaly Detection Using Transformer',
        'description': 'A page dedicated for new ways of exploring complex and large data, and getting useful insights.It helps to find hidden patterns and relationships in more intuitive manner. For a start, I have combined time-series segmentation, prompt engineerng and knowledge graph to quickly visualise factors affecting a lap time. It is still work in progress but enough as proof of concept.',
        'category': 'Anomaly Detection'
    },
    {
        'page': 'Anomaly Detection And Insights',
        'image': 'images/Image_17.png',
        'title': 'Anomaly Detection and Insights',
        'description': 'A page to demonstrate the ability of using foundation models to analyse data and provide insights and root cause analysis for an engine component. It uses fuel pump failure as an example and use AI to determine failure modes and factors contributing to the failure.',
        'category': 'Anomaly Detection'
    },
    {
        'page': 'https://farraen.github.io/streamlit_demo/',
        'image': 'images/stlite.png',
        'title': 'Streamlit without Python backend',
        'description': 'A page to showcase the ability of using streamlit without Python backend. Works by loading Pyodide, Stlite and other libraries directly into browser.',
        'category': 'Data Analysis'
    },
    {
        'page': 'Interactive Report Generator',
        'image': 'images/interactive.png',
        'title': 'Interactive Report Generator',
        'description': 'A tool for generating interactive reports for more intuitive experience compared to normal excel or pdf reports. This app generates a report showing an engine ML model performance.',
        'category': 'Data Analysis'
    },
    {
        'page': 'Multi Agent',
        'image': 'images/multi_agent.png',
        'title': 'Multi Agent Demonstration',
        'description': 'A tool for demonstrating the ability to orchestrate multiple agents to solve a problem.',
        'category': 'Agentic AI'
    },
]

def render_tool_card(tool, col):
    """Render a single tool card"""
    with col:
        if 'https://' in tool['page']:
            # For external links, create a button that opens in same tab
            button_html = f"""
            <div style="margin-bottom: 1rem;">
                <a href="{tool['page']}" target="_self" style="text-decoration: none;">
                    <button style="width: 100%; padding: 0.25rem 0.75rem; min-height: 2.5rem;
                    border: 1px solid rgba(250, 250, 250, 0.2); 
                    border-radius: 0.5rem; background-color: rgba(255, 255, 255, 0.05); 
                    color: white; cursor: pointer; font-weight: 400; font-size: 1rem;
                    font-family: 'Source Sans Pro', sans-serif; transition: all 0.2s ease;">Open tool</button>
                </a>
            </div>
            <style>
                button:hover {{
                    background-color: rgba(255, 255, 255, 0.1);
                    border-color: rgba(255, 75, 75, 0.5);
                }}
            </style>
            """
            st.markdown(button_html, unsafe_allow_html=True)
        else:
            # For internal pages, use switch_page
            if st.button('Open tool', use_container_width=True, key=tool['title']):
                switch_page(tool['page'])
        image = read_image(tool['image'])
        st.image(image)
        st.subheader(tool['title'])
        st.write(tool['description'])

if st.session_state.view_mode == 'all':
    # Original view - display in original order (3 columns per row)
    for i in range(0, len(tools_data), 3):
        gap,col1,gap,col2,gap,col3,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])
        
        if i < len(tools_data):
            render_tool_card(tools_data[i], col1)
        if i+1 < len(tools_data):
            render_tool_card(tools_data[i+1], col2)
        if i+2 < len(tools_data):
            render_tool_card(tools_data[i+2], col3)
        
        st.header(' ')

else:
    # Sorted view - display by category
    from collections import defaultdict
    
    # Group tools by category
    categories = defaultdict(list)
    for tool in tools_data:
        categories[tool['category']].append(tool)
    
    # Define category order
    category_order = [
        'Strategy & Optimization',
        'Optimisation',
        'Generative Design',
        'Data Analysis',
        'Anomaly Detection',
        'Agentic AI'
    ]
    
    # Display each category
    for category in category_order:
        if category in categories:
            st.markdown(f"### {category}")
            
            tools = categories[category]
            for i in range(0, len(tools), 3):
                gap,col1,gap,col2,gap,col3,gap = st.columns([gap_size,1,gap_size,1,gap_size,1,gap_size])
                
                if i < len(tools):
                    render_tool_card(tools[i], col1)
                if i+1 < len(tools):
                    render_tool_card(tools[i+1], col2)
                if i+2 < len(tools):
                    render_tool_card(tools[i+2], col3)
                
                st.write(' ')
            
            st.header(' ')



st.write('Copyright © 2025 Farraen. All rights reserved.')
