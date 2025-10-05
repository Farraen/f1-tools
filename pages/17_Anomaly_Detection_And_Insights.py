import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from PIL import Image
from streamlit_d3graph import d3graph
from openai import OpenAI
import json

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

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

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"
    #st.session_state["openai_model"] = "gpt-5-mini-2025-08-07"

    # ---------- Open AI ----------------------

# OpenAI 
@st.cache_resource 
def connect_openAI():
    openai_api_key = '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'
    openai_client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
    )
    return openai_client

openai_client = connect_openAI()


def send_message(messages):

    persona = [{"role":"system", "content":"You are a racing car engineer"}]
    persona.extend(messages)
    openai_response = openai_client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": m["role"], "content": m["content"]} for m in persona],
        stream=False)
    
    return openai_response


def openai_send_message(messages_dict):

    openai_response = openai_client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages_dict,
        stream=True)
    
    full_response = ""
    for response in openai_response:
        full_response += (response.choices[0].delta.content or "")
    
    json_str = full_response.strip("```json").strip()

    return json_str



def data_json(df):
  
    json_dict = df.to_dict(orient='records')

    json_string = json.dumps(json_dict)
    return json_string, df

def analyse(json_1,add_info):

    str0 = "modes = ['BRAKE', 'ACCEL', 'WOT', 'DECEL'], observations = ['Frequent Braking', 'Short Straights', 'Controlled Deceleration', 'Acceleration Zones'], nodes = modes + observations, adj_matrix = [[0, 2, 0, 0, 1, 0, 0, 0],[0, 0, 1, 0, 0, 1, 0, 0],[0, 0, 0, 1, 0, 0, 1, 0],[1, 0, 0, 0, 0, 0, 0, 1]]"


    str1 = f"""Using this data '{json_1}', 
    Check for anomalies, spikes or instabiltiy in the PWM.

    if there are anomalies, spikes or instabiltiy in the data, 
    then I would like to analyse the fuel pump pwm data, include any observations about 
    external factor that might affect the degradation of the PWM.
    If the PWM is clean step, then there is no anomaly. It is anomalous, if it is not a nice step.
    I want the general reason what causing race fuel pump degradation.
    And then convert the analysis and observations into a square matrix to generate a d3graph graph knowledge. 
    The square matrix should show the analysis of the factors affecting the performance. 

    Add also this extra factors {add_info}. 
    
    IMPORTANT: The adjacency matrix values represent WEIGHTS that show the STRENGTH of relationships between nodes:
    - Use 0 for no relationship
    - Use 1-3 for weak relationships (minor contributing factors)
    - Use 4-6 for moderate relationships (significant contributing factors)
    - Use 7-10 for strong relationships (major contributing factors, root causes)
    
    The weights should reflect how strongly one factor influences or causes another. For example:
    - 'High Usage' → 'Pump Degradation' might be weight 9 (very strong causal relationship)
    - 'Temperature' → 'Voltage Spikes' might be weight 6 (moderate influence)
    - 'Race Number' → 'Cumulative Wear' might be weight 10 (direct causation)
    
    CRITICAL: You must also include an "anomaly_detected" field in your JSON response:
    - true: If anomalies, spikes, or instability detected
    - false: If pump operating normally
    
    Base this on voltage spikes, instability severity, PWM duty cycle variations, degradation factor, and statistical deviations
    
    The only output i need from you is just the square matrix of your analysis that is compatible with streamlit_d3graph 
    in json format with the anomaly_detected field, nothing else. 
    This is the example output i want {str0} but don't follow the content because it is for a different tool.
    
    But if there is no anomaly in the selected PWM window, show factors contributing to normal fuel pump operation.
    
    If one of the failure mode is race number, that means it is the mileage of the fuel pump.
    So use mileage instead of race number.
    """
    #response = send_message({"role": "user", "content": "hello"})
    #response
    messages = [{"role": "user", "content": str1}]
    persona = [{"role":"system", "content":"You are a racing car and electricalengineer"}]
    persona.extend(messages)
    messages_dict = [{"role": m["role"], "content": m["content"]} for m in persona]
    


    # Example loop to check JSON validity
    attempt = 0
    max_attempts = 5
    

    valid = False
    n_error = 0
    while attempt < max_attempts:
        attempt += 1
        
        json_string = openai_send_message(messages_dict)
        
        # Check if it's valid
        if is_valid_json(json_string):
            valid = True
            break  # Exit loop if valid
        else:
            n_error = n_error+1
    if n_error>0:
        st.write(f'Repeated {n_error} time(s) due to error.')

    if valid:
        json_dict = json.loads(json_string)
        modes = []
        observations = []
        nodes = json_dict["nodes"]
        adj_matrix = json_dict["adj_matrix"]
        anomaly_detected = json_dict.get("anomaly_detected", None)
    else:
        modes = []
        observations = []
        nodes = []
        adj_matrix = []
        anomaly_detected = None


    return modes, observations, nodes, adj_matrix, anomaly_detected 


def is_valid_json(json_string):
    try:
        json.loads(json_string)  # Attempt to parse the JSON
        return True
    except json.JSONDecodeError:
        return False



def st_title(text):
    st.markdown(f'<p class="title_medium">{text}</p>', unsafe_allow_html=True)

def st_text(text):
    st.markdown(f'<p class="text_small">{text}</p>', unsafe_allow_html=True)


# Page title and description
st.subheader("Anomaly Detection and Insights - Pump PWM Analysis")

# MongoDB Connection
@st.cache_resource 
def connect_mongo():
    mongoUser = 'farraen'
    mongoPwd = 'rI68TwqYQTSDu5Pp'
    mongoDb = 'f1_analysis_max' 
    mongoDb2 = 'f1_info'  

    uri = f"mongodb+srv://{mongoUser}:{mongoPwd}@cluster0.rjtb7gz.mongodb.net/?retryWrites=true&w=majority"
    mongo_client = MongoClient(uri, server_api=ServerApi('1'))

    try:
        mongo_client.admin.command('ping')
        db_status = "Connected"
    except Exception as e:
        db_status = "Connection error"

    db = mongo_client[mongoDb]
    db_info = mongo_client[mongoDb2]

    return db, db_info, mongo_client, db_status

# Initialize database connection
db, db_info, mongo_client, db_status = connect_mongo()

# Session state initialization
if 'year' not in st.session_state:
    st.session_state.year = 2023

if 'race' not in st.session_state:
    st.session_state.race = 6

if 'lap' not in st.session_state:
    st.session_state.lap = 51

# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    return im

# Title and description styling
def st_title(text):
    st.markdown(f'<p class="title_medium">{text}</p>', unsafe_allow_html=True)

def st_text(text):
    st.markdown(f'<p class="text_small">{text}</p>', unsafe_allow_html=True)

# Simulate Pump PWM Signal for 4-Cylinder Engine
def generate_pump_pwm_signal(rpm=8000, duration=0.1, num_pulses=10, sampling_rate=100000, degradation_factor=0.0):
    """
    Generate PWM signal for fuel pump in a 4-cylinder engine
    
    Args:
        rpm: Engine RPM
        duration: Duration in seconds
        num_pulses: Number of PWM pulses per engine cycle
        sampling_rate: Sampling rate in Hz
        degradation_factor: 0.0 (new) to 1.0 (heavily degraded) - adds spikes at ramp-up
    
    Returns:
        time, pwm_signal, duty_cycle_per_pulse
    """
    # Calculate engine cycle time for 4-cylinder engine
    # 4-stroke engine: 2 rotations per cycle, 4 cylinders
    engine_cycle_time = (60 / rpm) * 2  # Time for one complete cycle (2 rotations)
    
    # Time array
    time = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Calculate number of complete engine cycles in the duration
    num_cycles = int(duration / engine_cycle_time)
    if num_cycles == 0:
        num_cycles = 1
    
    # PWM signal initialization
    pwm_signal = np.zeros_like(time)
    
    # Generate PWM pulses
    pulse_width_base = engine_cycle_time / num_pulses  # Base pulse width
    duty_cycle_per_pulse = []
    
    # Set random seed based on degradation for consistency
    np.random.seed(int(degradation_factor * 1000))
    
    for cycle in range(num_cycles):
        cycle_start_time = cycle * engine_cycle_time
        
        for pulse_idx in range(num_pulses):
            # Vary duty cycle slightly for each pulse (realistic variation)
            # Higher duty cycle at beginning of cycle (more fuel demand)
            duty_cycle = 0.3 + 0.4 * (1 - pulse_idx / num_pulses) + np.random.normal(0, 0.05)
            duty_cycle = np.clip(duty_cycle, 0.1, 0.9)  # Keep between 10% and 90%
            
            pulse_start = cycle_start_time + pulse_idx * (engine_cycle_time / num_pulses)
            pulse_on_duration = pulse_width_base * duty_cycle
            
            # Create PWM pulse (0V to 12V for automotive systems)
            mask = (time >= pulse_start) & (time < pulse_start + pulse_on_duration)
            pwm_signal[mask] = 12.0  # 12V when ON
            
            # Add degradation effects
            if degradation_factor > 0:
                # Early season (< 50%): No degradation - clean signal
                if degradation_factor < 0.5:
                    # No spikes or degradation before mid-season
                    pass
                
                # Late season (>= 50%): Exponential degradation with pulse instability
                else:
                    # Exponential degradation factor after mid-season
                    exp_degradation = (degradation_factor - 0.5) * 2  # 0 to 1 for second half
                    exp_factor = np.exp(3 * exp_degradation) - 1  # Exponential growth
                    exp_factor = exp_factor / (np.exp(3) - 1)  # Normalize to 0-1
                    
                    # Calculate probability of this pulse being affected
                    # More aggressive progression: start with 2-3 pulses at race 13, all by race 20
                    # Use steeper curve for faster progression
                    affected_pulse_ratio = exp_factor ** 0.7  # Power < 1 makes it grow faster early on
                    affected_pulse_count = int(2 + affected_pulse_ratio * 8)  # 2 to 10 pulses affected
                    
                    # Determine which pulses are affected (consistent per race)
                    pulse_hash = hash(f"{degradation_factor}_{pulse_idx}") % 100
                    pulse_threshold = (affected_pulse_count / num_pulses) * 100
                    
                    pulse_is_affected = pulse_hash < pulse_threshold
                    
                    # Only apply degradation to affected pulses
                    if pulse_is_affected:
                        # Add instability throughout the pulse (voltage fluctuations)
                        pulse_mask = (time >= pulse_start) & (time < pulse_start + pulse_on_duration)
                        pulse_time = time[pulse_mask] - pulse_start
                        
                        # Multiple frequency components for unstable signal
                        instability = 0
                        
                        # Low frequency drift (power supply instability)
                        instability += 0.5 * exp_factor * np.sin(2 * np.pi * 500 * pulse_time)
                        
                        # Medium frequency oscillation (driver circuit issues)
                        instability += 0.8 * exp_factor * np.sin(2 * np.pi * 2000 * pulse_time + pulse_idx)
                        
                        # High frequency jitter (noise and interference)
                        instability += 1.5 * exp_factor * np.sin(2 * np.pi * 8000 * pulse_time + pulse_idx * 2)
                        
                        # Random spikes/dropouts (severe degradation)
                        num_glitches = int(10 * exp_factor)
                        for glitch_idx in range(num_glitches):
                            glitch_seed = int((pulse_idx + glitch_idx) * 1234) % len(pulse_time)
                            glitch_pos = glitch_seed
                            glitch_width = max(1, int(len(pulse_time) * 0.02))
                            glitch_start = max(0, glitch_pos - glitch_width // 2)
                            glitch_end = min(len(pulse_time), glitch_pos + glitch_width // 2)
                            glitch_amplitude_choices = [-3, -2, -1, 1, 2, 3]
                            glitch_amplitude = glitch_amplitude_choices[(pulse_idx + glitch_idx) % len(glitch_amplitude_choices)] * exp_factor
                            
                            if glitch_start < glitch_end:
                                instability[glitch_start:glitch_end] += glitch_amplitude
                        
                        # Apply instability
                        pwm_signal[pulse_mask] += instability
                        
                        # Add rising edge spike (still present but less dominant)
                        spike_duration = pulse_width_base * 0.015
                        spike_mask = (time >= pulse_start) & (time < pulse_start + spike_duration)
                        spike_time = time[spike_mask] - pulse_start
                        spike_pattern = (2.0 + 1.5 * exp_factor) * np.exp(-60000 * spike_time) * np.sin(2 * np.pi * 12000 * spike_time)
                        pwm_signal[spike_mask] += spike_pattern
                
                # Clip to reasonable voltage range
                pwm_signal = np.clip(pwm_signal, 0, 18)
            
            duty_cycle_per_pulse.append({
                'cycle': cycle,
                'pulse': pulse_idx,
                'duty_cycle': duty_cycle * 100,
                'start_time': pulse_start,
                'duration': pulse_on_duration
            })
    
    # Reset random seed
    np.random.seed(None)
    
    return time, pwm_signal, pd.DataFrame(duty_cycle_per_pulse)

# Generate PWM with anomaly
def generate_pump_pwm_with_anomaly(rpm=8000, duration=0.1, num_pulses=10, anomaly_pulse=5, degradation_factor=0.0):
    """Generate PWM signal with an anomaly in one specific pulse"""
    time, pwm_signal, df_pulses = generate_pump_pwm_signal(rpm, duration, num_pulses, degradation_factor=degradation_factor)
    
    # Introduce anomaly: reduce duty cycle significantly for one pulse
    if anomaly_pulse < len(df_pulses):
        pulse_info = df_pulses.iloc[anomaly_pulse]
        anomaly_start = pulse_info['start_time']
        anomaly_duration = pulse_info['duration'] * 0.3  # Reduce to 30% of normal
        
        # Find the anomalous pulse region and reduce it
        mask = (time >= anomaly_start) & (time < anomaly_start + pulse_info['duration'])
        pwm_signal[mask] = 0  # Turn off
        
        # Turn on only for reduced duration
        mask_reduced = (time >= anomaly_start) & (time < anomaly_start + anomaly_duration)
        pwm_signal[mask_reduced] = 12.0
        
        df_pulses.loc[anomaly_pulse, 'duty_cycle'] = df_pulses.loc[anomaly_pulse, 'duty_cycle'] * 0.3
        df_pulses.loc[anomaly_pulse, 'anomaly'] = True
    
    df_pulses['anomaly'] = df_pulses.get('anomaly', False)
    
    return time, pwm_signal, df_pulses

# Introduction Section
with st.expander('Introduction', expanded=False):
    
    col1, col2 = st.columns([0.6,1])
    
    with col1:
        st.markdown("""
        This is a concept tool to demonstrate the ability of using foundation models to 
        analyse data and provide insights. It is still work in progress but enough as proof of concept.

        The idea is to use foundation models to gain in-sights of time-series data and provide insights
        that can be easily interpreted. As the performance of foundation models improve every year,
        it can analyse more complete data and provide more accurate insights to patterns and relationships
        in the data.

        Anomaly detection with root cause analysis is a powerful tool to identify faults and 
        why it happen. Foundation models can also explain how to prevent the issue from happening and 
        how we can slow down the degradtion rate.

        This dashboard a proof of concept of this idea. It generates aritificial fuel pump PWM signal for all races
        for the year and induced anomalies and instabilities in the PWM signal to simulate
        rotor failure. We then use this data to analyse the anomaly and provide insights of what could have caused 
        the degradation from racing perspective.

        """)
    
    with col2:
        # Display introduction image
        image = read_image("images/insights.png")
        st.image(image,use_column_width=True)


# Main Analysis Section
col1, col2 = st.columns(2)
with col1:
    with st.container(height=500,border=True):
        st.subheader("Fuel Pump PWM Simulation")
        
        # Controls
        # Fixed engine speed
        rpm = 14000  # Fixed at 14000 RPM
        
        # Race selection slider
        current_race = st.slider("Select Race (2026 F1 Season)", min_value=1, max_value=24, value=1, step=1)
    
        
        
        # Fixed parameters for clarity - ensure we see exactly 10 pulses
        num_pulses = 10
        
        # Calculate duration to fit exactly 10 pulses based on RPM
        engine_cycle_time = (60 / rpm) * 2  # Time for one complete cycle in seconds
        duration_per_race = engine_cycle_time * 1.1  # Add 10% buffer to show complete cycle
        
        # Calculate degradation factor for the selected race
        # Progressive degradation: 0.0 (race 1) to 1.0 (race 24)
        race_idx = current_race - 1
        degradation_factor = race_idx / 23.0  # 0 to 1 over 24 races
        
        # Generate signal for current race only
        time, pwm_signal, df_pulses = generate_pump_pwm_signal(rpm, duration_per_race, num_pulses, degradation_factor=degradation_factor)
        
        # Convert time to milliseconds for better readability
        time_ms = time * 1000
        
        fig = go.Figure()
        

        color = '#00d4ff'  # Cyan for black background
        fillcolor = 'rgba(0, 212, 255, 0.3)'
        signal_name = f'Race {current_race} PWM'

        # Add PWM signal for current race
        fig.add_trace(go.Scatter(
            x=time_ms,
            y=pwm_signal,
            mode='lines',
            name=signal_name,
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=fillcolor
        ))
        

        
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title='Time (ms)',
            yaxis_title='Voltage (V)',
            hovermode='x unified',
            height=300,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#333333',
                color='white',
                zerolinecolor='#555555'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#333333',
                range=[-1, 15],
                color='white',
                zerolinecolor='#555555'
            ),
            # Enable box selection mode
            dragmode='select',
            selectdirection='h'  # Horizontal selection for time window
        )
        
        # Display plot with selection capability
        selected_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="box", key=f"pwm_plot_{current_race}")
        
        # Process selection
        if selected_data and 'selection' in selected_data and 'box' in selected_data['selection']:
            selection_box = selected_data['selection']['box']
            if selection_box:
                for box in selection_box:
                    if 'x' in box and len(box['x']) >= 2:
                        # Extract time window
                        time_start = min(box['x'])
                        time_end = max(box['x'])
                        
                        # Store selection in session state
                        if 'time_selection' not in st.session_state:
                            st.session_state.time_selection = {}
                        
                        st.session_state.time_selection[current_race] = {
                            'start': time_start,
                            'end': time_end,
                            'duration': time_end - time_start
                        }
        
        # Display selected time window
        if 'time_selection' in st.session_state and current_race in st.session_state.time_selection:
            selection = st.session_state.time_selection[current_race]
            
            st.metric("Start Time", f"{selection['start']:.3f} ms")
            st.metric("End Time", f"{selection['end']:.3f} ms")
            st.metric("Duration", f"{selection['duration']:.3f} ms")
        
with col2:
    with st.container(height=500,border=True):
        st.subheader("Data in-sights")

        # Display selected time window
        if 'time_selection' in st.session_state and current_race in st.session_state.time_selection:
            selection = st.session_state.time_selection[current_race]
            
            st.write(f"Start: {selection['start']:.3f} ms, End: {selection['end']:.3f} ms, Duration: {selection['duration']:.3f} ms")
        
            # Add text input and button in columns
            col1, col2 = st.columns([3, 1])
            with col1:
                extra_info = st.text_input(
                    "Additional context", 
                    placeholder="Add extra instructions",
                    key=f"extra_info_{current_race}"
                )
            with col2:
                st.write("")  # Add spacing
                find_button = st.button("Find root cause")
        
            if find_button:
                # Convert selected time window (ms) to seconds
                start_time_s = selection['start'] / 1000
                end_time_s = selection['end'] / 1000
                
                # Extract PWM voltage values within the selected time window
                # Create mask for time array
                time_mask = (time >= start_time_s) & (time <= end_time_s)
                
                # Get time and voltage data within selection
                selected_time = time[time_mask]
                selected_voltage = pwm_signal[time_mask]
                
                # Convert to milliseconds for JSON
                selected_time_ms = selected_time * 1000
                
                # Create dataframe with time and voltage
                df_pwm_selected = pd.DataFrame({
                    'time_ms': selected_time_ms,
                    'voltage': selected_voltage
                })
                
                # Also get pulse information within the window
                pulse_mask = (df_pulses['start_time'] >= start_time_s) & (df_pulses['start_time'] <= end_time_s)
                df_pulses_selected = df_pulses[pulse_mask].copy()
                
                # Create comprehensive JSON with both PWM data and pulse info
                json_data = {
                    'selection_info': {
                        'start_time_ms': selection['start'],
                        'end_time_ms': selection['end'],
                        'duration_ms': selection['duration'],
                        'race_number': current_race,
                        'total_races': 24,
                        'rpm': rpm,
                        'degradation_factor': degradation_factor * 100
                    },
                    'pwm_voltage_data': df_pwm_selected.to_dict(orient='records'),
                    'pulse_info': df_pulses_selected.to_dict(orient='records'),
                    'statistics': {
                        'num_samples': len(selected_voltage),
                        'num_pulses': len(df_pulses_selected),
                        'avg_voltage': float(np.mean(selected_voltage)),
                        'max_voltage': float(np.max(selected_voltage)),
                        'min_voltage': float(np.min(selected_voltage)),
                        'std_voltage': float(np.std(selected_voltage))
                    }
                }
                
               
                # Option to download JSON
                json_str = json.dumps(json_data, indent=2)

                # Pass extra info to analysis
                modes_1, observations_1, nodes_1, adj_matrix_1, anomaly_detected = analyse(json_data, extra_info if extra_info else "")

                # Display anomaly alarm
                if anomaly_detected:
                    st.error("⚠️ Anomaly Detected")
                elif anomaly_detected is False:
                    st.success("✅ No Anomaly")
                else:
                    st.info("❓ Unable to determine anomaly status")

                if adj_matrix_1 != []:
                    df_matrix = pd.DataFrame(adj_matrix_1, columns=nodes_1, index=nodes_1)
                    
                    try:
                        d3 = d3graph()
                        d3.graph(df_matrix)
                        d3.show(figsize=(400, 500))
                    except Exception as e:
                        if "no links" in str(e).lower() or "weight > 0" in str(e):
                            st.success("✅ No anomalies detected in the selected PWM window")
                            st.info("The fuel pump appears to be operating normally in this time range.")
                            
                            # Show the factors identified even if no relationships
                            if nodes_1:
                                st.write("**Factors analyzed:**")
                                for node in nodes_1:
                                    st.write(f"- {node}")
                        else:
                            st.error(f"Error generating knowledge graph: {str(e)}")
                else:
                    st.warning("⚠️ Unable to generate analysis from the data")



           
        else:
            st.write("Select time window to see data in sights")

st.write('Copyright © 2025 Farraen. All rights reserved.')
