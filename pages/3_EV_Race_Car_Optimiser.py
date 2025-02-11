import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import time
import plotly.express as px
import math
import plotly.graph_objects as go



# --------  For page layout  ---------------
st.set_page_config(layout="wide")

st.subheader('EV Race Car Simulator')

# ----------- Functions --------------------------
    
# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

def saturation(value, lower_limit, upper_limit):
    if value<lower_limit:
        value = lower_limit
    if value>upper_limit:
        value = upper_limit
    return value

def rpm_to_rad_per_sec(rpm):
    return rpm * (2 * math.pi / 60)

def deg_to_rad(deg):
    return (math.pi/180) * deg


# Define time array
distance = np.linspace(0, 300.0, 300+1)
track_length_m = distance[-1]

# Create test cycle
throttle = np.zeros_like(distance)
brake = np.zeros_like(distance)
road_angle = np.zeros_like(distance)
throttle[distance <= 100.0] = 1
throttle[distance >= 200.0] = 1
brake[(distance >= 150) & (distance < 200)] = 1  
df_input = pd.DataFrame({
    'Distance':distance,
    'Throttle': throttle,
    'Brake':brake,
})
track_length = 300.0

def simulate_vehicle(
        motor_power_watt,
        motor_efficiency,
        vehicle_weight,
        drag_coeff,
        frontal_area,
        battery_capacity,
        eta_regen,
        wheel_radius_m,
        rolling_coeff,
        drivetrain_efficiency,
        reduction_ratio):
    

    # Motor limits
    torque_upper_limit = 260.0
    torque_lower_limit = 0.10
    rpm_upper_limit = 10000.0
    rpm_lower_limit = 1.0
    speed_upper_limit = 82.0/3.6 #m/s
    speed_lower_limit = 0.10   

    # Parameters that doesnt change
    air_density = 1.225 # kg/m3 
    dt = 0.1
    intial_soc = 0.9
    a_decel = -2
    a_gravity = 9.81

    # Initial conditions
    speed_mps = 0
    distance_m = 0
    t = 0
    soc = intial_soc

    # List to store results
    time = [t]
    speed_list = [speed_mps * 3.6]
    soc_list = [soc]
    motor_power_list = []
    torque_list = []
    efficiency_list = []
    torque_list = []
    distance_list = []

    throttle_cmd = 0

    # Simulation loop
    while distance_m <= track_length_m: 

        # Determine the drive mode
        closest_index = (df_input['Distance']-distance_m).abs().idxmin()
        pedal_position = df_input.loc[closest_index, 'Throttle']
        brake_position = df_input.loc[closest_index, 'Brake']
        if (pedal_position==0) & (brake_position==0):
            pedal_position = 'cruise'
        elif brake_position==1:
            pedal_position = 'brake'
        else:
            pedal_position = 'accelerate'

        # Get motor speed (direct drive)
        motor_speed_rpm = speed_mps * 60/(math.pi*2*wheel_radius_m)
        motor_speed_rpm = saturation(motor_speed_rpm,rpm_lower_limit,rpm_upper_limit)

        F_drag = 0.5 * drag_coeff * frontal_area * air_density * speed_mps**2
        F_rolling = rolling_coeff * vehicle_weight * a_gravity
        F_gradient = 0
        F_required_tractive_N = F_drag + F_rolling + F_gradient

        if pedal_position == 'accelerate':

            available_power = motor_power_watt * soc

            if speed_mps > 0:
                F_motor = (available_power * motor_efficiency) / speed_mps
            else:
                F_motor = available_power * motor_efficiency

            F_gearbox = F_motor * drivetrain_efficiency
            F_surplus = F_gearbox - F_required_tractive_N
            Acceleration = F_surplus / vehicle_weight

            P_motor = F_motor * speed_mps / 1000
            energy_consumed = available_power * dt
            soc -= energy_consumed / battery_capacity
            motor_speed_radps = speed_mps / wheel_radius_m

        elif pedal_position == 'cruise':

            F_surplus = -F_required_tractive_N
            Acceleration = F_surplus / vehicle_weight
            P_motor = 0
            energy_consumed = 0

        elif pedal_position == 'brake':

            F_surplus = 3.5*(vehicle_weight * a_decel) - F_required_tractive_N
            Acceleration = F_surplus / vehicle_weight

            energy_recovered = -vehicle_weight * Acceleration * speed_mps * dt * eta_regen

            soc += energy_recovered / battery_capacity
            soc = min(soc, 1)

            P_motor = F_surplus * speed_mps / 1000

            energy_consumed = 0
    
        speed_mps += Acceleration * dt
        distance_m += speed_mps * dt
        t += dt
        
        # Calculate torque using motor efficiency
        if speed_mps > 0:
            motor_speed_radps = speed_mps / wheel_radius_m
            torque = 0.1 * (available_power * motor_efficiency) / motor_speed_radps
        else:
            torque = 0

        # Simplified efficiency calculation
        efficiency = motor_efficiency if pedal_position == 'accelerate' else eta_regen if pedal_position == 'brake' else 0

        time.append(round(t,1))
        speed_list.append(speed_mps * 3.6)
        soc_list.append(soc)
        motor_power_list.append(P_motor)
        torque_list.append(torque)
        efficiency_list.append(efficiency)
        distance_list.append(distance_m)

    df_result = pd.DataFrame({
                'time':time[:-1],
                'speed':speed_list[:-1],
                'soc':soc_list[:-1],
                'motor_power':motor_power_list,
                'torque':torque_list,
                'efficiency':efficiency_list,
                'distance':distance_list})

    # Generate efficiency map data
    speed_range = np.linspace(0,140, 100)
    torque_range = np.linspace(0,500, 100)
    efficiency_map = np.zeros((len(speed_range),len(torque_range)))

    for i,s in enumerate(speed_range):
        for j,t in enumerate(torque_range):
            meff = 2.2 - (motor_efficiency * (1 - (s / 140) * (t / 400))/0.5)
            if meff < 0.97:
                efficiency_map[i, j] = meff
            else:
                efficiency_map[i, j] = None

    return df_result, efficiency_map,speed_range ,torque_range

# Competitor
motor_power_watt = 150 * 1000
motor_efficiency = 0.9
vehicle_weight_kg = 1200
drag_coeff = 0.3
frontal_area = 0.5
battery_capacity_joules = 50 * 1000 * 3600
eta_regen = 0.7
wheel_radius_m = 0.203
rolling_coeff = 0.015
drivetrain_efficiency = 0.81
reduction_ratio = 7.8

result_competitor, efficiency_map_competitor,speed_range ,torque_range = simulate_vehicle(
                                                            motor_power_watt,
                                                            motor_efficiency,
                                                            vehicle_weight_kg,
                                                            drag_coeff,
                                                            frontal_area,
                                                            battery_capacity_joules,
                                                            eta_regen,
                                                            wheel_radius_m,
                                                            rolling_coeff,
                                                            drivetrain_efficiency,
                                                            reduction_ratio,
                                                            )


with st.expander('Introduction', expanded=True):
    col1, col2 = st.columns([1,1])
    col1.write('A simple EV race car simulator with regenerative braking. At the moment, the throttle and brake pedal is fixed for simplicity. Future upgrade will include actual lap data from F1 or similar.')

    col1.write('Future plan:')
    col1.markdown(
        """
        - Simple open-source battery model
        - Incorporate motor design
        - Optimisation and decision engine
        - More realistic track data
        - Animations
        """
    )

    col1.write('Inspired by Jonathan Blissett bike simulator: https://github.com/jonblissett/bike-sim')

    image = read_image("images/ev.png")
    col2.image(image)


with st.expander('Vehicle parameter optimiser', expanded=True):
    col1, col2 = st.columns([1,1],gap='small')

    # Our car
    with col1:
        st.subheader('Our vehicle parameters')

        motor_power_watt = st.slider("Motor power (kW)",100.0,300.0,150.0) * 1000
        motor_efficiency = st.slider("Motor efficiency", 0.5, 1.0, 0.9)
        vehicle_weight_kg = st.slider("Vehicle weight (kg)", 700.0, 2000.0, 1100.0)
        drag_coeff = st.slider("Drag coefficient", 0.1, 1.0, 0.3)
        frontal_area = st.slider("Frontal area (m2)", 0.1, 5.0, 0.5)
        battery_capacity_joules = st.slider("Battery capacity(kWh)",20.0, 100.0, 50.0) * 1000 * 3600
        eta_regen = st.slider("Regenerative braking efficiency", 0.1, 1.0, 0.7)
        wheel_radius_m = st.slider("Wheel radius (m)", 0.15, 0.5, 0.203)
        rolling_coeff = st.slider("Rolling coefficients", 0.0, 0.1, 0.015)
        drivetrain_efficiency = st.slider("Drivetrain efficiency", 0.1, 1.0, 0.81)
        reduction_ratio = st.slider("Torque reduction ratio in gearbox", 0.0, 10.0, 7.8)

    result_optimised, efficiency_map_optimised,speed_range ,torque_range = simulate_vehicle(
                                                        motor_power_watt,
                                                        motor_efficiency,
                                                        vehicle_weight_kg,
                                                        drag_coeff,
                                                        frontal_area,
                                                        battery_capacity_joules,
                                                        eta_regen,
                                                        wheel_radius_m,
                                                        rolling_coeff,
                                                        drivetrain_efficiency,
                                                        reduction_ratio,
                                                        )
    
    # Calculate time delta
    y_new = np.interp(result_optimised['distance'], result_competitor['distance'], result_competitor['time'])
    list1 = list(y_new.copy())
    list2 = list(result_competitor['time'].values)
    max_length = max(len(list1),len(list2))
    list1.extend([0] * (max_length - len(list1)))
    list2.extend([0] * (max_length - len(list2)))
    time_lap = list1
    lap_delta = [a - b for a, b in zip(list1, list2)]

    # Plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=result_competitor['time'], y=result_competitor['speed'], mode='lines', name='Competitor'))
    fig1.add_trace(go.Scatter(x=result_optimised['time'], y=result_optimised['speed'], mode='lines', name='Our'))
    fig1.update_layout(title='Vehicle Speed', xaxis_title='Time (s)', yaxis_title='Speed (km/h)',template='plotly_dark')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=result_competitor['time'], y=result_competitor['soc'], mode='lines', name='Competitor'))
    fig2.add_trace(go.Scatter(x=result_optimised['time'], y=result_optimised['soc'], mode='lines', name='Our'))
    fig2.update_layout(title='Battery SoC', xaxis_title='Time (s)', yaxis_title='State of Charge (SoC)',template='plotly_dark')

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=result_competitor['time'], y=result_competitor['motor_power'], mode='lines', name='Competitor'))
    fig3.add_trace(go.Scatter(x=result_optimised['time'], y=result_optimised['motor_power'], mode='lines', name='Our'))
    fig3.update_layout(title='Motor Power', xaxis_title='Time (s)', yaxis_title='Power (kW)',template='plotly_dark')

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=result_competitor['time'], y=result_competitor['distance'], mode='lines', name='Competitor'))
    fig4.add_trace(go.Scatter(x=result_optimised['time'], y=result_optimised['distance'], mode='lines', name='Our'))
    fig4.add_trace(go.Scatter(x=list2[:-10], y=lap_delta[:-10], mode='lines', yaxis='y2', name='Lap delta',showlegend=True))
    fig4.update_layout(title='Lap Distance (m)', xaxis_title='Time (s)', yaxis=dict(title='Distance (m)'), yaxis2=dict(title='Time delta (s)', overlaying='y',side='right'),template='plotly_dark')

    fig5 = go.Figure()
    fig5.add_trace(
        go.Contour(
            z=efficiency_map_competitor,
            x=speed_range,
            y=torque_range,
            colorscale='Viridis',colorbar=dict(title='Efficiency'),contours=dict(showlabels=True))
        )
    fig5.add_trace(go.Scatter(x=[s for s in result_competitor['speed'] if s > 0], y=result_optimised['torque'], mode='markers', name='Operating Points'))

    fig5.update_layout(title='Motor Efficiency Map with Operating Points', xaxis_title='Speed (km/h)', yaxis=dict(title='Torque (N)'),template='plotly_dark')





    with col2:
        st.subheader('Vehicle performance plots')
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.plotly_chart(fig5, use_container_width=True)








