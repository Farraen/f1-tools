import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import time
import plotly.express as px
import math
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions import single_obj as fx

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

# ---------------- Initialisation ------------------

if 'x_rotor_v_angle_deg' not in st.session_state:
    st.session_state['x_rotor_v_angle_deg'] = np.array([90.,  100., 110., 120., 140., 150., 170., 180.])

if 'y_motor_torque_loss_v_angle' not in st.session_state:
    st.session_state['y_motor_torque_loss_v_angle'] =  (600-np.array([403., 410., 412., 411., 404., 394., 350., 325.])) /2

if 'y_mech_stress_mpa' not in st.session_state:
    st.session_state['y_mech_stress_mpa'] =  np.array([445., 440., 443., 440., 405., 375., 300., 245.])

if 'x_motor_dia_mm' not in st.session_state:
    st.session_state['x_motor_dia_mm'] =  np.array([160.,180.,200.,220.,240])

if 'y_motor_torque_loss_dia' not in st.session_state:
    st.session_state['y_motor_torque_loss_dia'] =  np.array([190.,170.,150.,130.,110.])

if 'y_mises_stress_mpa' not in st.session_state:
    st.session_state['y_mises_stress_mpa'] =  np.array([350.,374.,400.,425.,450.])

if 'Tmax' not in st.session_state:
    st.session_state['Tmax'] =  600.


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


# Functions

def filter_soh(x,y):

    filtered_x = [value for value in x if value > 0.5]
    filtered_y = [y[i] for i in range(len(x)) if x[i] > 0.5]

    return filtered_x, filtered_y

def calculate_time_delta(result_optimised,result_competitor):

    y_new = np.interp(result_optimised['distance'], result_competitor['distance'], result_competitor['time'])
    list1 = list(y_new.copy())
    list2 = list(result_competitor['time'].values)
    max_length = max(len(list1),len(list2))
    list1.extend([0] * (max_length - len(list1)))
    list2.extend([0] * (max_length - len(list2)))
    time_lap = list1
    lap_delta = [a - b for a, b in zip(list1, list2)]
    min_length = min(result_optimised['distance'].shape[0],result_competitor['distance'].shape[0])
    lap_delta = lap_delta[0:min_length]

    return list2[:-10], lap_delta[:-10]


def simulate_vehicle(
        rotor_v_angle_deg,
        rotor_dia_mm,
        motor_max_rpm_knee,
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
    

    x_rotor_v_angle_deg = st.session_state['x_rotor_v_angle_deg']
    y_motor_torque_loss_v_angle = st.session_state['y_motor_torque_loss_v_angle']
    y_mech_stress_mpa = st.session_state['y_mech_stress_mpa']

    x_motor_dia_mm = st.session_state['x_motor_dia_mm']
    y_motor_torque_loss_dia = st.session_state['y_motor_torque_loss_dia']
    y_mises_stress_mpa = st.session_state['y_mises_stress_mpa']

    Tmax = st.session_state['Tmax'] 


    f = interp1d(np.array(x_rotor_v_angle_deg),np.array(y_motor_torque_loss_v_angle),kind='cubic')
    torque_loss_1 = f([rotor_v_angle_deg])

    f = interp1d(np.array(x_motor_dia_mm),np.array(y_motor_torque_loss_dia),kind='cubic')
    torque_loss_2 = f([rotor_dia_mm])

    motor_torque_nm = Tmax - torque_loss_1[0] - torque_loss_2[0]
    motor_power_watt = float(1000. * motor_torque_nm * motor_max_rpm_knee / 9549.)


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
        torque = saturation(torque, torque_lower_limit, torque_upper_limit)

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

    lap_time = df_result['time'].values[-1]


    # Calculate degradation
    mileage = list(np.arange(0,100000,1000))
    max_mileage = mileage[-1]
    failure_point = mileage[-1]
    failure_point = mileage[-1]

    dmin = np.min(st.session_state['x_motor_dia_mm'])
    dmax = np.max(st.session_state['x_motor_dia_mm'])

    ageing_acceleration_factor = 1+(3-1)*(rotor_dia_mm-dmin)/(dmax-dmin)
    y = 1-ageing_acceleration_factor*np.square(mileage/(max_mileage))
    y_age,x_age = filter_soh(y,mileage)

    return df_result, efficiency_map,speed_range ,torque_range, motor_power_watt, motor_torque_nm, lap_time, x_age, y_age


# Competitor and baseline
rotor_v_angle_deg = 110.
rotor_dia_mm = 180.
motor_max_rpm_knee = 4930
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

result_competitor, efficiency_map_competitor,speed_range ,torque_range, max_motor_power_watt_competitor, max_motor_torque_nm_competitor,lap_time_competitor, x_ageing_competitor, y_ageing_competitor = simulate_vehicle(
                                                    rotor_v_angle_deg,
                                                    rotor_dia_mm,
                                                    motor_max_rpm_knee,
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

    col1.write('Future plans:')
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

with st.expander('Motor data and assumptions', expanded=True):

    col1, col2 = st.columns([1,1])
    st.session_state['Tmax'] = col1.number_input('Maximum torque',400,700,600)
    motor_max_rpm_knee = col2.number_input('Speed at max torque (knee point)',4000,6000,4930)

    x_rotor_v_angle_deg = st.session_state['x_rotor_v_angle_deg']
    y_motor_torque_loss_v_angle = st.session_state['y_motor_torque_loss_v_angle']
    y_mech_stress_mpa = st.session_state['y_mech_stress_mpa']
    x_motor_dia_mm = st.session_state['x_motor_dia_mm']
    y_motor_torque_loss_dia = st.session_state['y_motor_torque_loss_dia']
    y_mises_stress_mpa = st.session_state['y_mises_stress_mpa']

    col1,col2 = st.columns([1,1])
    fig4_motor = go.Figure()
    fig4_motor.add_trace(go.Scatter(x=x_rotor_v_angle_deg, y=y_motor_torque_loss_v_angle, mode='lines+markers'))
    fig4_motor.update_layout(title='Motor Torque', xaxis_title='Rotor V-angle (deg)', yaxis=dict(title='Torque loss (Nm)'),template='plotly_dark')
    col1.plotly_chart(fig4_motor, use_container_width=True)

    fig5_motor = go.Figure()
    fig5_motor.add_trace(go.Scatter(x=x_rotor_v_angle_deg, y=y_mech_stress_mpa, mode='lines+markers'))
    fig5_motor.update_layout(title='Motor Mechanical Stress', xaxis_title='Rotor V-angle (deg)', yaxis=dict(title='Mechanical Stress (MPa)'),template='plotly_dark')
    col2.plotly_chart(fig5_motor, use_container_width=True)


    col1,col2 = st.columns([1,1])
    fig6_motor = go.Figure()
    fig6_motor.add_trace(go.Scatter(x=x_motor_dia_mm, y=y_motor_torque_loss_dia, mode='lines+markers'))
    fig6_motor.update_layout(title='Torque loss for rotor diameter', xaxis_title='Rotor diameter (mm)', yaxis=dict(title='Torque loss (Nm)'),template='plotly_dark')
    col1.plotly_chart(fig6_motor, use_container_width=True)

    fig7_motor = go.Figure()
    fig7_motor.add_trace(go.Scatter(x=x_motor_dia_mm, y=y_mises_stress_mpa, mode='lines+markers'))
    fig7_motor.update_layout(title='Mises stress distribution for rotor diameter', xaxis_title='Rotor diameter (mm)', yaxis=dict(title='Mises stress (MPa)'),template='plotly_dark')
    col2.plotly_chart(fig7_motor, use_container_width=True)


with st.expander('Sanity check', expanded=True):
    col1, col2 = st.columns([1,1],gap='small')

    # Our car
    with col1:
        st.subheader('Our vehicle parameters')

        # Sliders
        rotor_v_angle_deg = st.slider("Rotor V-angle (deg)",90.0,180.0,110.0)
        rotor_dia_mm = st.slider("Rotor diameter (mm)",170.0,230.0,200.0)
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

    result_optimised, efficiency_map_optimised,speed_range ,torque_range, max_motor_power_watt, max_motor_torque_nm, lap_time, x_ageing, y_ageing = simulate_vehicle(
                                            rotor_v_angle_deg,
                                            rotor_dia_mm,
                                            motor_max_rpm_knee,
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
    x_lap_delta, y_lap_delta = calculate_time_delta(result_optimised,result_competitor)

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
    fig4.add_trace(go.Scatter(x=x_lap_delta, y=y_lap_delta, mode='lines', yaxis='y2', name='Lap delta',showlegend=True))
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

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=x_ageing_competitor, y=y_ageing_competitor, mode='lines',name='Competitor'))
    fig6.add_trace(go.Scatter(x=x_ageing, y=y_ageing, mode='lines',name='Ours'))
    fig6.update_yaxes(range=[0, 1])
    fig6.update_layout(title='Remaining Useful Life', xaxis_title='Mileage (miles)', yaxis=dict(title='State of Health (SoH)'),template='plotly_dark')

    st.plotly_chart(fig6, use_container_width=True)





    with col2:
        st.subheader('Vehicle performance metrics')
        
        Pmotor = np.round(max_motor_power_watt/1000,1)
        dPmotor = np.round((max_motor_power_watt-max_motor_power_watt_competitor)/1000,1)
        Tmotor = np.round(max_motor_torque_nm,1)
        dTmotor = np.round((max_motor_torque_nm-max_motor_torque_nm_competitor),1)

        col11, col22 = st.columns([1,1])
        col11.metric('Maximum motor power',f"{Pmotor} kW", dPmotor)
        col22.metric('Maximum motor torque',f"{Tmotor} Nm", dTmotor)

        col11, col22 = st.columns([1,1])
        col11.metric('Lap time',f"{lap_time} s")
        col22.metric('Lap time (competitor)',f"{lap_time_competitor} s")

        st.subheader('Vehicle performance plots')

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.plotly_chart(fig5, use_container_width=True)

# Define the objective function
def sphere_function(x):
    return np.sum(x ** 2, axis=1)


with st.expander('Vehicle parameter optimiser', expanded=True):

    start = st.button('Optimise')

    if start:
        # Set the bounds of the search space
        lower_bound = np.array([-5] * 2)  # 2-dimensional problem
        upper_bound = np.array([5] * 2)
        bounds = (lower_bound, upper_bound)

        # Configure the optimizer
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=30, dimensions=2, options=options, bounds=bounds)

        # Perform optimization
        best_cost, best_position = optimizer.optimize(sphere_function, iters=100)

        # Print the results
        st.write(f"Best cost: {best_cost}")
        st.write(f"Best position: {best_position}")
    




