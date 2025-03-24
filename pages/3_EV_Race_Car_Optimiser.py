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
import pygad
from datetime import datetime 
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


start_time = datetime.now() 


# --------  For page layout  ---------------
st.set_page_config(layout="wide")

st.subheader('EV Race Car Simulator and Motor Generative Design')

st.caption("Optimized for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")

# ----------- Functions --------------------------
    
# For loading images
@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

# For range
@st.cache_resource
def load_range(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df

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

if "fitness_gen" not in st.session_state:
    st.session_state.fitness_gen = []

if "pareto_plot_placeholder" not in st.session_state:
    st.session_state.pareto_plot_placeholder = None

if "fitness_plot_placeholder" not in st.session_state:
    st.session_state.fitness_plot_placeholder = None

if "fitness_gen_2" not in st.session_state:
    st.session_state.fitness_gen_2 = []


if "ml_plot_placeholder_1" not in st.session_state:
    st.session_state.ml_plot_placeholder_1 = None

if "ml_plot_placeholder_2" not in st.session_state:
    st.session_state.ml_plot_placeholder_2 = None

if "ml_plot_placeholder_3" not in st.session_state:
    st.session_state.ml_plot_placeholder_3 = None

if "ml_plot_placeholder_4" not in st.session_state:
    st.session_state.ml_plot_placeholder_4 = None

if "ml_plot_placeholder_5" not in st.session_state:
    st.session_state.ml_plot_placeholder_5 = None

if "ml_loss_fig_1" not in st.session_state:
    st.session_state.ml_loss_fig_1 = None

if "ml_loss_fig_2" not in st.session_state:
    st.session_state.ml_loss_fig_2 = None

if "ml_loss_fig_3" not in st.session_state:
    st.session_state.ml_loss_fig_3 = None

if "ml_loss_fig_4" not in st.session_state:
    st.session_state.ml_loss_fig_4 = None

if "ml_loss_fig_5" not in st.session_state:
    st.session_state.ml_loss_fig_5 = None


if "method_1" not in st.session_state:
    st.session_state.method_1 = '-'

if "method_2" not in st.session_state:
    st.session_state.method_2 = '-'

if "pareto_fig" not in st.session_state:
    st.session_state.pareto_fig = None

if "fitness_fig" not in st.session_state:
    st.session_state.fitness_fig = None



if "results" not in st.session_state:
    st.session_state.results = []

if "results_on_gen" not in st.session_state:
    st.session_state.results_on_gen = None

if "Tmax" not in st.session_state:
    st.session_state['Tmax'] = []

if "ml" not in st.session_state:
    st.session_state['ml'] = None


df = load_range('data/Page3_range.csv')
st.session_state.solution_baseline = df.iloc[0,:].to_dict()
st.session_state.solution_best_gen = df.iloc[3,:].to_dict()
st.session_state.columns = df.columns
st.session_state.df_gen = df

arr = df.loc[['Min','Max'],:]




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
steps = 50


# Functions

def plot_fitness():

    x = list(range(0,len(st.session_state.fitness_gen)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=st.session_state.fitness_gen,
        mode='lines',
        line = dict(width = 4, color = "lightblue"),
        marker = dict(color = "cyan", size = 15, opacity = 0.8),
    ))

    fig.update_layout(
        yaxis_title='Fitness',
        xaxis_title='Iteration',
        showlegend=False,
        height=250,
        margin=dict(l=20, r=20, t=20, b=20))
    return fig


def plot_pareto():

    df_pareto = st.session_state.results
    r = st.session_state.results_on_gen  


    fig = go.Figure(go.Scatter(
        x=r['Torque'],
        y=r['Durability'],
        mode='markers',
        showlegend=False,
        hoverinfo='skip',  # Disable hover info
        marker=dict(
            size=10,
            color=r['LapTime'],  # Use 'z' column to set marker color
            opacity=0.1,
            colorscale='Blues',  # Change 'Viridis' to any other colorscale you prefer
            colorbar=dict(title='Lap time (s)')  # Optional: Add a colorbar
        ),
    ))


    customdata = np.stack((df_pareto['rotorAngle'],df_pareto['rotorDiameter'],df_pareto['MotorPower']/1000,df_pareto['LapTime']), axis=-1)

    fig.add_trace(
        go.Scatter(
            x=df_pareto["Torque"],
            y=df_pareto["Durability"],
            mode='markers',
            showlegend=False,
            marker=dict(
                size=10,
                color='red'
                ),
            customdata=customdata,
            hovertemplate =
                        'Best solution:'+
                        '<br>Rotor angle</b>: %{customdata[0]:.1f} deg' +
                        '<br>Rotor diameter</b>: %{customdata[1]:.1f} mm' +
                        '<br>Performance</b>: %{x:.1f} Nm' +
                        '<br>Motor power</b>: %{customdata[2]:.1f} kW' +
                        '<br>Durability</b>: %{y:.1f} Miles' +
                        '<br><b>Lap time: %{customdata[3]} s</b>'
        )
    )
    fig.update_layout(height=450,margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(
        xaxis_title='Vehicle Performance (Torque)',
        yaxis=dict(title='Motor Durability (Miles)'),
        template='plotly_dark'
    )

    return fig



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

def handle_range(start_target, end_target):

    if start_target == 'Durability':
        start_target = 0.0
    elif start_target == 'Performance':
        start_target = 20.0
    else:
        start_target = np.double(start_target)

    if end_target == 'Durability':
        end_target = 0.0
    elif end_target == 'Performance':
        end_target = 20.0
    else:
        end_target = np.double(end_target)
    
    range_list = list(range(int(start_target), int(end_target+1)))

    return range_list

def simulate_vehicle(
        rotor_v_angle_deg: float = 110.0,
        rotor_dia_mm: float = 180.0,
        motor_max_rpm_knee: float = 4930,
        motor_efficiency: float = 0.9,
        vehicle_weight: float = 1200.0,
        drag_coeff: float = 0.3,
        frontal_area: float = 0.5,
        battery_capacity: float = 50 * 1000 * 3600,
        eta_regen: float = 0.7,
        wheel_radius_m: float = 0.203,
        rolling_coeff: float = 0.015,
        drivetrain_efficiency: float = 0.81,
        reduction_ratio: float = 7.8,
        ):
    

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

    v_spec = {
        "rotor_v_angle_deg":rotor_v_angle_deg,
        "rotor_dia_mm":rotor_dia_mm,
        "motor_max_rpm_knee":motor_max_rpm_knee,
        "motor_efficiency":motor_efficiency,
        "vehicle_weight":vehicle_weight,
        "drag_coeff":drag_coeff,
        "frontal_area":frontal_area,
        "battery_capacity":battery_capacity,
        "eta_regen":eta_regen,
        "wheel_radius_m":wheel_radius_m,
        "rolling_coeff":rolling_coeff,
        "drivetrain_efficiency":drivetrain_efficiency,
        "reduction_ratio":reduction_ratio,
    }

    output = {
        "df_result":df_result,
        "efficiency_map":efficiency_map,
        "speed_range":speed_range,
        "torque_range":torque_range,
        "motor_power_watt":motor_power_watt,
        "motor_torque_nm":motor_torque_nm,
        "lap_time":lap_time,
        "x_age":x_age,
        "y_age":y_age,
        "vehicle_spec":v_spec
    }

    return output



def find_pareto_front(df, maxX=True, maxY=True):
    """
    Identifies the Pareto frontier based on the first two columns of the DataFrame.
    Returns a filtered DataFrame containing only the Pareto front points.
    
    Parameters:
    - df: Input DataFrame. Only the first two columns will be used.
    - maxX: Boolean, whether to maximize the first column.
    - maxY: Boolean, whether to maximize the second column.
    """
    # Extract the first two columns for Pareto calculation
    Xs = df.iloc[:, 1].values  # First column
    Ys = df.iloc[:,4].values  # Second column

    # Sort points based on maxX and maxY preferences
    sorted_list = sorted(
        [[Xs[i], Ys[i], i] for i in range(len(Xs))],  # Store the index for later filtering
        key=lambda x: (-x[0] if maxX else x[0], -x[1] if maxY else x[1])
    )
    
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:  # Keep if it dominates in Y (maximizing)
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:  # Keep if it dominates in Y (minimizing)
                pareto_front.append(pair)

    # Extract the indices of Pareto points
    pareto_indices = [pair[2] for pair in pareto_front]

    # Filter the original DataFrame using the Pareto indices
    return df.iloc[pareto_indices].reset_index(drop=True)
@st.cache_data
def load_images_once():
    image1 = read_image("images/ev.png")
    image2 = read_image("images/ev_flow.png")
    image3 = read_image("images/overall_ev.png")
    image4 = read_image("images/catboost.png")

    return image1, image2, image3, image4

image1, image2, image3, image4 = load_images_once()

with st.expander('Introduction', expanded=True):
    col1, col2 = st.columns([1,1])
    col1.write('A simple EV race car simulator and motor design optimizer featuring regenerative braking. Currently, the throttle and brake pedal inputs are fixed for simplicity. Future updates will incorporate actual lap data from F1 or similar racing formats.')

    col1.write('Future plans:')
    col1.markdown(
        """
        - More realistic track data
        - Animations
        """
    )


    col2.image(image1)
    st.write('This dashboard introduces a novel optimization method powered by a decision engine, which is significantly faster than traditional optimization approaches. We leveraged all results and points visited by the optimizer to train a decision engine (an ML model). This innovation enables remarkably rapid design optimization (less than 1 second compared to 3 minutes) and generates hundreds of new designs. The output is a Pareto plot that illustrates the trade-off between durability and performance. A Pareto plot offers a more intuitive visualization compared to a single optimization solution..')

    col1, col2 = st.columns([1,0.5])
    col1.write('The model is a CatBoost model, which combines gradient boosting and ensemble techniques. Instead of interpolating, we are inputting as much data as possible—surpassing the limit set by the double-descent phenomenon to transform it into a decision engine.')
    col2.image(image4)
    
    st.image(image3)
    st.write('Inspired by Jonathan Blissett bike simulator: https://github.com/jonblissett/bike-sim')


with st.expander('Mathematical description',expanded=False):
    st.image(image2)


with st.expander('Motor data and assumptions', expanded=False):

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



def rescale_solution(solution):
    solution = np.double(solution)
    df = st.session_state.df_gen
    d = {}
    for index, var in enumerate(st.session_state.columns):
        dff = df[var]
        out= solution[index]*(dff['Max']-dff['Min'])/steps + dff['Min']
        d[var] =out
    return d

def rescale_solution_2d(solution):
    df = st.session_state.df_gen
    d = {}
    for index, var in enumerate(st.session_state.columns):
        dff = df[var]
        out= solution[:,index]*(dff['Max']-dff['Min'])/steps + dff['Min']
        d[var] =out
    return d

def on_generation(ga_instance):

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    st.session_state.fitness_gen.append(solution_fitness)

    if "data" in st.session_state.fitness_fig:
        x = list(range(0,len(st.session_state.fitness_gen)))
        st.session_state.fitness_fig.data[0].x = x
        st.session_state.fitness_fig.data[0].y = st.session_state.fitness_gen
    else:
        st.session_state.fitness_fig = plot_fitness()


    st.session_state.fitness_plot_placeholder.plotly_chart(st.session_state.fitness_fig, use_container_width=True)
    time.sleep(1.01)




def simulate_pipeline(solution):
    rotor_dia_mm = solution['rotorDiameter']
    rotor_v_angle_deg = solution['rotorAngle']
    out = simulate_vehicle(rotor_v_angle_deg,rotor_dia_mm)
    return out['lap_time'], out['x_age'], out['motor_power_watt'], out['motor_torque_nm'], out['vehicle_spec']

def fitness_func(bias):

    def fitness_func(ga_instance, solutionValue, solution_idx):

        solution = rescale_solution(solutionValue)
        lap_time, x_ageing, motor_power, torque, _ = simulate_pipeline(solution)
       
        perf_mod = 20/lap_time
        durab_mod = (x_ageing[-1]-20000)/(100000-20000)

        fitness = np.power((((10-bias)/10)),2)*durab_mod + np.power(((bias/10)),2)*perf_mod

        r1 = {'Bias':bias, 'Durability': x_ageing[-1], 'LapTime': lap_time, 'MotorPower': motor_power,'Torque': torque}
        r1.update(solution)
        st.session_state.results_on_gen.append(r1)
        

        return fitness

    return fitness_func



def train_ml(df_train,my_bar_ml):

    my_bar_ml.progress(10, text="Preparing training data...")

    # Split the data into features and target
    X = df_train[["rotorAngle","rotorDiameter","Bias"]]
    y = df_train[['LapTime','Durability','MotorPower','Torque']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    my_bar_ml.progress(30, text="Training model...")

    # Initialize the CatBoostRegressor
    model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=4, loss_function='MultiRMSE',
        verbose=200)

    # Fit the model
    model.fit(X_train, y_train)
    st.session_state.ml = model

    my_bar_ml.progress(70, text="Model is ready.")


    column_names = ['LapTime','Durability','MotorPower','Torque']
    predictions = model.predict(X_test)
    df_predictions = pd.DataFrame(predictions,columns=column_names)

    my_bar_ml.progress(80, text="Plotting results...")

    fig = go.Figure()
    loss = model.get_evals_result()['learn']['MultiRMSE']
    fig.add_trace(go.Scatter(y=loss, mode='lines',showlegend=False))
    fig.update_layout(height=200,margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(
        xaxis_title='Iteration',
        yaxis=dict(title='Loss'),
        template='plotly_dark')
    st.session_state.ml_loss_fig_1 = fig
    st.session_state.ml_plot_placeholder_1.plotly_chart(fig, use_container_width=True)


    col = 0
    fig = go.Figure()
    x,y = y_test[column_names[col]],df_predictions[column_names[col]]
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',showlegend=False))
    fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[y.min(), y.max()], mode='lines',showlegend=False))    
    fig.update_layout(height=200,margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(
        xaxis_title='Actual',
        yaxis=dict(title='Predicted'),
        template='plotly_dark')
    st.session_state.ml_loss_fig_2 = fig
    st.session_state.ml_plot_placeholder_2.plotly_chart(fig, use_container_width=True)



    col = 1
    fig = go.Figure()
    x,y = y_test[column_names[col]],df_predictions[column_names[col]]
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',showlegend=False))
    fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[y.min(), y.max()], mode='lines',showlegend=False))    
    fig.update_layout(height=200,margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(
        xaxis_title='Actual',
        yaxis=dict(title='Predicted'),
        template='plotly_dark')
    st.session_state.ml_loss_fig_3 = fig
    st.session_state.ml_plot_placeholder_3.plotly_chart(fig, use_container_width=True)


    col = 2
    fig = go.Figure()
    x,y = y_test[column_names[col]],df_predictions[column_names[col]]
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',showlegend=False))
    fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[y.min(), y.max()], mode='lines',showlegend=False))    
    fig.update_layout(height=200,margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(
        xaxis_title='Actual',
        yaxis=dict(title='Predicted'),
        template='plotly_dark')
    st.session_state.ml_loss_fig_4 = fig
    st.session_state.ml_plot_placeholder_4.plotly_chart(fig, use_container_width=True)

    col = 3
    fig = go.Figure()
    x,y = y_test[column_names[col]],df_predictions[column_names[col]]
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',showlegend=False))
    fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[y.min(), y.max()], mode='lines',showlegend=False))    
    fig.update_layout(height=200,margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(
        xaxis_title='Actual',
        yaxis=dict(title='Predicted'),
        template='plotly_dark')
    st.session_state.ml_loss_fig_5 = fig
    st.session_state.ml_plot_placeholder_5.plotly_chart(fig, use_container_width=True)

    my_bar_ml.progress(100, text="Done.")
    my_bar_ml.empty()    





with st.expander('Vehicle parameter optimiser', expanded=True):

    st.write('This is the panel run an optimiser to get best motor design depending on the function objective selected. The motor design parameters that can be optimised are v magnet angle and rotor diameter. Future version of the tool will include more motor design parameters.')



    st.write('Function objectives')
    col1, col2 = st.columns([0.5,1],gap='medium')
    with col1:
        st.caption('Use the slider to change the optmiser bias range. It is a range slider and the optimser will run a few times depending one the selected bias range.')

    with col2:
        start_target, end_target = st.select_slider(
            'Select optimisation target',
            options=['Durability', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Performance'],
            value=('Durability', 'Performance'))
        range_list = handle_range(start_target, end_target)

    st.divider()



    # Description
    col1, col2 = st.columns([1,1],gap='medium')
    with col1:
        st.write('Method A: Genetic Algorithm as the decision engine')
        st.caption('This method uses GA to create new designs.')

    with col2:
        st.write('Method B: Machine Learning mode as the decision engine')
        st.caption('Please run the optimiser once and press Train to initiate learning process.')
    


    m1 = st.empty()

    col1, col2 = st.columns([1,1],gap='medium')
    with col1:
        num_generations = st.number_input("Number of generation", value=20, placeholder="Type a number...")

        if st.session_state.fitness_plot_placeholder == None:
                fig = go.Figure()
                fig.update_layout(height=250,margin=dict(l=20, r=20, t=20, b=20))
                fig.update_layout(
                    xaxis_title='Iteration',
                    yaxis=dict(title='Fitness'),
                    template='plotly_dark')
                st.session_state.fitness_fig = fig
                st.session_state.fitness_plot_placeholder = st.plotly_chart(fig, use_container_width=True)


    
    with col2:

        col11, col22 = st.columns([1,1],gap='small')
        with col11:
            start_ml_train = st.button('Train use GA results')
        with col22:
            start_ml_train_2 = st.button('Train using previous file')

        my_bar_ml = st.empty()
        
        st.caption('Modeling performance')
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Loss", "LapTime", "Durability","MotorPower","Torque"])

        with tab1:
            if st.session_state.ml_plot_placeholder_1 == None:
                fig = go.Figure()
                fig.update_layout(height=200,margin=dict(l=20, r=20, t=20, b=20))
                fig.update_layout(
                    xaxis_title='Iteration',
                    yaxis=dict(title='Loss'),
                    template='plotly_dark')
                st.session_state.ml_loss_fig_1 = fig
                st.session_state.ml_plot_placeholder_1 = st.plotly_chart(fig, use_container_width=True)


        with tab2:
            if st.session_state.ml_plot_placeholder_2 == None:
                fig = go.Figure()
                fig.update_layout(height=200,margin=dict(l=20, r=20, t=20, b=20))
                fig.update_layout(
                    xaxis_title='Iteration',
                    yaxis=dict(title='Loss'),
                    template='plotly_dark')
                st.session_state.ml_loss_fig_2 = fig
                st.session_state.ml_plot_placeholder_2 = st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if st.session_state.ml_plot_placeholder_3 == None:
                fig = go.Figure()
                fig.update_layout(height=200,margin=dict(l=20, r=20, t=20, b=20))
                fig.update_layout(
                    xaxis_title='Iteration',
                    yaxis=dict(title='Loss'),
                    template='plotly_dark')
                st.session_state.ml_loss_fig_3 = fig
                st.session_state.ml_plot_placeholder_3 = st.plotly_chart(fig, use_container_width=True)

        with tab4:
            if st.session_state.ml_plot_placeholder_4 == None:
                fig = go.Figure()
                fig.update_layout(height=200,margin=dict(l=20, r=20, t=20, b=20))
                fig.update_layout(
                    xaxis_title='Iteration',
                    yaxis=dict(title='Loss'),
                    template='plotly_dark')
                st.session_state.ml_loss_fig_4 = fig
                st.session_state.ml_plot_placeholder_4 = st.plotly_chart(fig, use_container_width=True)

        with tab5:
            if st.session_state.ml_plot_placeholder_5 == None:
                fig = go.Figure()
                fig.update_layout(height=200,margin=dict(l=20, r=20, t=20, b=20))
                fig.update_layout(
                    xaxis_title='Iteration',
                    yaxis=dict(title='Loss'),
                    template='plotly_dark')
                st.session_state.ml_loss_fig_5 = fig
                st.session_state.ml_plot_placeholder_5 = st.plotly_chart(fig, use_container_width=True)







    col1, col2 = st.columns([1,1],gap='medium')
    with col1:

        start = st.button('Optimise')
        my_bar = st.empty()

    with col2:

        start_ml = st.button('Generate designs')
        my_bar_2 = st.empty()


    col1, col2 = st.columns([1,1],gap='medium')
    with col1:
        time_metric_1 = st.metric('Computational time:',f"{st.session_state.method_1}")

    with col2:
        time_metric_2 = st.metric('Computational time:',f"{st.session_state.method_2}")




    st.divider()






    st.write('Load a saved file instead of running the optimiser.')
    if st.button('Load file'):
        st.session_state.results = pd.read_pickle('ev_pareto_final.pkl')
        st.session_state.results_on_gen = pd.read_pickle('ev_pareto_on_gen.pkl')
        st.session_state.pareto_fig = plot_pareto()


    st.write('Pareto plot of the optimisation results')
    st.caption('Click on any red markers to view results')

    # Pareto plot: reload fig if it exist or plot an empty figure
    if st.session_state.pareto_plot_placeholder == None:
        st.session_state.pareto_plot_placeholder = st.empty()
        fig = go.Figure()
        fig.update_layout(height=450,margin=dict(l=20, r=20, t=20, b=20))
        fig.update_layout(
            xaxis_title='Vehicle Performance (Torque)',
            yaxis=dict(title='Motor Durability (Miles)'),
            template='plotly_dark')
        st.session_state.pareto_plot_placeholder.plotly_chart(fig, use_container_width=True)

    st.divider()

    t1 = st.empty()

    col1, col2 = st.columns([1,1])
    t2 = col1.empty()
    s1 = col1.empty()

    t3 = col2.empty()
    s2 = col2.empty()

    t4 = st.empty()
    t5 = st.empty()
    s3 = st.empty()

    col1, col2 = st.columns([1,1])
    p1 = col1.empty()
    p2 = col2.empty()

    col1, col2 = st.columns([1,1])
    p3 = col1.empty()
    p4 = col2.empty()

    col1, col2 = st.columns([1,1])
    p5 = col1.empty()
    p6 = col2.empty()


    if start_ml_train:
        if isinstance(st.session_state.results_on_gen,pd.DataFrame):
            df_train = st.session_state.results_on_gen
            train_ml(df_train,my_bar_ml)
        else:
            my_bar_ml.error('No training data available. Please run the GA optimiser once or use the button below to load a saved file.')

    
    if start_ml_train_2:
        df_train = pd.read_pickle('ev_pareto_on_gen.pkl')
        train_ml(df_train,my_bar_ml)


    if start_ml:

        start_time = datetime.now()

        out = simulate_vehicle()
        v_spec = out["vehicle_spec"]

        # Define ranges for each variable
        ranges = {
            "angle": (90.0, 180.0),
            "diameter": (170.0, 240.0),
        }

        # Number of random points to generate
        num_points = 100

        # Generate random points
        data = {
            key: np.random.uniform(low=val[0], high=val[1], size=num_points)
            for key, val in ranges.items()
        }

        df_doe = pd.DataFrame(data)

        #st.session_state.fitness_fig_2 
        #st.session_state.fitness_plot_placeholder_2
        progress_text = "Operation in progress. Please wait."
        st.session_state.results_on_gen = []
        result = {}
        index_dict = 0
        for index, bias in enumerate(range_list):

            my_bar_2.progress((index+1)/len(range_list), text=progress_text)
            st.session_state.fitness_gen_2 = []

         
            model = st.session_state.ml

            column_names = ['LapTime','Durability','MotorPower','Torque']
            
            for row in df_doe.iterrows():

                v_spec["rotor_v_angle_deg"] = row[1]['angle']
                v_spec["rotor_dia_mm"] = row[1]['diameter']

                predictions = model.predict([row[1]['angle'],row[1]['diameter'],bias])


                r1 = {'Bias':bias, 'Durability': predictions[1], 'LapTime': predictions[0], 'MotorPower': predictions[2],'Torque': predictions[3], 'VehicleSpec': v_spec,'rotorAngle':row[1]['angle'],'rotorDiameter':row[1]['diameter']}

                result[index_dict] = r1

                index_dict = index_dict + 1




        my_bar_2.empty()    


        st.session_state.results_on_gen =  pd.DataFrame.from_dict(result).T
        st.session_state.results = find_pareto_front(st.session_state.results_on_gen)

        st.session_state.pareto_fig = plot_pareto()

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
        # Format minutes and seconds with two digits
        minutes = str(int(minutes)).zfill(2)
        seconds = str(int(seconds)).zfill(2)
        st.session_state.method_2 = f"{minutes}:{seconds}"



    if start:
        start_time = datetime.now()
        
        st.session_state.fitness_fig = plot_fitness()
        st.session_state.fitness_plot_placeholder.plotly_chart(st.session_state.fitness_fig , use_container_width=True)

        if st.session_state.pareto_fig is not None:
            st.session_state.pareto_plot_placeholder.plotly_chart(st.session_state.pareto_fig, use_container_width=True)


        progress_text = "Operation in progress. Please wait."
        st.session_state.results_on_gen = []
        result = {}
        for index, bias in enumerate(range_list):

            my_bar.progress((index+1)/len(range_list), text=progress_text)
            st.session_state.fitness_gen = []

            num_parents_mating = 2  # Increase parents mating
            sol_per_pop = 5  # Increase population size
            num_genes = 2
            mutation_type = "random"
            fitness_function = fitness_func(bias)
            ga_instance = pygad.GA(num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                fitness_func=fitness_function,
                                sol_per_pop=sol_per_pop,
                                gene_space={'low': 0, 'high': steps, 'step': 1},                            
                                num_genes=num_genes,
                                gene_type=int,
                                stop_criteria="saturate_5",
                                mutation_type=mutation_type,
                                mutation_num_genes=1,
                                on_generation=on_generation,
                            )
            
            ga_instance.run()
            solution = ga_instance.best_solution()[0]
            solution = rescale_solution(ga_instance.best_solution()[0])


            lap_time, x_ageing, motor_power, torque, vehicle_spec = simulate_pipeline(solution)

            #st.write(f"{solution} - Laptime: {lap_time} - Durab: {x_ageing[-1]}")

            r1 = {'Bias':bias, 'Durability': x_ageing[-1], 'LapTime': lap_time, 'MotorPower': motor_power,'Torque': torque, 'VehicleSpec': vehicle_spec}

            r1.update(solution)
            result[index] = r1


            #st.session_state.fitness_fig = plot_fitness()
            #fitness_plot_placeholder.plotly_chart(st.session_state.fitness_fig, use_container_width=True)

        my_bar.empty()    

        st.session_state.results =  pd.DataFrame.from_dict(result).T
        st.session_state.results_on_gen =  pd.DataFrame.from_dict(st.session_state.results_on_gen)      

        st.session_state.pareto_fig = plot_pareto()

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
        # Format minutes and seconds with two digits
        minutes = str(int(minutes)).zfill(2)
        seconds = str(int(seconds)).zfill(2)
        st.session_state.method_1 = f"{minutes}:{seconds}"


time_metric_1.metric('Elaped time:',f"{st.session_state.method_1}")
time_metric_2.metric('Elaped time:',f"{st.session_state.method_2}")

# Fitness plot: reload fig if it exist or plot an empty figure
if st.session_state.fitness_fig is not None:  
    st.session_state.fitness_plot_placeholder.plotly_chart(st.session_state.fitness_fig, use_container_width=True, on_select="rerun")



# Ml plot replot

if st.session_state.ml_loss_fig_1 is not None:  
    st.session_state.ml_plot_placeholder_1.plotly_chart(st.session_state.ml_loss_fig_1, use_container_width=True)

if st.session_state.ml_loss_fig_2 is not None:  
    st.session_state.ml_plot_placeholder_2.plotly_chart(st.session_state.ml_loss_fig_2, use_container_width=True)

if st.session_state.ml_loss_fig_3 is not None:  
    st.session_state.ml_plot_placeholder_3.plotly_chart(st.session_state.ml_loss_fig_3, use_container_width=True)

if st.session_state.ml_loss_fig_4 is not None:  
    st.session_state.ml_plot_placeholder_4.plotly_chart(st.session_state.ml_loss_fig_4, use_container_width=True)

if st.session_state.ml_loss_fig_5 is not None:  
    st.session_state.ml_plot_placeholder_5.plotly_chart(st.session_state.ml_loss_fig_5, use_container_width=True)







# Pareto plot: reload fig if it exist or plot an empty figure
if st.session_state.pareto_plot_placeholder is not None:  
    if st.session_state.pareto_fig is not None:  
        events = st.session_state.pareto_plot_placeholder.plotly_chart(st.session_state.pareto_fig, use_container_width=True, on_select="rerun")

        if events["selection"]["points"]:

            points_index = events["selection"]["points"][0]["point_index"]
            s1.write(f'Selected point: {points_index}')

            x = events["selection"]["points"][0]["x"]
            y = events["selection"]["points"][0]["y"]


            df_results = st.session_state.results

            VehicleSpec = df_results.loc[points_index,'VehicleSpec']
            s1.table(VehicleSpec)

            Results = df_results.loc[points_index,:]
            Results.drop(['VehicleSpec','rotorAngle','rotorDiameter'],inplace=True)
            s2.table(Results)

            t5.write('Press the button below to run the vehicle model.')
            if s3.button('Simulate'):
                out_competitor = simulate_vehicle()
                out = simulate_vehicle(
                                VehicleSpec['rotor_v_angle_deg'],
                                VehicleSpec['rotor_dia_mm'],
                                VehicleSpec['motor_max_rpm_knee'],
                                VehicleSpec['motor_efficiency'],
                                VehicleSpec['vehicle_weight'],
                                VehicleSpec['drag_coeff'],
                                VehicleSpec['frontal_area'],
                                VehicleSpec['battery_capacity'],
                                VehicleSpec['eta_regen'],
                                VehicleSpec['wheel_radius_m'],
                                VehicleSpec['rolling_coeff'],
                                VehicleSpec['drivetrain_efficiency'],
                                VehicleSpec['reduction_ratio']
                                )
                
                df_comp = out_competitor['df_result']
                df_ours = out['df_result']
                lap_time = out['lap_time']
                lap_time_competitor = out_competitor['lap_time']
                x_lap_delta, y_lap_delta = calculate_time_delta(df_ours,df_comp)


                # Plot
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df_comp['time'], y=df_comp['speed'], mode='lines', name='Competitor'))
                fig1.add_trace(go.Scatter(x=df_ours['time'], y=df_ours['speed'], mode='lines', name='Ours'))
                fig1.update_layout(title='Vehicle Speed', xaxis_title='Time (s)', yaxis_title='Speed (km/h)',template='plotly_dark')

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_comp['time'], y=df_comp['soc'], mode='lines', name='Competitor'))
                fig2.add_trace(go.Scatter(x=df_ours['time'], y=df_ours['soc'], mode='lines', name='Ours'))
                fig2.update_layout(title='Battery SoC', xaxis_title='Time (s)', yaxis_title='State of Charge (SoC)',template='plotly_dark')

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df_comp['time'], y=df_comp['motor_power'], mode='lines', name='Competitor'))
                fig3.add_trace(go.Scatter(x=df_ours['time'], y=df_ours['motor_power'], mode='lines', name='Ours'))
                fig3.update_layout(title='Motor Power', xaxis_title='Time (s)', yaxis_title='Power (kW)',template='plotly_dark')

                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=df_comp['time'], y=df_comp['distance'], mode='lines', name='Competitor'))
                fig4.add_trace(go.Scatter(x=df_ours['time'], y=df_ours['distance'], mode='lines', name='Ours'))
                fig4.add_trace(go.Scatter(x=x_lap_delta, y=y_lap_delta, mode='lines', yaxis='y2', name='Lap delta',showlegend=True))
                fig4.update_layout(title='Lap Distance (m)', xaxis_title='Time (s)', yaxis=dict(title='Distance (m)'), yaxis2=dict(title='Time delta (s)', overlaying='y',side='right'),template='plotly_dark')

                fig5 = go.Figure()
                fig5.add_trace(
                    go.Contour(
                        z=out["efficiency_map"],
                        x=out["speed_range"],
                        y=out["torque_range"],
                        colorscale='Viridis',colorbar=dict(title='Efficiency'),contours=dict(showlabels=True))
                    )
                fig5.add_trace(go.Scatter(x=[s for s in df_comp['speed'] if s > 0], y=df_ours['torque'], mode='markers', name='Operating Points'))

                fig5.update_layout(title='Motor Efficiency Map with Operating Points', xaxis_title='Speed (km/h)', yaxis=dict(title='Torque (N)'),template='plotly_dark')

                fig6 = go.Figure()
                fig6.add_trace(go.Scatter(x=out_competitor["x_age"], y=out_competitor["y_age"], mode='lines',name='Competitor'))
                fig6.add_trace(go.Scatter(x=out["x_age"], y=out["y_age"], mode='lines',name='Ours'))
                fig6.update_yaxes(range=[0, 1])
                fig6.update_layout(title='Remaining Useful Life', xaxis_title='Mileage (Miles)', yaxis=dict(title='State of Health (SoH)'),template='plotly_dark')

                t1.write('Vehicle metrics')
                t2.write('Vehicle spefications:')
                t3.write('Results:')
                t4.write('Vehicle performance')


                p1.plotly_chart(fig1, use_container_width=True)
                p2.plotly_chart(fig2, use_container_width=True)
                p3.plotly_chart(fig3, use_container_width=True)
                p4.plotly_chart(fig4, use_container_width=True)
                p5.plotly_chart(fig5, use_container_width=True)
                p6.plotly_chart(fig6, use_container_width=True)


#  ---------------------  Diagnostics --------------------------------------

with st.expander('Sanity check', expanded=False):

    on = st.toggle("Activate feature",value=False)

    if on:
        col1, col2 = st.columns([1,1],gap='small')

        # Our car
        with col1:
            st.subheader('Our vehicle parameters')

            # Sliders
            rotor_v_angle_deg = st.slider("Rotor V-angle (deg)",90.0,180.0,110.0)
            rotor_dia_mm = st.slider("Rotor diameter (mm)",170.0,240.0,200.0)
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

        out_competitor = simulate_vehicle()
        out = simulate_vehicle(
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
        
        df_comp = out_competitor['df_result']
        df_ours = out['df_result']
        lap_time = out['lap_time']
        lap_time_competitor = out_competitor['lap_time']
        
        # Calculate time delta
        x_lap_delta, y_lap_delta = calculate_time_delta(df_ours,df_comp)

        # Plot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_comp['time'], y=df_comp['speed'], mode='lines', name='Competitor'))
        fig1.add_trace(go.Scatter(x=df_ours['time'], y=df_ours['speed'], mode='lines', name='Ours'))
        fig1.update_layout(title='Vehicle Speed', xaxis_title='Time (s)', yaxis_title='Speed (km/h)',template='plotly_dark')

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_comp['time'], y=df_comp['soc'], mode='lines', name='Competitor'))
        fig2.add_trace(go.Scatter(x=df_ours['time'], y=df_ours['soc'], mode='lines', name='Ours'))
        fig2.update_layout(title='Battery SoC', xaxis_title='Time (s)', yaxis_title='State of Charge (SoC)',template='plotly_dark')

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_comp['time'], y=df_comp['motor_power'], mode='lines', name='Competitor'))
        fig3.add_trace(go.Scatter(x=df_ours['time'], y=df_ours['motor_power'], mode='lines', name='Ours'))
        fig3.update_layout(title='Motor Power', xaxis_title='Time (s)', yaxis_title='Power (kW)',template='plotly_dark')

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df_comp['time'], y=df_comp['distance'], mode='lines', name='Competitor'))
        fig4.add_trace(go.Scatter(x=df_ours['time'], y=df_ours['distance'], mode='lines', name='Ours'))
        fig4.add_trace(go.Scatter(x=x_lap_delta, y=y_lap_delta, mode='lines', yaxis='y2', name='Lap delta',showlegend=True))
        fig4.update_layout(title='Lap Distance (m)', xaxis_title='Time (s)', yaxis=dict(title='Distance (m)'), yaxis2=dict(title='Time delta (s)', overlaying='y',side='right'),template='plotly_dark')

        fig5 = go.Figure()
        fig5.add_trace(
            go.Contour(
                z=out["efficiency_map"],
                x=out["speed_range"],
                y=out["torque_range"],
                colorscale='Viridis',colorbar=dict(title='Efficiency'),contours=dict(showlabels=True))
            )
        fig5.add_trace(go.Scatter(x=[s for s in df_comp['speed'] if s > 0], y=df_ours['torque'], mode='markers', name='Operating Points'))

        fig5.update_layout(title='Motor Efficiency Map with Operating Points', xaxis_title='Speed (km/h)', yaxis=dict(title='Torque (N)'),template='plotly_dark')

        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=out_competitor["x_age"], y=out_competitor["y_age"], mode='lines',name='Competitor'))
        fig6.add_trace(go.Scatter(x=out["x_age"], y=out["y_age"], mode='lines',name='Ours'))
        fig6.update_yaxes(range=[0, 1])
        fig6.update_layout(title='Remaining Useful Life', xaxis_title='Mileage (Miles)', yaxis=dict(title='State of Health (SoH)'),template='plotly_dark')

        st.plotly_chart(fig6, use_container_width=True)


        with col2:
            st.subheader('Vehicle performance metrics')
            
            Pmotor = np.round(out["motor_power_watt"]/1000,1)
            dPmotor = np.round((out["motor_power_watt"]-out_competitor["motor_power_watt"])/1000,1)
            Tmotor = np.round(out["motor_torque_nm"],1)
            dTmotor = np.round((out["motor_torque_nm"]-out_competitor["motor_torque_nm"]),1)

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


st.write('Copyright © 2025 Farraen. All rights reserved.')
