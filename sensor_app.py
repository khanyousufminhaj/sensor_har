import sensor_app as st
import pandas as pd
import time
from datetime import datetime
import random  # For simulation purposes

# Page config
st.set_page_config(page_title="Sensor Data Logger", layout="wide")
st.title("Sensor Data Logger")

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = False
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = []

# Create columns for sensor displays
acc_col, gyro_col = st.columns(2)

# Sensor displays
with acc_col:
    st.subheader("Accelerometer")
    acc_x = st.metric("X", "-")
    acc_y = st.metric("Y", "-")
    acc_z = st.metric("Z", "-")

with gyro_col:
    st.subheader("Gyroscope")
    gyro_x = st.metric("X", "-")
    gyro_y = st.metric("Y", "-")
    gyro_z = st.metric("Z", "-")

# Status display
status = st.empty()

# Control buttons
col1, col2, col3 = st.columns(3)
start_button = col1.button("Start Sensors")
stop_button = col2.button("Stop Sensors")
download_button = col3.button("Download Data")

def simulate_sensor_reading():
    """Simulate sensor readings (replace with actual sensor data in production)"""
    return {
        'timestamp': datetime.now(),
        'acc_x': round(random.uniform(-1, 1), 3),
        'acc_y': round(random.uniform(-1, 1), 3),
        'acc_z': round(random.uniform(-1, 1), 3),
        'gyro_x': round(random.uniform(-1, 1), 3),
        'gyro_y': round(random.uniform(-1, 1), 3),
        'gyro_z': round(random.uniform(-1, 1), 3)
    }

# Handle button actions
if start_button:
    st.session_state.running = True
    status.info("Sensors started")
    st.session_state.sensor_data = []

if stop_button:
    st.session_state.running = False
    status.info("Sensors stopped")

if download_button:
    if len(st.session_state.sensor_data) > 0:
        df = pd.DataFrame(st.session_state.sensor_data)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="sensor_data.csv",
            mime="text/csv"
        )
    else:
        status.warning("No data to download")

# Main loop for updating sensor data
if st.session_state.running:
    try:
        # Simulate sensor reading (replace with actual sensor code)
        reading = simulate_sensor_reading()
        
        # Update metrics
        acc_x.metric("X", reading['acc_x'])
        acc_y.metric("Y", reading['acc_y'])
        acc_z.metric("Z", reading['acc_z'])
        gyro_x.metric("X", reading['gyro_x'])
        gyro_y.metric("Y", reading['gyro_y'])
        gyro_z.metric("Z", reading['gyro_z'])
        
        # Store data
        st.session_state.sensor_data.append(reading)
        
        # Add a small delay to prevent excessive updates
        time.sleep(0.1)
        st.experimental_rerun()
        
    except Exception as e:
        status.error(f"Error reading sensors: {str(e)}")
        st.session_state.running = False