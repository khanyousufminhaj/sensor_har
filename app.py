import os
import logging
import time
import threading
import numpy as np
import tensorflow as tf
from flask import Flask, render_template
from flask_socketio import SocketIO
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
socketio = SocketIO(app)

# Load saved model and scaler
model = tf.keras.models.load_model('model.h5')
scaler = joblib.load('scaler.pkl')

# Constants
WINDOW_SIZE = 128
TIMEOUT = 3  # seconds
activities = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'
}

# Global variables
sensor_buffer = []
last_received_time = time.time()
buffer_lock = threading.Lock()

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    app.logger.info("Rendering 'get_data.html'")
    return render_template('get_data.html')

@socketio.on('sensor_data')
def handle_sensor_data(data):
    global sensor_buffer, last_received_time

    try:
        app.logger.debug(f"Received raw sensor data: {data}")
        
        # Validate and extract data
        reading = [
            float(data.get('acc_x', 0)), float(data.get('acc_y', 0)), float(data.get('acc_z', 0)),
            float(data.get('gyro_x', 0)), float(data.get('gyro_y', 0)), float(data.get('gyro_z', 0))
        ]
        
        with buffer_lock:
            # Add to buffer
            sensor_buffer.append(reading)
            if len(sensor_buffer) > WINDOW_SIZE:
                sensor_buffer = sensor_buffer[-WINDOW_SIZE:]
        
        last_received_time = time.time()

        # Check buffer size for prediction
        if len(sensor_buffer) == WINDOW_SIZE:
            make_prediction()

    except Exception as e:
        app.logger.error(f"Error processing sensor data: {e}", exc_info=True)

def make_prediction():
    global sensor_buffer
    try:
        with buffer_lock:
            X = np.array(sensor_buffer)

        app.logger.debug(f"Buffer before scaling: {X}")
        X = scaler.transform(X)
        X = X.reshape(1, WINDOW_SIZE, 6)

        pred = model.predict(X, verbose=0)[0]
        activity = activities[np.argmax(pred)]
        app.logger.info(f"Predicted activity: {activity}")

        socketio.emit('prediction', {'activity': activity})
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}", exc_info=True)

def check_for_timeout():
    global last_received_time
    while True:
        try:
            time_since_last_data = time.time() - last_received_time
            if time_since_last_data > TIMEOUT:
                app.logger.warning(f"No sensor data for {TIMEOUT} seconds. Sending NO_DATA signal.")
                socketio.emit('prediction', {'activity': 'NO_DATA'})
                last_received_time = time.time()  # Avoid spamming
            time.sleep(1)
        except Exception as e:
            app.logger.error(f"Error in timeout check: {e}", exc_info=True)

if __name__ == '__main__':
    app.logger.info("Starting Flask app")
    socketio.start_background_task(check_for_timeout)
    socketio.run(app, debug=True)
