import os
import logging
import time
import threading
import numpy as np
import tensorflow as tf
from flask import Flask, render_template
from flask_socketio import SocketIO
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
socketio = SocketIO(app)

# Constants
WINDOW_SIZE = 128
TIMEOUT = 3  # seconds
MODEL_PATH = 'CNN_BiLSTM.h5'
PREDICTION_INTERVAL = 0.25  # seconds

# Activity labels mapping
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
last_prediction_time = 0
buffer_lock = threading.Lock()
scaler = StandardScaler()  # Create a new StandardScaler instance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """Load the saved model"""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Load model at startup
model = load_model()

def preprocess_data(sensor_data):
    """Preprocess sensor data for prediction"""
    try:
        # Stack the data in the correct order
        data = np.column_stack([
            sensor_data['acc_x'], sensor_data['acc_y'], sensor_data['acc_z'],
            sensor_data['gyro_x'], sensor_data['gyro_y'], sensor_data['gyro_z']
        ])
        
        # Scale the data (fit_transform since we're using a new scaler each time)
        data_scaled = scaler.fit_transform(data)
        
        # Reshape for model input (add batch dimension)
        return data_scaled.reshape(1, WINDOW_SIZE, 6)
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None

@app.route('/')
def index():
    logger.info("Rendering 'get_data.html'")
    return render_template('get_data.html')

@socketio.on('sensor_data')
def handle_sensor_data(data):
    global sensor_buffer, last_received_time

    try:
        logger.debug(f"Received raw sensor data: {data}")
        
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
        logger.error(f"Error processing sensor data: {e}", exc_info=True)

def make_prediction():
    global sensor_buffer, last_prediction_time
    
    if model is None:
        logger.error("Model not loaded")
        return

    try:
        current_time = time.time()
        if current_time - last_prediction_time < PREDICTION_INTERVAL:
            return  # Skip prediction if interval has not passed

        with buffer_lock:
            # Create a dictionary with the sensor data
            sensor_data = {
                'acc_x': np.array([x[0] for x in sensor_buffer]),
                'acc_y': np.array([x[1] for x in sensor_buffer]),
                'acc_z': np.array([x[2] for x in sensor_buffer]),
                'gyro_x': np.array([x[3] for x in sensor_buffer]),
                'gyro_y': np.array([x[4] for x in sensor_buffer]),
                'gyro_z': np.array([x[5] for x in sensor_buffer])
            }

        # Preprocess the data
        X = preprocess_data(sensor_data)
        if X is None:
            return

        # Make prediction
        pred = model.predict(X, verbose=0)[0]
        activity = activities[np.argmax(pred)]
        
        logger.info(f"Predicted activity: {activity} in {time.time()-current_time:.2f} seconds")
        
        # Emit prediction
        socketio.emit('prediction', {'activity': activity})
        
        last_prediction_time = current_time

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)

def check_for_timeout():
    global last_received_time
    while True:
        try:
            time_since_last_data = time.time() - last_received_time
            if time_since_last_data > TIMEOUT:
                logger.warning(f"No sensor data for {TIMEOUT} seconds. Sending NO_DATA signal.")
                socketio.emit('prediction', {'activity': 'NO_DATA'})
                last_received_time = time.time()  # Avoid spamming
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in timeout check: {e}", exc_info=True)

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    if model is None:
        logger.error("Model not loaded")
        socketio.emit('error', {'message': 'Model not properly initialized'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

if __name__ == '__main__':
    if model is None:
        logger.error("Failed to load model. Please check the model file.")
    else:
        logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
    
    logger.info("Starting Flask app")
    socketio.start_background_task(check_for_timeout)
    socketio.run(app, debug=True)