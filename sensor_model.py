from sklearn.model_selection import KFold, train_test_split
import os
from datetime import datetime
import time
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
import joblib

class SensorModel:
    def __init__(self, window_size=128, architecture='LSTM'):
        # Model parameters
        self.window_size = window_size
        self.n_features = 6  # acc(x,y,z) + gyro(x,y,z)
        self.n_classes = 6
        self.architecture = architecture
        self.scaler = StandardScaler()

        # Metrics tracking
        self.metrics = ['accuracy', 'loss', 'val_accuracy', 'val_loss']

        # GPU setup
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print('GPUs Found')
        else:
            print('No gpus found',tf.config.experimental.list_physical_devices())

    def load_data(self, base_path):
        """Load and preprocess sensor data"""
        # Load accelerometer data
        acc_x = np.loadtxt(f"{base_path}/Inertial Signals/body_acc_x_train.txt")
        acc_y = np.loadtxt(f"{base_path}/Inertial Signals/body_acc_y_train.txt")
        acc_z = np.loadtxt(f"{base_path}/Inertial Signals/body_acc_z_train.txt")

        # Load gyroscope data
        gyro_x = np.loadtxt(f"{base_path}/Inertial Signals/body_gyro_x_train.txt")
        gyro_y = np.loadtxt(f"{base_path}/Inertial Signals/body_gyro_y_train.txt")
        gyro_z = np.loadtxt(f"{base_path}/Inertial Signals/body_gyro_z_train.txt")

        # Load labels
        y = np.loadtxt(f"{base_path}/y_train.txt")

        # Combine sensor data
        X = np.dstack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        
        # Normalize features
        X = self.scaler.fit_transform(X.reshape(-1, self.n_features))
        X = X.reshape(-1, self.window_size, self.n_features)

        return X, y

    def create_model(self):
        """Create model based on selected architecture"""
        if self.architecture == 'LSTM':
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, input_shape=(self.window_size, self.n_features),
                                   return_sequences=True),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(self.n_classes, activation='softmax')
            ])
        elif self.architecture == 'CNN_LSTM':
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(64, 3, activation='relu',
                                     input_shape=(self.window_size, self.n_features)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(128, 3, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(self.n_classes, activation='softmax')
            ])
        elif self.architecture == 'CNN_BiLSTM':
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(64, 3, activation='relu',
                                     input_shape=(self.window_size, self.n_features)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(128, 3, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Bidirectional(layers.LSTM(64)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(self.n_classes, activation='softmax')
            ])
        elif self.architecture == 'BiLSTM':
            model = tf.keras.Sequential([
                layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(self.window_size, self.n_features)),
                layers.BatchNormalization(),
                layers.Bidirectional(layers.LSTM(32)),
                layers.BatchNormalization(),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.n_classes, activation='softmax')
            ])
        elif self.architecture == 'GRU':
            model = tf.keras.Sequential([
                layers.GRU(64, input_shape=(self.window_size, self.n_features), return_sequences=True),
                layers.BatchNormalization(),
                layers.GRU(32),
                layers.BatchNormalization(),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.n_classes, activation='softmax')
            ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    def train(self, X, y, batch_size=32, epochs=50, validation_data=None):
        """Train model with optional validation data"""
        model = self.create_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3
            )
        ]

        # Configure training parameters
        train_params = {
            'x': X,
            'y': y-1,  # Adjust labels to start from 0
            'epochs': epochs,
            'batch_size': batch_size,
            'callbacks': callbacks,
            'verbose': 1
        }

        # Add validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            train_params['validation_data'] = (X_val, y_val-1)
        else:
            train_params['validation_split'] = 0.2

        history = model.fit(**train_params)
        self.model = model
        return history

    def train_with_cv(self, X, y, batch_size=32, epochs=50, k_folds=5):
        """Train with k-fold cross validation"""
        histories = {}
        # Make sure the number of samples is correct
        n_samples = len(X)
        
        # Create KFold object with correct parameters
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Convert y to 0-based indexing before splitting
        y = y - 1  # Subtract 1 from all labels at once
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold+1}/{k_folds}")
            print(f'train_idx: {train_idx}')
            print(f'val_idx: {val_idx}')
            
            # Split the data using the indices
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]  # No need to subtract 1 here anymore
            
            model = self.create_model()
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=5),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
                ]
            )
            
            histories[f"fold_{fold+1}"] = history.history
        
        self.plot_training_history(histories)
        return histories

    def plot_training_history(self, histories):
        """Plot training metrics"""
        for metric in self.metrics:
            plt.figure(figsize=(12, 6))
            for fold, history in histories.items():
                plt.plot(history[metric], label=f'{fold}')

            plt.title(f'{metric.capitalize()} Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.architecture}_{metric}.png')
            plt.show()
            plt.close()

    def save_model(self, save_dir):
        """
        Save the trained model and scaler to disk
        
        Parameters:
        -----------
        save_dir : str
            Directory where the model and scaler should be saved
        """
        if not hasattr(self, 'model'):
            raise ValueError("No trained model found. Please train the model first.")

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(save_dir, 'sensor_model.h5')
        self.model.save(model_path)
        
        # Save the scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Save model configuration
        config = {
            'window_size': self.window_size,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'architecture': self.architecture
        }
        config_path = os.path.join(save_dir, 'config.pkl')
        joblib.dump(config, config_path)
        
        print(f"Model saved to {save_dir}")

    @classmethod
    def load_saved_model(cls, model_dir):
        """
        Load a saved model from disk
        
        Parameters:
        -----------
        model_dir : str
            Directory containing the saved model files
            
        Returns:
        --------
        SensorModel
            Loaded model instance
        """
        # Load configuration
        config_path = os.path.join(model_dir, 'config.pkl')
        config = joblib.load(config_path)
        
        # Create model instance
        instance = cls(
            window_size=config['window_size'],
            architecture=config['architecture']
        )
        
        # Load model
        model_path = os.path.join(model_dir, 'sensor_model.h5')
        instance.model = load_model(model_path)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        instance.scaler = joblib.load(scaler_path)
        
        return instance

    def preprocess_raw_data(self, raw_data, sampling_rate=50):
        """
        Preprocess raw sensor data for inference
        
        Parameters:
        -----------
        raw_data : dict
            Dictionary containing raw sensor data with keys:
            'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'
            Each key should contain a numpy array of sensor readings
        sampling_rate : int
            Sampling rate of the sensor data in Hz
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed data ready for inference
        """
        # Check if all required sensors are present
        required_sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        if not all(sensor in raw_data for sensor in required_sensors):
            raise ValueError(f"Raw data must contain all sensors: {required_sensors}")
        
        # Stack all sensor data
        X = np.dstack([raw_data[sensor] for sensor in required_sensors])
        
        # Create windows with 50% overlap
        stride = self.window_size // 2
        n_samples = ((X.shape[0] - self.window_size) // stride) + 1
        windows = np.zeros((n_samples, self.window_size, self.n_features))
        
        for i in range(n_samples):
            start_idx = i * stride
            windows[i] = X[start_idx:start_idx + self.window_size]
        
        # Reshape for scaling
        X_scaled = self.scaler.transform(windows.reshape(-1, self.n_features))
        X_scaled = X_scaled.reshape(-1, self.window_size, self.n_features)
        
        return X_scaled

    def predict_activity(self, raw_data, sampling_rate=50):
        """
        Predict activity from raw sensor data
        
        Parameters:
        -----------
        raw_data : dict
            Dictionary containing raw sensor data with keys:
            'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'
            Each key should contain a numpy array of sensor readings
        sampling_rate : int
            Sampling rate of the sensor data in Hz
            
        Returns:
        --------
        tuple
            (predictions, probabilities)
            predictions: array of predicted activity labels
            probabilities: array of prediction probabilities for each class
        """
        if not hasattr(self, 'model'):
            raise ValueError("No trained model found. Please train or load a model first.")
        
        # Preprocess the raw data
        X_processed = self.preprocess_raw_data(raw_data, sampling_rate)
        
        # Get predictions
        probabilities = self.model.predict(X_processed)
        predictions = np.argmax(probabilities, axis=1) + 1  # Add 1 to match original labels
        
        return predictions, probabilities