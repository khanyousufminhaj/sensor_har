﻿# Human Activity Recognition System

## Overview

This project implements a Human Activity Recognition (HAR) system using sensor data from mobile devices. It leverages a Flask web application, Socket.IO for real-time communication, and a pre-trained TensorFlow model to predict human activities based on accelerometer and gyroscope data. The system includes a data logging interface, a real-time dashboard, and the capability to download collected sensor data.

## Functionality

*   **Data Logging:** A web interface (`get_data.html`) allows users to start and stop sensor data collection from their mobile devices. The collected data is displayed in real-time and can be downloaded as a CSV file.
*   **Real-time Prediction:** Sensor data is streamed to the server using Socket.IO. The data is preprocessed and fed into a TensorFlow model to predict the user's current activity.
*   **Dashboard:** A dedicated dashboard (`dashboard.html`) provides a real-time view of the predicted activity, activity history, and sensor data plots.
*   **Activity History:** The system maintains a history of predicted activities, which is displayed on the dashboard.
*   **Data Visualization:** Accelerometer and gyroscope data are visualized in real-time using Plotly.js on the dashboard.

## Technologies Used

*   **Flask:** A Python web framework for building the web application.
*   **Flask-SocketIO:** A Flask extension for real-time communication using WebSockets.
*   **TensorFlow:** An open-source machine learning framework for building and deploying the activity recognition model.
*   **NumPy:** A Python library for numerical computing, used for data preprocessing.
*   **Scikit-learn:** A Python library for machine learning, used for data scaling.
*   **Plotly.js:** A JavaScript library for creating interactive plots and charts.
*   **HTML, CSS, JavaScript:** For building the user interface and client-side logic.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/khanyousufminhaj/sensor_har.git
    cd sensor_har
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the pre-trained model:**

    *   Ensure the `CNN_BiLSTM.h5` model file is located in the root directory.

5.  **Run the Flask application:**

    ```bash
    flask run --debug
    ```

6.  **Access the application in your browser:**

    *   To access the application on your mobile device, you may need to use a port forwarding service like `ngrok`.  After installing ngrok, run:

        ```bash
        ngrok http 5000
        ```

    *   Use the `ngrok` provided URL to open `get_data.html` on your mobile device to start logging sensor data.
    *   Open `dashboard.html` on your PC to view the real-time dashboard.

## Code Structure

*   **`app.py`:** Contains the Flask application logic, Socket.IO event handlers, data preprocessing, and prediction functions.
*   **`templates/get_data.html`:** Defines the HTML structure for the data logging interface.
*   **`templates/dashboard.html`:** Defines the HTML structure for the real-time dashboard.
*   **`CNN_BiLSTM.h5`:** The pre-trained TensorFlow model for activity recognition.
*   **`README.md`:** This file, providing an overview of the project.

## Future Improvements

*   **Personalized Models:**
    *   Create personalized models for individual users to improve prediction accuracy. This could involve collecting user-specific data and fine-tuning the existing model or training new models from scratch.
*   **Enhanced Data Collection:**
    *   Collect more diverse and comprehensive sensor data to improve the model's ability to generalize across different activities and environments.
    *   Incorporate data from additional sensors, such as heart rate monitors or GPS, to provide a more complete picture of the user's context.
*   **Advanced Model Architectures:**
    *   Experiment with different model architectures, such as Transformers or more complex recurrent neural networks, to improve activity recognition performance.
    *   Explore techniques like transfer learning and domain adaptation to leverage existing datasets and improve the model's robustness.
*   **Improved User Interface:**
    *   Enhance the user interface of the data logging interface and dashboard to provide a more intuitive and user-friendly experience.
    *   Implement responsive design principles to ensure the application works seamlessly across different devices and screen sizes.
*   **Mobile Application:**
    *   Develop a native mobile application for data collection and activity recognition, providing a more integrated and seamless user experience.
    *   Utilize device-specific APIs for sensor data access and background processing.
*   **Expand Activity Recognition:**
    *   Increase the number of activities the model can recognize, including more complex and nuanced actions.
    *   Implement hierarchical activity recognition to identify both high-level activities (e.g., exercising) and more specific actions (e.g., running, weightlifting).
*   **Anomaly Detection:**
    *   Incorporate anomaly detection techniques to identify unusual or unexpected activity patterns.
    *   Use anomaly detection to provide alerts or notifications to the user in case of potential health issues or emergencies.

These improvements will enhance the accuracy, usability, and functionality of the HAR system, making it a more valuable tool for health monitoring, fitness tracking, and other applications.
