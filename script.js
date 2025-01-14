// script.js

// Set up the accelerometer chart
const accelCtx = document.getElementById('accelerometerChart').getContext('2d');
const accelerometerChart = new Chart(accelCtx, {
    type: 'line',
    data: {
        datasets: [
            {
                label: 'Acceleration X',
                data: [],
                borderColor: 'red',
                fill: false,
            },
            {
                label: 'Acceleration Y',
                data: [],
                borderColor: 'green',
                fill: false,
            },
            {
                label: 'Acceleration Z',
                data: [],
                borderColor: 'blue',
                fill: false,
            }
        ]
    },
    options: {
        animation: false,
        scales: {
            x: { display: false },
            y: { beginAtZero: true }
        }
    }
});

// Set up the gyroscope chart
const gyroCtx = document.getElementById('gyroscopeChart').getContext('2d');
const gyroscopeChart = new Chart(gyroCtx, {
    type: 'line',
    data: {
        datasets: [
            {
                label: 'Rotation Alpha',
                data: [],
                borderColor: 'orange',
                fill: false,
            },
            {
                label: 'Rotation Beta',
                data: [],
                borderColor: 'purple',
                fill: false,
            },
            {
                label: 'Rotation Gamma',
                data: [],
                borderColor: 'yellow',
                fill: false,
            }
        ]
    },
    options: {
        animation: false,
        scales: {
            x: { display: false },
            y: { beginAtZero: true }
        }
    }
});

// Listen for accelerometer data
if (window.DeviceMotionEvent) {
    window.addEventListener('devicemotion', (event) => {
        const acc = event.accelerationIncludingGravity;
        const time = Date.now();

        // Update accelerometer chart
        accelerometerChart.data.labels.push(time);
        accelerometerChart.data.datasets[0].data.push(acc.x);
        accelerometerChart.data.datasets[1].data.push(acc.y);
        accelerometerChart.data.datasets[2].data.push(acc.z);
        accelerometerChart.update();

        // Limit data points
        if (accelerometerChart.data.labels.length > 50) {
            accelerometerChart.data.labels.shift();
            accelerometerChart.data.datasets.forEach(dataset => dataset.data.shift());
        }
    });
}

// Listen for gyroscope data
if (window.DeviceOrientationEvent) {
    window.addEventListener('deviceorientation', (event) => {
        const time = Date.now();

        // Update gyroscope chart
        gyroscopeChart.data.labels.push(time);
        gyroscopeChart.data.datasets[0].data.push(event.alpha);
        gyroscopeChart.data.datasets[1].data.push(event.beta);
        gyroscopeChart.data.datasets[2].data.push(event.gamma);
        gyroscopeChart.update();

        // Limit data points
        if (gyroscopeChart.data.labels.length > 50) {
            gyroscopeChart.data.labels.shift();
            gyroscopeChart.data.datasets.forEach(dataset => dataset.data.shift());
        }
    });
}