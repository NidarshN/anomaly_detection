import numpy as np

APP_NAME="""Efficient Data Stream Anomaly Detection"""
APP_INFO="""This project demonstrates a real-time anomaly detection system for a simulated continuous data stream. 
The data stream simulates real-world metrics such as financial transactions or system performance metrics with 
seasonal patterns, noise, and anomalies. The main objective is to detect anomalies (e.g., unusually high or low values)
in real-time using different anomaly detection models.
"""

AUTHOR_INFO="""Create by Nidarsh Nithyananda"""

NUM_DATA_POINTS_INFO="""Determines how many of the most recent data points from the stream will be displayed on the real-time plot at any given moment."""

TREND_INFO="""Represents a gradual increase or decrease in the data values over time. 
It models the long-term movement in the data, like an upward or downward slope.
"""

SEASONAL_INFO="""Extent or height of the recurring patterns or cycles within the data stream. 
It refers to the degree of variation between the peaks and troughs of the seasonal pattern.
"""

NOISE_INFO="""Represents the amount of random variability or randomness added to the data stream. 
It introduces stochastic fluctuations that make the data less predictable, simulating real-world imperfections or variations.
"""

ANOMALY_RATE_INFO="""Refers to the frequency at which anomalies (unusual or unexpected data points) occur in the simulated data stream."""

STREAM_SPEED_INFO="""Represents the delay or interval between each data point in the real-time stream. 
It determines how fast the data points are being generated and processed.
"""


MODEL_DESC_DICT={
    'Z-score': """The Z-score is a statistical measure that quantifies the number of standard deviations a data point is from the mean of the dataset. 
                    In anomaly detection, if the Z-score of a data point exceeds a predefined threshold (positive or negative), it is flagged as an anomaly.""", 
    'EWMA': """ Exponentially Weighted Moving Average (EWMA) is a time series analysis method that gives more weight to recent observations while slowly discounting older data. It is useful for detecting shifts in the mean or trends in a time series. 
                    Anomalies are flagged when the EWMA deviates significantly from expected behavior.""", 
    'Isolation Forest': """Isolation Forest is an unsupervised machine learning algorithm specifically designed for anomaly detection. 
                    It isolates anomalies by randomly selecting features and splitting the data. Anomalies are easier to isolate because they are few and different from normal data."""
}

def simulate_data_stream(size, trend=0.001, seasonal_amplitude=10, noise_level=0.5, anomaly_rate=0.05):
    """
    Simulates a continuous data stream with configurable trend, seasonality, noise, 
    and injected anomalies.

    Parameters:
    ===========
    size (int): The number of data points to generate in the stream.
    trend (float, optional): A linear trend factor applied to the data stream. 
                                Default is 0.001.
    seasonal_amplitude (int, optional): The amplitude of the seasonal component 
                                        in the data stream. Default is 10.
    noise_level (float, optional): Standard deviation of the random noise added 
                                    to the data stream. Default is 0.5.
    anomaly_rate (float, optional): Proportion of data points that are anomalies, 
                                    ranging between 0 and 0.1. Default is 0.05.

    Returns:
    ========
    (tuple):
        - list: Simulated data stream with noise, trend, and seasonality applied.
        - list: Indices of injected anomalies in the data stream.
    """
    time_series = np.arange(size)
    
    # Trend component
    trend_component = trend * time_series
    
    # Seasonal component (sinusoidal)
    seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * time_series / 100)
    
    # Noise component
    noise_component = noise_level * np.random.randn(size)
    
    # Combine components
    data_stream = trend_component + seasonal_component + noise_component
    
    # Inject anomalies
    true_anomalies = []
    for i in range(size):
        if np.random.rand() < anomaly_rate:
            data_stream[i] += np.random.uniform(10, 20)  # Random spike as anomaly
            true_anomalies.append(i)
    
    return data_stream, true_anomalies
