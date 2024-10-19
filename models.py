import numpy as np
from sklearn.ensemble import IsolationForest

def z_score_detection(data_stream, threshold=3):
    """
    Detects anomalies in a data stream using the Z-score method.

    Parameters:
    ===========
    data_stream (list): The input data stream to analyze for anomalies.
    threshold (float, optional): The Z-score threshold above which points are considered anomalies. 
                                    Default is 3.

    Returns:
    ========
    (list): Detected anomalies with their indices in the data stream.
    """
    if len(data_stream) < 2:
        return [], None  # Not enough data to compute mean and std
    
    mean = np.mean(data_stream)
    std = np.std(data_stream)
    
    # Check if std is greater than zero to avoid division by zero
    if std <= 1e-10:  # Added a small tolerance to account for floating-point precision
        return [], mean  # Return empty anomalies if no variation

    anomalies = [(i, x) for i, x in enumerate(data_stream) if abs((x - mean) / std) > threshold]
    return anomalies, mean

def ewma_detection(data_stream, alpha=0.3, threshold=2.0):
    """
    Detects anomalies in a data stream using the Exponentially Weighted Moving Average (EWMA) method.

    Parameters:
    ===========
    data_stream (list): The input data stream to analyze for anomalies.
    span (int, optional): The span parameter for the EWMA. Default is 10.

    Returns:
    ========
    (list): Detected anomalies with their indices in the data stream.
    """
    if len(data_stream) < 2:
        return [], None  # Not enough data to compute EWMA
    
    ewma = np.zeros_like(data_stream)
    ewma[0] = data_stream[0]
    anomalies = []
    for i in range(1, len(data_stream)):
        ewma[i] = alpha * data_stream[i] + (1 - alpha) * ewma[i - 1]
        if abs(data_stream[i] - ewma[i]) > threshold:
            anomalies.append((i, data_stream[i]))
    return anomalies, ewma[-1]

def isolation_forest_detection(data_stream):
    """
    Detects anomalies in a data stream using the Isolation Forest method.

    Parameters:
    ===========
    data_stream (list): The input data stream to analyze for anomalies.
    contamination (float, optional): The proportion of observations to be considered as anomalies. 
                                        Default is 0.05.

    Returns:
    (list): Detected anomalies with their indices in the data stream.
    """
    if len(data_stream) < 2:
        return [], None  # Not enough data to fit the model
    
    clf = IsolationForest(contamination=0.05)
    data_stream_reshaped = np.array(data_stream).reshape(-1, 1)
    clf.fit(data_stream_reshaped)
    labels = clf.predict(data_stream_reshaped)
    anomalies = [(i, data_stream[i]) for i, label in enumerate(labels) if label == -1]
    return anomalies, None
