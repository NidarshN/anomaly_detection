import streamlit as st
import time
import matplotlib.pyplot as plt
from models import z_score_detection, ewma_detection, isolation_forest_detection
from utils import APP_INFO, AUTHOR_INFO, MODEL_DESC_DICT, NUM_DATA_POINTS_INFO, STREAM_SPEED_INFO, ANOMALY_RATE_INFO, NOISE_INFO, SEASONAL_INFO, TREND_INFO, simulate_data_stream
from sklearn.metrics import precision_score, recall_score, f1_score


st.set_page_config(
    page_title="Real Time Anomaly Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': APP_INFO + "\n" + AUTHOR_INFO
    }
)

# Sidebar: Model Selection
with st.sidebar:
    st.title('Anomaly Detection Models')
    model_option = st.selectbox(
        'Choose a Model',
        MODEL_DESC_DICT.keys()
    )

    # Sidebar: Streaming Data Parameters
    st.title('Stream Simulation Settings')
    data_size = st.slider(
        'Number of data points to show at a time', 50, 500, 100, help=NUM_DATA_POINTS_INFO)
    trend = st.slider('Trend factor', 0.0, 0.01, 0.001, help=TREND_INFO)
    seasonal_amplitude = st.slider(
        'Seasonal amplitude', 0, 20, 10, help=SEASONAL_INFO)
    noise_level = st.slider('Noise level', 0.0, 1.0, 0.5, help=NOISE_INFO)
    anomaly_rate = st.slider(
        'Anomaly rate', 0.0, 0.1, 0.05, help=ANOMALY_RATE_INFO)
    stream_speed = st.slider(
        'Stream speed (seconds per data point)', 0.01, 1.0, 0.1, help=STREAM_SPEED_INFO)

    left_column, right_column = st.columns(2)
    start_stream = left_column.button("Start Stream")
    stop_stream = right_column.button("Stop Stream")


# Global flag to stop the stream
run_stream = True

# Initialize placeholders for real-time plotting and anomaly detection
st.markdown("# Real Time Anomaly Detection")
st.write(APP_INFO)
st.markdown(f"### Model: {model_option}", help=MODEL_DESC_DICT[model_option])
if not start_stream:
    st.write(":red[Click on start stream to initiate the data stream!]")
plot_col, eval_col = st.columns([2, 2])
plot_placeholder = plot_col.empty()
evaluation_placeholder = eval_col.empty()


def evaluate_model(detected_anomalies, true_anomalies, data_length, tolerance=5):
    """Evaluates the performance of an anomaly detection model.

    Parameters:
    ===========
    detected_anomalies (list): Indices of anomalies detected by the model.
    true_anomalies (list): Indices of actual anomalies in the data stream.
    data_length (int): Length of the data stream being evaluated.
    tolerance (int, optional): The allowed tolerance (in number of data points) 
                                to account for small shifts in detected anomalies. 
                                Default is 5.

    Returns:
    ========
    (tuple):  Precision, Recall, and F1 score of the model's anomaly detection.
    """
    y_true = [0] * data_length
    for anomaly in true_anomalies:
        if anomaly < data_length:
            y_true[anomaly] = 1

    y_pred = [0] * data_length
    for anomaly in detected_anomalies:
        if anomaly < data_length:
            y_pred[anomaly] = 1

    y_pred_adj = [0] * data_length
    for pred_anomaly in detected_anomalies:
        for i in range(max(0, pred_anomaly - tolerance), min(data_length, pred_anomaly + tolerance + 1)):
            if y_true[i] == 1:
                y_pred_adj[i] = 1
                break

    precision = precision_score(y_true, y_pred_adj, zero_division=0)
    recall = recall_score(y_true, y_pred_adj, zero_division=0)
    f1 = f1_score(y_true, y_pred_adj, zero_division=0)

    return precision, recall, f1

def detect_model_anomalies(model_option, data_stream):
    """
    Detects anomalies in the data stream based on the selected model.

    Parameters:
    ===========
    model_option (str): The name of the anomaly detection model to use ('Z-score', 
                        'EWMA', or 'Isolation Forest').
    data_stream (list): A list of numeric data points representing the continuous data stream.

    Returns:
    ========
    (list): Detected anomalies with their indices.
    """
    if model_option == 'Z-score':
        return z_score_detection(data_stream)
    elif model_option == 'EWMA':
        return ewma_detection(data_stream)
    else:
        return isolation_forest_detection(data_stream)

# Function to run the real-time data stream


def run_real_time_stream(data_size,
                            trend,
                            seasonal_amplitude,
                            noise_level,
                            anomaly_rate,
                            stream_speed,
                            model_option):
    """
    Simulates and visualizes a real-time data stream and applies anomaly detection.

    Parameters:
    ===========
    data_size (int): Number of data points to display at any time in the plot.
    trend (float): Factor for adding a linear trend to the data stream.
    seasonal_amplitude (int): Amplitude of the seasonal component in the data stream.
    noise_level (float): Standard deviation of noise added to the data stream.
    anomaly_rate (float): Proportion of anomalies injected into the data stream.
    stream_speed (float): Time delay (in seconds) between each data point to mimic real-time streaming.
    model_option (str): The selected anomaly detection model.

    Returns:
    ========
    None
    """
    global run_stream
    data_stream = []
    true_anomalies = []
    detected_anomalies = []

    # Dark theme setup
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')

    # Continuously stream simulated data
    while run_stream:
        new_data, new_true_anomalies = simulate_data_stream(
            1, trend, seasonal_amplitude, noise_level, anomaly_rate)
        data_stream.extend(new_data)
        true_anomalies.extend(new_true_anomalies)

        if len(data_stream) > data_size:
            data_stream = data_stream[-data_size:]
            true_anomalies = [a for a in true_anomalies if a >= len(data_stream) - data_size]

        # Detect anomalies
        anomalies, _ = detect_model_anomalies(model_option, data_stream)
        detected_anomalies = [index for index, _ in anomalies]

        # Update real-time plot
        ax.clear()
        ax.grid(alpha=0.3)
        ax.plot(data_stream, color='white', label='Data Stream')

        # Highlight the sliding window
        ax.axvspan(0, data_size, color='grey', alpha=0.5, label='Sliding Window')

        # Highlight anomalies
        anomaly_values = [data_stream[idx]
                            for idx in detected_anomalies if idx < len(data_stream)]
        ax.scatter(detected_anomalies, anomaly_values, color='red', label='Anomalies')
        ax.set_xlabel('Time (Data Point Index)', color='white')
        ax.set_ylabel('Data Stream Value', color='white')
        ax.legend(loc='upper right')
        plot_placeholder.pyplot(fig)

        # Update model evaluation
        precision, recall, f1 = evaluate_model(
            detected_anomalies, true_anomalies, len(data_stream))
        container = evaluation_placeholder.container(border=True)
        container.text(f"Precision: {precision:.2f}")
        container.text(f"Recall: {recall:.2f}")
        container.text(f"F1: {f1:.2f}")

        plt.close('all')

        if not run_stream:
            break

        time.sleep(stream_speed)

# Stop the stream by changing the global flag
if stop_stream:
    run_stream = False


# Run the stream if the button is clicked
if start_stream and run_stream:
    run_real_time_stream(data_size, trend, seasonal_amplitude,
                            noise_level, anomaly_rate, stream_speed, model_option)
