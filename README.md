# Efficient Data Stream Anomaly Detection

## Project Description

This project demonstrates a real-time anomaly detection system for a simulated continuous data stream. The data stream simulates real-world metrics such as financial transactions or system performance metrics with seasonal patterns, noise, and anomalies. The main objective is to detect anomalies (e.g., unusually high or low values) in real-time using different anomaly detection models.

Click Here for [Project Live Link](https://datastream-anomalydetection-ntech.streamlit.app/)
Note: The project is deployed on Streamlit's Community Cloud (Limited to 1GB Resources), for a better performance, kindly follow the instructions [HERE](#installation-and-setup-instructions) and run it locally on your system.

## Objectives

1. **Algorithm Selection**: Implement appropriate anomaly detection algorithms that can handle concept drift and seasonal variations.
2. **Data Stream Simulation**: Generate a real-time data stream with patterns, seasonality, noise, and anomalies.
3. **Anomaly Detection**: Apply models to detect anomalies in real-time as the data streams.
4. **Optimization**: Ensure the detection algorithm is optimized for speed and efficiency.
5. **Visualization**: Provide real-time visualization of the data stream and detected anomalies.

## Project Structure

- **app.py**: The Streamlit web app that simulates and visualizes the data stream with anomaly detection.
- **models.py**: Contains implementations of multiple anomaly detection algorithms (Z-score, EWMA, Isolation Forest).
- **utils.py**: Utility functions for simulating data streams, injecting anomalies, and evaluating model performance.
  
## Algorithms Implemented

- **Z-score Detection**: Anomaly detection based on statistical deviation from the mean.
- **EWMA (Exponentially Weighted Moving Average)**: A method that weighs recent observations more heavily to detect anomalies.
- **Isolation Forest**: A tree-based algorithm that isolates outliers in a dataset.

## Evaluation Metrics

- **Precision**: The fraction of detected anomalies that are actual anomalies.
- **Recall**: The fraction of actual anomalies that were detected.
- **F1 Score**: The harmonic mean of precision and recall.

## Installation and Setup Instructions

1. Clone the repository:

   ```terminal
   git clone https://github.com/NidarshN/anomaly_detection.git
   cd anomaly_detection
   ```

2. Install the required dependencies:

   ```terminal
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```terminal
   streamlit run app.py
   ```

4. Use the sidebar options in the app to select the anomaly detection model and set parameters for the data stream (e.g., trend, noise, anomaly rate, etc.).
5. Start and stop the data stream with the provided buttons.

## How to Use

- Select a model from the sidebar.
- Adjust the data stream parameters like trend, noise level, and anomaly rate.
- Start the stream to visualize the real-time data and detected anomalies.
- Stop the stream whenever needed.
- Review the evaluation metrics (Precision, Recall, F1 Score) for the chosen model.

## Optimization Techniques

- The stream simulation and anomaly detection are optimized for real-time performance.
- Efficient algorithms are employed to maintain speed, especially with large or continuous data streams.

### License

[MIT License](./LICENSE)
