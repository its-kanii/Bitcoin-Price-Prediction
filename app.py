import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Streamlit Page Configuration
st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")
st.title("📈 Bitcoin Price Prediction using LSTM")
st.markdown("Predict Bitcoin closing prices using historical data and an LSTM neural network.")

# Load Bitcoin Data
@st.cache_data
def load_data():
    df = yf.download('BTC-USD', start='2017-01-01', end='2024-12-31')
    return df[['Close']].dropna()

df = load_data()

# Display historical data
st.subheader("Bitcoin Closing Price")
st.line_chart(df['Close'])

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Dataset creation function
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Create X, y datasets
time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape for LSTM input: [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train and Predict
if st.button("🚀 Train LSTM Model"):
    with st.spinner("Training LSTM Model..."):
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Predict
    y_pred = model.predict(X_test)
    y_predicted = scaler.inverse_transform(y_pred)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plotting
    st.subheader("📊 Actual vs Predicted Closing Price")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_actual, label='Actual Price')
    ax.plot(y_predicted, label='Predicted Price')
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # RMSE Evaluation
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    st.metric("📉 RMSE (Root Mean Squared Error)", f"${rmse:.2f}")



