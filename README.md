# 📈 Bitcoin Price Prediction using LSTM

A Streamlit web application that predicts **Bitcoin's closing prices** using a **Long Short-Term Memory (LSTM)** neural network. This project uses historical BTC-USD data and deep learning techniques to forecast future trends.

---

## 🚀 Demo

> 📌 *To launch the app locally:*

```bash
streamlit run app.py
```

---

## 🧠 Key Features

* 📉 Fetches historical **Bitcoin price data** using `yfinance`
* 📊 Visualizes actual vs predicted prices
* 🧪 Scales data and reshapes it for LSTM input
* 🧠 Trains a **2-layer LSTM model** with Keras
* 📈 Plots predicted vs actual prices
* 🧮 Calculates **RMSE** for model performance

---

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Data Source**: [Yahoo Finance](https://finance.yahoo.com/)
* **Machine Learning**:

  * `TensorFlow` & `Keras` for LSTM
  * `sklearn` for preprocessing and metrics
* **Visualization**: `Matplotlib`

---

## 📦 Installation

1. **Clone the repo:**

   ```bash
   git clone https://github.com/its-kanii/bitcoin-price-prediction.git
   cd bitcoin-lstm-predictor
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

```
bitcoin-lstm-predictor/
│
├── app.py               # Main Streamlit app
├── requirements.txt     # Dependencies
└── README.md            # Project overview
```

---

## 📈 Model Overview

* **Input**: Past 60 time steps (days)
* **Architecture**:

  * LSTM layer (50 units, return sequences)
  * LSTM layer (50 units)
  * Dense output layer (1 unit)
* **Loss Function**: Mean Squared Error
* **Optimizer**: Adam

---

## 📊 Output Example

* **Line chart** showing actual vs. predicted prices
* **RMSE metric** to measure prediction accuracy

---

## 📚 To-Do / Future Improvements

* [ ] Add multi-feature input (e.g., Open, High, Volume)
* [ ] Extend to other cryptocurrencies
* [ ] Include model saving/loading functionality
* [ ] Deploy to the cloud (e.g., Streamlit Cloud or Hugging Face Spaces)

---

## 🙌 Acknowledgements

* [Streamlit](https://streamlit.io/)
* [Yahoo Finance API via yfinance](https://pypi.org/project/yfinance/)
* [TensorFlow](https://www.tensorflow.org/)

---

## 🧠 Author

📫 [Kanimozhi Kathirvel](https://www.linkedin.com/in/kanimozhi-kathirvel)
💻 [Kanimozhi K](https://github.com/its-kanii)


