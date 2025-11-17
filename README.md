# Cryptocurrency Price Forecasting with Deep Learning

This repository contains a full deep learning pipeline for predicting short horizon Bitcoin price movements from historical OHLCV data. The project combines data engineering, feature extraction, denoising, sequence modeling, and backtesting in order to evaluate whether learned signals have trading relevance.

The work was originally developed as a course project and is written with a portfolio focus in mind. It is structured to make it easy to see the skills demonstrated in time series modeling, PyTorch, and quantitative research.

> Contributions, improvements, and extensions are welcome.

---

## Project Overview

The project implements a full end-to-end workflow for crypto price forecasting:

* Retrieves 10+ years of BTCUSDT 30m candles from the Binance API
* Builds technical indicators (Bollinger Bands, MACD, moving averages)
* Applies denoising techniques such as Kalman filtering
* Constructs supervised learning datasets with strict temporal separation
* Trains GRU/LSTM models to predict short-term log returns
* Evaluates results with statistical metrics and simple PnL simulations

The goal is to provide a clear, reproducible framework for experimentation in financial time-series prediction.

---

## Skills Demonstrated

* Data collection from the Binance API using python-binance with safe pagination
* Robust data cleaning and alignment using pandas
* Technical indicator generation with numpy and ta
* Signal smoothing and noise reduction using Kalman filtering (filterpy)
* Deep learning model training in PyTorch with custom datasets
* GRU/LSTM sequence modeling for time-series forecasting
* Backtesting and PnL simulation with vectorized numpy pipelines
* Visual analytics and diagnostic plotting using matplotlib and seaborn

---

## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Retrieve data

```
python Data\ Processing/01_Retrieve_API_Data.py
```

### 3. Preprocess data

```
python Data\ Processing/02_Data_Preprocessing.py
```

### 4. Train a model

```
cd GRU
python GRU.py
```

### 5. Run the PnL evaluation

```
python PnL.py
```

---

## Disclaimer

This project is strictly for research and educational purposes. It is not trading advice nor a trading bot.
