# Cryptocurrency Price Forecasting with Deep Learning

This repository contains a full deep learning pipeline for predicting short horizon Bitcoin price movements from historical OHLCV data. The project combines data engineering, feature extraction, denoising, sequence modeling, and backtesting in order to evaluate whether learned signals have trading relevance.

The work was originally developed as a course project and is written with a portfolio focus in mind. It is structured to make it easy to see the skills demonstrated in time series modeling, PyTorch, and quantitative research.

> **Disclaimer:** This project is strictly for research and educational purposes. It is not trading advice.

---

## Project Goals

The project aims to answer three main questions:

1. Can we predict short horizon log returns on BTCUSDT from historical 30 minute OHLCV data and technical indicators.  
2. How much does careful preprocessing and denoising help compared to naive baselines like predicting zero or using the last observed return.  
3. Do improvements in predictive metrics translate into more stable PnL when simulated as a simple long or short strategy.

To support this, the repository includes a complete pipeline:

- Data retrieval from the Binance API  
- Feature engineering and time series preprocessing  
- Multiple denoising strategies for noisy financial signals  
- Supervised sequence modeling with PyTorch (MLP and GRU based architectures)  
- Evaluation against baselines and simple PnL simulation

---

## Skills Demonstrated

This project is designed so that reviewers can quickly see what is being exercised.

### Time Series and Feature Engineering

- Working with long historical OHLCV datasets on 30 minute bars.  
- Log return construction for open, high, low, and close to stabilize training.  
- Technical indicator engineering with `pandas_ta`:
  - Bollinger band width and band position  
  - MACD and moving averages  
- Block-wise splitting and sequencing to prevent temporal leakage.  
- Careful normalization done with respect to the training set only, shared across train, validation, and test splits.

### Signal Denoising and Data Quality

- Implementation and comparison of multiple denoising strategies for financial time series:
  - Wavelet based denoising (DWT and SWT variants)  
  - Causal smoothing with exponential moving averages  
  - Kalman filter based smoothing for regime dependent volatility  
  - Butterworth low pass filtering on selected features  
- Explicit handling of features that should not be denoised, for example volume and number of trades.  
- Attention to causality and data leakage when designing transforms on time series.

### Deep Learning with PyTorch

- Implementation of several neural network architectures for time series regression:
  - Flattened MLP predicting future log return from fixed length sequences  
  - GRU based models that operate directly on sequences  
  - An `EncoderGRU` that first projects per timestep features through a feed forward encoder before temporal modeling  
  - LSTM variant for longer term sequence modeling  
- Custom training loops with:
  - Reproducible seeding and deterministic settings  
  - Gradient clipping to keep training stable  
  - OneCycleLR and ReduceLROnPlateau learning rate scheduling  
  - Early stopping hooks and checkpointing of the best validation model  
- Rich metric logging:
  - MSE, MAE, RMSE on log returns  
  - Directional accuracy based on the sign of the predicted return  
  - R squared style diagnostics and Pearson correlation  
  - Baselines for comparison (persistence and zero predicted return)

### Evaluation and PnL Backtesting

- Conversion from predicted log returns into simulated trades.  
- Simple long or short strategy based on predicted sign, with configurable fee and slippage in basis points.  
- Equity curve generation and export to CSV for further analysis.  
- Plots for:
  - Predicted vs true returns scatter  
  - Error histograms  
  - Tiny PnL curves and training or validation loss curves  

### Software Engineering and Reproducibility

- Modular code layout separating data processing, modeling, and experimentation.  
- Clear configuration blocks and hyperparameter sections at the top of scripts.  
- Consistent saving of:
  - Normalization statistics  
  - Preprocessed data as compressed NPZ files  
  - Model checkpoints  
  - Diagnostic plots and CSV artifacts  
- Use of type hints, helper utilities for batching and reshaping, and small abstractions for evaluation.