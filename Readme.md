# Stock Price Forecasting System

A comprehensive stock forecasting system that integrates data collection, preprocessing, feature engineering, multiple predictive models (including deep learning and statistical approaches), ensemble forecasting, and interactive visualization—all accessible via a command-line chatbot interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Forecasting Pipeline](#running-the-forecasting-pipeline)
  - [Chatbot Interface and Commands](#chatbot-interface-and-commands)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project is an **end-to-end stock forecasting system** that:

- **Downloads historical market data (OHLCV)** for multiple stock tickers using `yfinance`.
- **Preprocesses the data** to fill missing values, enforce business-day frequency, and interpolate gaps.
- **Extracts both basic and advanced features**, including technical indicators such as:
  - Moving Averages
  - RSI
  - MACD
  - OBV
  - Stochastic Oscillator
  - Bollinger Bands
  - Volatility measures
- **Trains multiple forecasting models**:
  - **Deep Learning:** LSTM and GRU neural networks to capture sequential dependencies.
  - **Statistical:** ARIMA for trend forecasting, Prophet for time series decomposition, and GARCH for volatility.
  - **Machine Learning:** XGBoost using lagged features.
- **Implements an ensemble mechanism** that weights individual model predictions based on their validation performance, including a stacking meta-model (via linear regression) for the final forecast.
- **Provides extensive visualization options** with support for multiple libraries (Matplotlib, Seaborn, Plotly, Bokeh, Altair, Plotnine, Holoviews, and Dash).
- **Features an interactive command-line chatbot** that allows users to:
  - Analyze stocks (technical indicators, evaluation metrics, ensemble forecasts)
  - Compare stock performances
  - Manage a portfolio
  - Select the preferred visualization library interactively

---

## Features

### Data Collection & Preprocessing
- **Historical Data:** Downloads over 7 years of data.
- **Missing Values:** Handled using forward-fill and linear interpolation.
- **Reindexing:** Ensures business-day frequency.

### Feature Engineering
- **Basic Indicators:** Returns, MA20/MA50, momentum, volume averages.
- **Technical Indicators:** RSI, MACD, OBV, Stochastic Oscillator.
- **Advanced Features:** Log-price transforms, multi-period volatility, moving average crossovers, Bollinger Bands, volume ratios.

### Model Training & Forecasting
- **Deep Learning:** 
  - LSTM/GRU models with sequence-based training.
  - Dropout regularization and early stopping.
- **Statistical Methods:**
  - ARIMA for non-seasonal trends.
  - Prophet for time series decomposition.
  - GARCH for volatility prediction.
- **Machine Learning:** 
  - XGBoost with lagged features and recursive forecasting.
- **Ensemble Strategy:**
  - Inverse error weighting.
  - Meta-model stacking via linear regression.

### Evaluation & Visualization
- **Plotting Capabilities:** 
  - Historical data, forecast comparisons, and error distributions.
- **Visualization Libraries:** 
  - Multiple libraries available for interactive or publication-quality charts.

### Interactive Chatbot
- **Command-based Interface:**
  - Analyze stock data.
  - Run forecasts.
  - Plot results.
  - Manage portfolios.
- **Built-in Help System:**
  - Commands such as `analyze`, `chart`, `predict`, etc.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock-forecasting-system.git
cd stock-forecasting-system

2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

3. Install Dependencies

Ensure you have Python 3.8+ installed, then run:

pip install -r requirements.txt

	Note: The requirements.txt should include libraries such as:
        •	numpy
	•	pandas
	•	matplotlib
	•	yfinance
	•	pmdarima
	•	prophet
	•	arch
	•	scikit-learn
	•	xgboost
	•	ta
	•	tensorflow
	•	seaborn (optional)
	•	plotly (optional)
	•	bokeh (optional)
	•	altair (optional)
	•	plotnine (optional)
	•	holoviews (optional)
	•	dash (optional)

4. Configure Output Directory

Ensure the output directory exists (for CSV files like raw data, forecasts, evaluations) or adjust the configuration in the code accordingly.

Usage

Running the Forecasting Pipeline

To execute the complete forecasting pipeline on selected tickers, run:

python your_script.py

This will:
	•	Download data for tickers (e.g., AAPL, GOOGL, MSFT)
	•	Preprocess and engineer features
	•	Train individual models and generate forecasts
	•	Build an ensemble forecast
	•	Visualize and evaluate the forecasts

You can customize the pipeline with parameters like so:

run_forecasting_pipeline(
    tickers=["AAPL", "GOOGL", "MSFT"],
    years=7,
    seq_length=40,
    train_frac=0.85,
    val_frac=0.1,
    xgb_lags=20,
    lstm_epochs=200,
    gru_epochs=200,
    batch_size=6
)

Chatbot Interface and Commands

Once the script is launched, the interactive command-line chatbot becomes available. Here are some sample commands:

Basic Analysis
	•	analyze AAPL
Displays analysis including evaluation metrics, technical indicators, and ensemble forecasts.
	•	raw AAPL
Shows raw stock data.

Visualization
	•	chart AAPL
Plots the raw price history.
	•	plot technical AAPL
Displays charts for technical indicators (e.g., RSI, MA20, MA50, MACD).
	•	plot forecast AAPL
Shows the ensemble (stacked) forecast.

Model-Specific Forecasts
	•	arima AAPL
Outputs ARIMA model forecasts.
	•	lstm AAPL
Outputs LSTM model forecasts.

Prediction & Comparison
	•	predict AAPL
Predicts the next 30 business days’ closing prices.
	•	compare AAPL GOOGL
Compares closing prices between two tickers.

Portfolio Management
	•	portfolio add AAPL 10
Adds 10 shares of AAPL to your portfolio.
	•	portfolio show
Displays the current portfolio.
	•	portfolio clear
Clears the portfolio.

Visualization Library Control
	•	set library plotly
Sets Plotly as the current visualization library.
	•	show library
Displays the currently selected visualization library.

For a full list of commands, type help in the chatbot interface.

File Structure

stock-forecasting-system/
├── output/                      # Directory for CSV outputs (raw data, forecasts, evaluations)
├── your_script.py               # Main script containing the forecasting pipeline and chatbot
├── requirements.txt             # List of Python dependencies
├── README.md                    # Project documentation (this file)
└── (additional modules or utilities, if any)

	•	Data Processing Functions: For downloading, preprocessing, and feature engineering.
	•	Model Training Modules: Separate modules for LSTM, GRU, ARIMA, Prophet, GARCH, and XGBoost.
	•	Ensemble & Evaluation: Functions for weighted ensembles and prediction evaluation.
	•	Chatbot & Command Processing: CLI for analysis, plotting, and portfolio management.

Contributing

Contributions are welcome! To contribute:
	1.	Fork the Repository
	2.	Create a New Branch: For your feature or bug fix.
	3.	Commit Your Changes: Ensure your code is well-documented and includes tests where applicable.
	4.	Submit a Pull Request
