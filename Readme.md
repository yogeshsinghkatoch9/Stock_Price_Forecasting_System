Stock Price Forecasting System

A comprehensive stock forecasting system that integrates data collection, preprocessing, feature engineering, multiple predictive models (including deep learning and statistical approaches), ensemble forecasting, and interactive visualization—all accessible via a command-line chatbot interface.

Table of Contents
	•	Overview
	•	Features
	•	Installation
	•	Usage
	•	Running the Forecasting Pipeline
	•	Chatbot Interface and Commands
	•	File Structure
	•	Contributing
	•	License

Overview

This project is an end-to-end stock forecasting system that:
	•	Downloads historical market data (OHLCV) for multiple stock tickers using yfinance.
	•	Preprocesses the data to fill missing values, enforce business-day frequency, and interpolate gaps.
	•	Extracts both basic and advanced features—including common technical indicators like Moving Averages, RSI, MACD, OBV, Stochastic Oscillator, Bollinger Bands, and volatility measures.
	•	Trains multiple forecasting models:
	•	Deep Learning Models: LSTM and GRU neural networks to capture sequential dependencies.
	•	Statistical Models: ARIMA for trend forecasting, Prophet for trend decomposition, and GARCH for volatility prediction.
	•	Machine Learning Models: XGBoost utilizing lagged features.
	•	Implements an ensemble mechanism that weights individual model predictions based on their validation performance, and also provides a stacking meta-model (via linear regression) for the final forecast.
	•	Provides extensive visualization options with support for several libraries (e.g., Matplotlib, Seaborn, Plotly, Bokeh, Altair, ggplot/Plotnine, Holoviews, and Dash).
	•	Includes a command-line chatbot that enables users to:
	•	Analyze stocks (showing technical indicators, evaluation metrics, and ensemble forecasts).
	•	Compare stock performances.
	•	Manage a portfolio (add, view, or clear stocks).
	•	Interactively select the preferred visualization library.

Features
	•	Data Collection & Preprocessing
	•	Downloads 7+ years of historical data.
	•	Handles missing values via forward-fill and linear interpolation.
	•	Reindexes data to maintain business-day frequency.
	•	Feature Engineering
	•	Basic Indicators: Returns, MA20/MA50, momentum, and volume averages.
	•	Technical Indicators: RSI, MACD, OBV, and Stochastic Oscillator.
	•	Advanced Features: Log-price transforms, multi-period volatility, moving average crossovers, Bollinger Bands, and volume ratios.
	•	Model Training & Forecasting
	•	Deep Learning: LSTM/GRU models with sequence-based training, dropout regularization, and early stopping.
	•	Statistical Methods: ARIMA for non-seasonal trends, Prophet for decomposing time series, and GARCH for volatility.
	•	Machine Learning: XGBoost using lagged features and recursive forecasting.
	•	Ensemble Strategy: Inverse error weighting and meta-model stacking.
	•	Evaluation & Visualization
	•	Plot historical data, forecast comparisons, and error distributions.
	•	Support for multiple plotting libraries to create interactive or publication-quality charts.
	•	Interactive Chatbot
	•	Command-based interface to analyze stock data, run forecasts, plot results, and manage a portfolio.
	•	Built-in help system with commands such as analyze, chart, predict, and more.

Installation
	1.	Clone the Repository

git clone https://github.com/yourusername/stock-forecasting-system.git
cd stock-forecasting-system


	2.	Create a Virtual Environment

python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate


	3.	Install Dependencies
Make sure you have Python 3.8+ installed. Then install the required packages:

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

	4.	Configure Output Directory
The system writes CSV files (raw data, forecasts, evaluations, etc.) into an output directory. Ensure that the directory exists or adjust the configuration in the code.

Usage

Running the Forecasting Pipeline

To run the complete forecasting pipeline on selected tickers, simply execute the script:

python your_script.py

The pipeline will:
	•	Download data for the specified tickers (e.g., AAPL, GOOGL, MSFT).
	•	Preprocess and engineer features.
	•	Train individual models and generate forecasts.
	•	Build an ensemble forecast.
	•	Visualize and evaluate the forecasts.

The function call is customizable; for example:

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

After launching the script, you enter an interactive command-line chatbot that accepts commands. Some examples include:
	•	Basic Analysis
	•	analyze AAPL
Displays comprehensive analysis including evaluation metrics, technical indicators, and ensemble forecasts.
	•	raw AAPL
Shows raw stock data.
	•	Visualization
	•	chart AAPL
Plots raw price history.
	•	plot technical AAPL
Displays charts for technical indicators like RSI, MA20, MA50, and MACD.
	•	plot forecast AAPL
Shows the ensemble (stacked) forecast.
	•	Model-Specific Forecasts
	•	arima AAPL
Outputs ARIMA model forecasts.
	•	lstm AAPL
Outputs LSTM model forecasts.
	•	Prediction & Comparison
	•	predict AAPL
Predicts the next 30 business days’ closing prices.
	•	compare AAPL GOOGL
Compares closing prices between two tickers.
	•	Portfolio Management
	•	portfolio add AAPL 10
Adds 10 shares of AAPL to your portfolio.
	•	portfolio show
Displays the current portfolio.
	•	portfolio clear
Clears the portfolio.
	•	Visualization Library Control
	•	set library plotly
Sets Plotly as the current visualization library.
	•	show library
Displays the currently selected visualization library.

For a full list of commands, type help within the chatbot.

File Structure

stock-forecasting-system/
├── output/                      # Directory for CSV outputs (raw data, forecasts, evaluations)
├── your_script.py               # Main Python script containing the forecasting pipeline and chatbot
├── requirements.txt             # List of Python dependencies
├── README.md                    # This README file
└── (additional modules or utilities if any)

	•	Data Processing Functions: Functions for downloading, preprocessing, and feature engineering.
	•	Model Training Modules: Separate functions for training LSTM, GRU, ARIMA, Prophet, GARCH, and XGBoost models.
	•	Ensemble & Evaluation: Functions to create weighted ensembles and to visualize/evaluate predictions.
	•	Chatbot & Command Processing: An interactive CLI to issue commands for analysis, plotting, and portfolio management.

Contributing

Contributions are welcome! If you’d like to contribute:
	1.	Fork the repository.
	2.	Create a new branch for your feature or bug fix.
	3.	Commit your changes and submit a pull request.
	4.	Please ensure your code is well-documented and includes tests where appropriate.
