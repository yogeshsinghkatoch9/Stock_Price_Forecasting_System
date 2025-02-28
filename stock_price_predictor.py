import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pmdarima as pm
from prophet import Prophet
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import ta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


 # ------------------- Configuration -------------------

 # Set random seed for reproducibility
SEED = 42   
np.random.seed(SEED) 
random.seed(SEED)
tf.random.set_seed(SEED)

 # Configure logging
logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
 # Get a logger instance
logger = logging.getLogger(__name__) 



 # ------------------- Data Download and Preprocessing ------------------- 
 
def download_stock_data(tickers, years=7):  # 7 years of data
     # Download stock data for given tickers using yfinance.
     # Returns a dict: {ticker: DataFrame}.
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    data_dict = {}
    
    for ticker in tickers:
        logger.info(f"Downloading data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            if 'Adj Close' in df.columns:
                df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            else:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            data_dict[ticker] = df
        else:
            logger.warning(f"No data found for {ticker}")
    
    return data_dict


 # Data Preprocessing
def preprocess_data(df):
     # Fill missing values and ensure a Business-Day DatetimeIndex.
    df = df.copy()
    df.ffill(inplace=True)                   # Forward-fill any missing data
    df.interpolate(method='linear', inplace=True)
    df.dropna(inplace=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.asfreq('B')                      # Convert to business-day frequency
    df.ffill(inplace=True)                   # Forward-fill again after reindexing
    return df


 # Feature Engineering     
def add_features(df):
     #Add technical indicators and basic features.
    df = df.copy()
    
     # Returns
    df['Returns'] = df['Close'].pct_change()
    
     # Rolling means
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
     # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    
     # Volume average
    if 'Volume' in df.columns:
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
     # Technical indicators
    close_series = df['Close'].squeeze()
    high_series = df['High'].squeeze()
    low_series = df['Low'].squeeze()
    volume_series = df['Volume'].squeeze() if 'Volume' in df.columns else pd.Series([0]*len(df), index=df.index)
    
    try:
         # RSI
        rsi_indicator = ta.momentum.RSIIndicator(close=close_series, window=14)
        df['RSI'] = rsi_indicator.rsi()
        
         # MACD
        macd_indicator = ta.trend.MACD(close=close_series)
        df['MACD'] = macd_indicator.macd()
        
         # OBV
        obv_indicator = ta.volume.OnBalanceVolumeIndicator(close=close_series, volume=volume_series)
        df['OBV'] = obv_indicator.on_balance_volume()
        
         # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=high_series,
            low=low_series,
            close=close_series,
            window=14,
            smooth_window=3
        )
        df['Stoch'] = stoch.stoch()
    except Exception as e:
        logger.error(f"Error calculating TA indicators: {e}")
    
    df.dropna(inplace=True)
    return df

 
 # ------------------- Enhanced Feature Engineering -------------------
def add_advanced_features(df):
     # Enhanced feature engineering
    df = df.copy()
    
     # Price transforms
    df['Log_Price'] = np.log(df['Close'])
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = df['Log_Price'].diff()
    
     # Volatility features
    for window in [5, 21, 63]:  # 1 week, 1 month, 3 months
        df[f'Volatility_{window}d'] = df['Returns'].rolling(window).std()
        df[f'Log_Volatility_{window}d'] = df['Log_Returns'].rolling(window).std()
    
     # Price momentum
    for window in [5, 21, 63]:
        df[f'Momentum_{window}d'] = df['Close'].pct_change(window)
        
     # Moving averages and crossovers
    for window in [10, 20, 50, 100]:
        df[f'MA_{window}'] = df['Close'].rolling(window).mean()
        df[f'MA_Vol_{window}'] = df['Volume'].rolling(window).mean()
    
     # Moving average crossover signals
    df['MA_Cross_10_50'] = (df['MA_10'] > df['MA_50']).astype(int)
    df['MA_Cross_20_50'] = (df['MA_20'] > df['MA_50']).astype(int)
    
     # Advanced technical indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
     # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
    
     # Volume-based features
    df['Volume_1d_chg'] = df['Volume'].pct_change()
    df['Volume_MA_ratio'] = df['Volume'] / df['MA_Vol_20']
    
    return df



    # ------------------- Model Training and Forecasting -------------------
def train_val_test_split(df, train_frac=0.6, val_frac=0.2):
    # Splits data into train, validation, and test by row counts.
    df = df.sort_index()
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


 # Sequence Creation
def create_sequences(data, seq_length=60):
     # Convert 1D array 'data' into sequences of length `seq_length`.
     # Returns X, y arrays for training (supervised).
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


 # LSTM/GRU Data Preparation
def prepare_lstm_test_data(full_series, scaler, seq_length=60):
     #Prepare sequences from the full_series (pd.Series) for LSTM/GRU testing/prediction.
    total_data = full_series.values
    inputs = scaler.transform(total_data.reshape(-1, 1))
    X_test = []
    for i in range(seq_length, len(inputs)):
        X_test.append(inputs[i - seq_length:i, 0])
    return np.array(X_test).reshape((-1, seq_length, 1))


 # LSTM and GRU Training
def train_lstm(train_series, seq_length=60, epochs=50, batch_size=32):
     # Train a LSTM model on a single series.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_series.values.reshape(-1, 1))
    X_train, y_train = create_sequences(scaled_data, seq_length)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    ]
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
    return model, scaler


 # Improved LSTM Architecture
def build_improved_lstm(seq_length, n_features):
     #Enhanced LSTM architecture
    model = Sequential([
         # Bidirectional LSTM layers
        Bidirectional(LSTM(128, return_sequences=True), 
                     input_shape=(seq_length, n_features)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),
        
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=huber_loss,  # More robust to outliers
        metrics=['mae', 'mse']
    )
    return model


 # Improved Training Process
def train_enhanced_lstm(X_train, y_train, X_val, y_val, epochs=100):
     # Train LSTM with better hyperparameters and callbacks
    model = build_enhanced_lstm_model(X_train.shape[1], X_train.shape[2])
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


 # GRU Training
def train_gru(train_series, seq_length=60, epochs=50, batch_size=32):
     # Train a GRU model on a single series.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_series.values.reshape(-1, 1))
    X_train, y_train = create_sequences(scaled_data, seq_length)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        GRU(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    ]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
    return model, scaler


 # Prophet Training
def process_garch(train_df, val_df, forecast_horizon_val, forecast_horizon_test): 
     #Fit GARCH on (train) or (train+val) and forecast returns for val/test.
     #Convert forecasted returns back to price by compounding from last known price.
    train_returns = train_df['Returns'].dropna()
    train_returns = train_returns[~np.isnan(train_returns) & ~np.isinf(train_returns)]
    train_returns = train_returns.values.flatten() * 100
    
    try:
        garch_model = arch_model(train_returns, mean='AR', lags=1, vol='Garch', p=1, q=1, dist='normal')
        garch_fit = garch_model.fit(disp='off')
        
        forecast_garch_val = garch_fit.forecast(horizon=forecast_horizon_val)
        forecast_returns_val = forecast_garch_val.mean.iloc[-1].values
        
        last_price = train_df['Close'].iloc[-1]
        garch_val_pred = []
        for r in forecast_returns_val:
            last_price *= (1 + r/100.0)
            garch_val_pred.append(last_price)
        garch_val_pred = np.array(garch_val_pred)
    except Exception as e:
        logger.error(f"GARCH validation error: {e}")
        

         # If GARCH fails, fallback to last known price
        garch_val_pred = np.repeat(train_df['Close'].iloc[-1], forecast_horizon_val)
    
     # For Test
    train_val_df = pd.concat([train_df, val_df])
    train_val_returns = train_val_df['Returns'].dropna()
    train_val_returns = train_val_returns[~np.isnan(train_val_returns) & ~np.isinf(train_val_returns)]
    train_val_returns = train_val_returns.values.flatten() * 100
    
    try:
        garch_model_final = arch_model(train_val_returns, mean='AR', lags=1, vol='Garch', p=1, q=1, dist='normal')
        garch_fit_final = garch_model_final.fit(disp='off')
        
        forecast_garch_test = garch_fit_final.forecast(horizon=forecast_horizon_test)
        forecast_returns_test = forecast_garch_test.mean.iloc[-1].values
        
        last_price_val = train_val_df['Close'].iloc[-1]
        garch_test_pred = []
        for r in forecast_returns_test:
            last_price_val *= (1 + r/100.0)
            garch_test_pred.append(last_price_val)
        garch_test_pred = np.array(garch_test_pred)
    except Exception as e:
        logger.error(f"GARCH test error: {e}")
        garch_test_pred = np.repeat(train_val_df['Close'].iloc[-1], forecast_horizon_test)
    
    return garch_val_pred, garch_test_pred


 # ARIMA Training
def prepare_xgb_features(series, lags=5):
     # Create lagged features for XGBoost from a single column Series.
     # Returns (X, y).
    if isinstance(series, pd.DataFrame):
         # If it's a DataFrame, ensure only 1 column
        if series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            raise ValueError("Expected a Series or a single-column DataFrame.")
        
    df = pd.DataFrame(series.copy())
    col_name = df.columns[0] if isinstance(df, pd.DataFrame) else df.name
    
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[col_name].shift(lag)
    
    df.dropna(inplace=True)
    X = df.drop(columns=[col_name])
    y = df[col_name]
    return X, y

 # XGBoost Training 
def train_xgboost(train_series, lags=5):  
     # Train an XGBoost regressor using lag features.
     # eturns (model, lags).
    X, y = prepare_xgb_features(train_series, lags)
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100,
        verbosity=0,
        random_state=SEED
    )
    model.fit(X, y)
    return model, lags

 #  Forecasting with XGBoost
def xgb_forecast(model, series, forecast_horizon, lags=5):
     # Generate forecasts for `forecast_horizon` steps using XGBoost. 
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            raise ValueError("Expected a single column.")
    
    history = series.values[-lags:].tolist()
    predictions = []
    for _ in range(forecast_horizon):
        x_input = np.array(history[-lags:]).reshape(1, -1)
        pred = model.predict(x_input)[0]
        predictions.append(pred)
        history.append(pred)
    return np.array(predictions)

 # Validation MSE Calculation
def train_meta_model(X_meta, y_meta):
     # Train a simple linear regression as the stacking meta-model.
    meta_model = LinearRegression()
    meta_model.fit(X_meta, y_meta)
    return meta_model

 # Forecasting with Meta-Model
def plot_forecasts(train_df, val_df, test_df, test_index, stacked_forecast, individual_forecasts, ticker, forecast_horizon_test):
     # Plot train, val, test, stacked ensemble predictions, and individual models.
    actual_test = test_df['Close'].values[:forecast_horizon_test]
    
     # Plot 1: Stacked forecast vs actual
    plt.figure(figsize=(14, 7))
    plt.plot(train_df.index, train_df['Close'], label='Train', color='blue')
    plt.plot(val_df.index, val_df['Close'], label='Validation', color='orange')
    plt.plot(test_df.index[:forecast_horizon_test], actual_test, label='Test (Actual)', color='green')
    plt.plot(test_index, stacked_forecast, label='Stacked Forecast', linewidth=2, color='red')
    plt.title(f"{ticker} - Stacked Ensemble Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
     # Plot 2: Individual forecasts vs actual (test portion)
    plt.figure(figsize=(14, 7))
    plt.plot(test_df.index[:forecast_horizon_test], actual_test, label='Actual', color='black', linewidth=2)
    for name, forecast in individual_forecasts.items():
        plt.plot(test_index, forecast, label=name, alpha=0.8)
    plt.title(f"{ticker} - Individual Model Forecasts")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


 
 # Visualize Results
def visualize_results(actual, predictions, ticker):
     # Visualize error distribution and scatter plot of actual vs predictions.
    errors = actual - predictions
    
     # Error histogram
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"{ticker} - Forecast Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()
    
     # Actual vs predicted scatter
    plt.figure(figsize=(10, 5))
    plt.scatter(actual, predictions, color='red', alpha=0.7)
    mn, mx = min(actual.min(), predictions.min()), max(actual.max(), predictions.max())
    plt.plot([mn, mx], [mn, mx], color='blue', lw=2)  # identity line
    plt.title(f"{ticker} - Actual vs Predicted")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.show()



 # Improved Weighted Ensemble
def weighted_ensemble_forecast(forecasts, weights=None):
    """Improved weighted ensemble strategy"""
    if weights is None:

         # Calculate weights based on validation performance
        val_errors = np.array([f['val_mse'] for f in forecasts])
        val_errors = np.where(val_errors == 0, np.inf, val_errors)  # Handle zero errors

         # Inverse error weighting with temperature scaling
        temperature = 2.0
        weights = np.exp(-val_errors / temperature)
        weights = weights / np.sum(weights)
     # Apply minimum weight threshold
    min_weight = 0.1
    weights = np.maximum(weights, min_weight)
    weights = weights / np.sum(weights)
    
    weighted_forecast = np.zeros_like(forecasts[0]['prediction'])
    for f, w in zip(forecasts, weights):
        weighted_forecast += w * f['prediction']
    return weighted_forecast, weights


 
 #  ------------------- Main Pipeline Function -------------------
def improved_forecasting_pipeline(ticker, data, forecast_horizon=30):
    """Enhanced forecasting pipeline with all improvements"""

     # Preprocess
    df = add_enhanced_features(data)
    df.dropna(inplace=True)
    
     # Train/Val/Test split
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

     # Scale features
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    
     # Prepare sequences
    seq_length = 60
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_val, y_val = create_sequences(val_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

     # Train models
    models = {
        'lstm': train_enhanced_lstm(X_train, y_train, X_val, y_val),
        'prophet': train_prophet(train_df['Close']),
        'arima': pm.auto_arima(
            train_df['Close'],
            seasonal=True,
            m=5,
            start_p=1,
            start_q=1,
            max_p=3,
            max_q=3,
            d=1,
            D=1,
            trace=True,
            error_action='ignore',
            suppress_warnings=True
        )
    }
    
     # Generate forecasts
    forecasts = []
    for name, model in models.items():
        pred = generate_forecast(model, name, test_df, forecast_horizon)
        val_mse = calculate_validation_mse(model, val_df)
        forecasts.append({
            'name': name,
            'prediction': pred,
            'val_mse': val_mse
        })
    
     # Create weighted ensemble
    ensemble_forecast = weighted_ensemble_forecast(forecasts)
    
    return ensemble_forecast, forecasts, models


 #  ------------------- Run the Pipeline -------------------
def run_forecasting_pipeline(
    tickers=["AAPL", "GOOGL"], 
    years=7,        # Number of years of data to download
    seq_length=60,  # Sequence length for LSTM/GRU
    train_frac=0.9, # Fraction of data for training
    val_frac=0.2,   # Fraction of data for validation
    xgb_lags=5,     # Number of lags for XGBoost
    lstm_epochs=10, # Number of epochs for LSTM
    gru_epochs=20,  # Number of epochs for GRU
    batch_size=32  # Batch size for LSTM/GRU
):
     #End-to-end pipeline to download data, train models, stack predictions, and plot results.
    logger.info("Starting the stock forecasting pipeline...")
    data_dict = download_stock_data(tickers, years=years)
    
    for ticker in tickers:
        logger.info(f"=== Processing {ticker} ===")
        if ticker not in data_dict or data_dict[ticker].empty:
            logger.warning(f"Skipping {ticker} because no data was downloaded.")
            continue
        
        df = data_dict[ticker].copy()
         # Preprocess
        df = preprocess_data(df)
         # Features
        df = add_features(df)
        df.dropna(inplace=True)
        
        if len(df) < 100:
            logger.warning(f"Skipping {ticker} because not enough data after preprocessing.")
            continue

    
         # Split
        train_df, val_df, test_df = train_val_test_split(df, train_frac=train_frac, val_frac=val_frac)
        forecast_horizon_val = len(val_df)
        forecast_horizon_test = len(test_df)
        
         # -------------------- ARIMA --------------------
        logger.info("Training ARIMA...")
        arima_model = pm.auto_arima(train_df['Close'], seasonal=False, error_action='ignore', suppress_warnings=True)
        arima_val_pred = arima_model.predict(n_periods=forecast_horizon_val)
        arima_test_pred = arima_model.predict(n_periods=forecast_horizon_test)
        
         # -------------------- Prophet --------------------
        logger.info("Training Prophet...")
         # Reset index and properly rename columns for Prophet
        train_reset = train_df.copy()
        train_reset.reset_index(inplace=True)
        train_reset['ds'] = pd.to_datetime(train_reset['Date']) 
        train_reset['y'] = train_reset['Close'].astype(float)    
        prophet_train_df = train_reset[['ds', 'y']]              
        
        val_reset = val_df.copy()
        val_reset.reset_index(inplace=True)
        val_reset['ds'] = pd.to_datetime(val_reset['Date'])
        val_reset['y'] = val_reset['Close'].astype(float)
        prophet_val_df = val_reset[['ds', 'y']]
        
        test_reset = test_df.copy()
        test_reset.reset_index(inplace=True)
        test_reset['ds'] = pd.to_datetime(test_reset['Date'])
        test_reset['y'] = test_reset['Close'].astype(float)
        
         # Create Prophet model and fit
        m_prophet = Prophet(daily_seasonality=True)
        m_prophet.fit(prophet_train_df)

        
         # Forecast for validation
        future_val = pd.DataFrame({'ds': prophet_val_df['ds']})
        prophet_val_pred_full = m_prophet.predict(future_val)
        prophet_val_pred = prophet_val_pred_full['yhat'].values

        
         # For final test forecasting, re-train on (train+val)
        train_val_reset = pd.concat([train_reset, val_reset], ignore_index=True)
        prophet_train_val_df = train_val_reset[['ds', 'y']]
        
        m_prophet_final = Prophet(daily_seasonality=True)
        m_prophet_final.fit(prophet_train_val_df)
        
        future_test = pd.DataFrame({'ds': test_reset['ds']})
        prophet_test_pred_full = m_prophet_final.predict(future_test)
        prophet_test_pred = prophet_test_pred_full['yhat'].values

                
        # -------------------- GARCH --------------------
        logger.info("Training GARCH...")
        try:
            garch_val_pred, garch_test_pred = process_garch(train_df, val_df, forecast_horizon_val, forecast_horizon_test)
        except Exception as e:
            logger.error(f"GARCH processing failed: {e}")
            fallback_val = train_df['Close'].iloc[-1]
            garch_val_pred = np.repeat(fallback_val, forecast_horizon_val)
            garch_test_pred = np.repeat(fallback_val, forecast_horizon_test)

        
        # -------------------- LSTM --------------------
        logger.info("Training LSTM...")
        lstm_model, lstm_scaler = train_lstm(train_df['Close'], seq_length=seq_length, epochs=lstm_epochs, batch_size=batch_size)
        
        combined_train_val = pd.concat([train_df['Close'], val_df['Close']])
        X_val_lstm = prepare_lstm_test_data(combined_train_val, lstm_scaler, seq_length=seq_length)
        lstm_val_pred_scaled = lstm_model.predict(X_val_lstm)
        lstm_val_pred = lstm_scaler.inverse_transform(lstm_val_pred_scaled).flatten()[-forecast_horizon_val:]

        
            # Re-train LSTM on train+val for final test
        lstm_model_final, lstm_scaler_final = train_lstm(combined_train_val, seq_length=seq_length, epochs=lstm_epochs, batch_size=batch_size)
        combined_series = pd.concat([combined_train_val, test_df['Close']])
        X_test_lstm = prepare_lstm_test_data(combined_series, lstm_scaler_final, seq_length=seq_length)
        lstm_test_pred_scaled = lstm_model_final.predict(X_test_lstm)
        lstm_test_pred = lstm_scaler_final.inverse_transform(lstm_test_pred_scaled).flatten()[-forecast_horizon_test:]

        
        # -------------------- GRU --------------------
        logger.info("Training GRU...")
        gru_model, gru_scaler = train_gru(train_df['Close'], seq_length=seq_length, epochs=gru_epochs, batch_size=batch_size)
        
        X_val_gru = prepare_lstm_test_data(combined_train_val, gru_scaler, seq_length=seq_length)
        gru_val_pred_scaled = gru_model.predict(X_val_gru)
        gru_val_pred = gru_scaler.inverse_transform(gru_val_pred_scaled).flatten()[-forecast_horizon_val:]
        
 
            # Re-train GRU on train+val for final test
        gru_model_final, gru_scaler_final = train_gru(combined_train_val, seq_length=seq_length, epochs=gru_epochs, batch_size=batch_size)
        X_test_gru = prepare_lstm_test_data(combined_series, gru_scaler_final, seq_length=seq_length)
        gru_test_pred_scaled = gru_model_final.predict(X_test_gru)
        gru_test_pred = gru_scaler_final.inverse_transform(gru_test_pred_scaled).flatten()[-forecast_horizon_test:]

        
        # -------------------- XGBoost --------------------
        logger.info("Training XGBoost...")
        xgb_model, xgb_lags_val = train_xgboost(train_df['Close'], lags=xgb_lags)
        xgb_val_pred = xgb_forecast(xgb_model, train_df['Close'], forecast_horizon_val, lags=xgb_lags_val)
        
        xgb_model_final, _ = train_xgboost(pd.concat([train_df['Close'], val_df['Close']]), lags=xgb_lags)
        xgb_test_pred = xgb_forecast(xgb_model_final, pd.concat([train_df['Close'], val_df['Close']]), forecast_horizon_test, lags=xgb_lags)

        
        # -------------------- Stacking Ensemble --------------------
        logger.info("Building stacking ensemble...")
        
         # Validation predictions (to train meta-model)
        val_index = val_df.index[:forecast_horizon_val]
        arima_val_pred_flat    = np.array(arima_val_pred).flatten()
        prophet_val_pred_flat  = np.array(prophet_val_pred).flatten()
        garch_val_pred_flat    = np.array(garch_val_pred).flatten()
        lstm_val_pred_flat     = np.array(lstm_val_pred).flatten()
        gru_val_pred_flat      = np.array(gru_val_pred).flatten()
        xgb_val_pred_flat      = np.array(xgb_val_pred).flatten()
        
        stacking_val_df = pd.DataFrame({
            'ARIMA': arima_val_pred_flat,
            'Prophet': prophet_val_pred_flat,
            'GARCH': garch_val_pred_flat,
            'LSTM': lstm_val_pred_flat,
            'GRU': gru_val_pred_flat,
            'XGBoost': xgb_val_pred_flat
        }, index=val_index)
        
        y_val_actual = val_df['Close'].values
        meta_model = train_meta_model(stacking_val_df, y_val_actual)
        
         # Test predictions (stack them)
        arima_test_pred_flat    = np.array(arima_test_pred).flatten()
        prophet_test_pred_flat  = np.array(prophet_test_pred).flatten()
        garch_test_pred_flat    = np.array(garch_test_pred).flatten()
        lstm_test_pred_flat     = np.array(lstm_test_pred).flatten()
        gru_test_pred_flat      = np.array(gru_test_pred).flatten()
        xgb_test_pred_flat      = np.array(xgb_test_pred).flatten()
        
        test_index = test_df.index[:forecast_horizon_test] 
        stacking_test_df = pd.DataFrame({
            'ARIMA': arima_test_pred_flat,
            'Prophet': prophet_test_pred_flat,
            'GARCH': garch_test_pred_flat,
            'LSTM': lstm_test_pred_flat,
            'GRU': gru_test_pred_flat,
            'XGBoost': xgb_test_pred_flat
        }, index=test_index)
        
         # Stacked ensemble forecast
        stacked_forecast = meta_model.predict(stacking_test_df)           
        
            # Evaluate
        logger.info("Evaluating stacked ensemble...")                     
        actual_test = test_df['Close'].values[:forecast_horizon_test]     
        mae_stacked = mean_absolute_error(actual_test, stacked_forecast)
        mse_stacked = mean_squared_error(actual_test, stacked_forecast)   
        rmse_stacked = np.sqrt(mse_stacked)                               

        
        logger.info(f"Stacked Ensemble for {ticker}:")    
        logger.info(f"  MAE  = {mae_stacked:.4f}")        
        logger.info(f"  MSE  = {mse_stacked:.4f}")       
        logger.info(f"  RMSE = {rmse_stacked:.4f}")       

        
        individual_forecasts = {
            'ARIMA': arima_test_pred_flat,     # ARIMA
            'Prophet': prophet_test_pred_flat, # Prophet
            'GARCH': garch_test_pred_flat,     # GARCH
            'LSTM': lstm_test_pred_flat,       # LSTM
            'GRU': gru_test_pred_flat,         # GRU
            'XGBoost': xgb_test_pred_flat      # XGBoost
        }
        
        plot_forecasts(
            train_df,             #Training data
            val_df,               # Validation data
            test_df,              # Test data
            test_index,           # Test index
            stacked_forecast,     # Stacked forecast
            individual_forecasts, # Individual model forecasts
            ticker,               # Ticker symbol
            forecast_horizon_test # Forecast horizon for test
        )
        visualize_results(actual_test, stacked_forecast, ticker)

    logger.info("Pipeline completed!")


 # Call the function below to run everything and produce charts. 
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
