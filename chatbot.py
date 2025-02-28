import os  # Module for interacting with the operating system (e.g., file paths)
import sys  # Module to interact with the Python interpreter (e.g., exit)
import re   # Regular expressions module for string pattern matching
import pandas as pd  # Library for data manipulation and analysis
import matplotlib.pyplot as plt  # Plotting library (Matplotlib)
import pmdarima as pm  # Library for ARIMA forecasting

# Try to import additional libraries (if installed) for enhanced visualization
try:
    import seaborn as sns  # Statistical data visualization library based on matplotlib
except ImportError:
    sns = None  # If not installed, set sns to None

try:
    import plotly.express as px  # High-level interactive plotting library
except ImportError:
    px = None  # If not installed, set px to None

try:
    from bokeh.plotting import figure, show as bokeh_show, output_notebook  # Interactive visualization library for web browsers
except ImportError:
    figure = None  # If not installed, set figure to None

try:
    import altair as alt  # Declarative statistical visualization library
except ImportError:
    alt = None  # If not installed, set alt to None

try:
    from plotnine import ggplot, aes, geom_line, labs, theme_minimal  # Grammar of graphics library (similar to ggplot2 in R)
except ImportError:
    ggplot = None  # If not installed, set ggplot to None

try:
    import holoviews as hv  # High-level data analysis and visualization library
    hv.extension('bokeh')  # Enable the Bokeh backend for Holoviews
except ImportError:
    hv = None  # If not installed, set hv to None

try:
    import dash  # Web application framework for building interactive dashboards
except ImportError:
    dash = None  # If not installed, set dash to None

# ======================================================
# Global Configuration and Visualization Library Settings
# ======================================================

OUTPUT_DIR = "output"  # Directory where CSV files from the forecasting pipeline are saved

# Mapping of file types to their corresponding folder or file name parts
FILE_MAPPING = {
    "raw": "raw",
    "preprocessed": "preprocessed",
    "features": "features",
    "train": "train",
    "validation": "validation",
    "test": "test",
    "arima validation": "arima_validation",
    "arima test": "arima_test",
    "prophet validation": "prophet_validation",
    "prophet test": "prophet_test",
    "garch validation": "garch_validation",
    "garch test": "garch_test",
    "lstm validation": "lstm_validation",
    "lstm test": "lstm_test",
    "gru validation": "gru_validation",
    "gru test": "gru_test",
    "xgboost validation": "xgboost_validation",
    "xgboost test": "xgboost_test",
    "ensemble": "stacked_forecast",
    "evaluation": "evaluation_metrics"
}

portfolio = {}  # Global dictionary to store the user's portfolio (ticker: shares)

# Global variable for the current visualization library (default is "matplotlib")
# Allowed options: "matplotlib", "seaborn", "plotly", "bokeh", "altair", "ggplot", "holoviews", "dash"
CURRENT_VIZ_LIB = "matplotlib"

# Function to set the current visualization library based on user preference
def set_library(lib):
    global CURRENT_VIZ_LIB  # Use the global variable to update the current library
    allowed = ["matplotlib", "seaborn", "plotly", "bokeh", "altair", "ggplot", "holoviews", "dash"]
    if lib.lower() in allowed:  # Check if the provided library is supported
        CURRENT_VIZ_LIB = lib.lower()  # Update the current visualization library
        print(f"Visualization library set to {CURRENT_VIZ_LIB}.")
    else:
        print(f"Library '{lib}' is not supported. Choose from: {', '.join(allowed)}")

# Function to display which visualization library is currently selected
def show_library():
    print(f"Current visualization library: {CURRENT_VIZ_LIB}")

# ======================================================
# Visualization Dispatchers
# ======================================================

# Dispatcher function to create a line plot using the selected visualization library
def line_plot_dispatch(x, y, title, xlabel, ylabel, legend_label):
    """Dispatch a simple line plot using the selected library."""
    global CURRENT_VIZ_LIB
    if CURRENT_VIZ_LIB == "matplotlib":
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label=legend_label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
    elif CURRENT_VIZ_LIB == "seaborn":
        if sns is None:
            print("Seaborn is not installed.")
            return
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 5))
        sns.lineplot(x=x, y=y, label=legend_label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
    elif CURRENT_VIZ_LIB == "plotly":
        if px is None:
            print("Plotly is not installed.")
            return
        df = pd.DataFrame({xlabel: x, ylabel: y})
        fig = px.line(df, x=xlabel, y=ylabel, title=title)
        fig.show()
    elif CURRENT_VIZ_LIB == "bokeh":
        if figure is None:
            print("Bokeh is not installed.")
            return
        from bokeh.io import output_notebook, show as bokeh_show
        output_notebook()
        p = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel,
                   plot_width=800, plot_height=400)
        p.line(x, y, legend_label=legend_label, line_width=2)
        bokeh_show(p)
    elif CURRENT_VIZ_LIB == "altair":
        if alt is None:
            print("Altair is not installed.")
            return
        df = pd.DataFrame({xlabel: x, ylabel: y})
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X(xlabel, title=xlabel),
            y=alt.Y(ylabel, title=ylabel)
        ).properties(title=title)
        chart.show()
    elif CURRENT_VIZ_LIB == "ggplot":
        if ggplot is None:
            print("Plotnine (ggplot) is not installed.")
            return
        df = pd.DataFrame({xlabel: x, ylabel: y})
        plot = (ggplot(df, aes(x=xlabel, y=ylabel)) +
                geom_line(color="blue") +
                labs(title=title, x=xlabel, y=ylabel) +
                theme_minimal())
        print(plot)
    elif CURRENT_VIZ_LIB == "holoviews":
        if hv is None:
            print("Holoviews is not installed.")
            return
        import hvplot.pandas
        df = pd.DataFrame({xlabel: x, ylabel: y})
        plot = df.hvplot.line(x=xlabel, y=ylabel, title=title)
        hv.save(plot, 'temp.html')
        hv.show(plot)
    elif CURRENT_VIZ_LIB == "dash":
        print("Dash visualization requires a running Dash server. Please use another library.")
    else:
        print(f"Visualization library {CURRENT_VIZ_LIB} not supported.")

# Dispatcher function to create a bar chart using the selected visualization library
def bar_plot_dispatch(categories, values, title, xlabel, ylabel):
    """Dispatch a simple bar chart using the selected library."""
    global CURRENT_VIZ_LIB
    categories = [str(c) for c in categories]
    if CURRENT_VIZ_LIB == "matplotlib":
        plt.figure(figsize=(6, 4))
        plt.bar(categories, values, color=["blue", "green", "red"])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for i, v in enumerate(values):
            plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
        plt.show()
    elif CURRENT_VIZ_LIB == "seaborn":
        if sns is None:
            print("Seaborn is not installed.")
            return
        df = pd.DataFrame({xlabel: categories, ylabel: values})
        sns.set(style="whitegrid")
        ax = sns.barplot(x=xlabel, y=ylabel, data=df, palette=["blue", "green", "red"])
        ax.set_title(title)
        plt.show()
    elif CURRENT_VIZ_LIB == "plotly":
        if px is None:
            print("Plotly is not installed.")
            return
        df = pd.DataFrame({xlabel: categories, ylabel: values})
        fig = px.bar(df, x=xlabel, y=ylabel, title=title)
        fig.show()
    elif CURRENT_VIZ_LIB == "bokeh":
        if figure is None:
            print("Bokeh is not installed.")
            return
        from bokeh.models import ColumnDataSource
        from bokeh.io import output_notebook, show as bokeh_show
        output_notebook()
        source = ColumnDataSource(data={xlabel: categories, ylabel: values})
        p = figure(x_range=categories, title=title, plot_height=400, plot_width=600)
        p.vbar(x=xlabel, top=ylabel, width=0.9, source=source, color=["blue", "green", "red"])
        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        bokeh_show(p)
    elif CURRENT_VIZ_LIB == "altair":
        if alt is None:
            print("Altair is not installed.")
            return
        df = pd.DataFrame({xlabel: categories, ylabel: values})
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(xlabel, title=xlabel),
            y=alt.Y(ylabel, title=ylabel)
        ).properties(title=title)
        chart.show()
    elif CURRENT_VIZ_LIB == "ggplot":
        if ggplot is None:
            print("Plotnine (ggplot) is not installed.")
            return
        from plotnine import ggplot, aes, geom_bar, labs, theme_minimal
        df = pd.DataFrame({xlabel: categories, ylabel: values})
        plot = (ggplot(df, aes(x=xlabel, y=ylabel)) +
                geom_bar(stat="identity", fill="skyblue") +
                labs(title=title, x=xlabel, y=ylabel) +
                theme_minimal())
        print(plot)
    elif CURRENT_VIZ_LIB == "holoviews":
        if hv is None:
            print("Holoviews is not installed.")
            return
        import hvplot.pandas
        df = pd.DataFrame({xlabel: categories, ylabel: values})
        chart = df.hvplot.bar(x=xlabel, y=ylabel, title=title)
        hv.save(chart, 'temp_bar.html')
        hv.show(chart)
    elif CURRENT_VIZ_LIB == "dash":
        print("Dash visualization requires a running Dash server. Please use another library.")
    else:
        print(f"Visualization library {CURRENT_VIZ_LIB} not supported.")

# ======================================================
# Additional Helper Functions Added to Fix Chatbot Errors
# ======================================================

def show_file(ticker, filetype):
    """
    Reads a CSV file for the given ticker and filetype from OUTPUT_DIR.
    Uses FILE_MAPPING to build the filename. Returns the file content as a string,
    or an error message if the file is not found.
    """
    ticker = ticker.upper()
    mapping = FILE_MAPPING.get(filetype.lower(), filetype.lower())
    filename = f"{ticker}_{mapping}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return f"File '{filename}' not found in {OUTPUT_DIR}."
    try:
        with open(filepath, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file '{filename}': {e}"

def list_tickers():
    """
    Scans the OUTPUT_DIR for files that follow the naming pattern (e.g. "AAPL_raw.csv")
    and returns a sorted list of unique ticker symbols.
    """
    tickers = set()
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            match = re.match(r"([A-Z]+)_", file)
            if match:
                tickers.add(match.group(1))
    return sorted(list(tickers))

def list_files_for_ticker(ticker):
    """
    Lists all files in OUTPUT_DIR that start with the ticker (e.g. "AAPL_").
    """
    ticker = ticker.upper()
    files = []
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            if file.startswith(ticker + "_"):
                files.append(file)
    return files

def plot_compare_stocks(ticker1, ticker2):
    """
    Reads raw CSV data for two tickers and plots their closing prices
    for the last 60 business days on the same chart for comparison.
    """
    ticker1 = ticker1.upper()
    ticker2 = ticker2.upper()
    file1 = os.path.join(OUTPUT_DIR, f"{ticker1}_raw.csv")
    file2 = os.path.join(OUTPUT_DIR, f"{ticker2}_raw.csv")
    if not os.path.exists(file1) or not os.path.exists(file2):
        print("Raw data file for one or both tickers not found.")
        return
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except Exception as e:
        print("Error reading raw data files:", e)
        return

    # Ensure a "Date" column exists and convert it to datetime
    for df in [df1, df2]:
        if "Date" not in df.columns:
            df.reset_index(inplace=True)
            if "index" in df.columns:
                df.rename(columns={"index": "Date"}, inplace=True)
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        except Exception as e:
            print("Error converting Date column:", e)
            return
    # Take the last 60 days of data
    df1 = df1.sort_values("Date").tail(60)
    df2 = df2.sort_values("Date").tail(60)

    plt.figure(figsize=(10, 6))
    plt.plot(df1["Date"], df1["Close"], label=ticker1)
    plt.plot(df2["Date"], df2["Close"], label=ticker2)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"Comparison of {ticker1} vs {ticker2}")
    plt.legend()
    plt.show()

# ======================================================
# Visualization Functions Using Dispatchers
# ======================================================

def plot_raw_data(ticker):
    """Plot raw closing price data using the current visualization library."""
    ticker = ticker.upper()
    filename = f"{ticker}_raw.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Raw data file '{filename}' not found.")
        return
    df = pd.read_csv(filepath)
    if "Date" not in df.columns:
        df = df.copy()
        df.reset_index(inplace=True)
        if "index" in df.columns:
            df.rename(columns={"index": "Date"}, inplace=True)
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df[df["Date"].notnull()]
    except Exception as e:
        print(f"Error converting Date column: {e}")
        return
    line_plot_dispatch(df["Date"], df["Close"],
                       title=f"Raw Close Price for {ticker}",
                       xlabel="Date", ylabel="Price", legend_label="Close Price")

def plot_evaluation(ticker):
    """Plot evaluation metrics as a bar chart using the current visualization library."""
    ticker = ticker.upper()
    filename = f"{ticker}_evaluation_metrics.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Evaluation metrics file '{filename}' not found.")
        return
    df = pd.read_csv(filepath)
    if df.empty:
        print("No evaluation metrics data available.")
        return
    metrics = ["MAE", "MSE", "RMSE"]
    values = []
    for metric in metrics:
        if metric in df.columns:
            values.append(df[metric].iloc[0])
        else:
            values.append(0)
    bar_plot_dispatch(metrics, values,
                      title=f"Evaluation Metrics for {ticker}",
                      xlabel="Metric", ylabel="Value")

def plot_technical_indicators(ticker):
    """Plot common technical indicators (RSI, MA20, MA50, MACD) from the features file."""
    ticker = ticker.upper()
    filename = f"{ticker}_features.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Features file '{filename}' not found.")
        return
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notnull()]
    cols_to_plot = [col for col in ["RSI", "MA20", "MA50", "MACD"] if col in df.columns]
    if not cols_to_plot:
        print("No recognized technical indicator columns found.")
        return
    line_plot_dispatch(df.index, df[cols_to_plot[0]],
                       title=f"{ticker} {cols_to_plot[0]} over Time",
                       xlabel="Date", ylabel=cols_to_plot[0], legend_label=cols_to_plot[0])

def plot_ensemble_forecast(ticker):
    """Plot the ensemble (stacked) forecast using the current visualization library (or fallback to ARIMA forecast)."""
    ticker = ticker.upper()
    filename = f"{ticker}_stacked_forecast.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notnull()]
        if "Stacked" in df.columns:
            line_plot_dispatch(df.index, df["Stacked"],
                               title=f"Ensemble Forecast for {ticker}",
                               xlabel="Date", ylabel="Price", legend_label="Stacked Forecast")
        else:
            print("Ensemble forecast column 'Stacked' not found. Plotting first numeric column.")
            line_plot_dispatch(df.index, df.iloc[:, 0],
                               title=f"Ensemble Forecast for {ticker}",
                               xlabel="Date", ylabel="Price", legend_label="Forecast")
    else:
        print(f"Ensemble forecast file not found for {ticker}. Using ARIMA forecast instead.")
        predict_stock(ticker)

def plot_model_forecast(ticker, model_type):
    """
    Plot forecast data for a specific model using the current visualization library.
    MODEL should be one of: arima, prophet, garch, lstm, gru, xgboost.
    """
    ticker = ticker.upper()
    model_type = model_type.lower().strip()
    if model_type not in ["arima", "prophet", "garch", "lstm", "gru", "xgboost"]:
        print(f"Unknown model type: {model_type}. Valid types: arima, prophet, garch, lstm, gru, xgboost")
        return
    filename = f"{ticker}_{model_type}_validation.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        filename = f"{ticker}_{model_type}_test.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"No forecast CSV found for model {model_type} and ticker {ticker}.")
            return
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notnull()]
    forecast_col = None
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            forecast_col = col
            break
    if forecast_col is None:
        print("No numeric forecast column found in the CSV.")
        return
    line_plot_dispatch(df.index, df[forecast_col],
                       title=f"{model_type.upper()} Forecast for {ticker}",
                       xlabel="Date", ylabel="Price", legend_label=f"{model_type.upper()} Forecast")

def plot_all(ticker):
    """
    Display multiple graphs in one multi-panel figure:
      - Raw price history
      - Technical indicators (if available)
      - Ensemble forecast (or fallback to ARIMA forecast)
    """
    ticker = ticker.upper()
    fig, axs = plt.subplots(3, 1, figsize=(12, 16))
    raw_file = os.path.join(OUTPUT_DIR, f"{ticker}_raw.csv")
    if os.path.exists(raw_file):
        df_raw = pd.read_csv(raw_file)
        if "Date" not in df_raw.columns:
            df_raw = df_raw.copy()
            df_raw.reset_index(inplace=True)
            if "index" in df_raw.columns:
                df_raw.rename(columns={"index": "Date"}, inplace=True)
        try:
            df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors='coerce')
            df_raw = df_raw[df_raw["Date"].notnull()]
            axs[0].plot(df_raw["Date"], df_raw["Close"], label="Close Price", color="blue")
            axs[0].set_title(f"{ticker} Raw Price History")
            axs[0].set_xlabel("Date")
            axs[0].set_ylabel("Price")
            axs[0].legend()
        except Exception as e:
            axs[0].text(0.5, 0.5, f"Error plotting raw data: {e}", transform=axs[0].transAxes)
    else:
        axs[0].text(0.5, 0.5, "Raw data not available", transform=axs[0].transAxes)
    feat_file = os.path.join(OUTPUT_DIR, f"{ticker}_features.csv")
    if os.path.exists(feat_file):
        df_feat = pd.read_csv(feat_file, index_col=0)
        try:
            df_feat.index = pd.to_datetime(df_feat.index, errors='coerce')
            df_feat = df_feat[df_feat.index.notnull()]
            for col in ["RSI", "MA20", "MA50", "MACD"]:
                if col in df_feat.columns:
                    axs[1].plot(df_feat.index, df_feat[col], label=col)
            axs[1].set_title(f"{ticker} Technical Indicators")
            axs[1].set_xlabel("Date")
            axs[1].set_ylabel("Value")
            axs[1].legend()
        except Exception as e:
            axs[1].text(0.5, 0.5, f"Error plotting technical indicators: {e}", transform=axs[1].transAxes)
    else:
        axs[1].text(0.5, 0.5, "Features data not available", transform=axs[1].transAxes)
    ensemble_file = os.path.join(OUTPUT_DIR, f"{ticker}_stacked_forecast.csv")
    if os.path.exists(ensemble_file):
        df_ens = pd.read_csv(ensemble_file, index_col=0)
        try:
            df_ens.index = pd.to_datetime(df_ens.index, errors='coerce')
            df_ens = df_ens[df_ens.index.notnull()]
            if "Stacked" in df_ens.columns:
                axs[2].plot(df_ens.index, df_ens["Stacked"], label="Ensemble Forecast", color="magenta")
            else:
                axs[2].plot(df_ens.index, df_ens.iloc[:, 0], label="Forecast", color="magenta")
            axs[2].set_title(f"{ticker} Ensemble Forecast")
            axs[2].set_xlabel("Date")
            axs[2].set_ylabel("Price")
            axs[2].legend()
        except Exception as e:
            axs[2].text(0.5, 0.5, f"Error plotting ensemble forecast: {e}", transform=axs[2].transAxes)
    else:
        try:
            pre_file = os.path.join(OUTPUT_DIR, f"{ticker}_preprocessed.csv")
            df_pre = pd.read_csv(pre_file, index_col=0)
            df_pre.index = pd.to_datetime(df_pre.index, errors='coerce')
            df_pre = df_pre[df_pre.index.notnull()]
            if "Close" in df_pre.columns:
                model = pm.auto_arima(df_pre["Close"], seasonal=False, error_action="ignore", suppress_warnings=True)
                forecast = model.predict(n_periods=30)
                last_date = df_pre.index[-1]
                forecast_dates = pd.bdate_range(last_date, periods=31)[1:]
                axs[2].plot(df_pre.index, df_pre["Close"], label="Historical Close", color="blue")
                axs[2].plot(forecast_dates, forecast, label="ARIMA Forecast", linestyle="--", color="red")
                axs[2].set_title(f"{ticker} ARIMA Forecast (Fallback)")
                axs[2].set_xlabel("Date")
                axs[2].set_ylabel("Price")
                axs[2].legend()
            else:
                axs[2].text(0.5, 0.5, "'Close' column not found in preprocessed data", transform=axs[2].transAxes)
        except Exception as e:
            axs[2].text(0.5, 0.5, f"Error in ARIMA fallback: {e}", transform=axs[2].transAxes)
    plt.tight_layout()
    plt.show()

def predict_stock(ticker):
    """
    Predict the closing prices for the next 30 business days using LSTM, GRU, and stacked forecast.
    First, tries to load a stacked forecast CSV file. If not available, then loads LSTM and GRU forecast CSV files,
    computes their average as a simple ensemble, and displays the result.
    """
    ticker = ticker.upper()
    stacked_file = os.path.join(OUTPUT_DIR, f"{ticker}_stacked_forecast.csv")
    if os.path.exists(stacked_file):
        df_stack = pd.read_csv(stacked_file, index_col=0)
        df_stack.index = pd.to_datetime(df_stack.index, errors='coerce')
        df_stack = df_stack[df_stack.index.notnull()]
        if "Stacked" in df_stack.columns:
            line_plot_dispatch(df_stack.index, df_stack["Stacked"],
                               title=f"Stacked Forecast for {ticker}",
                               xlabel="Date", ylabel="Price", legend_label="Stacked Forecast")
            print(f"\nStacked Forecast for the next 30 business days for {ticker}:")
            print(df_stack.to_string())
            return
        else:
            print("Stacked forecast column 'Stacked' not found in the CSV.")
    lstm_file = os.path.join(OUTPUT_DIR, f"{ticker}_lstm_test.csv")
    gru_file = os.path.join(OUTPUT_DIR, f"{ticker}_gru_test.csv")
    if os.path.exists(lstm_file) and os.path.exists(gru_file):
        df_lstm = pd.read_csv(lstm_file, index_col=0)
        df_gru = pd.read_csv(gru_file, index_col=0)
        df_lstm.index = pd.to_datetime(df_lstm.index, errors='coerce')
        df_gru.index = pd.to_datetime(df_gru.index, errors='coerce')
        df_lstm = df_lstm[df_lstm.index.notnull()]
        df_gru = df_gru[df_gru.index.notnull()]
        col_lstm = df_lstm.columns[0]
        col_gru = df_gru.columns[0]
        common_index = df_lstm.index.intersection(df_gru.index)
        df_lstm = df_lstm.loc[common_index]
        df_gru = df_gru.loc[common_index]
        avg_forecast = (df_lstm[col_lstm] + df_gru[col_gru]) / 2
        line_plot_dispatch(common_index, avg_forecast,
                           title=f"Average Forecast (LSTM & GRU) for {ticker}",
                           xlabel="Date", ylabel="Price", legend_label="Avg Forecast")
        combined_df = pd.DataFrame({"Avg Forecast": avg_forecast}, index=common_index)
        print(f"\nCombined Forecast (average of LSTM and GRU) for {ticker}:")
        print(combined_df.to_string())
        return
    print("No forecast data available from stacked, LSTM, or GRU models. Please run the forecasting pipeline.")

# ======================================================
# Command Functions (Text Output)
# ======================================================

def analyze_stock(ticker):
    """Display a comprehensive analysis of the stock (text output)."""
    ticker = ticker.upper()
    print(f"\n=== Comprehensive Analysis for {ticker} ===\n")
    print(">> Evaluation Metrics:")
    print(show_file(ticker, "evaluation"))
    print("\n>> Technical Indicators (Features):")
    print(show_file(ticker, "features"))
    print("\n>> Ensemble Forecast (Stacked Forecast):")
    print(show_file(ticker, "ensemble"))
    print("")

def chart_stock(ticker):
    """Display raw price history and volume charts (graph output)."""
    plot_raw_data(ticker)

def technical_stock(ticker):
    """Display technical indicator data (text output)."""
    ticker = ticker.upper()
    print(f"\n--- Technical Indicators for {ticker} ---")
    print(show_file(ticker, "features"))
    print("")

def forecast_stock(ticker):
    """Display ensemble forecast (text output)."""
    ticker = ticker.upper()
    print(f"\n--- Ensemble Forecast for {ticker} ---")
    print(show_file(ticker, "ensemble"))
    print("")

def evaluation_stock(ticker):
    """Display evaluation metrics (text output)."""
    ticker = ticker.upper()
    print(f"\n--- Evaluation Metrics for {ticker} ---")
    print(show_file(ticker, "evaluation"))
    print("")

def model_forecast_stock(ticker, model_type):
    """
    Display forecast data for a specific model (text output).
    Valid types: arima, prophet, garch, lstm, gru, xgboost.
    """
    ticker = ticker.upper()
    model_type = model_type.lower().strip()
    if model_type not in ["arima", "prophet", "garch", "lstm", "gru", "xgboost"]:
        print(f"Unknown model type: {model_type}. Valid types: arima, prophet, garch, lstm, gru, xgboost")
        return
    summary = show_file(ticker, f"{model_type} validation")
    if "not found" in summary:
        summary = show_file(ticker, f"{model_type} test")
    print(f"\n--- {model_type.upper()} Forecast for {ticker} ---")
    print(summary)
    print("")

def compare_stocks(ticker1, ticker2):
    """Compare two stocks by printing key figures (text output)."""
    ticker1 = ticker1.upper()
    ticker2 = ticker2.upper()
    file1 = os.path.join(OUTPUT_DIR, f"{ticker1}_raw.csv")
    file2 = os.path.join(OUTPUT_DIR, f"{ticker2}_raw.csv")
    if not os.path.exists(file1) or not os.path.exists(file2):
        print("Raw data for one or both tickers is not available.")
        return
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except Exception as e:
        print(f"Error reading raw data: {e}")
        return
    for df in [df1, df2]:
        if "Date" not in df.columns:
            df.reset_index(inplace=True)
            if "index" in df.columns:
                df.rename(columns={"index": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df[df["Date"].notnull()]
    last1 = df1["Close"].iloc[-1]
    last2 = df2["Close"].iloc[-1]
    print(f"\n=== Comparison: {ticker1} vs {ticker2} ===")
    print(f"{ticker1} latest close: {last1}")
    print(f"{ticker2} latest close: {last2}")
    if len(df1) >= 30 and len(df2) >= 30:
        change1 = (df1["Close"].iloc[-1] - df1["Close"].iloc[-30]) / df1["Close"].iloc[-30] * 100
        change2 = (df2["Close"].iloc[-1] - df2["Close"].iloc[-30]) / df2["Close"].iloc[-30] * 100
        print(f"{ticker1} 30-day change: {change1:.2f}%")
        print(f"{ticker2} 30-day change: {change2:.2f}%")
    else:
        print("Not enough data to calculate 30-day change.")
    print("")

# ======================================================
# Portfolio Management Functions
# ======================================================

def portfolio_add(ticker, shares):
    """Add a given number of shares for a stock to the portfolio."""
    ticker = ticker.upper()
    try:
        shares = float(shares)
    except ValueError:
        print("Invalid number of shares.")
        return
    portfolio[ticker] = portfolio.get(ticker, 0) + shares
    print(f"Added {shares} shares of {ticker} to your portfolio.\n")

def portfolio_show():
    """Display the current portfolio."""
    if not portfolio:
        print("Your portfolio is empty.\n")
    else:
        print("\n=== Your Portfolio ===")
        for t, s in portfolio.items():
            print(f"{t}: {s} shares")
        print("")

def portfolio_clear():
    """Clear your portfolio."""
    portfolio.clear()
    print("Portfolio cleared.\n")

# ======================================================
# New Commands for Visualization Library Control
# ======================================================

def process_set_command(tokens):
    if len(tokens) < 3:
        print("Usage: set library LIBRARY_NAME")
        return
    subcmd = tokens[1].lower()
    if subcmd == "library":
        lib = tokens[2]
        set_library(lib)
    else:
        print("Unknown set command. Use: set library LIBRARY_NAME")

def process_show_command(tokens):
    if len(tokens) < 2:
        print("Usage: show library")
        return
    subcmd = tokens[1].lower()
    if subcmd == "library":
        show_library()
    else:
        print("Unknown show command. Use: show library")

# ======================================================
# Help Message
# ======================================================

def print_help():
    help_text = """
Welcome to the Advanced Stock Analysis Chatbot!
Below are the available commands:

Basic Analysis:
  analyze TICKER
      → Show comprehensive analysis (evaluation metrics, technical indicators, ensemble forecast).
  raw TICKER
      → Display raw data (text).
  preprocessed TICKER
      → Display preprocessed data (text).
  features TICKER
      → Display technical indicator data (text).
  evaluation TICKER
      → Display evaluation metrics (text).
  ensemble TICKER or stacked TICKER
      → Display the stacked ensemble forecast data (text).

Advanced Model Forecasts:
  arima TICKER
  prophet TICKER
  garch TICKER
  lstm TICKER
  gru TICKER
  xgboost TICKER
      → Display forecast outputs from the specified model (text).
  forecast TICKER
      → Display overall ensemble forecast (text).

Visualization:
  technical TICKER
      → Display technical indicator data (text).
  chart TICKER
      → Display raw price history and volume charts (graph).
  plot raw TICKER
      → Plot raw close price data.
  plot technical TICKER
      → Plot technical indicators as a chart.
  plot evaluation TICKER
      → Plot evaluation metrics as a bar chart.
  plot compare TICKER1 TICKER2
      → Plot closing price comparison (last 60 days).
  plot forecast TICKER or plot stacked TICKER
      → Plot the ensemble (stacked) forecast (graph).
  plot model TICKER MODEL
      → Plot forecast output from a specific model (MODEL = arima, prophet, garch, lstm, gru, or xgboost).
  plot all TICKER
      → Display multiple graphs (raw data, technical indicators, and forecast) in one figure.

Prediction:
  predict TICKER
      → Predict the stock’s closing prices for the next 30 business days using LSTM, GRU and the stacked forecast (with a chart).

Comparison:
  compare TICKER1 TICKER2
      → Compare two stocks (text output).

Portfolio Management:
  portfolio add TICKER SHARES
      → Add stock to your portfolio.
  portfolio show
      → Display your portfolio.
  portfolio clear
      → Clear your portfolio.

Visualization Library Control:
  set library LIBRARY_NAME
      → Set the visualization library (matplotlib, seaborn, plotly, bokeh, altair, ggplot, holoviews, dash).
  show library
      → Show the currently selected visualization library.

Utility:
  list tickers
      → List available tickers (based on CSV files).
  list files for TICKER
      → List all CSV files available for a given ticker.

Other:
  help
      → Show this help message.
  quit or exit
      → Exit the chatbot.
"""
    print(help_text)

# ======================================================
# Command Processing and Chatbot Loop
# ======================================================

def process_command(command):
    tokens = command.strip().split()
    if not tokens:
        return
    cmd = tokens[0].lower()
    if cmd == "help":
        print_help()
    elif cmd == "set":
        process_set_command(tokens)
    elif cmd == "show":
        process_show_command(tokens)
    elif cmd == "list":
        if len(tokens) >= 2 and tokens[1].lower() == "tickers":
            tickers = list_tickers()
            if tickers:
                print("Available tickers:")
                for t in tickers:
                    print("  " + t)
            else:
                print("No tickers found in output.")
        elif len(tokens) >= 3 and tokens[1].lower() == "files" and tokens[2].lower() == "for":
            if len(tokens) < 4:
                print("Usage: list files for TICKER")
            else:
                ticker = tokens[3].upper()
                files = list_files_for_ticker(ticker)
                if files:
                    print(f"Files for {ticker}:")
                    for f in files:
                        print("  " + f)
                else:
                    print(f"No files found for {ticker}.")
        else:
            print("Usage: list tickers OR list files for TICKER")
    elif cmd == "analyze":
        if len(tokens) < 2:
            print("Usage: analyze TICKER")
        else:
            analyze_stock(tokens[1])
    elif cmd in ["raw", "preprocessed", "features", "evaluation", "ensemble", "stacked"]:
        if len(tokens) < 2:
            print(f"Usage: {cmd} TICKER")
        else:
            if cmd == "stacked":
                print(show_file(tokens[1], "ensemble"))
            else:
                print(show_file(tokens[1], cmd))
    elif cmd in ["arima", "prophet", "garch", "lstm", "gru", "xgboost"]:
        if len(tokens) < 2:
            print(f"Usage: {cmd} TICKER")
        else:
            model_forecast_stock(tokens[1], cmd)
    elif cmd == "forecast":
        if len(tokens) < 2:
            print("Usage: forecast TICKER")
        else:
            forecast_stock(tokens[1])
    elif cmd == "technical":
        if len(tokens) < 2:
            print("Usage: technical TICKER")
        else:
            technical_stock(tokens[1])
    elif cmd == "chart":
        if len(tokens) < 2:
            print("Usage: chart TICKER")
        else:
            chart_stock(tokens[1])
    elif cmd == "predict":
        if len(tokens) < 2:
            print("Usage: predict TICKER")
        else:
            predict_stock(tokens[1])
    elif cmd == "compare":
        if len(tokens) < 3:
            print("Usage: compare TICKER1 TICKER2")
        else:
            compare_stocks(tokens[1], tokens[2])
    elif cmd == "plot":
        if len(tokens) < 3:
            print("Usage: plot <type> TICKER or plot compare TICKER1 TICKER2 or plot model TICKER MODEL or plot all TICKER")
        else:
            subcmd = tokens[1].lower()
            if subcmd == "evaluation":
                if len(tokens) < 3:
                    print("Usage: plot evaluation TICKER")
                else:
                    plot_evaluation(tokens[2])
            elif subcmd == "technical":
                if len(tokens) < 3:
                    print("Usage: plot technical TICKER")
                else:
                    plot_technical_indicators(tokens[2])
            elif subcmd == "compare":
                if len(tokens) < 4:
                    print("Usage: plot compare TICKER1 TICKER2")
                else:
                    plot_compare_stocks(tokens[2], tokens[3])
            elif subcmd == "raw":
                if len(tokens) < 3:
                    print("Usage: plot raw TICKER")
                else:
                    plot_raw_data(tokens[2])
            elif subcmd in ["forecast", "stacked"]:
                if len(tokens) < 3:
                    print("Usage: plot forecast TICKER")
                else:
                    plot_ensemble_forecast(tokens[2])
            elif subcmd == "model":
                if len(tokens) < 4:
                    print("Usage: plot model TICKER MODEL")
                else:
                    plot_model_forecast(tokens[2], tokens[3])
            elif subcmd == "all":
                if len(tokens) < 3:
                    print("Usage: plot all TICKER")
                else:
                    plot_all(tokens[2])
            else:
                print("Unknown plot type. Valid types: evaluation, technical, compare, raw, forecast (or stacked), model, all")
    elif cmd == "portfolio":
        if len(tokens) < 2:
            print("Usage: portfolio [add/show/clear] ...")
        else:
            subcmd = tokens[1].lower()
            if subcmd == "add":
                if len(tokens) < 4:
                    print("Usage: portfolio add TICKER SHARES")
                else:
                    portfolio_add(tokens[2], tokens[3])
            elif subcmd == "show":
                portfolio_show()
            elif subcmd == "clear":
                portfolio_clear()
            else:
                print("Unknown portfolio command. Use: portfolio add/show/clear")
    elif cmd in ["quit", "exit"]:
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Unknown command. Type 'help' to see available commands.")

# Main chatbot loop to continuously accept and process user input
def chatbot():
    print("Welcome to the Advanced Stock Analysis Chatbot!")
    print("Type 'help' to see available commands.\n")
    while True:
        try:
            user_input = input("You: ")
            process_command(user_input)
        except Exception as e:
            print(f"An error occurred: {e}")

# Start the chatbot if this script is executed as the main program
if __name__ == "__main__":
    chatbot()