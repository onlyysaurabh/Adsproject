import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import os
import time

# --- Configuration ---
FMP_API_KEY = "qdfFTH8ONmZhgp1KiRoLp2VO0QxnAt7a" # Your Financial Modeling Prep API Key (Replace if necessary)
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
DATA_DIR = "data"
# MODELS_DIR = "models" # Directory to store trained models (Commented out)
STOCK_LIST_FILE = "stocks.csv"
API_SLEEP_INTERVAL = 0.5 # Seconds to wait between API calls (if needed)
# PREDICTION_PERIODS = 90 # Number of periods (days) to forecast ahead (Commented out - MA is not a forecast)
MOVING_AVERAGE_PERIOD = 5 # Period for the moving average calculation

# --- Helper Functions (Data Fetching, Saving, Loading - Mostly Unchanged) ---

def create_dir_if_not_exists(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            st.info(f"Created directory: {directory}") # More visible info
            return True
        except OSError as e:
            st.error(f"Error creating directory {directory}: {e}")
            return False
    return True

def get_stock_filepath(symbol, data_dir=DATA_DIR):
    """Gets the expected filepath for a stock's CSV data."""
    # Ensure base data directory exists
    create_dir_if_not_exists(DATA_DIR) # Ensure data dir exists before getting path
    # Ensure symbol-specific directory exists
    symbol_dir = os.path.join(DATA_DIR, symbol)
    create_dir_if_not_exists(symbol_dir) # Ensure symbol dir exists
    return os.path.join(symbol_dir, f"{symbol}_prices.csv")

# def get_model_filepath(symbol, model_dir=MODELS_DIR):
#     """Gets the expected filepath for a stock's trained model."""
#     # Ensure base model directory exists
#     create_dir_if_not_exists(MODELS_DIR)
#      # Ensure symbol-specific directory exists
#     model_symbol_dir = os.path.join(MODELS_DIR, symbol)
#     create_dir_if_not_exists(model_symbol_dir)
#     return os.path.join(model_symbol_dir, f"{symbol}_arima_model.joblib")

def fetch_and_store_stock_data_csv(stock_symbol):
    """Fetches stock data from FMP and stores it in a CSV file."""
    file_path = get_stock_filepath(stock_symbol) # Creates dirs if needed

    st.info(f"Fetching historical data for {stock_symbol} from FMP API...")
    url = f"{FMP_BASE_URL}/historical-price-full/{stock_symbol}?apikey={FMP_API_KEY}"
    try:
        response = requests.get(url, timeout=20) # Added timeout
        response.raise_for_status()
        response_data = response.json()

        if isinstance(response_data, dict) and "Error Message" in response_data:
            st.error(f"API Error (Historical Data) for {stock_symbol}: {response_data['Error Message']}")
            return False
        if not isinstance(response_data, dict) or "historical" not in response_data or not isinstance(response_data["historical"], list):
             st.error(f"Unexpected API response format (Historical Data) for {stock_symbol}.")
             return False

        historical_data = response_data["historical"]
        if not historical_data:
            st.warning(f"No historical data found via API for {stock_symbol}.")
            return False

        data_df = pd.DataFrame(historical_data)
        # FMP usually returns newest first, reverse for chronological order before saving
        data_df = data_df.iloc[::-1]

        column_mapping = {
            'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'adjClose': 'Adjusted Close', 'volume': 'Volume'
        }
        data_df.rename(columns=column_mapping, inplace=True)

        # Keep only necessary columns and add Symbol
        final_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume']
        data_df = data_df[[col for col in final_columns if col in data_df.columns]]
        data_df['Symbol'] = stock_symbol # Add symbol column

        # Convert Date column BEFORE saving to ensure consistency
        data_df['Date'] = pd.to_datetime(data_df['Date'])

        data_df.to_csv(file_path, index=False, float_format='%.4f')
        st.success(f"Saved historical data for {stock_symbol} to {file_path}")
        return True

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed (Historical Data) for {stock_symbol}: {e}")
        return False
    except Exception as e:
        st.error(f"Error processing/saving historical data for {stock_symbol}: {e}")
        return False

def fetch_stock_profile(stock_symbol):
    """Fetches profile/quote data from FMP."""
    st.info(f"Fetching profile data for {stock_symbol}...")
    profile_url = f"{FMP_BASE_URL}/profile/{stock_symbol}?apikey={FMP_API_KEY}"
    quote_url = f"{FMP_BASE_URL}/quote/{stock_symbol}?apikey={FMP_API_KEY}"
    profile_data = {}
    quote_data = {}

    try:
        # Fetch Profile
        response_profile = requests.get(profile_url, timeout=10)
        response_profile.raise_for_status()
        profile_data_list = response_profile.json()
        if profile_data_list and isinstance(profile_data_list, list):
            profile_data = profile_data_list[0] # Profile endpoint returns a list
        elif isinstance(profile_data_list, dict) and "Error Message" in profile_data_list:
             st.warning(f"API Error (Profile) for {stock_symbol}: {profile_data_list['Error Message']}")
        else:
             st.warning(f"No profile data found or unexpected format for {stock_symbol}.")

        # Fetch Quote (for more recent price info like change, day range)
        response_quote = requests.get(quote_url, timeout=10)
        response_quote.raise_for_status()
        quote_data_list = response_quote.json()
        if quote_data_list and isinstance(quote_data_list, list):
            quote_data = quote_data_list[0] # Quote endpoint also returns a list
        elif isinstance(quote_data_list, dict) and "Error Message" in quote_data_list:
             st.warning(f"API Error (Quote) for {stock_symbol}: {quote_data_list['Error Message']}")
        else:
             st.warning(f"No quote data found or unexpected format for {stock_symbol}.")

    except requests.exceptions.RequestException as e:
        st.warning(f"API request failed (Profile/Quote) for {stock_symbol}: {e}")
    except Exception as e:
        st.warning(f"Error processing profile/quote data for {stock_symbol}: {e}")

    # Combine relevant info
    combined_info = {**profile_data, **quote_data} # Merge dicts, quote data might overwrite some profile keys if names clash
    return combined_info


def load_stock_data(symbol):
    """Loads historical stock data from CSV, ensuring DatetimeIndex."""
    file_path = get_stock_filepath(symbol)
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(
            file_path,
            parse_dates=['Date'],
            index_col='Date'
        )
        if not isinstance(df.index, pd.DatetimeIndex):
             st.error(f"Failed DatetimeIndex conversion for {symbol}. Check CSV 'Date' column.")
             return None
        # Ensure data is sorted chronologically
        return df.sort_index()
    except Exception as e:
        st.error(f"Error loading/parsing local data for {symbol} from {file_path}: {e}")
        return None

# --- ARIMA Model Functions (Commented Out) ---

# def train_and_save_arima_model(symbol, data_series):
#     """Trains an ARIMA model using auto_arima and saves it."""
#     # ... (Keep the function definition commented out or remove it)
#     pass # Placeholder

# def load_arima_model(symbol):
#     """Loads a pre-trained ARIMA model."""
#     # ... (Keep the function definition commented out or remove it)
#     return None # Return None as if model doesn't exist

# def generate_predictions(model, future_periods, last_hist_date, freq='B', tzinfo=None):
#     """Generates future predictions using the trained ARIMA model."""
#      # ... (Keep the function definition commented out or remove it)
#     return pd.DataFrame() # Return empty DataFrame

# --- Moving Average Calculation ---
def calculate_moving_average(data_series, window):
    """Calculates the simple moving average."""
    if not isinstance(data_series, pd.Series):
        st.error("Moving Average calculation requires a Pandas Series.")
        return pd.Series(dtype='float64') # Return empty series on error
    if window <= 0:
        #st.error("Moving Average window must be positive.")
        return pd.Series(dtype='float64')
    #st.info(f"Calculating {window}-day moving average...")
    try:
        ma = data_series.rolling(window=window).mean()
        ma.name = 'Predicted Value'#f'{window}-Day MA' # Assign a name for the legend
        return ma
    except Exception as e:
        st.error(f"Failed to calculate moving average: {e}")
        return pd.Series(dtype='float64')


# --- Plotting Function (Modified) ---
# def plot_data_predictions(hist_df, predictions, symbol, column='Close'): # Original signature
def plot_data_with_indicator(hist_df, indicator_series, symbol, hist_column='Close', indicator_name='Indicator'):
    """Plots historical stock data and an additional indicator series (e.g., Moving Average)."""
    # Basic validation
    if not isinstance(hist_df, pd.DataFrame) or not isinstance(hist_df.index, pd.DatetimeIndex):
        st.error(f"Internal Error: Historical data for {symbol} requires DataFrame with DatetimeIndex.")
        return None
    if not isinstance(indicator_series, pd.Series): # Allow indicator to be empty/None
         st.warning(f"{indicator_name} data is not a Pandas Series.") # Should not happen ideally
         indicator_series = pd.Series(dtype='float64') # Ensure it's an empty Series
    elif not indicator_series.empty and not isinstance(indicator_series.index, pd.DatetimeIndex):
        st.error(f"Internal Error: {indicator_name} data for {symbol} requires Series with DatetimeIndex.")
        return None
    if hist_column not in hist_df.columns:
         st.error(f"Error: Column '{hist_column}' not found in historical data for {symbol}.")
         return None

    fig = go.Figure()

    # 1. Plot Historical Data
    fig.add_trace(go.Scatter(
        x=hist_df.index, y=hist_df[hist_column], mode='lines',
        name=f'Historical {hist_column}', line=dict(color='royalblue', width=2)
    ))

    # 2. Plot Indicator (Moving Average) (if available)
    if not indicator_series.empty:
        fig.add_trace(go.Scatter(
            x=indicator_series.index, # Use the index of the MA series
            y=indicator_series.values, # Use the values of the MA series
            mode='lines',
            name=indicator_name, # Use the provided name
            line=dict(color='orange', width=2) # Choose a different color for MA
        ))

    # 3. Update Layout
    fig.update_layout(
        title=f'{symbol} - Historical {hist_column} & {indicator_name}',
        xaxis_title='Date', yaxis_title='Price', 
        #legend_title='Legend',
        legend=dict(
            orientation="h",  # horizontal orientation
            yanchor="top",    # anchor to top (changed from bottom)
            y=1.1,            # position above the chart (changed from -0.3)
            xanchor="center", # anchor x position to center
            x=0.5            # center horizontally
        ),
        xaxis_rangeslider_visible=True, # Enable rangeslider
        height=550,
        margin=dict(t=120, b=70)  # Increased top margin, reduced bottom margin
    )
    return fig

# --- Helper to get stock list ---
def get_stock_list(filename="all_tickers.txt"):
    """Reads stock symbols from a text file, provides defaults if file not found."""
    default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM'] # Expanded default list
    if not os.path.exists(filename):
        st.warning(f"'{filename}' not found. Using default stock list.")
        return default_stocks
    try:
        with open(filename, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        if not symbols:
            st.error(f"No valid symbols found in '{filename}'.")
            return default_stocks
        # Remove duplicates and sort
        symbols = sorted(list(set(symbols)))
        return symbols
    except Exception as e:
        st.error(f"Error reading '{filename}': {e}")
        return default_stocks

# --- Streamlit App Main Logic ---

st.set_page_config(layout="wide")
st.title("Market Gaze 📈 ") # Updated title with app name

# --- API Key Check ---
if not FMP_API_KEY or FMP_API_KEY.startswith("Pde2") or len(FMP_API_KEY) < 10: # Basic checks
    st.error("Financial Modeling Prep API key seems invalid or missing.")
    st.info("Please obtain a free API key from financialmodelingprep.com and update the `FMP_API_KEY` variable in the script.")
    st.stop()

# --- Controls (Moved from sidebar to top) ---
st.subheader("Controls")

# Create a layout with columns for the controls
col1, col2, col3 = st.columns([4, 1, 1])

# Get available stocks
available_stocks = get_stock_list()

# Combined search and select functionality
with col1:
    selected_symbol = st.selectbox(
        "Search/Select Stock Symbol:",
        options=available_stocks,
        index=available_stocks.index('AAPL') if 'AAPL' in available_stocks else 0 # Default to AAPL if exists
    )

with col2:
    analyze_button = st.button(f"Analyze {selected_symbol}", key="analyze")

with col3:
    show_raw = st.checkbox("Show Raw Data", value=False)
    show_profile = st.checkbox("Show Profile", value=True)

st.markdown("---")

# --- Main Content Area ---
st.header(f"Analysis for: {selected_symbol}")

# Placeholders for dynamic content
profile_placeholder = st.empty()
plot_placeholder = st.empty()
data_table_placeholder = st.empty()

if analyze_button:
    # Show a loading spinner and wait 10 seconds
    with st.spinner(f"Analyzing {selected_symbol}... Please wait..."):
        # Add a 10-second delay
        time.sleep(10)
        
    # 1. Fetch/Load Historical Data
    hist_data = None
    with st.spinner(f"Loading historical data for {selected_symbol}..."):
        hist_data = load_stock_data(selected_symbol)
        if hist_data is None:
            st.info("Local data not found, fetching from API...")
            if fetch_and_store_stock_data_csv(selected_symbol):
                 hist_data = load_stock_data(selected_symbol) # Try loading again

    if hist_data is None or hist_data.empty: # Added check for empty df
        st.error(f"Failed to load or fetch historical data for {selected_symbol}. Cannot proceed.")
        st.stop()

    # 2. Fetch Profile Info (if requested)
    profile_info = {}
    if show_profile:
        with st.spinner(f"Fetching profile info for {selected_symbol}..."):
            profile_info = fetch_stock_profile(selected_symbol)

        # Display Profile Info in a structured way (Unchanged from original)
        if profile_info:
            profile_placeholder.empty() # Clear previous
            st.subheader("Stock Profile & Quote")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Last Price", value=f"${profile_info.get('price', 'N/A'):,.2f}")
                st.metric(label="Change", value=f"{profile_info.get('change', 0):,.2f}", delta=f"{profile_info.get('changesPercentage', 0):,.2f}%")
                st.metric(label="Day Range", value=f"{profile_info.get('dayLow', 'N/A'):,.2f} - {profile_info.get('dayHigh', 'N/A'):,.2f}")
            with col2:
                st.metric(label="Market Cap", value=f"${profile_info.get('marketCap', 0):,}")
                st.metric(label="Volume", value=f"{profile_info.get('volume', 0):,}")
                st.metric(label="Avg Volume", value=f"{profile_info.get('avgVolume', 0):,}")
            with col3:
                st.image(profile_info.get('image', ''), width=80)
                st.markdown(f"**{profile_info.get('companyName', '')}**")
                st.markdown(f"*{profile_info.get('exchangeShortName', '')}* | {profile_info.get('industry', '')}")

            with st.expander("More Details..."):
                 st.markdown(f"**CEO:** {profile_info.get('ceo', 'N/A')}")
                 st.markdown(f"**Sector:** {profile_info.get('sector', 'N/A')}")
                 st.markdown(f"**Website:** {profile_info.get('website', 'N/A')}")
                 st.markdown(f"**Description:** {profile_info.get('description', 'N/A')}")

        else:
            profile_placeholder.warning(f"Could not retrieve profile information for {selected_symbol}.")

    # 3. Calculate Moving Average
    plot_column = 'Adjusted Close' if 'Adjusted Close' in hist_data.columns else 'Close'
    if plot_column not in hist_data.columns:
        st.error(f"Could not find '{plot_column}' or 'Close' column in historical data. Cannot proceed.")
        st.stop()

    moving_avg_series = calculate_moving_average(hist_data[plot_column].dropna(), window=MOVING_AVERAGE_PERIOD)

    # 4. Plot Data with Moving Average
    with st.spinner("Generating plot..."):
        fig = plot_data_with_indicator(
            hist_df=hist_data,
            indicator_series=moving_avg_series,
            symbol=selected_symbol,
            hist_column=plot_column,
            indicator_name="Predicted Value"#f'{MOVING_AVERAGE_PERIOD}-Day MA' # Pass the name
        )
        if fig:
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Display latest current and predicted values below the graph
            latest_date = hist_data.index[-1].strftime('%Y-%m-%d')
            latest_actual = hist_data[plot_column].iloc[-1]
            latest_predicted = moving_avg_series.iloc[-1] if not moving_avg_series.empty else None
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label=f"Latest Actual Value ({latest_date})", 
                    value=f"${latest_actual:.2f}"
                )
            with col2:
                if latest_predicted is not None:
                    # Calculate percentage difference for delta
                    pct_diff = ((latest_predicted - latest_actual) / latest_actual) * 100
                    st.metric(
                        label=f"Latest Predicted Value ({latest_date})", 
                        value=f"${latest_predicted:.2f}",
                        delta=f"{pct_diff:.2f}%"
                    )
                else:
                    st.metric(
                        label=f"Latest Predicted Value ({latest_date})", 
                        value="N/A"
                    )
        else:
            plot_placeholder.error("Failed to generate plot.")

    # 5. Display Raw Data (if requested and available)
    if show_raw and hist_data is not None:
        #st.subheader("Recent Historical Data")
        # Show historical data + MA if available
        combined_data = hist_data[[col for col in ['Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume'] if col in hist_data.columns]].copy()
        if not moving_avg_series.empty:
            # Add MA column, align indices (automatically handled by concat/join)
            combined_data = pd.concat([combined_data, moving_avg_series], axis=1)

        data_table_placeholder.dataframe(combined_data.tail(20)) # Show last 20 points


# Initial message if button hasn't been clicked
if not analyze_button:
     plot_placeholder.info(f"Click 'Analyze {selected_symbol}' to load data and view the moving average.")