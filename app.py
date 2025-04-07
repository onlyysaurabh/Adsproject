import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import os
import time
import joblib # For saving/loading models
import pmdarima as pm # For auto-ARIMA
from pmdarima import model_selection # For train/test split if needed later

# --- Configuration ---
FMP_API_KEY = "qdfFTH8ONmZhgp1KiRoLp2VO0QxnAt7a"#"Pde28rQkeMw9IQlvQ7tfIyikvFCVM2zr" # Your Financial Modeling Prep API Key
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
DATA_DIR = "data"
MODELS_DIR = "models" # Directory to store trained models
STOCK_LIST_FILE = "stocks.csv"
API_SLEEP_INTERVAL = 0.5 # Seconds to wait between API calls (if needed)
PREDICTION_PERIODS = 90 # Number of periods (days) to forecast ahead

# --- Helper Functions (Data Fetching, Saving, Loading - Modified for clarity) ---

def create_dir_if_not_exists(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            st.sidebar.info(f"Created directory: {directory}") # More visible info
            return True
        except OSError as e:
            st.sidebar.error(f"Error creating directory {directory}: {e}")
            return False
    return True

def get_stock_filepath(symbol, data_dir=DATA_DIR):
    """Gets the expected filepath for a stock's CSV data."""
    return os.path.join(data_dir, symbol, f"{symbol}_prices.csv")

def get_model_filepath(symbol, model_dir=MODELS_DIR):
    """Gets the expected filepath for a stock's trained model."""
    return os.path.join(model_dir, symbol, f"{symbol}_arima_model.joblib")

def fetch_and_store_stock_data_csv(stock_symbol):
    """Fetches stock data from FMP and stores it in a CSV file."""
    # Ensure base data directory exists
    if not create_dir_if_not_exists(DATA_DIR): return False
    # Ensure symbol-specific directory exists
    symbol_dir = os.path.join(DATA_DIR, stock_symbol)
    if not create_dir_if_not_exists(symbol_dir): return False

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

        file_path = get_stock_filepath(stock_symbol)
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

# --- ARIMA Model Functions ---

def train_and_save_arima_model(symbol, data_series):
    """Trains an ARIMA model using auto_arima and saves it."""
    if not isinstance(data_series, pd.Series):
        st.error("Training function requires a Pandas Series.")
        return False
    if data_series.isnull().any():
        st.warning("Data contains null values. Attempting to fill with forward fill.")
        data_series = data_series.ffill().bfill() # Forward fill then back fill remaining NaNs
    if data_series.isnull().any():
        st.error("Could not fill all null values. Training aborted.")
        return False
    if len(data_series) < 30: # Need sufficient data for ARIMA
        st.error(f"Insufficient data ({len(data_series)} points) to train ARIMA model for {symbol}.")
        return False

    st.info(f"Starting ARIMA model training for {symbol} (this might take some time)...")
    status_placeholder = st.empty()
    status_placeholder.info("Running auto_arima...")

    try:
        # Use auto_arima to find the best ARIMA model parameters
        # Train on the entire history for forecasting
        model = pm.auto_arima(data_series,
                              start_p=1, start_q=1,
                              test='adf',       # Use ADF test to find 'd'
                              max_p=5, max_q=5, # Max non-seasonal orders
                              m=1,              # Non-seasonal data
                              d=None,           # Let ADF test determine 'd'
                              seasonal=False,   # No seasonality for daily stock data usually
                              start_P=0, D=0,
                              trace=False,      # Don't print intermediate results
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)    # Use stepwise algorithm for speed

        status_placeholder.success(f"Auto ARIMA found best model: {model.order}")
        # print(model.summary()) # Optional: print summary to console

        # Ensure model directory exists
        model_symbol_dir = os.path.join(MODELS_DIR, symbol)
        if not create_dir_if_not_exists(MODELS_DIR): return False
        if not create_dir_if_not_exists(model_symbol_dir): return False

        # Save the trained model
        model_filepath = get_model_filepath(symbol)
        joblib.dump(model, model_filepath)
        st.success(f"ARIMA model for {symbol} trained and saved to {model_filepath}")
        return True

    except Exception as e:
        status_placeholder.error(f"ARIMA training failed for {symbol}: {e}")
        import traceback
        print(traceback.format_exc()) # Print detailed error to console for debugging
        return False

def load_arima_model(symbol):
    """Loads a pre-trained ARIMA model."""
    model_filepath = get_model_filepath(symbol)
    if not os.path.exists(model_filepath):
        return None
    try:
        model = joblib.load(model_filepath)
        # st.info(f"Loaded pre-trained ARIMA model for {symbol}") # Removed for cleaner UI
        return model
    except Exception as e:
        st.error(f"Error loading model for {symbol} from {model_filepath}: {e}")
        return None

def generate_predictions(model, future_periods, last_hist_date, freq='B', tzinfo=None):
    """Generates future predictions using the trained ARIMA model."""
    st.info(f"Generating {future_periods}-day forecast...")
    try:
        # Use model.predict for pmdarima models
        forecast_values = model.predict(n_periods=future_periods)

        # Create future dates
        future_dates = pd.date_range(
            start=last_hist_date + pd.Timedelta(days=1),
            periods=future_periods,
            freq=freq, # Business day frequency default
            tz=tzinfo # Match timezone of historical data if it exists
        )

        # Create predictions DataFrame
        predictions_df = pd.DataFrame(
            {'Predicted Close': forecast_values},
            index=future_dates
        )
        predictions_df.index.name = 'Date'
        st.success(f"Generated {len(predictions_df)} forecast points.")
        return predictions_df

    except Exception as e:
        st.error(f"Failed to generate predictions: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure


# --- Plotting Function ---
def plot_data_predictions(hist_df, predictions, symbol, column='Close'):
    """Plots historical stock data and model predictions."""
    # Basic validation
    if not isinstance(hist_df, pd.DataFrame) or not isinstance(hist_df.index, pd.DatetimeIndex):
        st.error(f"Internal Error: Historical data for {symbol} requires DataFrame with DatetimeIndex.")
        return None
    if not isinstance(predictions, pd.DataFrame): # Allow predictions to be empty
         st.warning("Predictions data is not a DataFrame.") # Should not happen ideally
         predictions = pd.DataFrame() # Ensure it's an empty DF
    elif not predictions.empty and not isinstance(predictions.index, pd.DatetimeIndex):
        st.error(f"Internal Error: Predictions data for {symbol} requires DataFrame with DatetimeIndex.")
        return None
    if column not in hist_df.columns:
         st.error(f"Error: Column '{column}' not found in historical data for {symbol}.")
         return None

    fig = go.Figure()

    # 1. Plot Historical Data
    fig.add_trace(go.Scatter(
        x=hist_df.index, y=hist_df[column], mode='lines',
        name=f'Historical {column}', line=dict(color='royalblue', width=2)
    ))

    # 2. Plot Predictions (if available)
    if not predictions.empty:
        is_future_pred = False
        pred_column_name = predictions.columns[0]
        pred_label = f'{pred_column_name} (ARIMA Forecast)'

        # Check if predictions start after historical data ends (basic check)
        if not hist_df.empty and predictions.index.min() > hist_df.index.max():
            is_future_pred = True

        fig.add_trace(go.Scatter(
            x=predictions.index, y=predictions[pred_column_name], mode='lines',
            name=pred_label, line=dict(color='red', dash='dot', width=2)
        ))
        # Add markers for forecast points for visibility
        fig.add_trace(go.Scatter(
            x=predictions.index, y=predictions[pred_column_name], mode='markers',
            name='Forecast Points', marker=dict(color='red', size=4), showlegend=False
        ))

    # 3. Update Layout
    fig.update_layout(
        title=f'{symbol} - Historical {column} & ARIMA Forecast',
        xaxis_title='Date', yaxis_title='Price', legend_title='Legend',
        xaxis_rangeslider_visible=True, # Enable rangeslider
        height=500 # Adjust height
    )
    return fig

# --- Helper to get stock list ---
def get_stock_list(filename=STOCK_LIST_FILE):
    """Reads stock symbols from a CSV file, provides defaults if file not found."""
    default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM'] # Expanded default list
    if not os.path.exists(filename):
        st.sidebar.warning(f"'{filename}' not found. Using default stock list.")
        return default_stocks
    try:
        df = pd.read_csv(filename)
        if 'Symbol' not in df.columns:
            st.sidebar.error(f"'{filename}' must contain a 'Symbol' column.")
            return default_stocks
        symbols = sorted(df['Symbol'].astype(str).str.strip().str.upper().unique().tolist())
        if not symbols:
             st.sidebar.error(f"No valid symbols found in '{filename}'.")
             return default_stocks
        # st.sidebar.info(f"Loaded {len(symbols)} symbols from '{filename}'.") # Less verbose
        return symbols
    except Exception as e:
        st.sidebar.error(f"Error reading '{filename}': {e}")
        return default_stocks

# --- Streamlit App Main Logic ---

st.set_page_config(layout="wide")
st.title("📈 Stock Analysis & ARIMA Forecast")

# --- API Key Check ---
if not FMP_API_KEY or FMP_API_KEY == "Pde28rQkeMw9IQlvQ7tfIyikvFCVM2zr" or len(FMP_API_KEY) < 10: # Basic length check
    st.error("Financial Modeling Prep API key seems invalid or missing.")
    st.info("Please obtain a free API key from financialmodelingprep.com and update the `FMP_API_KEY` variable in the script.")
    # Optionally, allow user input for API key if not hardcoded:
    # FMP_API_KEY = st.text_input("Enter FMP API Key:", type="password")
    # if not FMP_API_KEY: st.stop()
    st.stop()


# --- Sidebar ---
st.sidebar.header("Controls")
available_stocks = get_stock_list()
selected_symbol = st.sidebar.selectbox(
    "Select Stock Symbol:",
    options=available_stocks,
    index=available_stocks.index('AAPL') if 'AAPL' in available_stocks else 0 # Default to AAPL if exists
)

# Check if model exists for the selected symbol
model_exists = os.path.exists(get_model_filepath(selected_symbol))
train_button_placeholder = st.sidebar.empty()

if not model_exists:
    st.sidebar.warning(f"No pre-trained ARIMA model found for {selected_symbol}.")
    if train_button_placeholder.button(f"Train ARIMA Model for {selected_symbol}", key=f"train_{selected_symbol}"):
        # --- Training Logic ---
        st.subheader(f"🏋️ Training Model for {selected_symbol}")
        hist_data = load_stock_data(selected_symbol)
        if hist_data is None:
            st.info("Historical data not found locally, fetching...")
            if fetch_and_store_stock_data_csv(selected_symbol):
                hist_data = load_stock_data(selected_symbol)
            else:
                st.error("Failed to fetch historical data. Cannot train model.")
                st.stop()

        if hist_data is not None and 'Adjusted Close' in hist_data.columns:
            # Use 'Adjusted Close' for training as it accounts for splits/dividends
            data_to_train = hist_data['Adjusted Close'].dropna()
            with st.spinner("Training in progress..."):
                train_success = train_and_save_arima_model(selected_symbol, data_to_train)
            if train_success:
                # Trigger a rerun to update the sidebar (remove train button)
                st.rerun()
            else:
                st.error("Model training failed. Check errors above.")
        else:
            st.error("Could not load valid historical data ('Adjusted Close') for training.")
else:
    st.sidebar.success(f"✅ Pre-trained ARIMA model found for {selected_symbol}.")
    train_button_placeholder.empty() # Remove button placeholder if model exists

st.sidebar.markdown("---")
# Main action button
analyze_button = st.sidebar.button(f"Analyze & Predict {selected_symbol}", key="analyze")
st.sidebar.markdown("---")
# Optional Raw Data Display
show_raw = st.sidebar.checkbox("Show Raw Data Sample", value=False)
show_profile = st.sidebar.checkbox("Show Stock Profile Info", value=True)


# --- Main Content Area ---
st.header(f"Analysis for: {selected_symbol}")

# Placeholders for dynamic content
profile_placeholder = st.empty()
plot_placeholder = st.empty()
data_table_placeholder = st.empty()

if analyze_button:
    # 1. Fetch/Load Historical Data
    hist_data = None
    with st.spinner(f"Loading historical data for {selected_symbol}..."):
        hist_data = load_stock_data(selected_symbol)
        if hist_data is None:
            st.info("Local data not found, fetching from API...")
            if fetch_and_store_stock_data_csv(selected_symbol):
                 hist_data = load_stock_data(selected_symbol) # Try loading again

    if hist_data is None:
        st.error(f"Failed to load or fetch historical data for {selected_symbol}. Cannot proceed.")
        st.stop()

    # 2. Fetch Profile Info (if requested)
    profile_info = {}
    if show_profile:
        with st.spinner(f"Fetching profile info for {selected_symbol}..."):
            profile_info = fetch_stock_profile(selected_symbol)

        # Display Profile Info in a structured way
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


    # 3. Load Model & Generate Predictions
    predictions = pd.DataFrame() # Initialize empty
    model = load_arima_model(selected_symbol)
    if model:
        with st.spinner(f"Generating {PREDICTION_PERIODS}-day forecast..."):
            # Determine frequency and timezone from historical data
            hist_freq = getattr(hist_data.index, 'freq', 'B') # Default to Business Day if no freq
            if hist_freq is None: hist_freq = 'B' # Handle None case explicitly
            hist_tz = hist_data.index.tz # Get timezone info

            predictions = generate_predictions(
                model,
                future_periods=PREDICTION_PERIODS,
                last_hist_date=hist_data.index.max(),
                freq=hist_freq,
                tzinfo=hist_tz
            )
    else:
        st.warning(f"No trained model loaded for {selected_symbol}. Train the model using the sidebar button to see forecasts.")

    # 4. Plot Data
    with st.spinner("Generating plot..."):
        plot_column = 'Adjusted Close' if 'Adjusted Close' in hist_data.columns else 'Close'
        fig = plot_data_predictions(hist_data, predictions, selected_symbol, column=plot_column)
        if fig:
            plot_placeholder.plotly_chart(fig, use_container_width=True)
        else:
            plot_placeholder.error("Failed to generate plot.")

    # 5. Display Raw Data (if requested and available)
    if show_raw and hist_data is not None:
        st.subheader("Recent Historical Data")
        # Show historical data + predictions if available
        combined_data = hist_data[[col for col in ['Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume'] if col in hist_data.columns]].copy()
        if not predictions.empty:
            # Add prediction column, align indices if possible (might create NaNs in hist part)
            combined_data = pd.concat([combined_data, predictions], axis=1)

        data_table_placeholder.dataframe(combined_data.tail(15)) # Show last 15 points


# Initial message if button hasn't been clicked
if not analyze_button:
     plot_placeholder.info(f"Click 'Analyze & Predict {selected_symbol}' in the sidebar to load data and generate forecasts.")