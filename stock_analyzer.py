import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging # Import logging

# --- Configuration ---
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA'] # Added a few more
MODELS_DIR = "models"
PREDICTION_DAYS = 30
MIN_TRAIN_DATA_POINTS = 50 # Minimum data points needed for training

# --- Logging Setup ---
# Configure logging to provide more details in the console/logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Stock Analyzer & Forecaster", layout="wide")
st.title("Stock Price Analysis and ARIMA Forecasting")

# --- Helper Functions ---

# Use st.cache_data for functions that return data (like dataframes, dicts)
@st.cache_data(ttl=3600, show_spinner="Fetching historical stock data...") # Cache data for 1 hour
def get_stock_data(ticker_symbol, period="max", interval="1d"):
    """Fetches historical stock data using yfinance."""
    logging.info(f"Attempting to fetch data for {ticker_symbol} with period={period}, interval={interval}")
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Adjust start date based on period for more reliable fetching
        end_date = datetime.now().date()
        start_date = None
        if period != "max":
             # Map user-friendly periods to timedelta/start dates if needed
             # yfinance handles '1d', '5d', '1mo', '6mo', '1y', '5y' directly with 'period'
             pass # yfinance handles these periods directly

        history = ticker.history(period=period, interval=interval, start=start_date, end=end_date)

        if history.empty:
            logging.warning(f"No historical data returned for {ticker_symbol} (period={period}).")
            st.error(f"Could not fetch historical data for {ticker_symbol}. The ticker might be invalid, delisted, or have no data for the selected period.")
            return pd.DataFrame() # Return empty DataFrame on error

        history.index = history.index.tz_localize(None) # Remove timezone for compatibility
        logging.info(f"Successfully fetched {len(history)} data points for {ticker_symbol}")
        return history.dropna(subset=['Close']) # Drop rows where 'Close' is NaN

    except Exception as e:
        logging.error(f"Error fetching historical data for {ticker_symbol}: {e}", exc_info=True)
        st.error(f"An error occurred while fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner="Fetching company information...") # Cache company info for 1 day
def get_company_info(ticker_symbol):
    """Fetches company information using yfinance."""
    logging.info(f"Attempting to fetch company info for {ticker_symbol}")
    try:
        ticker = yf.Ticker(ticker_symbol)
        # The .info attribute can sometimes be slow or fail; use basic_info for faster essentials
        info = ticker.info

        if not info: # Check if info dict is empty
             logging.warning(f"No company info returned for {ticker_symbol}")
             st.warning(f"Could not retrieve detailed company info for {ticker_symbol}.")
             return {}

        # Format values safely using .get() with defaults
        market_cap_raw = info.get('marketCap')
        pe_ratio_raw = info.get('trailingPE')
        dividend_yield_raw = info.get('dividendYield')
        low_52w = info.get('fiftyTwoWeekLow')
        high_52w = info.get('fiftyTwoWeekHigh')

        relevant_info = {
            "Company Name": info.get('longName', 'N/A'),
            "Symbol": info.get('symbol', ticker_symbol),
            "Sector": info.get('sector', 'N/A'),
            "Industry": info.get('industry', 'N/A'),
            "Market Cap": f"{market_cap_raw:,}" if market_cap_raw else 'N/A',
            "P/E Ratio": f"{pe_ratio_raw:.2f}" if pe_ratio_raw else 'N/A',
            "Dividend Yield": f"{dividend_yield_raw * 100:.2f}%" if dividend_yield_raw else 'N/A',
            "52 Week Range": f"{low_52w} - {high_52w}" if low_52w and high_52w else 'N/A',
            "Summary": info.get('longBusinessSummary', 'No summary available.')
        }
        logging.info(f"Successfully fetched company info for {ticker_symbol}")
        return relevant_info
    except Exception as e:
        logging.error(f"Error fetching company info for {ticker_symbol}: {e}", exc_info=True)
        st.error(f"Could not fetch company info for {ticker_symbol}: {e}")
        return {} # Return empty dict on error

def get_model_path(ticker_symbol):
    """Constructs the file path for a given ticker's model."""
    return os.path.join(MODELS_DIR, f"arima_model_{ticker_symbol}.pkl")

# Use st.session_state to manage model loading status and results per ticker
# Initialize keys if they don't exist
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {} # Store model, predictions, metrics per ticker
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None

# Use @st.cache_resource for things that shouldn't be re-created on every script run
# BUT loading models this way can be tricky if you switch tickers often.
# Let's manage loading/saving manually with session state for more control.
def load_model(ticker_symbol):
    """Loads a pre-trained ARIMA model for the ticker."""
    model_path = get_model_path(ticker_symbol)
    if os.path.exists(model_path):
        logging.info(f"Attempting to load model from {model_path}")
        try:
            model_data = joblib.load(model_path)
            logging.info(f"Model loaded successfully for {ticker_symbol}")
            # Store in session state cache
            st.session_state.model_cache[ticker_symbol] = {
                'model': model_data['model'],
                'mae': model_data.get('mae'), # Use .get for backward compatibility
                'rmse': model_data.get('rmse'),
                'order': model_data.get('order'),
                'predictions': None # Predictions need regeneration based on current data end date
            }
            return st.session_state.model_cache[ticker_symbol]
        except Exception as e:
            logging.error(f"Error loading model for {ticker_symbol} from {model_path}: {e}", exc_info=True)
            st.error(f"Error loading model file for {ticker_symbol}. It might be corrupted. Please retrain. Error: {e}")
            # Clean up corrupted file? Optional.
            # try: os.remove(model_path) except OSError: pass
            return None
    else:
        logging.info(f"No pre-trained model found at {model_path}")
        return None

# This function involves computation and should not be cached with st.cache_data
# as it depends on the *latest* data. Caching could lead to stale models.
def train_and_save_model(ticker_symbol, data):
    """Trains an ARIMA model using auto_arima and saves it."""
    model_path = get_model_path(ticker_symbol)
    close_prices = data['Close'].dropna() # Ensure no NaNs in the series

    if len(close_prices) < MIN_TRAIN_DATA_POINTS:
         st.warning(f"Not enough historical data ({len(close_prices)} points) for {ticker_symbol} to train a reliable model (minimum required: {MIN_TRAIN_DATA_POINTS}).")
         return None # Return None to indicate failure

    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_placeholder.info("Finding best ARIMA model using auto_arima...")
    logging.info(f"Starting auto_arima for {ticker_symbol}")

    try:
        # Consider using stepwise=False for a more thorough search, but it takes longer
        auto_model = pm.auto_arima(close_prices,
                                   start_p=1, start_q=1,
                                   test='adf', # Augmented Dickey-Fuller test to find 'd'
                                   max_p=5, max_q=5, # Limit search space
                                   m=1,              # No seasonality for daily stock data
                                   d=None,           # Let ADF test find 'd'
                                   seasonal=False,
                                   start_P=0, D=0,
                                   trace=False,      # Don't print stepwise results to console
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True,    # Faster search
                                   information_criterion='aic', # Use AIC for model selection
                                   n_jobs=-1)        # Use all available CPU cores

        best_order = auto_model.order
        logging.info(f"Auto ARIMA found best order {best_order} for {ticker_symbol}")
        progress_bar.progress(50)
        status_placeholder.info(f"Best model found (order={best_order}). Fitting final model...")

        # Fit the final ARIMA model with the best order found
        final_model = ARIMA(close_prices, order=best_order).fit()
        logging.info(f"Final ARIMA model fitted for {ticker_symbol}")
        progress_bar.progress(80)
        status_placeholder.info("Calculating in-sample metrics...")

        # Calculate in-sample predictions and metrics
        # Note: In-sample metrics are not a perfect measure of future performance
        in_sample_preds = final_model.predict(start=close_prices.index[0], end=close_prices.index[-1])
        # Ensure alignment before calculating metrics if indices differ slightly
        aligned_actuals, aligned_preds = close_prices.align(in_sample_preds, join='inner')
        mae = mean_absolute_error(aligned_actuals, aligned_preds)
        rmse = np.sqrt(mean_squared_error(aligned_actuals, aligned_preds))
        logging.info(f"In-sample metrics for {ticker_symbol} (Order: {best_order}): MAE={mae:.4f}, RMSE={rmse:.4f}")

        progress_bar.progress(90)
        status_placeholder.info("Saving model...")

        # Save the fitted model and metrics
        model_data_to_save = {
            'model': final_model,
            'mae': mae,
            'rmse': rmse,
            'order': best_order,
            'training_date': datetime.now().isoformat()
        }
        joblib.dump(model_data_to_save, model_path)
        logging.info(f"Model for {ticker_symbol} saved to {model_path}")

        progress_bar.progress(100)
        status_placeholder.success(f"Model trained (Order: {best_order}), evaluated (MAE: {mae:.4f}), and saved for {ticker_symbol}.")
        st.success(f"Model trained for {ticker_symbol} with order {best_order} and saved.")

        # Update session state immediately after training
        st.session_state.model_cache[ticker_symbol] = {
             'model': final_model,
             'mae': mae,
             'rmse': rmse,
             'order': best_order,
             'predictions': None # Reset predictions
        }
        return st.session_state.model_cache[ticker_symbol]

    except Exception as e:
        logging.error(f"Error training model for {ticker_symbol}: {e}", exc_info=True)
        st.error(f"Error during model training for {ticker_symbol}: {e}")
        # Attempt to remove potentially corrupted model file if save failed partially
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                logging.info(f"Removed potentially corrupted model file: {model_path}")
            except OSError as rm_err:
                 logging.error(f"Error removing model file {model_path} after training error: {rm_err}")
        # Clear cache for this ticker if training fails
        if ticker_symbol in st.session_state.model_cache:
            del st.session_state.model_cache[ticker_symbol]
        return None # Return None on failure
    finally:
        # Ensure progress bar and status message are removed
        progress_bar.empty()
        status_placeholder.empty()


def predict_prices(model_obj, steps):
    """Generates future price predictions using the fitted model."""
    if model_obj is None:
        st.error("Cannot predict: Model object is not available.")
        return None
    model = model_obj['model'] # Extract the actual model
    logging.info(f"Generating forecast for {steps} steps ahead.")
    try:
        forecast_result = model.get_forecast(steps=steps)
        # Use alpha=0.10 for 90% CI, alpha=0.05 for 95% CI
        forecast_summary = forecast_result.summary_frame(alpha=0.10)

        # Generate future dates starting from the day AFTER the last date in the model's data
        last_date = model.data.dates[-1]
        # Use pandas' Business Day frequency if possible, else fallback to daily
        try:
            # Ensure last_date is Timestamp if it's not
            if not isinstance(last_date, pd.Timestamp):
                 last_date = pd.Timestamp(last_date)
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')
            # Check if Bday frequency resulted in skipping (e.g., forecast starts Monday after Friday data)
            if forecast_dates[0].date() != (last_date + pd.Timedelta(days=1)).date() and \
               (last_date + pd.Timedelta(days=1)).weekday() < 5 : # Check if next day is weekday
                 # If B freq skipped a valid next weekday, maybe stick to calendar days
                 logging.warning("Business day frequency skipped expected next day. Falling back to daily frequency.")
                 forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')

        except ValueError: # Handle potential frequency issues
             logging.warning("Could not use 'B' frequency for date range. Falling back to 'D' (daily).")
             forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')

        # Ensure the number of dates matches the forecast steps
        if len(forecast_dates) != steps:
             logging.warning(f"Date range generated {len(forecast_dates)} dates, expected {steps}. Adjusting.")
             # If too many dates (e.g., due to freq='D'), trim
             forecast_dates = forecast_dates[:steps]
             # If too few (less likely but possible), regenerate with 'D' to be safe
             if len(forecast_dates) < steps:
                  forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')


        forecast_summary.index = forecast_dates[:steps] # Assign dates, ensure length matches forecast

        logging.info(f"Forecast generated successfully. Range: {forecast_summary.index.min()} to {forecast_summary.index.max()}")
        # Select relevant columns standardizing names
        return forecast_summary[['mean', 'mean_ci_lower', 'mean_ci_upper']]

    except Exception as e:
        logging.error(f"Error generating forecast: {e}", exc_info=True)
        st.error(f"Error generating forecast: {e}")
        return None

def plot_data(hist_data, forecast_df, ticker_symbol, range_str):
    """Creates an interactive Plotly chart of historical data and forecast."""
    fig = go.Figure()

    # --- Add Historical Data Trace ---
    # Ensure index is datetime for plotting
    hist_data.index = pd.to_datetime(hist_data.index)
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='royalblue', width=2),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>' # Custom hover text
    ))

    # --- Add Forecast Traces (if available) ---
    if forecast_df is not None and not forecast_df.empty:
        forecast_df.index = pd.to_datetime(forecast_df.index) # Ensure index is datetime

        # Forecasted Mean Line
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['mean'],
            mode='lines',
            name='Forecasted Mean',
            line=dict(color='darkorange', dash='dash', width=2),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Forecast: $%{y:.2f}<extra></extra>'
        ))

        # Confidence Interval Area (using fill='tonexty')
        # Important: Add Upper bound trace FIRST, then Lower bound trace for fill to work correctly
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['mean_ci_upper'],
            mode='lines',
            line=dict(width=0), # No line for the bounds themselves
            hoverinfo='skip', # Don't show hover for this trace itself
            showlegend=False # Hide from legend
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['mean_ci_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty', # Fill area to the trace defined previously (Upper bound)
            fillcolor='rgba(255, 165, 0, 0.2)', # Orange with transparency
            name='Confidence Interval (90%)', # Will appear below 'Forecasted Mean' in hover
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Lower CI: $%{y:.2f}<extra></extra>',
            showlegend=False # Hide from legend, maybe add annotation later if needed
        ))

    # --- Update Layout ---
    fig.update_layout(
        title=f'{ticker_symbol} - Historical Prices & {PREDICTION_DAYS}-Day ARIMA Forecast ({range_str} View)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified', # Show unified hover info across traces for a given x-value
        legend=dict(
            orientation="h", # Horizontal legend
            yanchor="bottom", y=1.02, # Position legend above chart
            xanchor="left", x=0
        ),
        margin=dict(l=50, r=20, t=60, b=20), # Adjust margins (left, right, top, bottom)
        xaxis_rangeslider_visible=False # Optional: Hide the range slider
    )
    # Add annotation for confidence interval if forecast exists
    if forecast_df is not None and not forecast_df.empty:
         fig.add_annotation(
             xref="paper", yref="paper", x=0.98, y=0.95, showarrow=False,
             text="Orange shaded area: 90% Confidence Interval",
             font=dict(size=10, color="grey"), align="right"
         )

    return fig

# --- Main Application Logic & UI ---

# --- Sidebar for Controls ---
st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", TICKERS, key="ticker_select")

# Update current ticker in session state if changed
if st.session_state.current_ticker != selected_ticker:
    logging.info(f"Ticker changed from {st.session_state.current_ticker} to {selected_ticker}")
    st.session_state.current_ticker = selected_ticker
    # Do NOT clear the entire model_cache here, just ensure the correct model is loaded/used later

# Time Range Selection for Chart Display
time_range_options = {
    # "1 Day": "1d", # 1d interval can be tricky, let's use larger periods
    "5 Days": "5d",
    "1 Month": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
    "Max": "max"
}
selected_range_str = st.sidebar.radio(
    "Select Time Range for Chart Display:",
    options=list(time_range_options.keys()),
    index=len(time_range_options) - 2, # Default to '5 Years'
    key="range_select"
)
selected_period = time_range_options[selected_range_str]

st.sidebar.markdown("---")
st.sidebar.header("ARIMA Model")
# Prediction horizon (days) - Make it configurable maybe?
st.sidebar.write(f"Forecast Horizon: **{PREDICTION_DAYS} days**")

# --- Main Panel ---
st.header(f"Analysis for: {selected_ticker}")

# --- Display Company Information ---
with st.spinner(f"Loading company details for {selected_ticker}..."):
    company_info = get_company_info(selected_ticker)

if company_info:
    st.subheader("Company Overview")
    col1, col2 = st.columns([1, 2]) # Adjust column widths
    with col1:
        st.metric("Company Name", company_info.get("Company Name", "N/A"))
        st.metric("Symbol", company_info.get("Symbol", "N/A"))
        st.metric("Sector", company_info.get("Sector", "N/A"))
        st.metric("Industry", company_info.get("Industry", "N/A"))
    with col2:
        st.metric("Market Cap", company_info.get("Market Cap", "N/A"))
        st.metric("P/E Ratio (TTM)", company_info.get("P/E Ratio", "N/A"))
        st.metric("Dividend Yield", company_info.get("Dividend Yield", "N/A"))
        st.metric("52 Week Range", company_info.get("52 Week Range", "N/A"))

    with st.expander("Business Summary"):
        st.markdown(company_info.get("Summary", "No summary available.")) # Use markdown for potentially long text
else:
    st.warning(f"Could not retrieve company information for {selected_ticker}.")


# --- Fetch Historical Data ---
# Fetch full history for potential training, and display-specific history for chart
# Using different cache keys implicitly by changing the 'period' argument
hist_data_full = get_stock_data(selected_ticker, period="max") # For training
hist_data_display = get_stock_data(selected_ticker, period=selected_period) # For display


# --- Model Training and Prediction Section ---
st.subheader("ARIMA Forecasting")
model_status_placeholder = st.container() # Use container for better layout control
metrics_placeholder = st.container()

# Check session state cache first
current_model_data = st.session_state.model_cache.get(selected_ticker)

if current_model_data is None:
    # If not in session state, try loading from disk
    current_model_data = load_model(selected_ticker) # This updates session_state if load is successful

# Display Model Status and Train Button
if current_model_data:
    model_status_placeholder.success(f"ARIMA model ready for {selected_ticker} (Order: {current_model_data.get('order', 'N/A')}).")
    mae = current_model_data.get('mae')
    rmse = current_model_data.get('rmse')
    if mae is not None and rmse is not None:
         col_m1, col_m2 = metrics_placeholder.columns(2)
         col_m1.metric(label="In-Sample MAE", value=f"${mae:.4f}", help="Mean Absolute Error on the data the model was trained on. Lower is generally better.")
         col_m2.metric(label="In-Sample RMSE", value=f"${rmse:.4f}", help="Root Mean Squared Error on the data the model was trained on. Lower is generally better, penalizes large errors more.")
    else:
         metrics_placeholder.info("Model loaded, but training metrics were not saved in this version.")

else:
    model_status_placeholder.warning(f"No pre-trained model found or loaded for {selected_ticker}. Training is required to generate forecasts.")

# Train Button Logic - Placed below status for clarity
if st.button(f"Train/Retrain ARIMA Model for {selected_ticker}", key=f"train_{selected_ticker}"):
    if hist_data_full is not None and not hist_data_full.empty:
        # Show spinner during training
        with st.spinner(f"Training ARIMA model for {selected_ticker}... This may take several minutes."):
            training_result = train_and_save_model(selected_ticker, hist_data_full)
        # Update the model data reference after training attempt
        current_model_data = st.session_state.model_cache.get(selected_ticker)
        # Rerun script slightly to update metrics display if training succeeded
        if training_result:
             st.experimental_rerun() # Force rerun to update MAE/RMSE display

    else:
        st.error(f"Cannot train model: Failed to fetch sufficient historical data for {selected_ticker}.")

# Generate Predictions if model is available
# Check if predictions are already generated and stored for the current model
if current_model_data and current_model_data.get('predictions') is None:
     with st.spinner(f"Generating {PREDICTION_DAYS}-day forecast for {selected_ticker}..."):
        predictions_df = predict_prices(current_model_data, PREDICTION_DAYS)
        # Store predictions back into the session state cache for this ticker
        if predictions_df is not None:
             st.session_state.model_cache[selected_ticker]['predictions'] = predictions_df
             logging.info(f"Predictions generated and cached for {selected_ticker}")
        else:
             logging.warning(f"Prediction generation failed for {selected_ticker}")
             # Ensure predictions key exists but is None if failed
             st.session_state.model_cache[selected_ticker]['predictions'] = None

# --- Display Chart ---
st.subheader(f"Price Chart & Forecast ({selected_range_str})")
chart_placeholder = st.empty() # Placeholder for the chart

if hist_data_display is not None and not hist_data_display.empty:
    # Get predictions from session state cache
    forecast_data_to_plot = current_model_data.get('predictions') if current_model_data else None

    fig = plot_data(hist_data_display, forecast_data_to_plot, selected_ticker, selected_range_str)
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # Optionally display the forecast data table below the chart
    if forecast_data_to_plot is not None and not forecast_data_to_plot.empty:
        with st.expander(f"View {PREDICTION_DAYS}-Day Forecast Data Table"):
             # Format dataframe for better display
             display_forecast_df = forecast_data_to_plot.copy()
             display_forecast_df.index = display_forecast_df.index.strftime('%Y-%m-%d') # Format date index
             display_forecast_df = display_forecast_df.rename(columns={
                 'mean': 'Forecasted Price',
                 'mean_ci_lower': 'Lower CI (90%)',
                 'mean_ci_upper': 'Upper CI (90%)'
             })
             st.dataframe(display_forecast_df.style.format("${:,.2f}")) # Format numbers as currency
    elif current_model_data:
         st.info("Forecast is being generated or model needs training.")
    # else: # No model loaded at all
         # Handled by the 'no pre-trained model' warning earlier


elif selected_ticker: # Only show warning if a ticker is actually selected
    chart_placeholder.warning(f"Could not retrieve historical stock data for {selected_ticker} for the selected range '{selected_range_str}'.")
else:
     chart_placeholder.info("Select a ticker symbol from the sidebar to begin.")

st.sidebar.markdown("---")
st.sidebar.info("Note: ARIMA forecasts are based purely on historical price patterns and do not account for news, events, or fundamental changes.")