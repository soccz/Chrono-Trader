import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from utils.config import config
from utils.logger import logger
from data.database import load_data, get_db_connection

FUTURE_WINDOW_SIZE = 6
MAJOR_COINS = ['KRW-BTC', 'KRW-ETH'] # Coins to build the market index

def get_market_weights() -> dict:
    """
    Dynamically calculates market weights for major coins based on recent trading value.
    """
    logger.info("Dynamically calculating market weights based on recent 30-day trading value...")
    weights = {}
    total_value = 0
    
    try:
        for coin in MAJOR_COINS:
            query = f"""
                SELECT AVG(close * volume) 
                FROM crypto_data 
                WHERE market = '{coin}' 
                AND timestamp >= date('now', '-30 days')
            """
            avg_value_df = load_data(query)
            if avg_value_df.empty or avg_value_df.iloc[0, 0] is None:
                logger.warning(f"Could not calculate average trading value for {coin}. It will be excluded from dynamic weighting.")
                avg_value = 0
            else:
                avg_value = avg_value_df.iloc[0, 0]
            
            weights[coin] = avg_value
            total_value += avg_value

        if total_value > 0:
            for coin in weights:
                weights[coin] = weights[coin] / total_value
            logger.info(f"Calculated dynamic weights: {weights}")
            return weights
            
    except Exception as e:
        logger.error(f"Failed to calculate dynamic market weights: {e}")

    # Fallback to default weights if calculation fails
    logger.warning("Using default market cap weights (70/30).")
    return {'KRW-BTC': 0.7, 'KRW-ETH': 0.3}

def get_market_index() -> pd.DataFrame:
    """
    Calculates a market-cap weighted market index based on major coins.
    Weights are now calculated dynamically.
    """
    logger.info(f"Calculating market index from {MAJOR_COINS}...")
    
    # Get dynamic weights instead of using hardcoded ones
    market_weights = get_market_weights()
    index_df = pd.DataFrame()
    
    try:
        for coin in MAJOR_COINS:
            query = f"SELECT timestamp, close FROM crypto_data WHERE market = '{coin}' ORDER BY timestamp ASC"
            df = load_data(query)
            if df.empty:
                logger.warning(f"No data for {coin} to calculate market index.")
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df[f'{coin}_pct_change'] = df['close'].pct_change()
            
            if index_df.empty:
                index_df = df[[f'{coin}_pct_change']]
            else:
                index_df = index_df.join(df[[f'{coin}_pct_change']], how='outer')
        
        if index_df.empty or not all(f'{c}_pct_change' in index_df.columns for c in MAJOR_COINS):
            logger.warning("Could not calculate market index, data missing for major coins.")
            return pd.DataFrame()

        # Fill NaNs before calculation
        index_df.fillna(0, inplace=True)

        # Calculate weighted average using dynamic weights
        index_df['market_index_return'] = (index_df[f'{MAJOR_COINS[0]}_pct_change'] * market_weights[MAJOR_COINS[0]] +
                                           index_df[f'{MAJOR_COINS[1]}_pct_change'] * market_weights[MAJOR_COINS[1]])
        
        logger.info("Market-cap weighted index calculated successfully.")
        return index_df[['market_index_return']]

    except Exception as e:
        logger.error(f"Failed to calculate market index: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators for the given data using pandas-ta."""
    # Calculate all indicators and store them temporarily
    rsi = ta.rsi(df['close'])
    macd = ta.macd(df['close'])
    adx = ta.adx(df['high'], df['low'], df['close'])
    obv = ta.obv(df['close'], df['volume'])
    bbands = ta.bbands(df['close'], length=20)
    volume_ma = ta.sma(df['volume'], length=20)

    # Assign to the dataframe with the exact column names the project expects
    df['rsi'] = rsi
    if macd is not None and not macd.empty:
        df['macd'] = macd['MACD_12_26_9']
        df['macdsignal'] = macd['MACDs_12_26_9']
        df['macdhist'] = macd['MACDh_12_26_9']
    if adx is not None and not adx.empty:
        df['adx'] = adx['ADX_14']
    df['obv'] = obv
    if bbands is not None and not bbands.empty and bbands.shape[1] >= 3:
        # Access by position for robustness against naming changes
        df['bb_lower'] = bbands.iloc[:, 0]  # Lower band
        df['bb_middle'] = bbands.iloc[:, 1] # Middle band
        df['bb_upper'] = bbands.iloc[:, 2]  # Upper band

    df['volume_ma'] = volume_ma

    df.fillna(0, inplace=True)
    return df

def create_sequences(data, sequence_length, future_window):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - future_window):
        x = data[i:(i + sequence_length)]
        y = data[(i + sequence_length):(i + sequence_length + future_window), 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_processed_data(market: str, market_index_df: pd.DataFrame):
    """Loads, preprocesses, and prepares data, now including Alpha and Beta."""
    logger.info(f"Processing data for market: {market}")
    
    query = f"SELECT * FROM crypto_data WHERE market = '{market}' ORDER BY timestamp ASC"
    df = load_data(query)
    if df.empty:
        logger.warning(f"No data found for market {market}. Skipping.")
        return None, None, None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    df = calculate_technical_indicators(df)

    # Join the market-cap weighted index
    df = df.join(market_index_df, how='left')
    df['market_index_return'] = df['market_index_return'].fillna(0)

    # --- Calculate Alpha and Beta ---
    df['coin_return'] = df['close'].pct_change().fillna(0)
    beta_window = 30 # 30-hour rolling window for beta calculation

    # Rolling covariance between coin and market
    rolling_cov = df['coin_return'].rolling(window=beta_window).cov(df['market_index_return'])
    # Rolling variance of the market
    rolling_var = df['market_index_return'].rolling(window=beta_window).var()

    df['beta'] = (rolling_cov / rolling_var).fillna(0)
    df['alpha'] = (df['coin_return'] - (df['beta'] * df['market_index_return'])).fillna(0)
    # --- End Alpha and Beta Calculation ---

    df['future_pct_change'] = (df['close'].shift(-1) - df['close']) / df['close']
    df.fillna(0, inplace=True)

    features_to_scale = [
        'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'adx', 'obv', 
        'market_index_return', 'bb_upper', 'bb_middle', 'bb_lower', 'volume_ma',
        'alpha', 'beta' # Add new features
    ]
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features_to_scale])

    data_for_sequences = np.c_[df[['future_pct_change']], scaled_features]

    X, y = create_sequences(data_for_sequences, config.SEQUENCE_LENGTH, FUTURE_WINDOW_SIZE)
    
    X = X[:, :, 1:]

    logger.info(f"Data processing complete for {market}. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    return X, y, scaler

def get_recent_pattern(market: str, current_time: datetime, hours: int = 24) -> np.ndarray:
    """
    Loads recent historical data for a market and returns its price change pattern.
    Returns a 1D numpy array of percentage changes for the last 'hours' period.
    """
    # Need hours + 1 data points to calculate 'hours' percentage changes
    query = f"SELECT timestamp, close FROM crypto_data WHERE market = '{market}' AND timestamp <= '{current_time}' ORDER BY timestamp DESC LIMIT {hours + 1}"
    df = load_data(query)
    
    if df.empty or len(df) < (hours + 1):
        # logger.warning(f"Not enough recent data for {market} to form a {hours}-hour pattern.")
        return np.array([]) # Return empty array if not enough data points

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp', ascending=True) # Ensure chronological order

    pattern = df['close'].pct_change().dropna().values
    
    if len(pattern) != hours:
        # This should ideally not happen if len(df) >= hours + 1 and no NaNs
        return np.array([])

    return pattern # Return the 'hours' percentage changes

# --- Functions for Pump Prediction Dataset ---

def create_pump_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates features that might indicate a future pump."""
    # Volume Spikes (volume change vs. rolling average)
    df['volume_pct_change'] = df['volume'].pct_change()
    df['volume_spike_score'] = df['volume_pct_change'] / (df['volume_pct_change'].rolling(window=24).mean() + 1e-9)

    # Bollinger Band Squeeze
    if 'bb_middle' in df.columns and df['bb_middle'].notna().any():
        bb_bandwidth = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)
        # A squeeze is when the current bandwidth is at a 24-hour low
        df['squeeze_on'] = (bb_bandwidth.rolling(window=24).min() == bb_bandwidth).astype(int)
    else:
        df['squeeze_on'] = 0

    # Price momentum (rate of change over different periods)
    df['roc_3'] = ta.roc(df['close'], length=3)
    df['roc_6'] = ta.roc(df['close'], length=6)

    return df

def create_pump_labels(df: pd.DataFrame, time_horizon: int = 6) -> pd.DataFrame:
    """Creates multi-class ground truth labels for pump events."""
    # Find the max high price in the next `time_horizon` hours
    future_max_high = df['high'].rolling(window=time_horizon, min_periods=1).max().shift(-time_horizon)
    
    # Calculate the maximum percentage rise from the current close
    max_future_rise = (future_max_high / df['close']) - 1
    
    # Define conditions for each class
    # Label 0: < 10% (No Pump)
    # Label 1: 10% to 15%
    # Label 2: 15% to 20%
    # Label 3: >= 20%
    conditions = [
        (max_future_rise < 0.10),
        (max_future_rise >= 0.10) & (max_future_rise < 0.15),
        (max_future_rise >= 0.15) & (max_future_rise < 0.20),
        (max_future_rise >= 0.20)
    ]
    choices = [0, 1, 2, 3]
    
    # Use numpy.select for conditional labeling
    df['pump_label'] = np.select(conditions, choices, default=0)
    
    return df

def get_pump_dataset(days: int = None):
    """
    Generates a complete dataset for training the pump prediction model.
    If 'days' is specified, it only processes data from the last N days.
    """
    if days:
        logger.info(f"--- Starting Generation of Pump Prediction Dataset for last {days} days ---")
    else:
        logger.info("--- Starting Generation of Full Pump Prediction Dataset ---")
    
    # First, get the market index that will be joined to all individual dataframes
    market_index_df = get_market_index()

    all_markets_df = load_data("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
    all_markets = all_markets_df['market'].tolist() if not all_markets_df.empty else []

    full_dataset = []

    for market in all_markets:
        
        if days:
            query = f"SELECT * FROM crypto_data WHERE market = '{market}' AND timestamp >= date('now', '-{days} days') ORDER BY timestamp ASC"
        else:
            query = f"SELECT * FROM crypto_data WHERE market = '{market}' ORDER BY timestamp ASC"
            
        df = load_data(query)
        
        if df.empty or len(df) < 100: # Need enough data for rolling windows
            continue

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Join the market index here
        df = df.join(market_index_df, how='left')
        df['market_index_return'] = df['market_index_return'].fillna(0)

        # 1. Calculate standard and new pump-specific features
        df = calculate_technical_indicators(df)
        df = create_pump_features(df)

        # 2. Create the pump labels
        df = create_pump_labels(df, time_horizon=6)

        # 3. Select features and labels
        feature_cols = [
            'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'adx', 'obv',
            'bb_upper', 'bb_middle', 'bb_lower', 'volume_ma',
            'market_index_return', # Add market trend as a feature
            'volume_spike_score', 'squeeze_on', 'roc_3', 'roc_6'
        ]
        # Ensure all columns exist before trying to select them
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        df_subset = df[feature_cols + ['pump_label']].copy()
        df_subset['market'] = market # Add market identifier
        
        full_dataset.append(df_subset)

    if not full_dataset:
        logger.error("Could not generate any data for the pump dataset.")
        return

    # Combine all dataframes and save to a single file
    final_df = pd.concat(full_dataset, ignore_index=True)
    final_df.dropna(inplace=True) # Drop rows with NaNs from rolling calculations
    
    output_path = "data/pump_dataset.csv"
    final_df.to_csv(output_path, index=False)
    logger.info(f"Pump prediction dataset created successfully with {len(final_df)} samples.")
    logger.info(f"Saved to {output_path}")

    return final_df
