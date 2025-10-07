import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from utils.config import config
from utils.logger import logger
from data.database import load_data, get_db_connection

FUTURE_WINDOW_SIZE = 6
MAJOR_COINS = ['KRW-BTC', 'KRW-ETH'] # Coins to build the market index

def get_market_index() -> pd.DataFrame:
    """Calculates a market index based on major coins."""
    logger.info(f"Calculating market index from {MAJOR_COINS}...")
    index_df = pd.DataFrame()
    try:
        for coin in MAJOR_COINS:
            query = f"SELECT timestamp, close FROM crypto_data WHERE market = '{coin}' ORDER BY timestamp ASC"
            df = load_data(query)
            if df.empty:
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df[f'{coin}_pct_change'] = df['close'].pct_change()
            if index_df.empty:
                index_df = df[[f'{coin}_pct_change']]
            else:
                index_df = index_df.join(df[[f'{coin}_pct_change']], how='outer')
        
        if index_df.empty:
            logger.warning("Could not calculate market index, no data for major coins.")
            return pd.DataFrame()

        index_df['market_index'] = index_df.mean(axis=1)
        index_df['market_index'] = index_df['market_index'].fillna(0)
        return index_df[['market_index']]

    except Exception as e:
        logger.error(f"Failed to calculate market index: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators for the given data."""
    df['rsi'] = talib.RSI(df['close'])
    df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'])
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
    df['obv'] = talib.OBV(df['close'], df['volume'])
    
    # Add Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20)
    
    # Add Volume Moving Average
    df['volume_ma'] = talib.SMA(df['volume'], timeperiod=20)

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
    """Loads, preprocesses, and prepares data, now including the market index."""
    logger.info(f"Processing data for market: {market}")
    
    query = f"SELECT * FROM crypto_data WHERE market = '{market}' ORDER BY timestamp ASC"
    df = load_data(query)
    if df.empty:
        logger.warning(f"No data found for market {market}. Skipping.")
        return None, None, None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    df = calculate_technical_indicators(df)

    df = df.join(market_index_df, how='left')
    df['market_index'] = df['market_index'].fillna(0)

    df['future_pct_change'] = (df['close'].shift(-1) - df['close']) / df['close']
    df.fillna(0, inplace=True)

    features_to_scale = [
        'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'adx', 'obv', 
        'market_index', 'bb_upper', 'bb_middle', 'bb_lower', 'volume_ma'
    ]
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features_to_scale])

    data_for_sequences = np.c_[df[['future_pct_change']], scaled_features]

    X, y = create_sequences(data_for_sequences, config.SEQUENCE_LENGTH, FUTURE_WINDOW_SIZE)
    
    X = X[:, :, 1:]

    logger.info(f"Data processing complete for {market}. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    return X, y, scaler

def get_recent_pattern(market: str, current_time: datetime, hours: int = 6) -> np.ndarray:
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