
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from utils.config import config
from utils.logger import logger
from data.database import save_data, get_db_connection

def get_last_timestamp(market: str):
    """Get the last timestamp for a given market from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(timestamp) FROM crypto_data WHERE market = ?", (market,))
    result = cursor.fetchone()
    conn.close()
    if result and result[0]:
        ts_str = result[0]
        if 'T' in ts_str:
            return datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S')
        else:
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
    return None

def collect_market_data(market: str, days: int = 90):
    """Collects historical data for a single market from Upbit."""
    logger.info(f"Starting data collection for market: {market}")
    url = config.UPBIT_API_URL
    last_ts = get_last_timestamp(market)

    if last_ts:
        to_datetime = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Last timestamp for {market} is {last_ts}. Fetching new data up to {to_datetime}.")
    else:
        to_datetime = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"No existing data for {market}. Fetching last {days} days of data.")

    all_data = []
    while True:
        params = {
            'market': market,
            'count': 200,
            'to': to_datetime
        }
        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            data = res.json()

            if not data:
                logger.info(f"No more data to fetch for {market}.")
                break

            # Filter data before appending
            if last_ts:
                new_data = []
                stop_collecting = False
                for candle in data:
                    candle_ts = datetime.strptime(candle['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
                    if candle_ts > last_ts:
                        new_data.append(candle)
                    else:
                        stop_collecting = True
                        break
                all_data.extend(new_data)
                if stop_collecting:
                    logger.info("Reached the last saved timestamp. Stopping collection.")
                    break
            else:
                all_data.extend(data)

            oldest_ts_str = data[-1]['candle_date_time_utc']
            oldest_ts = datetime.strptime(oldest_ts_str, '%Y-%m-%dT%H:%M:%S')
            logger.info(f"Fetched {len(data)} records for {market}. Oldest timestamp: {oldest_ts}")

            if not last_ts and (datetime.utcnow() - oldest_ts).days >= days:
                logger.info(f"Collected approximately {days} days of data. Stopping collection.")
                break

            to_datetime = oldest_ts.strftime('%Y-%m-%d %H:%M:%S')
            time.sleep(0.2)

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {market}: {e}")
            break

    if all_data:
        df = pd.DataFrame(all_data)
        df.rename(columns={
            'candle_date_time_utc': 'timestamp',
            'opening_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'trade_price': 'close',
            'candle_acc_trade_volume': 'volume'
        }, inplace=True)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['market'] = market
        
        df = df.drop_duplicates(subset=['timestamp', 'market'])
        save_data(df, 'crypto_data')
    else:
        logger.info(f"No new data collected for {market}.")


def run(days: int = 90):
    logger.info("=== Starting Data Collection ===")
    for market in config.TARGET_MARKETS:
        collect_market_data(market, days)
    logger.info("=== Data Collection Finished ===")
