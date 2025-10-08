
import requests
import pandas as pd
import time
from utils.logger import logger

UPBIT_MARKET_API_URL = "https://api.upbit.com/v1/market/all"
UPBIT_CANDLE_API_URL = "https://api.upbit.com/v1/candles/days"

def get_trending_markets(limit: int = 20, lookback_days: int = 3):
    """ 
    Finds markets with the highest volatility (absolute price change) over the last few days.
    This identifies coins with 'potential' for significant moves.
    """
    logger.info(f"--- Starting Market Screening (last {lookback_days} days) ---")
    try:
        # 1. Get all KRW markets
        res = requests.get(UPBIT_MARKET_API_URL, params={"isWarning": "false"})
        res.raise_for_status()
        all_markets = res.json()
        krw_markets = [m['market'] for m in all_markets if m['market'].startswith('KRW')]
        logger.info(f"Found {len(krw_markets)} KRW markets.")
        time.sleep(0.2)

        market_volatility = []
        for market in krw_markets:
            # 2. Get daily candles for each market
            params = {
                'market': market,
                'count': lookback_days + 1 # Need n+1 days to calculate n days of change
            }
            candle_res = requests.get(UPBIT_CANDLE_API_URL, params=params)
            candle_res.raise_for_status()
            candles = candle_res.json()
            time.sleep(0.2)

            if len(candles) < lookback_days + 1:
                continue

            # 3. Calculate price change percentage
            start_price = candles[lookback_days]['opening_price']
            end_price = candles[0]['trade_price']
            
            if start_price == 0:
                continue

            change_pct = (end_price - start_price) / start_price
            trade_volume = candles[0]['candle_acc_trade_price']

            # Filter out low-volume markets
            if trade_volume > config.SCREENER_MIN_VOLUME_KRW:
                market_volatility.append({
                    'market': market,
                    'change_pct': change_pct,
                    'abs_change': abs(change_pct)
                })
        
        # 4. Sort by absolute change to find the 'hottest' markets
        market_volatility.sort(key=lambda x: x['abs_change'], reverse=True)

        top_markets = [m['market'] for m in market_volatility[:limit]]
        logger.info(f"Screening complete. Top {len(top_markets)} trending markets: {top_markets}")
        return top_markets

    except Exception as e:
        logger.error(f"Market screening failed: {e}")
        return []

