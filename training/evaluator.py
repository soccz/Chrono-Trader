import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

from utils.config import config
from utils.logger import logger
from data.database import load_data
from data.preprocessor import get_market_index
# We will reuse the already-functional predictor and recommender
from inference import predictor, recommender

# --- Backtest Configuration ---
INITIAL_BALANCE = 10000000  # 1,000만원

def run(days_to_backtest: int = 30):
    """Runs a realistic backtest for the given number of days."""
    logger.info(f"=== Starting Backtest for the last {days_to_backtest} days ===")

    # 1. Load all necessary data for the backtest period
    end_time = datetime.now()
    # Ensure enough historical data for initial processing (SEQUENCE_LENGTH + 24h for screening)
    data_load_start_time = end_time - timedelta(days=days_to_backtest) - timedelta(hours=config.SEQUENCE_LENGTH + 24)
    
    logger.info(f"Loading all market data from {data_load_start_time} to {end_time}...")
    all_data_query = f"SELECT * FROM crypto_data WHERE timestamp >= '{data_load_start_time}'"
    all_df = load_data(all_data_query)
    if all_df.empty:
        logger.error("Not enough data to run backtest. Please collect more data.")
        return

    all_df['timestamp'] = pd.to_datetime(all_df['timestamp'])
    logger.info(f"Loaded {len(all_df)} total records.")

    # 2. Setup backtest environment
    balance = INITIAL_BALANCE
    portfolio_value = [INITIAL_BALANCE]
    positions = {} # {market: {entry_price: float, amount: float, signal: 'Long'/'Short'}}
    trade_log = []

    market_index_df = get_market_index()

    # Define the actual start time for the backtest loop (after warm-up)
    backtest_loop_start_time = end_time - timedelta(days=days_to_backtest)

    # 3. Loop through each hour of the backtest period
    for hour in tqdm(range(days_to_backtest * 24), desc="Backtesting Progress"):
        current_time = backtest_loop_start_time + timedelta(hours=hour)
        
        # Get data available at the current simulated time
        current_market_data = all_df[all_df['timestamp'] <= current_time]

        # Ensure enough data for screening and prediction (at least SEQUENCE_LENGTH + 24 hours)
        if len(current_market_data.groupby('market').size()) < 5 or current_time < (backtest_loop_start_time + timedelta(hours=config.SEQUENCE_LENGTH + 24)):
            logger.info(f"[{current_time}] Warm-up period. Skipping trading logic.")
            portfolio_value.append(balance) # Keep portfolio value consistent during warm-up
            continue

        # --- Simulate Daily Run Logic ---
        # a) Simulate Screening: Find top 5 volatile markets in the last 24 hours
        lookback_24h = current_time - timedelta(hours=24)
        recent_data = current_market_data[current_market_data['timestamp'] >= lookback_24h]
        
        if recent_data.empty:
            logger.info(f"[{current_time}] DEBUG: recent_data is empty. Skipping screening.")
            continue

        # Ensure enough data points for std() calculation (at least 2)
        market_counts = recent_data.groupby('market').size()
        markets_with_enough_data = market_counts[market_counts >= 2].index
        
        if markets_with_enough_data.empty:
            logger.info(f"[{current_time}] DEBUG: No markets with enough data for volatility calculation. Skipping screening.")
            continue

        volatility = recent_data[recent_data['market'].isin(markets_with_enough_data)].groupby('market')['close'].std() / recent_data[recent_data['market'].isin(markets_with_enough_data)].groupby('market')['close'].mean()
        
        if volatility.empty:
            logger.info(f"[{current_time}] DEBUG: Volatility calculation resulted in empty. Skipping screening.")
            continue

        trending_markets = volatility.nlargest(5).index.tolist()

        logger.info(f"[{current_time}] Volatility calculated. Top 5: {volatility.nlargest(5).to_dict()}")
        logger.info(f"[{current_time}] Trending Markets for prediction: {trending_markets}")

        if not trending_markets:
            logger.info(f"[{current_time}] DEBUG: No trending markets found after volatility calculation. Skipping prediction.")
            continue

        # b) Simulate Prediction & Recommendation (reusing existing modules)
        # Note: This is a simplified version. A full backtest would re-train/fine-tune the model here.
        # For speed, we use the currently trained model to make predictions.
        predictions = predictor.run(markets=trending_markets)
        recommendations = recommender.run(predictions)

        logger.info(f"[{current_time}] Trending Markets: {trending_markets}")
        logger.info(f"[{current_time}] Predictions: {predictions}")
        logger.info(f"[{current_time}] Recommendations: {recommendations}")

        # c) Simulate Trading based on recommendations
        if recommendations:
            # Exit any positions that are no longer recommended
            recommended_markets = [r['market'] for r in recommendations]
            markets_to_exit = [m for m in positions if m not in recommended_markets]
            for market in markets_to_exit:
                pos = positions.pop(market)
                exit_price = current_market_data[current_market_data['market'] == market].iloc[-1]['close']
                pnl = (exit_price - pos['entry_price']) * pos['amount'] if pos['signal'] == 'Long' else (pos['entry_price'] - exit_price) * pos['amount']
                balance += (pos['entry_price'] * pos['amount']) + pnl
                trade_log.append({'exit_time': current_time, 'market': market, 'pnl': pnl})

            # Enter new positions
            for rec in recommendations:
                if rec['market'] not in positions and len(positions) < config.MAX_POSITIONS:
                    entry_price = rec['current_price']
                    amount_to_invest = balance * config.KELLY_FRACTION / (len(recommendations) or 1)
                    amount = amount_to_invest / entry_price
                    positions[rec['market']] = {'entry_price': entry_price, 'amount': amount, 'signal': rec['signal'], 'pnl': 0}
                    balance -= amount_to_invest
                    trade_log.append({'entry_time': current_time, 'market': rec['market'], 'signal': rec['signal'], 'pnl': 0})

        # Update portfolio value for this hour
        current_portfolio_value = balance
        for market, pos in positions.items():
            current_price = current_market_data[current_market_data['market'] == market].iloc[-1]['close']
            current_portfolio_value += pos['amount'] * current_price
        portfolio_value.append(current_portfolio_value)

    # 4. Calculate and Log Final Metrics
    trade_df = pd.DataFrame(trade_log)
    wins = 0
    losses = 0
    win_rate = 0
    if not trade_df.empty:
        wins = trade_df[trade_df['pnl'] > 0].shape[0]
        losses = trade_df[trade_df['pnl'] <= 0].shape[0]
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    final_balance = portfolio_value[-1]
    total_return_pct = (final_balance / INITIAL_BALANCE - 1) * 100
    
    returns = pd.Series(portfolio_value).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365*24) if returns.std() > 0 else 0

    rolling_max = pd.Series(portfolio_value).cummax()
    daily_drawdown = pd.Series(portfolio_value) / rolling_max - 1.0
    max_drawdown = daily_drawdown.min() * 100

    logger.info("\n--- Backtest Results ---")
    logger.info(f"Period: {days_to_backtest} days")
    logger.info(f"Final Portfolio Value: {final_balance:, .0f} KRW")
    logger.info(f"Total Return: {total_return_pct:.2f}%")
    logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"Win Rate: {win_rate:.2f}% ({wins} wins / {losses} losses)")
    logger.info(f"Total Trades: {wins + losses}")
    logger.info("========================")
