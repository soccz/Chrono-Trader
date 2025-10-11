import argparse
from utils.logger import logger
from utils.config import config
from data import database, collector, preprocessor
from training import trainer
from inference import predictor, recommender
from utils import screener
from datetime import datetime
import pandas as pd
import time
import os

def display_pump_candidates(potential_pumps):
    """Helper function to print pump candidates in a structured format."""
    if not potential_pumps:
        logger.info("--- No Potential Pump Candidates Found ---")
        return

    logger.info("--- 🚀 급등 가능성 포착 결과 ---")
    for pump in potential_pumps:
        market = pump['market']
        current_price = pump['current_price']
        target_price = current_price * 1.10
        probs = pump['probabilities']
        total_pump_prob = pump['total_pump_prob']

        logger.info(f"▶ {market}")
        logger.info(f"  - 현재가: {current_price:,.0f}원 | 10% 상승 목표가: {target_price:,.0f}원")
        logger.info(f"  - 급등 확률 (총합): {total_pump_prob:.2%}")
        logger.info(f"  - 분포: [10-15%]: {probs[1]:.2%} | [15-20%]: {probs[2]:.2%} | [20%+]: {probs[3]:.2%}")
    logger.info("------------------------------------")

def save_pump_predictions_to_csv(pump_predictions: list):
    """Saves pump prediction results to a CSV file."""
    if not pump_predictions:
        logger.info("No pump predictions to save to CSV.")
        return

    df_to_save = []
    for pump in pump_predictions:
        row = {
            'market': pump['market'],
            'current_price': pump['current_price'],
            'target_price_10_pct_up': pump['current_price'] * 1.10,
            'total_pump_probability': pump['total_pump_prob'],
            'prob_0_10_pct': pump['probabilities'][0], # No pump
            'prob_10_15_pct': pump['probabilities'][1],
            'prob_15_20_pct': pump['probabilities'][2],
            'prob_20_plus_pct': pump['probabilities'][3]
        }
        df_to_save.append(row)

    df = pd.DataFrame(df_to_save)
    output_dir = 'predictions' # Using 'predictions' folder for pump predictions
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"pump_preds_{timestamp}.csv")
    
    try:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Pump predictions successfully saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save pump predictions to CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Crypto Predictor CLI v3")
    parser.add_argument(
        '--mode',
        choices=['init_db', 'train', 'daily', 'screen', 'quick-recommend', 'backtest', 'train-pump', 'find-pumps', 'explain'],
        required=True,
        help="The mode to run the script in."
    )
    parser.add_argument('--days', type=int, default=30, help="Number of days for data collection or backtesting.")
    parser.add_argument('--symbol', type=str, help="A specific crypto symbol to predict (e.g., KRW-BTC).")
    parser.add_argument('--tune', action='store_true', help="Enable hyperparameter tuning during training.")
    parser.add_argument('--model_path', type=str, default='models/model_1.pth', help="Path to the model file for analysis.")
    parser.add_argument('--lr', type=float, help="Override learning rate for training.")
    parser.add_argument('--epochs', type=int, help="Override number of epochs for training.")
    parser.add_argument('--d_model', type=int, help="Override d_model for Transformer/GAN.")
    parser.add_argument('--n_layers', type=int, help="Override n_layers for Transformer.")
    parser.add_argument('--n_heads', type=int, help="Override n_heads for Transformer.")
    parser.add_argument('--batch_size', type=int, help="Override batch size for training.")

    args = parser.parse_args()
    logger.info(f"--- Running in {args.mode.upper()} mode ---")

    if args.lr is not None: config.LEARNING_RATE = args.lr
    if args.epochs is not None: config.EPOCHS = args.epochs
    if args.d_model is not None: config.D_MODEL = args.d_model
    if args.n_layers is not None: config.N_LAYERS = args.n_layers
    if args.n_heads is not None: config.N_HEADS = args.n_heads
    if args.batch_size is not None: config.BATCH_SIZE = args.batch_size

    if args.mode == 'init_db':
        database.init_db()
    
    elif args.mode == 'train':
        logger.info("Starting full model training...")
        if args.days < 90: logger.warning(f"For initial training, a period of at least 90 days is recommended. Using --days={args.days}.")
        for market in config.TARGET_MARKETS:
             collector.collect_market_data(market, days=args.days)
             time.sleep(0.5)
        trainer.run(tune=args.tune)

    elif args.mode == 'train-pump':
        from training import pump_trainer
        pump_trainer.run(fine_tune=False)

    elif args.mode == 'find-pumps':
        from inference import pump_predictor
        potential_pumps = pump_predictor.run()
        display_pump_candidates(potential_pumps)
        save_pump_predictions_to_csv(potential_pumps)

    elif args.mode == 'backtest':
        from training import evaluator
        evaluator.run(days_to_backtest=args.days)

    elif args.mode == 'daily':
        from training import pump_trainer
        from inference import pump_predictor
        logger.info("Starting daily run (including model reinforcement)...")
        trending_markets = screener.get_trending_markets()
        
        if trending_markets:
            logger.info(f"Collecting latest data for {len(trending_markets)} trending markets...")
            for market in trending_markets:
                collector.collect_market_data(market, days=30)
                time.sleep(0.5)
            
            logger.info("Fine-tuning main trend model...")
            trainer.run(markets=trending_markets)

            logger.info("Fine-tuning pump prediction model...")
            pump_trainer.run(fine_tune=True)

            logger.info("--- Running Main Trend Prediction Module ---")
            all_krw_markets_df = database.load_data("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
            all_krw_markets = all_krw_markets_df['market'].tolist() if not all_krw_markets_df.empty else []
            initial_predictions = predictor.run(markets=trending_markets)

            pattern_following_candidates = []
            if initial_predictions:
                target_pattern = initial_predictions[0]['predicted_pattern']
                other_markets = [m for m in all_krw_markets if m not in trending_markets]
                logger.info(f"Searching for pattern-following coins among {len(other_markets)} markets...")
                for other_market in other_markets:
                    recent_historical_pattern = preprocessor.get_recent_pattern(other_market, datetime.now(), hours=config.PATTERN_LOOKBACK_HOURS)
                    if len(recent_historical_pattern) == config.PATTERN_LOOKBACK_HOURS:
                        similarity = predictor.get_pattern_similarity(target_pattern, recent_historical_pattern)
                        pattern_following_candidates.append({'market': other_market, 'similarity': similarity})
                
                pattern_following_candidates.sort(key=lambda x: x['similarity'])
                top_pattern_followers = pattern_following_candidates[:3]
                
                logger.info(f"Found {len(top_pattern_followers)} pattern-following candidates: {[c['market'] for c in top_pattern_followers]}")

                pattern_follower_predictions = []
                if top_pattern_followers:
                    logger.info("Collecting latest data for pattern-following coins...")
                    for candidate in top_pattern_followers:
                        collector.collect_market_data(candidate['market'], days=30)
                        time.sleep(0.5)
                    logger.info("Making predictions for pattern-following coins...")
                    pattern_follower_predictions = predictor.run(markets=[c['market'] for c in top_pattern_followers])
                
                for pred in initial_predictions: pred['strategy'] = 'trending'
                for pred in pattern_follower_predictions: pred['strategy'] = 'pattern'
                
                all_predictions = initial_predictions + pattern_follower_predictions
                recommender.run(predictions=all_predictions)
            else:
                logger.info("No initial trend predictions to base pattern following on.")

            # --- Running Pump Prediction Module ---
            potential_pumps = pump_predictor.run()
            display_pump_candidates(potential_pumps)
            save_pump_predictions_to_csv(potential_pumps)

        else:
            logger.info("No trending markets found today.")
        logger.info("Daily run finished.")

    elif args.mode == 'quick-recommend':
        logger.info("Starting quick recommendation run (no training)...")
        trending_markets = screener.get_trending_markets()
        if trending_markets:
            logger.info(f"Collecting latest data for {len(trending_markets)} trending markets...")
            for market in trending_markets:
                collector.collect_market_data(market, days=30)
                time.sleep(0.5)
            logger.info("Making predictions with the existing model...")
            predictions = predictor.run(markets=trending_markets)
            for pred in predictions: pred['strategy'] = 'trending'
            recommender.run(predictions=predictions)
        else:
            logger.info("No trending markets found today.")
        logger.info("Quick recommendation run finished.")

    elif args.mode == 'screen':
        screener.get_trending_markets()

if __name__ == "__main__":
    main()