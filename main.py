import argparse
from utils.logger import logger
from utils.config import config
from data import database, collector, preprocessor
from training import trainer
from inference import predictor, recommender
from utils import screener
from datetime import datetime
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Crypto Predictor CLI v3")
    parser.add_argument(
        '--mode',
        choices=['init_db', 'train', 'daily', 'screen', 'quick-recommend', 'backtest'],
        required=True,
        help="The mode to run the script in."
    )
    parser.add_argument('--days', type=int, default=30, help="Number of days for data collection or backtesting.")
    parser.add_argument('--symbol', type=str, help="A specific crypto symbol to predict (e.g., KRW-BTC).")
    parser.add_argument('--tune', action='store_true', help="Enable hyperparameter tuning during training.")

    # --- New CLI arguments for configuration overrides ---
    parser.add_argument('--lr', type=float, help="Override learning rate for training.")
    parser.add_argument('--epochs', type=int, help="Override number of epochs for training.")
    parser.add_argument('--d_model', type=int, help="Override d_model for Transformer/GAN.")
    parser.add_argument('--n_layers', type=int, help="Override n_layers for Transformer.")
    parser.add_argument('--n_heads', type=int, help="Override n_heads for Transformer.")
    parser.add_argument('--batch_size', type=int, help="Override batch size for training.")
    # --- End new CLI arguments ---

    args = parser.parse_args()
    logger.info(f"--- Running in {args.mode.upper()} mode ---")

    # --- Apply CLI overrides to config ---
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
        logger.info(f"Config: Learning Rate overridden to {config.LEARNING_RATE}")
    if args.epochs is not None:
        config.EPOCHS = args.epochs
        logger.info(f"Config: Epochs overridden to {config.EPOCHS}")
    if args.d_model is not None:
        config.D_MODEL = args.d_model
        logger.info(f"Config: D_MODEL overridden to {config.D_MODEL}")
    if args.n_layers is not None:
        config.N_LAYERS = args.n_layers
        logger.info(f"Config: N_LAYERS overridden to {config.N_LAYERS}")
    if args.n_heads is not None:
        config.N_HEADS = args.n_heads
        logger.info(f"Config: N_HEADS overridden to {config.N_HEADS}")
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
        logger.info(f"Config: Batch Size overridden to {config.BATCH_SIZE}")
    # --- End CLI overrides ---

    if args.mode == 'init_db':
        database.init_db()
    
    elif args.mode == 'train':
        logger.info("Starting full model training...")
        if args.days < 90:
            logger.warning(f"For initial training, a period of at least 90 days is recommended. Using --days={args.days}.")
        
        for market in config.TARGET_MARKETS:
             collector.collect_market_data(market, days=args.days)
        trainer.run(tune=args.tune)

    elif args.mode == 'backtest':
        from training import evaluator # Import here to avoid potential circular dependencies
        evaluator.run(days_to_backtest=args.days)

    elif args.mode == 'daily':
        logger.info("Starting daily run (including model reinforcement)...")
        trending_markets = screener.get_trending_markets()
        
        all_krw_markets_df = database.load_data("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
        all_krw_markets = all_krw_markets_df['market'].tolist() if not all_krw_markets_df.empty else []

        if trending_markets:
            logger.info(f"Collecting latest data for {len(trending_markets)} trending markets...")
            for market in trending_markets:
                collector.collect_market_data(market, days=30)
            
            logger.info("Fine-tuning model with new data...")
            trainer.run(markets=trending_markets)

            logger.info("Making predictions with the newly fine-tuned model...")
            initial_predictions = predictor.run(markets=trending_markets) # Store initial predictions

            # --- START of Pattern Following Logic ---
            pattern_following_candidates = []
            if initial_predictions:
                # Use the predicted pattern of the top initial recommendation as the target pattern
                # Or, if no initial predictions, skip pattern following
                target_pattern = initial_predictions[0]['predicted_pattern'] if initial_predictions else None
                
                if target_pattern is not None:
                    # Find other markets not in trending_markets
                    other_markets = [m for m in all_krw_markets if m not in trending_markets]
                    
                    logger.info(f"Searching for pattern-following coins among {len(other_markets)} markets...")
                    
                    for other_market in other_markets:
                        # Get recent historical pattern for the other market
                        recent_historical_pattern = preprocessor.get_recent_pattern(other_market, datetime.now(), hours=6)
                        
                        if len(recent_historical_pattern) == 6: # Ensure valid pattern length
                            similarity = predictor.get_pattern_similarity(target_pattern, recent_historical_pattern)
                            pattern_following_candidates.append({
                                'market': other_market,
                                'similarity': similarity,
                                'target_pattern': target_pattern # Store for later prediction
                            })
                    
                    # Sort by similarity (lower distance means higher similarity)
                    pattern_following_candidates.sort(key=lambda x: x['similarity'])
                    
                    # Select top 3 pattern-following candidates
                    top_pattern_followers = pattern_following_candidates[:3]
                    
                    logger.info(f"Found {len(top_pattern_followers)} pattern-following candidates: {[c['market'] for c in top_pattern_followers]}")

                    # Collect data and predict for these pattern followers
                    pattern_follower_predictions = []
                    if top_pattern_followers:
                        logger.info("Collecting latest data for pattern-following coins...")
                        for candidate in top_pattern_followers:
                            collector.collect_market_data(candidate['market'], days=30) # Collect data for them
                        
                        logger.info("Making predictions for pattern-following coins...")
                        # Predict for these pattern followers using the fine-tuned model
                        pattern_follower_predictions = predictor.run(markets=[c['market'] for c in top_pattern_followers])
                
                # Add strategy tags before combining
                for pred in initial_predictions:
                    pred['strategy'] = 'trending'
                for pred in pattern_follower_predictions:
                    pred['strategy'] = 'pattern'
                
                # Combine initial predictions and pattern-follower predictions
                all_predictions = initial_predictions + pattern_follower_predictions
                recommender.run(predictions=all_predictions) # Pass combined predictions
            else:
                logger.info("No initial predictions to base pattern following on.")
            # --- END of Pattern Following Logic ---

        else:
            logger.info("No trending markets found today.")
        logger.info("Daily run finished.")

    elif args.mode == 'quick-recommend':
        logger.info("Starting quick recommendation run (no training)...")
        trending_markets = screener.get_trending_markets()
        if trending_markets:
            logger.info(f"Collecting latest data for {len(trending_markets)} trending markets...")
            for market in trending_markets:
                collector.collect_market_data(market, days=30) # Still need fresh data for prediction

            logger.info("Making predictions with the existing model...")
            predictions = predictor.run(markets=trending_markets)
            # Add strategy tag for recommender
            for pred in predictions:
                pred['strategy'] = 'trending'
            recommender.run(predictions=predictions)
        else:
            logger.info("No trending markets found today.")
        logger.info("Quick recommendation run finished.")

    elif args.mode == 'screen':
        screener.get_trending_markets()

if __name__ == "__main__":
    main()