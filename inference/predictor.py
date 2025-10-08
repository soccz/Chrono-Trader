import torch
import numpy as np
import glob
import os
import json

from utils.config import config
from utils.logger import logger
from data import database, preprocessor
from models.hybrid_model import build_model

def load_and_apply_model_config():
    """Checks for a saved model config and applies it to ensure consistency."""
    config_path = os.path.join("models", "model_config.json")
    if os.path.exists(config_path):
        logger.info(f"Loading model configuration from {config_path}")
        try:
            with open(config_path, 'r') as f:
                best_params = json.load(f)
            
            # Update config with loaded parameters
            config.LEARNING_RATE = best_params.get('lr', config.LEARNING_RATE)
            config.D_MODEL = best_params.get('d_model', config.D_MODEL)
            config.N_LAYERS = best_params.get('n_layers', config.N_LAYERS)
            config.N_HEADS = best_params.get('n_heads', config.N_HEADS)
            config.BATCH_SIZE = best_params.get('batch_size', config.BATCH_SIZE)
            logger.info("Configuration updated with parameters from saved model config.")
        except Exception as e:
            logger.error(f"Failed to load or apply model config: {e}. Using default config.")
    else:
        logger.info("No model_config.json found. Using default configuration.")

from dtaidistance import dtw

def get_pattern_similarity(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
    """
    Calculates the similarity between two price change patterns using Dynamic Time Warping (DTW).
    DTW is robust to time shifts and scaling. Lower values indicate higher similarity.
    """
    if len(pattern1) == 0 or len(pattern2) == 0:
        return float('inf')

    # Ensure patterns are numpy arrays of the correct type
    pattern1 = np.array(pattern1, dtype=np.double)
    pattern2 = np.array(pattern2, dtype=np.double)

    # Calculate DTW distance. The library handles varying lengths, but we expect similar lengths.
    # The 'distance' is the value of the last cell in the warping path matrix.
    distance = dtw.distance(pattern1, pattern2)
    
    return distance

N_INFERENCES = 30 # Number of times to run prediction for MC Dropout

def run(markets: list):
    """Makes ensembled, probabilistic predictions for a given list of markets."""
    # Load the same configuration that the models were trained with
    load_and_apply_model_config()

    logger.info(f"--- Making ensembled predictions for {len(markets)} markets ---")

    model_paths = glob.glob(os.path.join("models", "model_*.pth"))
    if not model_paths:
        logger.error("No trained models found in /models directory. Please run training first.")
        return []

    logger.info(f"Found {len(model_paths)} ensemble models.")
    models = []
    for path in model_paths:
        model = build_model(
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            n_layers=config.N_LAYERS,
            input_dim=config.N_FEATURES,
            noise_dim=config.GAN_NOISE_DIM,
            output_dim=6 # Corresponds to FUTURE_WINDOW_SIZE
        )
        try:
            model.load_state_dict(torch.load(path, map_location=config.DEVICE))
            models.append(model)
        except RuntimeError as e:
            logger.error(f"Failed to load model from {path}. It might be incompatible. Error: {e}")
            logger.error("Please retrain the model, potentially with the --tune option.")
            return []

    if not models:
        logger.error("No models were successfully loaded. Aborting prediction.")
        return []

    market_index_df = preprocessor.get_market_index()
    
    all_predictions = []
    for market in markets:
        X, _, scaler = preprocessor.get_processed_data(market, market_index_df)
        if X is None:
            continue
        
        last_sequence = X[-1]
        sequence_tensor = torch.FloatTensor([last_sequence]).to(config.DEVICE)

        ensemble_patterns = []
        ensemble_uncertainties = []

        for model in models:
            model.train() # Activate dropout for MC Dropout
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.eval()

            with torch.no_grad():
                mc_predictions = []
                for _ in range(N_INFERENCES):
                    predicted_pattern = model(sequence_tensor)[0].cpu().numpy()
                    mc_predictions.append(predicted_pattern)
                
                mc_predictions = np.array(mc_predictions)
                mean_pattern = mc_predictions.mean(axis=0)
                uncertainty_score = np.sum(mc_predictions.std(axis=0))
                
                ensemble_patterns.append(mean_pattern)
                ensemble_uncertainties.append(uncertainty_score)

        # Average the results from all models in the ensemble
        final_pattern = np.mean(ensemble_patterns, axis=0)
        final_uncertainty = np.mean(ensemble_uncertainties)

        # Get current price for the output
        current_price = database.load_data(f"SELECT close FROM crypto_data WHERE market = '{market}' ORDER BY timestamp DESC LIMIT 1").iloc[0]['close']

        all_predictions.append({
            "market": market,
            "predicted_pattern": final_pattern,
            "uncertainty": final_uncertainty,
            "current_price": current_price
        })
        logger.info(f"Ensemble prediction for {market} generated (Uncertainty: {final_uncertainty:.6f}).")

    return all_predictions