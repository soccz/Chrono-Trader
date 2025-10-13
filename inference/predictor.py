import torch
import numpy as np
import glob
import os
import json
import torch.serialization
import torch.nn
import models.hybrid_model
import models.transformer_encoder

from utils.config import config
from utils.logger import logger
from data import database, preprocessor
# build_model is no longer needed as we load the whole model object
# from models.hybrid_model import build_model

from dtaidistance import dtw

def get_pattern_similarity(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
    """
    Calculates the similarity between two price change patterns using Dynamic Time Warping (DTW).
    """
    if len(pattern1) == 0 or len(pattern2) == 0:
        return float('inf')

    pattern1 = np.array(pattern1, dtype=np.double)
    pattern2 = np.array(pattern2, dtype=np.double)
    distance = dtw.distance(pattern1, pattern2)
    return distance

N_INFERENCES = 30 # Number of times to run prediction for MC Dropout

def run(markets: list):
    """Makes ensembled, probabilistic predictions for a given list of markets."""
    logger.info(f"--- Making ensembled predictions for {len(markets)} markets ---")

    model_paths = glob.glob(os.path.join("models", "model_*.pth"))
    if not model_paths:
        logger.error("No trained models found in /models directory. Please run training first.")
        return []

    logger.info(f"Found {len(model_paths)} ensemble models.")
    models = []
    for path in model_paths:
        try:
            # Load the entire model object directly by setting weights_only=False.
            # This is required for recent PyTorch versions.
            model = torch.load(path, map_location=config.DEVICE, weights_only=False)
            models.append(model)
            logger.info(f"Successfully loaded model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}. It might be incompatible or corrupt. Error: {e}")
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

        final_pattern = np.mean(ensemble_patterns, axis=0)
        final_uncertainty = np.mean(ensemble_uncertainties)

        current_price_df = database.load_data(f"SELECT close FROM crypto_data WHERE market = '{market}' ORDER BY timestamp DESC LIMIT 1")
        if not current_price_df.empty:
            current_price = current_price_df.iloc[0]['close']
        else:
            logger.warning(f"Could not retrieve current price for {market}. Skipping.")
            continue

        all_predictions.append({
            "market": market,
            "predicted_pattern": final_pattern,
            "uncertainty": final_uncertainty,
            "current_price": current_price
        })
        logger.info(f"Ensemble prediction for {market} generated (Uncertainty: {final_uncertainty:.6f}).")

    return all_predictions
