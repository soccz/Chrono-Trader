import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import optuna
import functools
import json

from utils.config import config
from utils.logger import logger
from data.preprocessor import get_processed_data, get_market_index
from models.hybrid_model import build_model

def objective(trial, X, y):
    """Optuna objective function to find the best hyperparameters."""
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 2, 6)
    n_heads = trial.suggest_categorical("n_heads", [4, 8])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # For this trial, we only need to train one model, not the full ensemble
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 - config.TRAIN_SPLIT, random_state=42)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        input_dim=config.N_FEATURES,
        noise_dim=config.GAN_NOISE_DIM,
        output_dim=6 # FUTURE_WINDOW_SIZE
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    # Train for a limited number of epochs for speed during tuning
    for epoch in range(15): # Reduced epochs for tuning
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(config.DEVICE), batch_y.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(config.DEVICE), batch_y.to(config.DEVICE)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        # Pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

def run(markets: list = None, tune: bool = False):
    # Load model config if available (for both full training and fine-tuning)
    config_path = os.path.join("models", "model_config.json")
    if os.path.exists(config_path):
        logger.info(f"Loading model configuration from {config_path}")
        try:
            with open(config_path, 'r') as f:
                best_params = json.load(f)
            
            config.LEARNING_RATE = best_params.get('lr', config.LEARNING_RATE)
            config.D_MODEL = best_params.get('d_model', config.D_MODEL)
            config.N_LAYERS = best_params.get('n_layers', config.N_LAYERS)
            config.N_HEADS = best_params.get('n_heads', config.N_HEADS)
            config.BATCH_SIZE = best_params.get('batch_size', config.BATCH_SIZE)
            logger.info("Configuration updated with parameters from saved model config.")
        except Exception as e:
            logger.error(f"Failed to load or apply model config: {e}. Using default config.")

    if markets:
        logger.info(f"--- Starting Fine-tuning on {len(markets)} trending markets ---")
        market_index_df = get_market_index()
        
        # Load existing ensemble models
        model_paths = [os.path.join("models", f"model_{i+1}.pth") for i in range(config.N_ENSEMBLE_MODELS)]
        models = []
        for path in model_paths:
            model = build_model(
                d_model=config.D_MODEL,
                n_heads=config.N_HEADS,
                n_layers=config.N_LAYERS,
                input_dim=config.N_FEATURES,
                noise_dim=config.GAN_NOISE_DIM,
                output_dim=6 # FUTURE_WINDOW_SIZE
            )
            try:
                model.load_state_dict(torch.load(path, map_location=config.DEVICE))
                models.append(model)
            except RuntimeError as e:
                logger.error(f"Failed to load model from {path} for fine-tuning. It might be incompatible. Error: {e}")
                logger.error("Please retrain the base model, potentially with the --tune option.")
                return
        
        if not models:
            logger.error("No base models loaded for fine-tuning. Aborting.")
            return

        # Fine-tune each model in the ensemble
        for i, model in enumerate(models):
            logger.info(f"\n--- Fine-tuning Ensemble Model {i+1}/{config.N_ENSEMBLE_MODELS} ---")
            
            # Collect and process data for the specific trending markets
            X_list, y_list = [], []
            for market in markets:
                X_market, y_market, _ = get_processed_data(market, market_index_df)
                if X_market is not None:
                    X_list.append(X_market)
                    y_list.append(y_market)
            
            if not X_list:
                logger.warning(f"No data available for trending markets for fine-tuning. Skipping model {i+1}.")
                continue

            X_fine_tune = np.concatenate(X_list, axis=0)
            y_fine_tune = np.concatenate(y_list, axis=0)

            # Use a smaller portion of data for validation during fine-tuning
            X_train, X_val, y_train, y_val = train_test_split(X_fine_tune, y_fine_tune, test_size=0.1, random_state=42 + i)
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

            optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE / 10) # Use a smaller LR for fine-tuning
            criterion = nn.MSELoss()
            best_val_loss = float('inf')
            model_save_path = os.path.join("models", f"model_{i+1}.pth")

            for epoch in range(config.EPOCHS // 5): # Fine-tune for fewer epochs
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(config.DEVICE), batch_y.to(config.DEVICE)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(config.DEVICE), batch_y.to(config.DEVICE)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)
            
            logger.info(f"Ensemble model {i+1} fine-tuned and saved to {model_save_path} with best val loss: {best_val_loss:.6f}")

        logger.info("=== Fine-tuning Finished ===")

    else:
        logger.info("--- Preparing for Full Model Training ---")
        market_index_df = get_market_index()
        
        # For tuning and base training, use the primary market data
        X, y, scaler = get_processed_data(config.TARGET_MARKETS[0], market_index_df)
        if X is None:
            logger.error("Training failed: No data available for the primary market.")
            return

        if tune:
            logger.info("--- Starting Hyperparameter Tuning with Optuna ---")
            study = optuna.create_study(direction='minimize')
            objective_with_data = functools.partial(objective, X=X, y=y)
            study.optimize(objective_with_data, n_trials=50) # Run 50 trials

            best_params = study.best_params
            logger.info(f"Tuning finished. Best parameters found: {best_params}")

            # Save the best parameters to a JSON file
            config_save_path = os.path.join("models", "model_config.json")
            try:
                with open(config_save_path, 'w') as f:
                    json.dump(best_params, f, indent=4)
                logger.info(f"Best parameters saved to {config_save_path}")
            except Exception as e:
                logger.error(f"Failed to save best parameters: {e}")

            # Update config with best parameters for the final training
            config.LEARNING_RATE = best_params['lr']
            config.D_MODEL = best_params['d_model']
            config.N_LAYERS = best_params['n_layers']
            config.N_HEADS = best_params['n_heads']
            config.BATCH_SIZE = best_params['batch_size']
            logger.info("Configuration updated with best parameters for this run.")

        logger.info("--- Starting Full Ensemble Model Training ---")
        for i in range(config.N_ENSEMBLE_MODELS):
            logger.info(f"\n--- Training Ensemble Model {i+1}/{config.N_ENSEMBLE_MODELS} ---")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 - config.TRAIN_SPLIT, random_state=42 + i)
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

            model = build_model(
                d_model=config.D_MODEL,
                n_heads=config.N_HEADS,
                n_layers=config.N_LAYERS,
                input_dim=config.N_FEATURES,
                noise_dim=config.GAN_NOISE_DIM,
                output_dim=6 # FUTURE_WINDOW_SIZE
            )

            optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
            criterion = nn.MSELoss()
            best_val_loss = float('inf')
            model_save_path = os.path.join("models", f"model_{i+1}.pth")

            for epoch in range(config.EPOCHS):
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(config.DEVICE), batch_y.to(config.DEVICE)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(config.DEVICE), batch_y.to(config.DEVICE)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)
            
            logger.info(f"Ensemble model {i+1} saved to {model_save_path} with best val loss: {best_val_loss:.6f}")

        logger.info("=== Full Ensemble Model Training Finished ===")