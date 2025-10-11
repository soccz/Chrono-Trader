import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
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
from models.critic import build_critic

# --- WGAN-GP Hyperparameters ---
CRITIC_ITERATIONS = 5  # Train critic 5 times per generator train
LAMBDA_GP = 10         # Gradient penalty lambda

def compute_gradient_penalty(critic, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN-GP."""
    alpha = torch.rand(real_samples.size(0), 1).to(config.DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).to(config.DEVICE),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def objective(trial, X, y):
    """Optuna objective function to find the best hyperparameters for WGAN-GP."""
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    d_model = trial.suggest_categorical("d_model", [128, 256])
    n_layers = trial.suggest_int("n_layers", 2, 4)
    n_heads = trial.suggest_categorical("n_heads", [4, 8])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    lambda_recon = trial.suggest_float("lambda_recon", 10, 200, log=True)

    # 2. Setup Dataloaders
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Init models and optimizers
    generator = build_model(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        input_dim=config.N_FEATURES, noise_dim=config.GAN_NOISE_DIM, output_dim=6
    )
    critic = build_critic()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    recon_criterion = nn.MSELoss()

    # 4. Simplified Training Loop (15 epochs for tuning)
    for epoch in range(15):
        generator.train()
        critic.train()
        for batch_idx, (real_sequences, real_paths) in enumerate(train_loader):
            real_sequences, real_paths = real_sequences.to(config.DEVICE), real_paths.to(config.DEVICE)
            
            # Train Critic
            optimizer_C.zero_grad()
            fake_paths = generator(real_sequences).detach()
            real_validity = critic(real_paths)
            fake_validity = critic(fake_paths)
            gradient_penalty = compute_gradient_penalty(critic, real_paths, fake_paths)
            loss_C = fake_validity.mean() - real_validity.mean() + LAMBDA_GP * gradient_penalty
            loss_C.backward()
            optimizer_C.step()

            # Train Generator
            if batch_idx % CRITIC_ITERATIONS == 0:
                optimizer_G.zero_grad()
                gen_paths = generator(real_sequences)
                fake_validity = critic(gen_paths)
                loss_G_adv = -fake_validity.mean()
                loss_G_recon = recon_criterion(gen_paths, real_paths)
                loss_G = loss_G_adv + lambda_recon * loss_G_recon
                loss_G.backward()
                optimizer_G.step()

        # --- Validation ---
        generator.eval()
        val_recon_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(config.DEVICE), batch_y.to(config.DEVICE)
                outputs = generator(batch_X)
                loss = recon_criterion(outputs, batch_y)
                val_recon_loss += loss.item()
        val_recon_loss /= len(val_loader)

        trial.report(val_recon_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_recon_loss

def run(markets: list = None, tune: bool = False):
    # Load model config if available
    config_path = os.path.join("models", "model_config.json")
    if os.path.exists(config_path) and not tune: # Do not load if tuning
        logger.info(f"Loading model configuration from {config_path}")
        # ... (config loading logic as before)

    market_index_df = get_market_index()
    
    # --- Hyperparameter Tuning (if specified) ---
    if tune:
        logger.info("--- Starting Hyperparameter Tuning with Optuna ---")
        X, y, _ = get_processed_data(config.TARGET_MARKETS[0], market_index_df)
        if X is None:
            logger.error("Tuning failed: No data available for the primary market.")
            return

        study = optuna.create_study(direction='minimize')
        objective_with_data = functools.partial(objective, X=X, y=y)
        study.optimize(objective_with_data, n_trials=50) # n_trials can be adjusted

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

        # Update config with best parameters for the final training run
        config.LEARNING_RATE = best_params['lr']
        config.D_MODEL = best_params['d_model']
        config.N_LAYERS = best_params['n_layers']
        config.N_HEADS = best_params['n_heads']
        config.BATCH_SIZE = best_params['batch_size']
        config.LAMBDA_RECON = best_params['lambda_recon'] # Store the new hyperparameter
        logger.info("Configuration updated with best parameters for this run.")

    # --- Main Training / Fine-tuning Logic ---
    # (The rest of the run function remains the same, using the potentially updated config)
    is_finetuning = markets is not None
    mode_log = "Fine-tuning" if is_finetuning else "Full Training"
    logger.info(f"--- Preparing for {mode_log} ---")
    # ... (The rest of the logic is the same as before)
    # ... it will now use the tuned hyperparameters if tune was True ...
    if is_finetuning:
        X_list, y_list = [], []
        for market in markets:
            X_market, y_market, _ = get_processed_data(market, market_index_df)
            if X_market is not None:
                X_list.append(X_market)
                y_list.append(y_market)
        if not X_list:
            logger.error(f"No data available for fine-tuning markets. Aborting.")
            return
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        epochs = config.EPOCHS // 5
        lr = config.LEARNING_RATE / 10
    else: # Full training
        X, y, _ = get_processed_data(config.TARGET_MARKETS[0], market_index_df)
        if X is None:
            logger.error("Training failed: No data available for the primary market.")
            return
        epochs = config.EPOCHS
        lr = config.LEARNING_RATE

    # --- WGAN-GP Training Loop ---
    for i in range(config.N_ENSEMBLE_MODELS):
        logger.info(f"\n--- Training Ensemble Model {i+1}/{config.N_ENSEMBLE_MODELS} ---")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 - config.TRAIN_SPLIT, random_state=42 + i)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        # Add drop_last=True to prevent batch size of 1, which causes BatchNorm errors.
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # If fine-tuning, load the entire model object. Otherwise, build a new one.
        model_save_path = os.path.join("models", f"model_{i+1}.pth")
        if is_finetuning and os.path.exists(model_save_path):
            try:
                generator = torch.load(model_save_path, map_location=config.DEVICE)
                logger.info(f"Loaded existing model object {i+1} for fine-tuning.")
            except Exception as e:
                logger.error(f"Could not load model object {i+1}: {e}. Rebuilding from scratch.")
                generator = build_model(
                    d_model=config.D_MODEL, n_heads=config.N_HEADS, n_layers=config.N_LAYERS,
                    input_dim=config.N_FEATURES, noise_dim=config.GAN_NOISE_DIM, output_dim=6
                )
        else:
            generator = build_model(
                d_model=config.D_MODEL, n_heads=config.N_HEADS, n_layers=config.N_LAYERS,
                input_dim=config.N_FEATURES, noise_dim=config.GAN_NOISE_DIM, output_dim=6
            )
        
        critic = build_critic()

        optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
        optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
        recon_criterion = nn.MSELoss()
        best_val_recon_loss = float('inf')

        # Determine lambda_recon for this run
        lambda_recon_run = getattr(config, 'LAMBDA_RECON', 100) # Default to 100 if not set by tune

        for epoch in range(epochs):
            generator.train()
            critic.train()
            for batch_idx, (real_sequences, real_paths) in enumerate(train_loader):
                # Safety guard for the last batch if drop_last=False were used
                if real_sequences.size(0) <= 1:
                    continue

                real_sequences, real_paths = real_sequences.to(config.DEVICE), real_paths.to(config.DEVICE)
                
                # Train Critic
                optimizer_C.zero_grad()
                fake_paths = generator(real_sequences).detach()
                real_validity = critic(real_paths)
                fake_validity = critic(fake_paths)
                gradient_penalty = compute_gradient_penalty(critic, real_paths, fake_paths)
                loss_C = fake_validity.mean() - real_validity.mean() + LAMBDA_GP * gradient_penalty
                loss_C.backward()
                optimizer_C.step()

                # Train Generator
                if batch_idx % CRITIC_ITERATIONS == 0:
                    optimizer_G.zero_grad()
                    gen_paths = generator(real_sequences)
                    fake_validity = critic(gen_paths)
                    loss_G_adv = -fake_validity.mean()
                    loss_G_recon = recon_criterion(gen_paths, real_paths)
                    loss_G = loss_G_adv + lambda_recon_run * loss_G_recon
                    loss_G.backward()
                    optimizer_G.step()

            # Validation
            generator.eval()
            val_recon_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(config.DEVICE), batch_y.to(config.DEVICE)
                    outputs = generator(batch_X)
                    loss = recon_criterion(outputs, batch_y)
                    val_recon_loss += loss.item()
            val_recon_loss /= len(val_loader)

            logger.info(f"Epoch [{epoch+1}/{epochs}] | Model [{i+1}] | Val Recon Loss: {val_recon_loss:.6f}")

            if val_recon_loss < best_val_recon_loss:
                best_val_recon_loss = val_recon_loss
                torch.save(generator, model_save_path)
                logger.info(f"Validation loss improved. Model {i+1} saved to {model_save_path}")

        logger.info(f"Finished training model {i+1}. Best validation reconstruction loss: {best_val_recon_loss:.6f}")

    logger.info(f"=== {mode_log} Finished ===")