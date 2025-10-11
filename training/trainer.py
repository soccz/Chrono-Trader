
import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json

from utils.config import config
from utils.logger import logger
from data.preprocessor import get_processed_data, get_market_index
from models.hybrid_model import build_model
from models.critic import build_critic

# --- WGAN-GP Hyperparameters ---
CRITIC_ITERATIONS = 5  # Train critic 5 times per generator train
LAMBDA_GP = 10         # Gradient penalty lambda
LAMBDA_RECON = 100     # Reconstruction loss lambda (to keep predictions accurate)

def compute_gradient_penalty(critic, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN-GP."""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1).to(config.DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    critic_interpolates = critic(interpolates)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).to(config.DEVICE),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def run(markets: list = None, tune: bool = False):
    if tune:
        logger.warning("Hyperparameter tuning is not supported with WGAN-GP in this version. Ignoring --tune flag.")

    # Load model config if available
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

    # Determine if we are fine-tuning or doing a full training
    is_finetuning = markets is not None
    mode_log = "Fine-tuning" if is_finetuning else "Full Training"
    logger.info(f"--- Preparing for {mode_log} ---")

    market_index_df = get_market_index()
    
    if is_finetuning:
        # Fine-tuning on a specific list of markets
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
        epochs = config.EPOCHS // 5 # Fewer epochs for fine-tuning
        lr = config.LEARNING_RATE / 10 # Smaller LR for fine-tuning
    else:
        # Full training on the primary market
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
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Initialize Generator and Critic
        generator = build_model(
            d_model=config.D_MODEL, n_heads=config.N_HEADS, n_layers=config.N_LAYERS,
            input_dim=config.N_FEATURES, noise_dim=config.GAN_NOISE_DIM, output_dim=6
        )
        critic = build_critic()

        # Load existing model weights if fine-tuning
        model_save_path = os.path.join("models", f"model_{i+1}.pth")
        if is_finetuning and os.path.exists(model_save_path):
            try:
                generator.load_state_dict(torch.load(model_save_path, map_location=config.DEVICE))
                logger.info(f"Loaded existing weights for model {i+1} for fine-tuning.")
            except Exception as e:
                logger.error(f"Could not load weights for fine-tuning model {i+1}: {e}. Training from scratch.")

        # Optimizers
        optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
        optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
        
        recon_criterion = nn.MSELoss()
        best_val_recon_loss = float('inf')

        for epoch in range(epochs):
            generator.train()
            critic.train()
            
            for batch_idx, (real_sequences, real_paths) in enumerate(train_loader):
                real_sequences = real_sequences.to(config.DEVICE)
                real_paths = real_paths.to(config.DEVICE)

                # -------------------
                #  Train Critic
                # -------------------
                optimizer_C.zero_grad()
                
                # Generate a batch of fake paths
                fake_paths = generator(real_sequences).detach()
                
                # Get critic scores
                real_validity = critic(real_paths)
                fake_validity = critic(fake_paths)
                
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(critic, real_paths, fake_paths)
                
                # Critic loss
                loss_C = fake_validity.mean() - real_validity.mean() + LAMBDA_GP * gradient_penalty
                
                loss_C.backward()
                optimizer_C.step()

                # Train the generator only once every CRITIC_ITERATIONS batches
                if batch_idx % CRITIC_ITERATIONS == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizer_G.zero_grad()
                    
                    # Generate a batch of fake paths
                    gen_paths = generator(real_sequences)
                    
                    # Get critic score for generated paths
                    fake_validity = critic(gen_paths)
                    
                    # Generator's adversarial loss
                    loss_G_adv = -fake_validity.mean()
                    
                    # Generator's reconstruction loss (to ensure it predicts accurately)
                    loss_G_recon = recon_criterion(gen_paths, real_paths)
                    
                    # Total generator loss
                    loss_G = loss_G_adv + LAMBDA_RECON * loss_G_recon
                    
                    loss_G.backward()
                    optimizer_G.step()

            # --- End of Epoch Validation ---
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
                torch.save(generator.state_dict(), model_save_path)
                logger.info(f"Validation loss improved. Model {i+1} saved to {model_save_path}")

        logger.info(f"Finished training model {i+1}. Best validation reconstruction loss: {best_val_recon_loss:.6f}")

    logger.info(f"=== {mode_log} Finished ===")
