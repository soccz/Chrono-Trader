import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from models.hybrid_model import build_model
from data import preprocessor, collector, database
from utils.config import config
from utils.logger import logger
from inference.predictor import load_and_apply_model_config

def run_grad_cam(market: str, model_path: str):
    logger.info(f"--- Running Manual Grad-CAM analysis for {market} ---")

    # 0. Load Model Configuration
    load_and_apply_model_config()

    # 1. Load Model
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    model = build_model(
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        input_dim=config.N_FEATURES,
        noise_dim=config.GAN_NOISE_DIM,
        output_dim=6
    )
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    if config.CNN_MODE != '1D':
        logger.error("Manual Grad-CAM is currently implemented for '1D' CNN mode only.")
        return

    # 2. Prepare Input Data
    collector.collect_market_data(market, days=5)
    market_index_df = preprocessor.get_market_index()
    X, _, _ = preprocessor.get_processed_data(market, market_index_df)
    if X is None:
        logger.error(f"Could not process data for {market}")
        return

    last_sequence = X[-1]
    input_tensor = torch.FloatTensor([last_sequence]).to(config.DEVICE)
    input_tensor.requires_grad = True

    # 3. Implement and Register Hooks
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Target the last conv layer in the cnn_encoder
    target_layer = model.cnn_encoder[2]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # 4. Forward and Backward Pass
    output = model(input_tensor)
    score = output.mean() # Use the mean of the output sequence as the score
    model.zero_grad()
    score.backward()

    # VERY IMPORTANT: Remove hooks after use to prevent memory leaks
    forward_handle.remove()
    backward_handle.remove()

    # 5. Calculate Grad-CAM Heatmap
    if not gradients or not feature_maps:
        logger.error("Failed to capture gradients or feature maps from hooks.")
        return

    # Get the captured values (for a single-item batch)
    grads = gradients[0].data.cpu().numpy()[0]
    fmaps = feature_maps[0].data.cpu().numpy()[0]

    # Calculate weights (alpha)
    weights = np.mean(grads, axis=1)
    
    # Compute heatmap
    cam = np.zeros(fmaps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmaps[i, :]

    # Apply ReLU
    cam = np.maximum(cam, 0)
    
    # Upsample to original sequence length
    cam = cv2.resize(cam, (1, config.SEQUENCE_LENGTH)) # Target shape is (width, height)

    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8) # Add epsilon to avoid division by zero

    # 6. Visualize and Save
    # Use the 'close' price from the original (unprocessed) data for visualization
    raw_data_query = f"SELECT timestamp, close FROM crypto_data WHERE market = '{market}' ORDER BY timestamp DESC LIMIT {config.SEQUENCE_LENGTH}"
    raw_data = database.load_data(raw_data_query).sort_values(by='timestamp', ascending=True)
    raw_sequence = raw_data['close'].values

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(raw_sequence, color='blue', label='Price')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    for i in range(len(raw_sequence) - 1):
        ax1.fill_between([i, i+1], raw_sequence.min(), raw_sequence.max(), color=heatmap[i]/255., alpha=0.4)

    plt.title(f"Manual Grad-CAM Analysis for {market}")
    plt.xlabel("Time Steps (Hours)")
    
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"grad_cam_{market}.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Grad-CAM analysis saved to {save_path}")