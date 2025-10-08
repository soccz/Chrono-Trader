import torch
import numpy as np
import os
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.hybrid_model import build_model
from data import preprocessor, collector
from utils.config import config
from utils.logger import logger
import matplotlib.pyplot as plt

class ReshapeForCAM(torch.nn.Module):
    """A wrapper to reshape the model output for Grad-CAM."""
    def __init__(self, model):
        super(ReshapeForCAM, self).__init__()
        self.model = model

    def forward(self, x):
        # The model's raw output is (batch, seq_len). 
        # We need a single score for Grad-CAM. We'll use the mean of the predicted sequence.
        output = self.model(x)
        return output.mean(dim=1) # Return a single score per batch item

def run_grad_cam(market: str, model_path: str):
    logger.info(f"--- Running Grad-CAM analysis for {market} ---")

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
        output_dim=6 # Corresponds to FUTURE_WINDOW_SIZE
    )
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    # Grad-CAM for 1D CNN requires specific layer targeting
    if config.CNN_MODE != '1D':
        logger.error("Grad-CAM is currently implemented for '1D' CNN mode only.")
        return
    
    # Target the last conv layer in the cnn_encoder
    target_layer = model.cnn_encoder[2]

    # 2. Prepare Input Data
    collector.collect_market_data(market, days=5) # Ensure we have fresh data
    market_index_df = preprocessor.get_market_index()
    X, _, scaler = preprocessor.get_processed_data(market, market_index_df)
    if X is None:
        logger.error(f"Could not process data for {market}")
        return

    last_sequence = X[-1]
    input_tensor = torch.FloatTensor([last_sequence]).to(config.DEVICE)

    # 3. Setup and Run Grad-CAM
    # We need a wrapper because the library expects a single output score, not a sequence.
    model_for_cam = ReshapeForCAM(model)
    targets = [ClassifierOutputTarget(0)] # Target the single output score

    cam = GradCAM(model=model_for_cam, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # grayscale_cam is a 1D heatmap. We take the first one from the batch.
    grayscale_cam = grayscale_cam[0, :]

    # 4. Visualize and Save
    # Use the 'close' price from the original (unprocessed) data for visualization
    raw_data, _ = preprocessor.get_raw_data(market)
    raw_sequence = raw_data['close'].values[-config.SEQUENCE_LENGTH:]

    # Normalize heatmap for better visualization
    heatmap = cv2.normalize(grayscale_cam, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.squeeze() # from (48, 1, 3) to (48, 3)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(raw_sequence, color='blue', label='Price')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Overlay heatmap
    for i in range(len(raw_sequence) - 1):
        ax1.fill_between([i, i+1], raw_sequence.min(), raw_sequence.max(), color=heatmap[i]/255., alpha=0.4)

    plt.title(f"Grad-CAM Analysis for {market}")
    plt.xlabel("Time Steps (Hours)")
    
    # Save the figure
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"grad_cam_{market}.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Grad-CAM analysis saved to {save_path}")
