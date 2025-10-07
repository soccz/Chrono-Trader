import torch
import torch.nn as nn
from utils.config import config
from models.transformer_encoder import build_transformer_encoder
from models.gan_decoder import build_gan_decoder

class HybridModel(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, input_dim, noise_dim, output_dim):
        super(HybridModel, self).__init__()
        self.encoder = build_transformer_encoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )
        self.decoder = build_gan_decoder(
            context_dim=d_model,
            noise_dim=noise_dim,
            output_dim=output_dim
        )
        # Store noise_dim for use in the forward pass
        self.noise_dim = noise_dim

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        
        # 1. Encode the time-series context
        encoded_context = self.encoder(src)
        # Use the context from the last time step
        last_step_context = encoded_context[:, -1, :] # Shape: (batch_size, d_model)

        # 2. Generate prediction using GAN decoder
        # Generate random noise for the GAN
        noise = torch.randn(src.size(0), self.noise_dim).to(config.DEVICE)
        
        # Prediction is now a sequence of future values
        prediction = self.decoder(last_step_context, noise)
        # prediction shape: (batch_size, FUTURE_WINDOW_SIZE)
        
        return prediction

def build_model(d_model, n_heads, n_layers, input_dim, noise_dim, output_dim) -> HybridModel:
    """Builds and returns the hybrid model with specified parameters."""
    model = HybridModel(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        input_dim=input_dim,
        noise_dim=noise_dim,
        output_dim=output_dim
    )
    model.to(config.DEVICE)
    return model
