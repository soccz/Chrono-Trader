import torch
import torch.nn as nn

class GANDecoder(nn.Module):
    """A Generator model for GAN, used as a decoder to produce predictions."""
    def __init__(self, context_dim, noise_dim, output_dim):
        super(GANDecoder, self).__init__()
        self.noise_dim = noise_dim
        
        self.model = nn.Sequential(
            nn.Linear(context_dim + noise_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
        )

    def forward(self, context_vector, noise):
        combined_input = torch.cat((context_vector, noise), dim=1)
        output = self.model(combined_input)
        return output

def build_gan_decoder(context_dim, noise_dim, output_dim):
    """Builds and returns the GAN decoder."""
    return GANDecoder(
        context_dim=context_dim,
        noise_dim=noise_dim,
        output_dim=output_dim
    )
