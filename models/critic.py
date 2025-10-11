
import torch.nn as nn
from utils.config import config

class Critic(nn.Module):
    """
    The Critic (or Discriminator) model for the WGAN-GP.
    It takes a sequence of 6 future price changes and outputs a single
    scalar score indicating how 'real' it thinks the sequence is.
    """
    def __init__(self):
        super(Critic, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(6, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1) # Outputs a single scalar score
        )

    def forward(self, x):
        return self.model(x)

def build_critic():
    """Builds and returns the Critic."""
    critic = Critic()
    critic.to(config.DEVICE)
    return critic
