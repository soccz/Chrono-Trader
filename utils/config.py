
import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # System Settings
    DEVICE: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DB_PATH: str = "crypto_data.db"
    LOG_DIR: str = "logs"
    PREDICTION_DIR: str = "predictions"
    RECOMMENDATION_DIR: str = "recommendations"
    MODEL_SAVE_PATH: str = "models/best_model.pth"

    # Data Collection
    UPBIT_API_URL: str = "https://api.upbit.com/v1/candles/minutes/60" # 60-minute candles
    TARGET_MARKETS: List[str] = field(default_factory=lambda: ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-DOGE", "KRW-SOL"]) # Example markets

    # Screener Settings
    SCREENER_MIN_VOLUME_KRW: int = 100000000 # 1억 KRW

    # Data Preprocessing
    SEQUENCE_LENGTH: int = 48  # Use 48 hours of data to predict the next
    N_FEATURES: int = 15 # Number of features after preprocessing (incl. alpha, beta)

    # Model Parameters (tuned)
    CNN_MODE: str = '1D' # '1D' or '2D'. Determines the CNN architecture to use.
    IMAGE_SIZE: int = 24 # For 2D CNN, the size of the GAF image (e.g., 24x24)
    N_ENSEMBLE_MODELS: int = 3
    D_MODEL: int = 256  # Transformer model dimension
    N_HEADS: int = 8    # Number of attention heads
    N_LAYERS: int = 4   # Number of transformer layers
    GAN_NOISE_DIM: int = 100 # GAN noise dimension

    # Training Settings (tuned)
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.0095
    EPOCHS: int = 50
    TRAIN_SPLIT: float = 0.8 # 80% for training, 20% for validation

    # Prediction & Recommendation
    PATTERN_LOOKBACK_HOURS: int = 24 # Hours of historical data for pattern matching
    PREDICTION_THRESHOLD: float = 0.7  # Confidence threshold for making a trade
    MAX_POSITIONS: int = 3 # Max number of concurrent trades
    KELLY_FRACTION: float = 0.2 # Kelly criterion fraction for position sizing
    STOP_LOSS_PCT: float = 0.05 # 5% stop-loss
    TAKE_PROFIT_PCT: float = 0.10 # 10% take-profit

# Instantiate the config
config = Config()
