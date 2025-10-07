# Chrono-Trader: Hybrid Transformer-GAN Crypto Forecasting Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Chrono-Trader** is a sophisticated cryptocurrency forecasting and recommendation engine. It leverages a state-of-the-art hybrid model combining a **Transformer Encoder** with a **GAN (Generative Adversarial Network) Decoder** to analyze market trends and generate actionable trading signals.

The system continuously learns from new market data, fine-tuning its ensemble of models to adapt to ever-changing market dynamics.

## Key Features

- **Hybrid AI Model**: Utilizes a Transformer Encoder to understand market context and a GAN Decoder to generate realistic future price scenarios.
- **Ensemble Learning**: Employs an ensemble of three hybrid models to ensure robust and stable predictions.
- **Dual-Strategy Recommendations**: Generates trading signals based on two distinct strategies:
    1.  **High-Confidence Trends**: Identifies trending assets and recommends those with the highest prediction confidence.
    2.  **Dynamic Pattern Following**: Finds assets whose recent price action mimics the predicted future of top-performing assets.
- **Continuous Learning**: A `daily` pipeline automatically collects new data, fine-tunes the models, and generates updated recommendations.
- **Extensible & Configurable**: Key model parameters and operational modes are easily configurable.

## Architecture

The core of Chrono-Trader is its hybrid model architecture, which processes time-series data to forecast future price movements.

```mermaid
graph TD
    A[Input: Time-Series Data <br> (Price, Volume, Indicators)] --> B{Transformer Encoder};
    B --"Encoded Context<br>(Market Understanding)"--> C{GAN Decoder};
    D[Random Noise] --> C;
    C --"Generated Future Pattern<br>(6-hour Forecast)"--> E[Output: Prediction];
```

## Project Structure

```
/
├─── data/              # Data collection, preprocessing, and database logic
├─── inference/         # Prediction and recommendation generation logic
├─── models/            # Model architecture (Transformer, GAN) and saved weights (.pth)
├─── training/          # Model training and evaluation scripts
├─── utils/             # Utility scripts (config, logger, screener)
├─── recommendations/   # Directory for saving recommendation CSVs
├─── main.py            # Main CLI entry point for the project
├─── requirements.txt   # Python dependencies
└─── README.md          # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/Chrono-Trader.git
    cd Chrono-Trader
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    *Note: The `TA-Lib` library requires the underlying C library to be installed first.*
    
    On macOS:
    ```bash
    brew install ta-lib
    pip install -r requirements.txt
    ```
    On Debian/Ubuntu:
    ```bash
    sudo apt-get install -y libta-lib-dev
    pip install -r requirements.txt
    ```

## Usage

Chrono-Trader is operated via the `main.py` script with different modes.

-   **Initialize the Database:**
    *Must be run once before any other operation.*
    ```bash
    python main.py --mode init_db
    ```

-   **Initial Model Training:**
    *Run this to train the models from scratch. Requires a significant amount of data.*
    ```bash
    # Collect data for 90 days and train
    python main.py --mode train --days 90
    ```

-   **Run Daily Recommendation Pipeline:**
    *This is the main operational mode. It collects new data, fine-tunes the models, and generates new recommendations.*
    ```bash
    python main.py --mode daily
    ```

-   **Generate Quick Recommendations:**
    *Generates recommendations using existing models without fine-tuning. Faster than `daily` mode.*
    ```bash
    python main.py --mode quick-recommend
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.