# Research Log & Methodology Evolution

This document tracks the evolution of the core methodology for the Crypto Predictor project, from its initial conception to the current hybrid architecture. It serves as a log of our thought process and the rationale behind key architectural decisions.

## 1. Initial State: Transformer-Conditioned GAN

The project began with a sophisticated and novel architecture: a **Transformer-Conditioned Generative Adversarial Network (GAN)**.

-   **Core Idea**: Use a Transformer Encoder to learn the rich temporal context of a time-series and feed this context into a GAN Decoder (Generator) to produce realistic future predictions.
-   **Strength**: This was already a strong baseline, moving beyond simple prediction to a generative approach conditioned on learned patterns.

## 2. Phase 1: Formalization for Reproducibility

With the goal of developing a publishable academic paper, we identified several areas in the initial implementation that needed to be formalized to ensure **reproducibility and methodological soundness**.

### 2.1. Similarity Metric: From Euclidean to DTW

-   **Initial State**: The `Pattern Following` strategy used Euclidean distance to measure the similarity between a predicted future pattern and other assets' historical patterns.
-   **Problem**: Euclidean distance is not robust to time-shifts. Two patterns with identical shapes but slight temporal offsets would be considered very different.
-   **Solution**: We replaced the Euclidean distance with **Dynamic Time Warping (DTW)**, the academic standard for time-series similarity measurement. This makes the pattern-matching process far more robust and methodologically sound.

### 2.2. Model Architecture Specification

-   **Initial State**: The `HybridModel` was implemented, but the exact architecture was not explicitly documented within the code.
-   **Problem**: For a paper, the model architecture must be described in unambiguous detail.
-   **Solution**: We analyzed the data flow and formally named the architecture a **"Transformer-Conditioned GAN"**. We added a detailed docstring to the `HybridModel` class explaining the flow: Transformer encodes context -> GAN generates a future sequence based on that context.

### 2.3. Screener Logic Formalization

-   **Initial State**: The logic for selecting "trending" markets used a hardcoded "magic number" for the minimum trading volume (e.g., `100,000,000` KRW).
-   **Problem**: Magic numbers obscure important experimental parameters and hinder reproducibility.
-   **Solution**: We moved the minimum volume threshold into the central `config.py` file as `SCREENER_MIN_VOLUME_KRW`. This defines it as a clear, explicit parameter of our trading strategy.

## 3. A New Idea: From Time-Series to Images & The Power of CNNs

A key creative insight was proposed: **What if we treat time-series patterns as images?** This opened the door to leveraging the immense power of Convolutional Neural Networks (CNNs).

-   **Core Insight**: The strength of CNNs lies in detecting local patterns in data (like pixels in an image). This same strength can be applied to time-series data to find short-term motifs.
-   **The Explainability Breakthrough**: By using a CNN, we can apply well-established explainability techniques like **CAM (Class Activation Mapping) or Grad-CAM**. This would allow us to *visualize* which parts of the historical data (which "pixels" in our pattern "image") the model focused on to make its prediction. This directly addresses the "black box" problem in financial AI and is a massive contribution.

## 4. Current Architecture: A Unified Model for Two Competing Approaches

This new insight led to a crucial research question: What is the *best* way to apply CNNs here?

-   **Approach A: 1D CNN on Raw Time-Series**: The most direct method. A 1D CNN scans the 1D time-series to find local, temporal patterns (e.g., sharp dips, spikes).
    -   *Pros*: Fast, intuitive, no complex data transformation.
    -   *Cons*: May miss non-local or more complex structural patterns.

-   **Approach B: 2D CNN on GAF Images**: A more complex but potentially more powerful method. The 1D time-series is converted into a 2D image using a Gramian Angular Field (GAF). A 2D CNN then finds spatial patterns in the image, which correspond to complex, long-range temporal correlations.
    -   *Pros*: Can potentially capture more complex, global patterns.
    -   *Cons*: Computationally more expensive and adds the complexity of the GAF transformation.

To properly investigate this research question, we have refactored the `HybridModel` to support both approaches.

-   **Implementation**: The model now includes a `config.CNN_MODE` setting. By changing this setting to `'1D'` or `'2D'`, the model will dynamically use the corresponding parallel path (1D CNN or 2D CNN) alongside the Transformer.
-   **Next Step**: This flexible architecture allows us to rigorously conduct comparative experiments (Phase 2) to determine which approach yields better results for this specific problem, strengthening the final paper.
