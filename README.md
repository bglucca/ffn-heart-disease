# PyTorch Sandbox - Heart Disease prediction

A sandbox environment for exploring PyTorch fundamentals through a practical case: predicting heart disease using a Feed-Forward Neural Network (MLP).

## Quick Start

1. **Install dependencies** (using [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync
   ```

2. **Run the training script**:
   ```bash
   uv run python train_network.py
   ```
   This trains a simple classifier and saves the model to `models/` along with training metrics in `data/`.

3. **Explore the notebooks**:
   - `notebooks/01-data-ingestion.ipynb` — Data loading and preprocessing exploration
   - `notebooks/02-results.ipynb` — Visualize training results and model performance

## Project Structure

```
├── src/                 # Core modules (model, data loader, early stopping)
├── notebooks/           # Jupyter notebooks for exploration
├── train_network.py     # Main training script
├── models/              # Saved model checkpoints
└── data/                # Training metrics and outputs
```
