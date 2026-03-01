# BioLENS

This repository implements the MVP described in the paper "Towards Transparent Embeddings for Biomarker Discovery": encode EEG recordings, z-score and PCA1-project embeddings, fit a cubic B-spline over ordered outcomes, evaluate constraints (proximity, monotonicity, reproducibility) plus fit adequacy vs. a null MSE, and train a constraint-aware encoder.

There are six primary scripts:
- `main.py` — entrypoint; runs the MVP using synthetic data and prints key metrics
- `embed.py` — functions to generate embeddings from raw signals, preprocess, and project
- `spline.py` — functions to fit a cubic spline and extract key characteristics
- `constraints.py` — functions to evaluate constraints
- `constants.py` — constants used in the MVP
- `train_constraint_aware_encoder.py` - Lightning implementation of the constraint-aware learning framework (Section 4.4) with synthetic data

The code to generate figures can be found in the `figures/` subdirectory.

## Running the code

 Install uv:
 ```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
 ```

Environment setup:

```shell
uv venv .venv
uv sync
source .venv/bin/activate
```

Run the scripts:

```shell
# MVP
python main.py
python train_constraint_aware_encoder.py

# Generate figures
cd figures
python figure_1.py
```
