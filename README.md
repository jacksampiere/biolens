# BioLENS

This repository implements the MVP described in the paper "Towards Transparent Embeddings for Biomarker Discovery": encode EEG recordings, z-score and PCA1-project embeddings, fit a cubic B-spline $g$ over ordered outcomes, and evaluate motif-derived constraints (proximity, monotonicity, reproducibility) plus fit adequacy vs. a null MSE.

There are five primary scripts:
- `main.py` — entry point; runs the MVP using synthetic data and prints key metrics
- `embed.py` — functions to generate embeddings from raw signals, preprocess, and project
- `spline.py` — functions to fit a cubic spline and extract key characteristics
- `constraints.py` — functions to evaluate constraints as described in the paper
- `constants.py` — constants used in the MVP

The code to generate figures can be found in the `figures/` subdirectory.

## Running the code

**Environment setup:**
```shell
# Install pyenv
curl -fsSL https://pyenv.run | bash
# Install python 3.12.6
pyenv install 3.12.6
# Set as local version
pyenv local 3.12.6

# Create and activate env
python -m venv .venv
source .venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

**Implementation:**
```shell
# Run the MVP
python main.py

# Generate figures
cd figures
python figure_1.py
```
