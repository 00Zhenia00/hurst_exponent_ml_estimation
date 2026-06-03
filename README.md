# Machine Learning Estimation of the Hurst Exponent

This repository contains the code for the paper **"Machine Learning Estimation of the Hurst Exponent"**. It provides a complete pipeline for generating synthetic fractional Brownian motion (fBm) data, training and tuning machine learning models to estimate the Hurst exponent from short time-series, and evaluating performance against classical statistical estimators.

## Overview

The Hurst exponent *H* characterizes long-range dependence in time series:
- *H* = 0.5 — random walk (Brownian motion)
- *H* > 0.5 — persistent (trending) process
- *H* < 0.5 — mean-reverting process

Classical estimators (R/S analysis, DFA) require long time series to be reliable. This work investigates whether machine learning models trained on synthetic fBm data can accurately estimate *H* from short sequences (25–100 observations) and generalizes the approach to real financial data.

### Models evaluated

| Category | Models |
|---|---|
| Linear | Ridge Regression, Lasso Regression |
| Tree-based | Decision Tree, Random Forest, LightGBM, XGBoost |
| Neural networks | MLP, RNN, CNN |
| Classical baselines | R/S Analysis, Detrended Fluctuation Analysis (DFA) |

## Repository Structure

```
├── 1-data-generation.ipynb   # Generate synthetic fBm training and test data
├── 2-tuning.ipynb            # Hyperparameter optimization with Optuna
├── 3-training.ipynb          # Train models on the full dataset
├── 4-scoring.ipynb           # Evaluate models on the test set with SHAP analysis
├── 5-feature-selection.ipynb # Feature fraction ablation study
├── 6-inference.ipynb         # Apply models to real financial data (S&P 500, Dow Jones)
├── 7-extra.ipynb             # Summary pivot tables and cross-model comparisons
├── config.yaml               # Central configuration (models, sample sizes, seeds, etc.)
├── requirements.txt          # Python dependencies
├── data/
│   ├── train/                # 100,000 training samples per series length (25/50/100)
│   └── test/                 # 1,000 test samples per series length
├── models/                   # Trained model artifacts (joblib format)
├── hyperparams/              # Tuned hyperparameter sets (JSON format)
├── artifacts/                # Generated figures (feature importance, SHAP plots)
└── utils/
    ├── config.py             # YAML config loader
    ├── shared.py             # Model factory and save/load utilities
    ├── keras_models.py       # RNN and CNN architecture definitions
    ├── dfa_estimator.py      # DFA classical estimator (scikit-learn API)
    ├── rs_estimator.py       # R/S classical estimator (scikit-learn API)
    ├── tuning.py             # Optuna objective functions with MLflow logging
    ├── training.py           # Model training with feature importance extraction
    ├── scoring.py            # Model evaluation with SHAP explanations
    ├── mlflow_utils.py       # MLflow experiment and run management
    └── visualization.py      # Correlation and heatmap plotting utilities
```

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/<your-org>/hurst_exponent_ml_estimation.git
cd hurst_exponent_ml_estimation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the Experiments

Run the numbered notebooks in order. Pre-generated data and trained models are included so individual stages can be run independently.

### 1. Data generation

[1-data-generation.ipynb](1-data-generation.ipynb) — generates synthetic fBm trajectories with Hurst exponents sampled uniformly from (0, 1). Each trajectory is subsampled to lengths of 25, 50, and 100 observations and normalized. Outputs:
- `data/train/fbm_{25,50,100}x100000.csv` — 100,000 training samples per length
- `data/test/fbm_{25,50,100}x1000.csv` — 1,000 test samples per length

### 2. Hyperparameter tuning

[2-tuning.ipynb](2-tuning.ipynb) — runs Optuna over each (model, series length) combination. Tuning uses 3-fold cross-validation for classical ML models and an 80/20 train/validation split for RNN and CNN. Results are logged to MLflow and saved to `hyperparams/`.

### 3. Training

[3-training.ipynb](3-training.ipynb) — trains each model using the best hyperparameters on the full training set. Models are saved to `models/` and feature importance plots are saved to `artifacts/training/`.

### 4. Scoring

[4-scoring.ipynb](4-scoring.ipynb) — evaluates all models on the held-out test set. Computes RMSE, generates SHAP explanations for all model types, and saves plots to `artifacts/scoring/shap/`.

### 5. Feature fraction ablation

[5-feature-selection.ipynb](5-feature-selection.ipynb) — studies the effect of using a fraction of the time-series observations (0.3–1.0, selected symmetrically from both ends) on estimation accuracy.

### 6. Inference on financial data

[6-inference.ipynb](6-inference.ipynb) — applies all trained models to historical S&P 500 and Dow Jones price data using a rolling window of 200 observations. Produces Hurst exponent estimates over time and inter-model correlation heatmaps.

### 7. Summary tables

[7-extra.ipynb](7-extra.ipynb) — loads all scoring metrics from MLflow and produces summary pivot tables (RMSE and error std) for models × series lengths.

## Experiment Tracking

All experiments are tracked with MLflow. To inspect results locally:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser. Four experiments are present:
- `tune_hyperparameters`
- `training`
- `scoring`
- `tuning_fs_1` (feature fraction ablation)

## Configuration

All experiment parameters are controlled through [config.yaml](config.yaml):

```yaml
sample_sizes: [25, 50, 100]     # Time-series lengths evaluated
train_samples_num: 100000       # Training samples per length
test_samples_num: 1000          # Test samples per length
models: [Ridge, Lasso, DecisionTree, RandomForest,
         LightGBM, XGBoost, MLP, RNN, CNN, RS, DFA]
random_seed: 42
n_jobs: 6
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
