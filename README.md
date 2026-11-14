# House Price Prediction

A reproducible, modular regression project for the Ames Housing / Kaggle House Prices dataset. It includes data checks, exploratory analysis, engineered housing features, leakage-conscious preprocessing, six model families, cross-validation, model selection, metrics, diagnostics, a persisted pipeline, and a command-line prediction interface.

## Dataset and scope

The provided archive was extracted into `data/raw/`; no extra dataset download or manual extraction is needed. Training data has **1,460 rows and 81 columns** (including target `SalePrice`); test data has **1,459 rows and 80 columns**. The training target ranges from **$34,900 to $755,000**, with a median of **$163,000** and mean of approximately **$180,921**. The dataset contains a mix of numeric and categorical property features and is appropriate for supervised tabular regression. It is a competition dataset, not a live appraisal source.

## Repository layout

```text
├── data/
│   ├── raw/                 # Extracted original CSV/TXT files
│   └── processed/           # Optional generated data artifacts
├── models/                  # Saved fitted pipeline
├── notebooks/               # Three workflow notebooks
├── reports/
│   ├── figures/             # Generated plots
│   └── metrics/             # JSON/CSV metrics
├── src/
│   ├── __init__.py          # Package exports
│   ├── config.py            # Centralized portable paths and seed
│   ├── data_loader.py       # Dataset readers
│   ├── data_validation.py   # Schema and quality checks
│   ├── exploratory_analysis.py # EDA figures and overview
│   ├── preprocessing.py     # sklearn preprocessing
│   ├── feature_engineering.py # Derived property features
│   ├── modeling.py          # Candidate estimators and pipeline builder
│   ├── train.py             # CV comparison, tuning, validation, persistence
│   ├── evaluate.py          # Regression metrics and diagnostics
│   └── predict.py           # CLI and prediction generation
├── tests/                   # pytest checks
├── .github/workflows/       # CI/CD pipelines
├── Dockerfile               # Container support
└── requirements*.txt        # Dependencies
```

## Setup

### Local Development

Python 3.10+ is required.

```bash
# Clone and setup
git clone https://github.com/alirezahs81/house-price-prediction.git
cd house-price-prediction

# Create virtual environment
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Docker

```bash
# Build image
docker build -t house-price-prediction .

# Run training
docker run --rm house-price-prediction train

# Run prediction with volume mounts
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  house-price-prediction predict
```

## Global paths

All filesystem locations are rooted dynamically in `src/config.py` using `Path(__file__).resolve().parents[1]`. No personal machine paths or current-working-directory assumptions are used for default data/model/report locations. The prediction CLI accepts custom input/output/model paths.

## Exploratory analysis findings

- No duplicate training rows were found.
- The target is right-skewed; training uses `log1p(SalePrice)` and converts predictions back with `expm1`.
- Several columns have substantial missingness, including `PoolQC`, `MiscFeature`, `Alley`, `Fence`, and `MasVnrType`.
- For features whose missingness denotes absence (e.g., pool, alley, basement, garage, fence, fireplace features), missing categories are encoded as `None`. Other categorical values are imputed using the most frequent category; numeric values use training-fold medians.
- EDA figures are generated in `reports/figures/`.

## Feature engineering

- `TotalSF`: basement + first-floor + second-floor square footage.
- `TotalBathrooms`: full baths plus half-baths weighted by 0.5, including basement bathrooms.
- `HouseAgeAtSale`: sale year minus year built, clipped at zero.
- `YearsSinceRemodel`: sale year minus remodel year, clipped at zero.
- `HasGarage`: whether garage area is positive.
- `QualityTimesLivingArea`: overall quality multiplied by above-ground living area.

The same deterministic feature function is applied to train and inference data. `Id` is retained only for output and is excluded from modeling.

## Modeling and validation

The project compares a median baseline, Ridge, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and HistGradientBoosting. It first reserves a final holdout that is excluded from all model and hyperparameter selection. Each tunable family receives the same bounded eight-iteration randomized search using shuffled five-fold CV on the training partition. The selected configuration from every family is then compared using the same repeated five-fold CV (five folds repeated twice) on that training partition. The untouched holdout is evaluated once for final diagnostics, after which the selected pipeline is refit on all labeled rows for test inference. Imputers, scalers, and encoders are fitted inside each cross-validation fold. All preprocessing is part of the saved sklearn Pipeline.

Metrics include MAE, MSE, RMSE, R² and RMSLE on the original price scale, plus cross-validation log-RMSE, its fold-to-fold standard deviation, training log-RMSE, validation log-RMSE, and their generalization gap. The training report records the selected model, search parameters, number of validation rows, target transform, and confirms that the holdout was not used for model selection. Metrics are generated on each run and saved in `reports/model_comparison.csv`, `reports/metrics/validation_metrics.json`, and related files; previous run-specific scores are intentionally not hard-coded here.

## Run

```bash
# Generate EDA reports only
python -m main eda

# Cross-validate candidates, tune the selected model, evaluate and save model
python -m main train

# Predict the supplied competition test set (requires a trained model)
python -m main predict

# Custom paths
python -m src.predict --input path/to/features.csv --output path/to/predictions.csv --model path/to/model.joblib
```

Prediction output has exactly `Id` and `SalePrice`, one row per input row. Values are checked for finite, positive output.

## Makefile Commands

```bash
# Install dependencies
make install

# Run tests
make test

# Generate EDA
make eda

# Train model
make train

# Predict
make predict
```

## Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_preprocessing.py -v
```

GitHub Actions runs pytest and linting on pushes and pull requests. The tests cover dataset loading, validation, missing-value handling, engineered features, preprocessing, exploratory analysis, prediction, and a small end-to-end estimator fit.

## CI/CD

- **Tests workflow** (`.github/workflows/tests.yml`): Runs pre-commit checks (isort, flake8, mypy, and file validation) plus pytest with coverage on every push/PR
- **Train workflow** (`.github/workflows/train.yml`): Runs full training pipeline on pushes to main, uploads model and reports as artifacts

## Development

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint
flake8 src tests
mypy src

# Run all pre-commit hooks
pre-commit run --all-files
```

### Adding New Models

To add a new candidate model:

1. Add the estimator to `candidate_models()` in `src/modeling.py`
2. Add hyperparameter search space in `src/train.py` if it's a tree ensemble
3. Ensure it works with the existing pipeline structure
4. Add tests if needed

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## Limitations and future work

This is a small historical competition dataset; predictions should not be treated as professional appraisals. Validation scores are split-dependent and may not represent real-world generalization. Potential extensions include:

- Residual-based error analysis
- Model-specific feature importance (SHAP values)
- Broader hyperparameter search (Optuna, Bayesian optimization)
- Robust outlier sensitivity experiments
- Model card describing deployment constraints
- Support for additional model families (XGBoost, LightGBM, CatBoost)
- Cross-validation with grouped splits (e.g., by neighborhood)

## License and data rights

The MIT license applies to original project source code only. The dataset and its description remain subject to their original terms; this repository does not claim ownership or relicense them.

## Acknowledgments

- Ames Housing dataset by Dean De Cock
- Kaggle House Prices competition
- scikit-learn, pandas, numpy, matplotlib communities
