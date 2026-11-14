# Contributing to House Price Prediction

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/house-price-prediction.git`
3. Create a virtual environment: `python -m venv .venv`
4. Activate it:
   - Windows: `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Install pre-commit hooks: `pre-commit install`

## Development Workflow

### Making Changes

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Run linting: `pre-commit run --all-files`
5. Commit your changes: `git commit -m "feat: description of your change"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a Pull Request

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or modifying tests
- `chore:` Maintenance tasks

Example: `feat: add support for XGBoost model`

## Testing

Run the full test suite:

```bash
pytest -v
```

Run with coverage:

```bash
pytest --cov=src --cov-report=html
```

## Code Style

We use:
- **Black** for formatting (line length: 88)
- **isort** for import sorting
- **flake8** for linting

Run formatters manually:

```bash
black src tests
isort src tests
```

Or use pre-commit which runs automatically on commit.

## Adding New Models

To add a new candidate model:

1. Add the estimator to `candidate_models()` in `src/modeling.py`
2. Add hyperparameter search space in `train.py` if it's a tree ensemble
3. Ensure it works with the existing pipeline structure
4. Add tests if needed

## Data Handling

- Raw data in `data/raw/` is included for reproducibility
- Do not commit large generated files to `data/processed/`
- Generated reports go to `reports/` (already in .gitignore for large files)

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add comments for complex logic

## Reporting Issues

Use GitHub Issues with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

## Questions?

Open a GitHub Discussion or issue for any questions.