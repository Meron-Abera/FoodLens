"""Locations of data and outputs, relative to the repository root.

Everything is resolved from this file's location, so the notebooks and scripts
work wherever the repository is cloned.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / 'data'
FOOD101_DIR = DATA_DIR / 'food-101'          # downloaded by `python -m src.data.download`
CLEAN_INDEX = DATA_DIR / 'food101_clean.csv'  # written by notebooks/01_eda.ipynb

OUTPUTS_DIR = REPO_ROOT / 'outputs'           # checkpoints and predictions from notebooks/02_models.ipynb
IOS_MODEL = REPO_ROOT / 'ios' / 'FoodLens' / 'Food101Classifier.mlpackage'  # Core ML model bundled with the app
