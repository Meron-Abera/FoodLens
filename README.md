# FoodLens

FoodLens recognizes a dish from a photo and audits whether the model is equally reliable across cuisines.

Food recognition models are usually judged by a single overall accuracy number on benchmarks like Food-101, whose categories lean toward Western dishes. FoodLens looks past that number: it fine-tunes a CNN on Food-101, then compares accuracy and confidence across cuisine groups, borrowing the subgroup-auditing approach from face-recognition fairness research (Buolamwini & Gebru, 2018).

The project has three parts:

1. **Recognize the dish:** an ImageNet-pretrained EfficientNet-B0 fine-tuned on the 101 Food-101 categories.
2. **Estimate nutrition:** look up typical calories, protein, carbs and fat for the predicted dish.
3. **Audit across cuisines:** group the 101 dishes into cuisine families and compare accuracy, calibration and errors between groups.

An iOS app runs the trained model on the phone.

CS663 Machine Learning, Fall 2026. **Team:** Aravind Bilugu (modeling and evaluation), Meron Shibiru (data, fairness audit and systems).

## Repository structure

```
FoodLens/
├── notebooks/
│   ├── 01_eda.ipynb             # data quality checks, cleaning, train/val/test index
│   ├── 02_models.ipynb          # model training, test predictions, overall results
│   └── 03_audit.ipynb           # cuisine fairness audit of both models
├── src/
│   ├── paths.py                 # all data and output locations, relative to the repo
│   ├── data/
│   │   ├── download.py          # downloads Food-101 into data/food-101
│   │   ├── food101_data.py      # loads the cleaned index, PyTorch Dataset and DataLoaders
│   │   └── transforms.py        # preprocessing and augmentation
│   ├── audit/
│   │   ├── metrics.py           # per-cuisine accuracy, bootstrap intervals, calibration, error analysis
│   │   └── plots.py             # audit figures
│   └── models/
│       ├── architectures.py     # EfficientNet-B0 with a 101-class head; checkpoint loading
│       ├── losses.py            # focal loss
│       ├── train.py             # fine-tuning loop
│       ├── evaluate.py          # test predictions and overall metrics
│       └── export_coreml.py     # converts a trained model to Core ML for the app
├── tests/                       # unit tests (pytest)
├── ios/                         # SwiftUI app (see ios/README.md)
├── data/
│   ├── cuisine_map.csv          # cuisine group of each of the 101 dishes
│   ├── nutrition.csv            # approximate nutrition per typical serving of each dish
│   ├── cuisine_reliability.csv  # per-cuisine accuracy of the app's model (written by 03_audit.ipynb)
│   └── food-101/                # dataset, downloaded by src/data/download.py (not committed)
├── outputs/full/                # test predictions and results (checkpoints not committed)
└── requirements.txt
```

## Setup

Tested with Python 3.14 on macOS (Apple silicon).

```bash
git clone https://github.com/Meron-Abera/FoodLens.git
cd FoodLens

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.download        # ~5 GB; skipped if data/food-101 already exists
```

`coremltools` is only needed to export the model for the iOS app and isn't installed on Windows. Everything else runs on macOS, Linux and Windows.

## Running

Run the notebooks in order from Jupyter, VS Code or PyCharm, using the `.venv` environment. The notebooks explain each step and call the reusable code in `src/`.

| Step | What it does | Output | Time |
|---|---|---|---|
| `01_eda.ipynb` | Checks completeness, duplicates, image sizes and color; removes 8 exact duplicates; splits off a stratified 10% validation set | `data/food101_clean.csv` | ~1 min |
| `02_models.ipynb` | Trains both models, predicts on the test set, computes overall metrics | `outputs/full/` | ~3.5–4 h on an Apple M4 |
| `03_audit.ipynb` | Compares accuracy (with 95% bootstrap intervals), calibration and errors across cuisine groups for both models; flags groups significantly below average; tests whether focal loss narrows the gap | figures in `outputs/full/figures/`, `data/cuisine_reliability.csv` | ~1 min |

**Cuisine grouping rules (03_audit):** dishes are grouped by country or region of origin into six groups (Western, East Asian, South Asian, Middle Eastern, Latin American, Other); Southeast Asian dishes count as East Asian, Tex-Mex counts as Western, and dishes with no single origin go to Other.

**Quick test:** training on all 101 classes takes hours. To check that everything runs first, set `QUICK_TEST = True` in the hyperparameters cell of `02_models.ipynb`. It trains on 10 classes for 1 epoch in a few minutes and writes to `outputs/quick_test/` instead.

**Hardware:** training uses an NVIDIA GPU (CUDA) or Apple silicon GPU (MPS) when available, and falls back to the CPU otherwise, which is much slower.

**Re-running:** `02_models.ipynb` loads a model from `outputs/` if it has already been trained, instead of training it again. Delete the `.pt` file to retrain it.

`02_models.ipynb` writes, for each model:
- `predictions_<model>.csv`: one row per test image, with the true label, predicted label, confidence, the probability given to the true label, and the top-5 predictions
- `probs_<model>.npy`: the full probability matrix, for calibration analysis
- `<model>.pt`: the best checkpoint by validation accuracy

and `overall_results.csv` comparing both models. The CSV and `.npy` files from the full run are committed, so the audit can be run without retraining; checkpoints are not.

## Tests

```bash
python -m pytest
```

The tests cover the data loading, preprocessing, focal loss, training loop, checkpoint loading, evaluation, Core ML wrapper, data download and cuisine audit metrics. They use small generated data, so they run in about 10 seconds on a CPU and don't need the Food-101 download.

## Models

Both models are an ImageNet-pretrained **EfficientNet-B0** (4.1M parameters) with its final layer replaced by Dropout(0.2) → Linear(1280 → 101). They use identical data, hyperparameters and training code, and differ only in the loss, so any difference between them can be attributed to the loss:

| Model | Loss | Purpose |
|---|---|---|
| `efficientnet_b0_ce` | Cross-entropy | Baseline |
| `efficientnet_b0_focal` | Focal loss (γ = 2) | Fairness-aware: down-weights images the model already gets right so training focuses on hard dishes |

Hyperparameters: 224×224 input, batch size 64, 5 epochs, AdamW (weight decay 1e-4), learning rate 1e-4 for the pretrained layers and 1e-3 for the new layer, cosine schedule. Training augmentation: random resized crop, horizontal flip, color jitter.

## iOS app

`ios/` contains a SwiftUI app that takes or chooses a photo and shows the predicted dish with its confidence. The model runs on the device with Core ML. For the top prediction, the app also shows:

- approximate nutrition per serving, from `nutrition.csv`
- a "Heads up: this dish type was less accurate in our testing" warning when the dish's cuisine was flagged by the audit, with that cuisine's test accuracy, from `cuisine_map.csv` and `cuisine_reliability.csv`
- a "Not sure" note and the top 3 guesses when its confidence is below 60%

The app reads copies of the three CSV files bundled in `ios/FoodLens/Data/`. After rerunning `03_audit.ipynb` or editing the data files, copy them again:

```bash
cp data/nutrition.csv data/cuisine_map.csv data/cuisine_reliability.csv ios/FoodLens/Data/
```

`cuisine_reliability.csv` must describe the model bundled in the app (`DEPLOYED` in step 8 of `03_audit.ipynb`; currently the focal loss model). To add a trained model:

```bash
python -m src.models.export_coreml --checkpoint outputs/full/efficientnet_b0_focal.pt
```

This writes `ios/FoodLens/Food101Classifier.mlpackage`, which the app picks up on the next build. The repository already includes the exported focal loss model. Then open `ios/FoodLens.xcodeproj` in Xcode 16 or later. See [ios/README.md](ios/README.md) for details, including running on an iPhone.

## Data

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (Bossard, Guillaumin & Van Gool, ECCV 2014): 101,000 images of 101 dishes, 750 training and 250 test images per class, collected from foodspotting.com. The test images were manually cleaned; the training images were deliberately left with some label noise. The dataset is for non-commercial research use (see `data/food-101/license_agreement.txt` after downloading), so it is downloaded from the original source rather than stored in this repository.

### Cuisine map, nutrition and reliability files

- `data/cuisine_map.csv`: the `cuisine_group` (6 groups, rules above) and a finer `food_group` for each Food-101 label. Used by the audit and by the app's warning.
- `data/nutrition.csv`: approximate calories, protein, carbs and fat per typical serving for each of the 101 dishes, compiled from general nutrition knowledge and not verified dish by dish against a database. Each row was checked for consistency (calories from protein, carbs and fat, at 4/4/9 kcal per gram, within 25%). Replacing these with exact USDA FoodData Central lookups is future work. Used only by the app.
- `data/cuisine_reliability.csv`: accuracy, calibration error (ECE) and the "flagged" result for each cuisine group, for the model the app uses. Written by step 8 of `03_audit.ipynb`.

## References

- L. Bossard, M. Guillaumin, L. Van Gool. *Food-101 – Mining Discriminative Components with Random Forests.* ECCV 2014.
- M. Tan, Q. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
- T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. *Focal Loss for Dense Object Detection.* ICCV 2017.
- J. Buolamwini, T. Gebru. *Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification.* FAT\* 2018.
