# FoodLens iOS app

A SwiftUI app that takes or chooses a photo of a dish and shows the model's top 3 predictions with their confidence. The model runs on the phone with Core ML, so it works offline and photos never leave the device.

Requires Xcode 16 or later and iOS 17 or later.

## Adding the model

The app looks for a Core ML model named **`Food101Classifier`**. Without one, it still builds and runs but shows a setup message.

1. Export a trained checkpoint from the repository root:
   ```bash
   python -m src.models.export_coreml --checkpoint outputs/full/efficientnet_b0_focal.pt
   ```
   This writes `ios/FoodLens/Food101Classifier.mlpackage` (~8 MB), with ImageNet normalization, softmax and the 101 class labels built in. The project syncs with that folder, so Xcode picks the model up automatically; no project settings need to change.
2. Build and run.

The repository already includes the exported focal loss model, so the app works without retraining.

Any model works as long as it keeps the same contract: an RGB image input of 224×224 and classifier output labels that match the Food-101 folder names (e.g. `pad_thai`).

## Nutrition and cuisine warnings

Under the top prediction, the app shows approximate nutrition per serving and, for cuisines the audit flagged as less accurate, a "Heads up" warning with that cuisine's test accuracy. It reads three CSV files bundled in `FoodLens/Data/`, copied from `data/`. After rerunning `notebooks/03_audit.ipynb` or editing the data files, copy them again from the repository root:

```bash
cp data/nutrition.csv data/cuisine_map.csv data/cuisine_reliability.csv ios/FoodLens/Data/
```

`cuisine_reliability.csv` must come from the same model as the bundled one (`DEPLOYED` in step 8 of the audit notebook).

## Running

- **Simulator:** open `FoodLens.xcodeproj`, pick an iPhone simulator, and press Run. Use **Photos**; drag food images onto the simulator window to add them to its photo library. (Older simulators have no camera, so **Camera** is hidden there.)
- **iPhone:** select your Apple ID team under *Signing & Capabilities*, connect the phone, and press Run. **Camera** appears on a real device.

## Code

| File | Purpose |
|---|---|
| `FoodLensApp.swift` | App entry point |
| `ContentView.swift` | Main screen: photo, buttons, results |
| `ClassifierViewModel.swift` | Holds the photo and predictions; runs the classifier off the main thread |
| `FoodClassifier.swift` | Loads the Core ML model and classifies images with Vision |
| `PredictionView.swift` | Top prediction, confidence, low-confidence and cuisine warnings, nutrition and runner-ups |
| `DishInfo.swift` | Reads the bundled CSVs: nutrition per dish and accuracy of flagged cuisines |
| `Data/` | Copies of `nutrition.csv`, `cuisine_map.csv` and `cuisine_reliability.csv` |
| `CameraPicker.swift` | Wraps the system camera for SwiftUI |
