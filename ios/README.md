# FoodLens iOS app

A SwiftUI app that takes or chooses a photo of a dish and shows the model's top 3 predictions with their confidence. The model runs on the phone with Core ML, so it works offline and photos never leave the device.

Requires Xcode 16 or later and iOS 17 or later.

## Adding the model

The app looks for a Core ML model named **`Food101Classifier`**. Until one is added, it builds and runs but shows a setup message.

1. Export a trained checkpoint from the repository root:
   ```bash
   python -m src.models.export_coreml --checkpoint outputs/full/efficientnet_b0_focal.pt
   ```
   This writes `models/Food101Classifier.mlpackage` (~8 MB), with ImageNet normalization, softmax and the 101 class labels built in.
2. Copy `Food101Classifier.mlpackage` into `ios/FoodLens/`. The project syncs with that folder, so Xcode picks the model up automatically; no project settings need to change.
3. Build and run.

Any model works as long as it keeps the same contract: an RGB image input of 224×224 and classifier output labels that match the Food-101 folder names (e.g. `pad_thai`).

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
| `PredictionView.swift` | Top prediction, confidence, low-confidence warning and runner-ups |
| `CameraPicker.swift` | Wraps the system camera for SwiftUI |
