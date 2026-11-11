import CoreML
import UIKit
import Vision

/// One candidate dish for a photo, with the model's probability for it.
struct Prediction: Identifiable, Hashable {
    /// Food-101 label, e.g. `pad_thai`. Matches the labels used by the notebooks.
    let label: String
    /// Probability between 0 and 1.
    let confidence: Double

    var id: String { label }

    /// Readable name, e.g. "Pad thai".
    var displayName: String {
        let words = label.replacingOccurrences(of: "_", with: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }
}

enum ClassifierError: LocalizedError {
    case modelNotFound
    case invalidImage
    case noResults

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "No model found. Add \(FoodClassifier.modelName).mlpackage to the FoodLens folder in Xcode (see ios/README.md)."
        case .invalidImage:
            "Couldn't read this photo."
        case .noResults:
            "The model returned no predictions."
        }
    }
}

/// Classifies food photos with the Core ML model exported by `src/models/export_coreml.py`.
///
/// The model is loaded by file name at runtime rather than through Xcode's generated class,
/// so the app still builds and runs (showing a setup message) before a model is added.
final class FoodClassifier: @unchecked Sendable {
    /// File name of the model, without extension, that the app looks for in its bundle.
    static let modelName = "Food101Classifier"

    private let model: VNCoreMLModel

    /// Loads the compiled model from the app bundle.
    /// - Throws: `ClassifierError.modelNotFound` if the model hasn't been added to the project.
    init() throws {
        guard let url = Bundle.main.url(forResource: Self.modelName, withExtension: "mlmodelc") else {
            throw ClassifierError.modelNotFound
        }
        model = try VNCoreMLModel(for: MLModel(contentsOf: url))
    }

    /// Returns the `topK` most likely dishes for `image`, most likely first.
    ///
    /// Runs synchronously, so call it off the main thread.
    func classify(_ image: UIImage, topK: Int = 3) throws -> [Prediction] {
        guard let cgImage = image.cgImage else { throw ClassifierError.invalidImage }

        let request = VNCoreMLRequest(model: model)
        // Scale the shorter side to 224 px and crop the center, like the notebook's evaluation transform.
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cgImage: cgImage,
                                            orientation: CGImagePropertyOrientation(image.imageOrientation))
        try handler.perform([request])

        guard let results = request.results as? [VNClassificationObservation], !results.isEmpty else {
            throw ClassifierError.noResults
        }
        return results.prefix(topK).map { Prediction(label: $0.identifier, confidence: Double($0.confidence)) }
    }
}

extension CGImagePropertyOrientation {
    /// Converts UIKit's orientation so Vision reads camera photos the right way up.
    init(_ orientation: UIImage.Orientation) {
        switch orientation {
        case .up: self = .up
        case .upMirrored: self = .upMirrored
        case .down: self = .down
        case .downMirrored: self = .downMirrored
        case .left: self = .left
        case .leftMirrored: self = .leftMirrored
        case .right: self = .right
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}
