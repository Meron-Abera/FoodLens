import Observation
import UIKit

/// Holds the current photo and its predictions, and runs the classifier.
@MainActor
@Observable
final class ClassifierViewModel {
    private(set) var image: UIImage?
    private(set) var predictions: [Prediction] = []
    private(set) var isClassifying = false
    private(set) var errorMessage: String?

    private let classifier: FoodClassifier?
    private let loadError: Error?

    /// Below this confidence the app tells the user the model is unsure.
    static let lowConfidenceThreshold = 0.5

    init() {
        do {
            classifier = try FoodClassifier()
            loadError = nil
        } catch {
            classifier = nil
            loadError = error
        }
    }

    /// Whether a model was found in the app bundle.
    var isModelLoaded: Bool { classifier != nil }

    /// Message explaining why the model couldn't be loaded, if it couldn't.
    var modelLoadMessage: String? { loadError?.localizedDescription }

    /// Shows `newImage` and replaces the predictions with the model's output for it.
    func classify(_ newImage: UIImage) async {
        image = newImage
        predictions = []
        errorMessage = nil

        guard let classifier else {
            errorMessage = modelLoadMessage
            return
        }

        isClassifying = true
        defer { isClassifying = false }
        do {
            predictions = try await Task.detached(priority: .userInitiated) {
                try classifier.classify(newImage)
            }.value
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
