import SwiftUI

/// Shows the top prediction, its confidence, nutrition, any warnings, and the other likely dishes.
struct PredictionView: View {
    let predictions: [Prediction]
    /// Approximate nutrition for the top prediction, if known.
    var nutrition: Nutrition?
    /// Set when the top prediction's cuisine was less accurate in testing.
    var lowReliability: CuisineReliability?

    private static let percent = FloatingPointFormatStyle<Double>.Percent().precision(.fractionLength(0))

    private var isUnsure: Bool {
        (predictions.first?.confidence ?? 0) < ClassifierViewModel.lowConfidenceThreshold
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let top = predictions.first {
                Text(top.displayName)
                    .font(.largeTitle.bold())
                Text("\(top.confidence.formatted(Self.percent)) confident")
                    .foregroundStyle(.secondary)

                if isUnsure {
                    Label("Not sure — the model's top guesses were close", systemImage: "questionmark.circle")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                if let lowReliability {
                    Label("Heads up: this dish type was less accurate in our testing "
                          + "(\(lowReliability.accuracy.formatted(Self.percent)) vs "
                          + "\(lowReliability.overallAccuracy.formatted(Self.percent)) overall).",
                          systemImage: "exclamationmark.triangle")
                        .font(.subheadline)
                        .foregroundStyle(.orange)
                }

                if let nutrition {
                    nutritionCard(nutrition)
                }
            }

            // When unsure, list the top guess too so all three can be compared side by side.
            let others = isUnsure ? Array(predictions.prefix(3)) : Array(predictions.dropFirst())
            if !others.isEmpty {
                Divider()
                Text(isUnsure ? "Top guesses" : "Also possible")
                    .font(.headline)
                ForEach(others) { prediction in
                    HStack {
                        Text(prediction.displayName)
                        Spacer()
                        Text(prediction.confidence.formatted(Self.percent))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
    }

    /// Serving and calories, then protein, carbs and fat on one line.
    private func nutritionCard(_ nutrition: Nutrition) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(nutrition.serving) · \(nutrition.calories) kcal")
                .font(.headline)
            Text("Protein \(nutrition.proteinGrams) g · Carbs \(nutrition.carbsGrams) g · Fat \(nutrition.fatGrams) g")
                .font(.subheadline)
            Text("approximate values")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 4)
    }
}

#Preview {
    PredictionView(predictions: [
        Prediction(label: "prime_rib", confidence: 0.44),
        Prediction(label: "steak", confidence: 0.40),
        Prediction(label: "filet_mignon", confidence: 0.07),
    ],
    nutrition: Nutrition(serving: "1 steak (225 g)", calories: 570, proteinGrams: 55, carbsGrams: 0, fatGrams: 38),
    lowReliability: CuisineReliability(accuracy: 0.78, overallAccuracy: 0.8346))
    .padding()
}
