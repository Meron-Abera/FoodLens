import SwiftUI

/// Shows the top prediction, its confidence, and the runner-up dishes.
struct PredictionView: View {
    let predictions: [Prediction]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let top = predictions.first {
                Text(top.displayName)
                    .font(.largeTitle.bold())
                Text("\(top.confidence, format: .percent.precision(.fractionLength(0))) confident")
                    .foregroundStyle(.secondary)

                if top.confidence < ClassifierViewModel.lowConfidenceThreshold {
                    Label("The model isn't sure about this one.", systemImage: "questionmark.circle")
                        .font(.subheadline)
                        .foregroundStyle(.orange)
                }
            }

            if predictions.count > 1 {
                Divider()
                Text("Also possible")
                    .font(.headline)
                ForEach(predictions.dropFirst()) { prediction in
                    HStack {
                        Text(prediction.displayName)
                        Spacer()
                        Text(prediction.confidence, format: .percent.precision(.fractionLength(0)))
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
}

#Preview {
    PredictionView(predictions: [
        Prediction(label: "pad_thai", confidence: 0.87),
        Prediction(label: "pho", confidence: 0.06),
        Prediction(label: "ramen", confidence: 0.03),
    ])
    .padding()
}
