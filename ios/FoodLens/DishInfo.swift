import Foundation

/// Approximate nutrition for one typical serving of a dish.
struct Nutrition: Hashable {
    /// Serving the values are for, e.g. "1 slice (125 g)".
    let serving: String
    let calories: Int
    let proteinGrams: Int
    let carbsGrams: Int
    let fatGrams: Int
}

/// How accurate the bundled model was, in testing, on dishes from one cuisine group.
struct CuisineReliability: Hashable {
    /// Test-set accuracy for the group, between 0 and 1.
    let accuracy: Double
    /// Test-set accuracy over all dishes, between 0 and 1.
    let overallAccuracy: Double
}

/// Nutrition and cuisine reliability for each dish, read from the CSV files bundled in `Data/`.
///
/// The files are copies of `data/nutrition.csv`, `data/cuisine_map.csv` and
/// `data/cuisine_reliability.csv` (written by `notebooks/03_audit.ipynb`).
struct DishInfoStore {
    private let nutrition: [String: Nutrition]
    private let cuisineByLabel: [String: String]
    private let flaggedAccuracy: [String: Double]
    private let overallAccuracy: Double

    /// Loads the bundled files. Missing or unreadable files leave the store empty,
    /// so the app still shows predictions without the extra information.
    init(bundle: Bundle = .main) {
        let nutritionRows = Self.readCSV("nutrition", bundle: bundle)
        let cuisineRows = Self.readCSV("cuisine_map", bundle: bundle)
        let reliabilityRows = Self.readCSV("cuisine_reliability", bundle: bundle)

        nutrition = Dictionary(uniqueKeysWithValues: nutritionRows.compactMap { row in
            guard let label = row["label"], let serving = row["serving"],
                  let calories = row["calories_kcal"].flatMap(Int.init),
                  let protein = row["protein_g"].flatMap(Int.init),
                  let carbs = row["carbs_g"].flatMap(Int.init),
                  let fat = row["fat_g"].flatMap(Int.init) else { return nil }
            return (label, Nutrition(serving: serving, calories: calories,
                                     proteinGrams: protein, carbsGrams: carbs, fatGrams: fat))
        })
        cuisineByLabel = Dictionary(uniqueKeysWithValues: cuisineRows.compactMap { row in
            guard let label = row["label"], let group = row["cuisine_group"] else { return nil }
            return (label, group)
        })

        var accuracyByGroup: [String: Double] = [:]
        var flagged: [String: Double] = [:]
        for row in reliabilityRows {
            guard let group = row["cuisine_group"], let accuracy = row["accuracy"].flatMap(Double.init) else { continue }
            accuracyByGroup[group] = accuracy
            if row["flagged"] == "True" { flagged[group] = accuracy }
        }
        flaggedAccuracy = flagged

        // Every dish has the same number of test images, so overall accuracy is the
        // group accuracies weighted by each group's number of dishes.
        let dishCounts = Dictionary(grouping: cuisineByLabel.values, by: { $0 }).mapValues(\.count)
        let weighted = dishCounts.reduce(0.0) { $0 + Double($1.value) * (accuracyByGroup[$1.key] ?? 0) }
        overallAccuracy = cuisineByLabel.isEmpty ? 0 : weighted / Double(cuisineByLabel.count)
    }

    /// Approximate nutrition for `label`, if known.
    func nutrition(for label: String) -> Nutrition? {
        nutrition[label]
    }

    /// Reliability for the cuisine of `label`, only if that cuisine was flagged as less accurate.
    func lowReliability(for label: String) -> CuisineReliability? {
        guard let group = cuisineByLabel[label], let accuracy = flaggedAccuracy[group] else { return nil }
        return CuisineReliability(accuracy: accuracy, overallAccuracy: overallAccuracy)
    }

    /// Reads a bundled CSV with a header row into one dictionary per row.
    ///
    /// The files have no quoted fields, so splitting on commas is enough.
    private static func readCSV(_ name: String, bundle: Bundle) -> [[String: String]] {
        guard let url = bundle.url(forResource: name, withExtension: "csv")
                ?? bundle.url(forResource: name, withExtension: "csv", subdirectory: "Data"),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        let lines = text.components(separatedBy: .newlines).filter { !$0.isEmpty }
        guard let header = lines.first?.components(separatedBy: ",") else { return [] }
        return lines.dropFirst().map { line in
            Dictionary(uniqueKeysWithValues: zip(header, line.components(separatedBy: ",")))
        }
    }
}
