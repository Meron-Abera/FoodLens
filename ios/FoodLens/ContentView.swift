import PhotosUI
import SwiftUI

/// Main screen: take or choose a photo, then show what dish the model thinks it is.
struct ContentView: View {
    @State private var viewModel = ClassifierViewModel()
    @State private var isShowingCamera = false
    @State private var selectedPhoto: PhotosPickerItem?

    /// Devices without a camera (and older simulators) only offer the photo library.
    private let isCameraAvailable = UIImagePickerController.isSourceTypeAvailable(.camera)

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    if let message = viewModel.modelLoadMessage {
                        Label(message, systemImage: "shippingbox")
                            .font(.footnote)
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.orange.opacity(0.15), in: RoundedRectangle(cornerRadius: 12))
                    }

                    photoArea

                    if viewModel.isClassifying {
                        ProgressView("Identifying…")
                    } else if let message = viewModel.errorMessage, viewModel.isModelLoaded {
                        Label(message, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                    } else if !viewModel.predictions.isEmpty {
                        PredictionView(predictions: viewModel.predictions)
                    }
                }
                .padding()
            }
            .navigationTitle("FoodLens")
            .safeAreaInset(edge: .bottom) {
                actionButtons
            }
            .fullScreenCover(isPresented: $isShowingCamera) {
                CameraPicker { image in
                    Task { await viewModel.classify(image) }
                }
                .ignoresSafeArea()
            }
            .onChange(of: selectedPhoto) { _, item in
                Task { await classifyLibraryPhoto(item) }
            }
        }
    }

    /// The selected photo, or a placeholder before one is chosen.
    @ViewBuilder
    private var photoArea: some View {
        if let image = viewModel.image {
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .clipShape(RoundedRectangle(cornerRadius: 16))
        } else {
            ContentUnavailableView("No photo yet",
                                   systemImage: "fork.knife",
                                   description: Text("Take or choose a photo of a dish to identify it."))
                .frame(minHeight: 300)
        }
    }

    private var actionButtons: some View {
        HStack(spacing: 12) {
            if isCameraAvailable {
                Button {
                    isShowingCamera = true
                } label: {
                    Label("Camera", systemImage: "camera")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            }

            PhotosPicker(selection: $selectedPhoto, matching: .images) {
                Label("Photos", systemImage: "photo.on.rectangle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
        }
        .controlSize(.large)
        .lineLimit(1)
        .padding()
        .background(.bar)
    }

    /// Loads a photo picked from the library and classifies it.
    private func classifyLibraryPhoto(_ item: PhotosPickerItem?) async {
        guard let item,
              let data = try? await item.loadTransferable(type: Data.self),
              let image = UIImage(data: data) else { return }
        await viewModel.classify(image)
    }
}

#Preview {
    ContentView()
}
