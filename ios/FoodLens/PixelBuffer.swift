import CoreGraphics
import CoreVideo

/// Draws `image` into a new 32BGRA pixel buffer of the same size.
///
/// Vision converts a `CGImage` to a pixel buffer in the image's own layout, and fails for
/// layouts Core Video doesn't support, such as the RGBA of many decoded JPEGs. Drawing into
/// 32BGRA first, which Core Video and Core ML always accept, makes every photo work.
///
/// - Returns: The pixel buffer, or `nil` if it couldn't be created.
func makeBGRAPixelBuffer(from image: CGImage) -> CVPixelBuffer? {
    let attributes = [kCVPixelBufferCGImageCompatibilityKey: true,
                      kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
    var buffer: CVPixelBuffer?
    guard CVPixelBufferCreate(kCFAllocatorDefault, image.width, image.height,
                              kCVPixelFormatType_32BGRA, attributes, &buffer) == kCVReturnSuccess,
          let buffer else { return nil }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

    let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
    guard let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                  width: image.width, height: image.height, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: bitmapInfo) else { return nil }
    context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    return buffer
}
