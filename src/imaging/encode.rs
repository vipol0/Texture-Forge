use crate::imaging::pipeline::flatten_rgba;
use crate::model::export::OutputFormat;
use image::{DynamicImage, ImageFormat};

/// Concrete encoder for one output format, carrying the settings it needs.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ImageEncoder {
    /// JPEG loses alpha, so it composites onto `background` before encoding.
    Jpeg {
        quality: u8,
        background: [u8; 3],
    },
    Png,
    WebP,
}

impl ImageEncoder {
    pub(crate) fn for_format(format: OutputFormat, quality: u8, background: [u8; 3]) -> Self {
        match format {
            OutputFormat::Jpeg => Self::Jpeg {
                quality,
                background,
            },
            OutputFormat::Png => Self::Png,
            OutputFormat::WebP => Self::WebP,
        }
    }

    /// Encode `img` in memory using this encoder's format and settings.
    pub(crate) fn encode(&self, img: &DynamicImage) -> image::ImageResult<Vec<u8>> {
        match *self {
            Self::Jpeg {
                quality,
                background,
            } => {
                // JPEG does not support alpha; composite onto the selected background.
                let rgba = img.to_rgba8();
                let rgb = flatten_rgba(&rgba, background);
                let mut buf = Vec::new();
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality).encode(
                    &rgb,
                    rgb.width(),
                    rgb.height(),
                    image::ExtendedColorType::Rgb8,
                )?;
                Ok(buf)
            }
            Self::Png => {
                let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
                img.write_to(&mut cursor, ImageFormat::Png)?;
                Ok(cursor.into_inner())
            }
            Self::WebP => {
                let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
                img.write_to(&mut cursor, ImageFormat::WebP)?;
                Ok(cursor.into_inner())
            }
        }
    }
}

/// Encode `img` into `format` in memory, applying JPEG `quality` where relevant.
#[cfg(test)]
pub(crate) fn encode_image(
    img: &DynamicImage,
    format: OutputFormat,
    quality: u8,
    background: [u8; 3],
) -> image::ImageResult<Vec<u8>> {
    ImageEncoder::for_format(format, quality, background).encode(img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::export::OutputFormat;

    use image::{DynamicImage, RgbaImage};

    #[test]
    fn jpeg_encoding_flattens_alpha_onto_background() {
        let image =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(16, 16, image::Rgba([200, 20, 20, 0])));
        let bytes =
            encode_image(&image, OutputFormat::Jpeg, 100, [0, 0, 255]).expect("encode JPEG");
        let decoded = image::load_from_memory(&bytes)
            .expect("decode JPEG")
            .to_rgb8();
        let pixel = decoded.get_pixel(8, 8).0;
        assert!(pixel[2] > 200, "background blue missing: {pixel:?}");
        assert!(pixel[0] < 40, "transparent red leaked through: {pixel:?}");
        assert!(pixel[1] < 40, "transparent green leaked through: {pixel:?}");
    }
}
