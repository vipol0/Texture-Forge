use serde::{Deserialize, Serialize};

#[derive(PartialEq, Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum OutputFormat {
    Png,
    Jpeg,
    WebP,
}

impl OutputFormat {
    /// File extension used when writing this format to disk.
    pub(crate) fn extension(self) -> &'static str {
        match self {
            OutputFormat::Png => "png",
            OutputFormat::Jpeg => "jpg",
            OutputFormat::WebP => "webp",
        }
    }
}
