use crate::imaging::dither::DitherMode;
use crate::model::export::OutputFormat;
use crate::model::settings::{PaletteMethod, Settings};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A named bundle of processing settings plus the export format and quality they target.
///
/// Presets are the unit of the preset manager: built-ins are defined in code, while user presets
/// are serialized to disk and restored on the next launch.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub(crate) struct Preset {
    pub(crate) name: String,
    pub(crate) settings: Settings,
    pub(crate) format: OutputFormat,
    pub(crate) quality: u8,
}

impl Preset {
    pub(crate) fn new(name: &str, settings: Settings, format: OutputFormat, quality: u8) -> Self {
        Self {
            name: name.to_owned(),
            settings,
            format,
            quality,
        }
    }
}

/// Ready-made presets that ship with the application.
pub(crate) fn builtin_presets() -> Vec<Preset> {
    vec![
        Preset::new(
            "Пиксель-арт",
            Settings {
                resolution: Some(64),
                palette_size: 16,
                palette_method: PaletteMethod::Wu,
                dither: DitherMode::None,
                dither_strength: 100,
                ..Default::default()
            },
            OutputFormat::Png,
            88,
        ),
        Preset::new(
            "Ретро-текстура",
            Settings {
                resolution: Some(128),
                palette_size: 32,
                palette_method: PaletteMethod::Wu,
                dither: DitherMode::FloydSteinberg,
                dither_strength: 100,
                ..Default::default()
            },
            OutputFormat::Png,
            88,
        ),
        Preset::new(
            "Постер",
            Settings {
                resolution: Some(512),
                palette_size: 8,
                palette_method: PaletteMethod::KMeans,
                dither: DitherMode::Ordered,
                dither_strength: 60,
                ..Default::default()
            },
            OutputFormat::Png,
            88,
        ),
        Preset::new(
            "Сглаженная",
            Settings {
                resolution: Some(256),
                palette_size: 64,
                palette_method: PaletteMethod::Wu,
                dither: DitherMode::Ordered,
                dither_strength: 40,
                ..Default::default()
            },
            OutputFormat::Png,
            88,
        ),
    ]
}

/// Path of the JSON file that stores user presets, following platform conventions.
pub(crate) fn preset_config_path() -> PathBuf {
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|home| {
            PathBuf::from(home)
                .join("Library")
                .join("Application Support")
        })
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    }
    .unwrap_or_else(|| PathBuf::from("."));
    base.join("texture_forge").join("presets.json")
}

/// Persisted collection of user-defined presets.
pub(crate) struct PresetStore {
    pub(crate) user: Vec<Preset>,
}

impl PresetStore {
    /// Load user presets from disk, falling back to an empty set when missing or unreadable.
    pub(crate) fn load() -> Self {
        Self::load_from(&preset_config_path())
    }

    pub(crate) fn load_from(path: &Path) -> Self {
        let user = std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str::<Vec<Preset>>(&text).ok())
            .unwrap_or_default();
        Self { user }
    }

    /// Persist the current user presets, creating parent directories as needed.
    pub(crate) fn save(&self) -> Result<(), String> {
        self.save_to(&preset_config_path())
    }

    pub(crate) fn save_to(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let text = serde_json::to_string_pretty(&self.user).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| e.to_string())
    }

    /// Built-in presets followed by user presets.
    pub(crate) fn all(&self) -> Vec<Preset> {
        let mut presets = builtin_presets();
        presets.extend(self.user.iter().cloned());
        presets
    }

    pub(crate) fn is_builtin(name: &str) -> bool {
        builtin_presets().iter().any(|p| p.name == name)
    }

    /// Add or replace a user preset by name. Callers persist with [`PresetStore::save`].
    pub(crate) fn insert(&mut self, preset: Preset) {
        match self.user.iter_mut().find(|p| p.name == preset.name) {
            Some(existing) => *existing = preset,
            None => self.user.push(preset),
        }
    }

    /// Remove a user preset by name. Callers persist with [`PresetStore::save`].
    pub(crate) fn remove(&mut self, name: &str) {
        self.user.retain(|p| p.name != name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::geometry::{CropRect, ResizeFilter, ResizeMode};
    use crate::model::settings::AlphaMode;
    use crate::model::settings::Settings;
    use crate::test_support::temp_dir;
    use std::collections::HashSet;

    #[test]
    fn builtin_presets_have_unique_names_and_valid_settings() {
        let presets = builtin_presets();
        assert!(!presets.is_empty());
        let names: HashSet<&str> = presets.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names.len(),
            presets.len(),
            "duplicate built-in preset names"
        );
        for preset in &presets {
            assert!((2..=256).contains(&preset.settings.palette_size));
            assert!((1..=100).contains(&preset.settings.dither_strength));
            assert!((-100..=100).contains(&preset.settings.brightness));
            assert!((0..=200).contains(&preset.settings.saturation));
            assert!((0..=200).contains(&preset.settings.contrast));
            assert!((10..=300).contains(&preset.settings.gamma));
            assert!((-100..=100).contains(&preset.settings.temperature));
            assert!((-100..=100).contains(&preset.settings.tint));
            assert!(preset.settings.black_point < preset.settings.white_point);
            assert!(!preset.settings.curves.is_empty());
            assert_eq!(preset.settings.alpha_mode, AlphaMode::Save);
        }
    }

    #[test]
    fn preset_json_round_trips() {
        let presets = builtin_presets();
        let text = serde_json::to_string(&presets).expect("serialize presets");
        let restored: Vec<Preset> = serde_json::from_str(&text).expect("deserialize presets");
        assert_eq!(presets, restored);
    }

    #[test]
    fn curve_points_survive_preset_json_round_trip() {
        let preset = Preset::new(
            "Кривая",
            Settings {
                curves: vec![[0.0, 0.05], [0.5, 0.7], [1.0, 0.95]],
                gamma: 120,
                temperature: 20,
                tint: -10,
                black_point: 12,
                white_point: 240,
                sepia: true,
                crop: CropRect {
                    x: 0.1,
                    y: 0.2,
                    w: 0.7,
                    h: 0.6,
                },
                output_size: Some([320, 180]),
                resize_mode: ResizeMode::Fill,
                resize_filter: ResizeFilter::Lanczos,
                ..Default::default()
            },
            OutputFormat::Png,
            90,
        );
        let text = serde_json::to_string(&preset).expect("serialize preset");
        let restored: Preset = serde_json::from_str(&text).expect("deserialize preset");
        assert_eq!(restored, preset);
    }

    #[test]
    fn preset_store_save_and_load_round_trip() {
        let dir = temp_dir("presets");
        let path = dir.join("presets.json");

        let mut store = PresetStore { user: Vec::new() };
        assert!(PresetStore::load_from(&path).user.is_empty());

        let preset = Preset::new(
            "Мой пресет",
            Settings {
                resolution: Some(128),
                palette_size: 24,
                dither: DitherMode::Ordered,
                ..Default::default()
            },
            OutputFormat::WebP,
            70,
        );
        store.insert(preset.clone());
        store.save_to(&path).expect("save preset");

        let reloaded = PresetStore::load_from(&path);
        assert_eq!(reloaded.user.len(), 1);
        assert_eq!(reloaded.user[0], preset);
        assert!(reloaded.all().iter().any(|p| p.name == "Мой пресет"));

        store.remove("Мой пресет");
        store.save_to(&path).expect("save after remove");
        assert!(PresetStore::load_from(&path).user.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn preset_store_ignores_corrupt_and_builtin_names() {
        assert!(PresetStore::is_builtin(&builtin_presets()[0].name));
        assert!(!PresetStore::is_builtin("Это не встроенный пресет"));

        let dir = temp_dir("presets_corrupt");
        let path = dir.join("presets.json");
        std::fs::write(&path, "{ not valid json").unwrap();
        assert!(PresetStore::load_from(&path).user.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
