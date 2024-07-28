use configuration_system::Configuration;
use configuration_system_derive::Merge;
use serde::Deserialize;

pub mod keybind_plugin;
pub mod keys;

const DEFAULT_CONFIG: &str = include_str!("../config_default.toml");

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug, Deserialize, Merge)]
pub struct TwixConfiguration {
    pub keys: keys::Keybinds,
}

impl Configuration for TwixConfiguration {
    const DEFAULT_FILENAME: &'static str = "twix.toml";
}

impl Default for TwixConfiguration {
    fn default() -> Self {
        toml::from_str(DEFAULT_CONFIG).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use configuration_system::Merge;

    #[test]
    fn parse_default_config() {
        toml::from_str::<TwixConfiguration>(DEFAULT_CONFIG).expect("failed to parse default.toml");
    }

    #[test]
    fn merge_configs() {
        let mut config_1: TwixConfiguration = toml::from_str(
            r#"
                [keys]
                C-a = "focus_left"
                C-S-a = "reconnect"
            "#,
        )
        .unwrap();

        let config_2: TwixConfiguration = toml::from_str(
            r#"
                [keys]
                C-b = "focus_left"
                C-A = "focus_right"
            "#,
        )
        .unwrap();

        config_1.merge(config_2);

        assert_eq!(
            config_1,
            toml::from_str(
                r#"
                    [keys]
                    C-a = "focus_left"
                    C-A = "focus_right"
                    C-b = "focus_left"
                "#
            )
            .unwrap()
        );
    }
}
