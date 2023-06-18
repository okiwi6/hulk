use std::{
    fmt::{self, Display, Formatter},
    net::{IpAddr, Ipv4Addr},
    path::Path,
    time::Duration,
};

use color_eyre::{
    eyre::{bail, eyre, WrapErr},
    Report, Result,
};
use surge_ping::SurgeError;
use tokio::{process::Command, time};

pub const PING_TIMEOUT: Duration = Duration::from_secs(2);
pub const PING_RETRIES: usize = 2;

pub struct Nao {
    host: Ipv4Addr,
}

impl Nao {
    pub fn new(host: Ipv4Addr) -> Self {
        Self { host }
    }

    pub async fn try_new_with_ping(host: Ipv4Addr) -> Result<Self> {
        Self::try_new_with_ping_and_arguments(host, PING_RETRIES, PING_TIMEOUT).await
    }

    pub async fn try_new_with_ping_and_arguments(
        host: Ipv4Addr,
        retries: usize,
        timeout: Duration,
    ) -> Result<Self> {
        let pinger = async {
            for _ in 0..retries {
                let ping_result = surge_ping::ping(IpAddr::V4(host), &[]).await;
                match ping_result {
                    Ok(_) => return Ok(Nao::new(host)),
                    Err(SurgeError::Timeout { .. }) => continue,
                    surge_error @ Err(_) => return surge_error.wrap_err("could not execute ping"),
                }
            }
            bail!("No route to {host}")
        };

        time::timeout(timeout, pinger)
            .await
            .map_err(|_| Report::msg(format!("No route to {host}")))?
    }

    pub async fn get_os_version(&self) -> Result<String> {
        let output = self
            .ssh_to_nao()
            .arg("cat /etc/os-release")
            .output()
            .await
            .wrap_err("failed to execute cat ssh command")?;

        let stdout = String::from_utf8(output.stdout).wrap_err("failed to decode UTF-8")?;
        extract_version_number(&stdout).ok_or_else(|| eyre!("could not extract version number"))
    }

    fn get_ssh_flags(&self) -> Vec<String> {
        vec![
            "-lnao".to_string(),
            "-oLogLevel=quiet".to_string(),
            "-oStrictHostKeyChecking=no".to_string(),
            "-oUserKnownHostsFile=/dev/null".to_string(),
        ]
    }

    fn ssh_to_nao(&self) -> Command {
        let mut command = Command::new("ssh");
        for flag in self.get_ssh_flags() {
            command.arg(flag);
        }
        command.arg(self.host.to_string());
        command
    }

    fn rsync_with_nao(&self, mkpath: bool) -> Command {
        let mut command = Command::new("rsync");
        let ssh_flags = self.get_ssh_flags().join(" ");
        command
            .arg("--compress")
            .arg("--recursive")
            .arg("--times")
            .arg(format!("--rsh=ssh {ssh_flags}"));
        if mkpath {
            command.arg("--mkpath");
        }
        command
    }

    pub async fn execute_shell(&self) -> Result<()> {
        let status = self
            .ssh_to_nao()
            .status()
            .await
            .wrap_err("failed to execute shell ssh command")?;

        if !status.success() {
            bail!("shell ssh command exited with {status}");
        }

        Ok(())
    }

    pub async fn execute_systemctl(&self, action: SystemctlAction, unit: &str) -> Result<String> {
        let output = self
            .ssh_to_nao()
            .arg("systemctl")
            .arg(match action {
                SystemctlAction::Disable => "disable",
                SystemctlAction::Enable => "enable",
                SystemctlAction::Restart => "restart",
                SystemctlAction::Start => "start",
                SystemctlAction::Status => "status",
                SystemctlAction::Stop => "stop",
            })
            .arg(unit)
            .output()
            .await
            .wrap_err("failed to execute systemctl ssh command")?;

        let status = output.status;

        if !status.success() {
            let systemctl_status_successful = matches!(action, SystemctlAction::Status)
                && status
                    .code()
                    .ok_or_else(|| eyre!("failed to extract exit code from {status:?}"))?
                    != 255;
            if !systemctl_status_successful {
                bail!("systemctl ssh command exited with {status}");
            }
        }

        String::from_utf8(output.stdout).wrap_err("failed to decode UTF-8")
    }

    pub async fn delete_logs(&self) -> Result<()> {
        let status = self
            .ssh_to_nao()
            .arg("rm")
            .arg("--recursive")
            .arg("--force")
            .arg("/home/nao/hulk/logs/*")
            .status()
            .await
            .wrap_err("failed to remove the log directory")?;

        if !status.success() {
            bail!("rm ssh command exited with {status}");
        }

        Ok(())
    }

    pub async fn download_logs(&self, local_directory: impl AsRef<Path>) -> Result<()> {
        let status = self
            .ssh_to_nao()
            .arg("dmesg > /home/nao/hulk/logs/kernel.log")
            .status()
            .await
            .wrap_err("failed to write dmesg to kernel.log")?;

        if !status.success() {
            bail!("dmesg pipe ssh command exited with {status}");
        }

        let status = self
            .rsync_with_nao(true)
            .arg("--quiet")
            .arg(format!("{}:hulk/logs/", self.host))
            .arg(local_directory.as_ref().to_str().unwrap())
            .status()
            .await
            .wrap_err("failed to execute rsync command")?;

        if !status.success() {
            bail!("rsync command exited with {status}");
        }

        Ok(())
    }

    pub async fn retrieve_logs(&self) -> Result<String> {
        let output = self
            .ssh_to_nao()
            .arg("tail")
            .arg("-n+1")
            .arg("hulk/logs/hulk.{out,err}")
            .output()
            .await
            .wrap_err("failed to execute cat command")?;

        if !output.status.success() {
            bail!("cat ssh command exited with {}", output.status);
        }

        String::from_utf8(output.stdout).wrap_err("failed to decode UTF-8")
    }

    pub async fn power_off(&self) -> Result<()> {
        let status = self
            .ssh_to_nao()
            .arg("systemctl")
            .arg("poweroff")
            .status()
            .await
            .wrap_err("failed to execute poweroff ssh command")?;

        if !status.success() {
            bail!("poweroff ssh command exited with {status}");
        }

        Ok(())
    }

    pub async fn reboot(&self) -> Result<()> {
        let status = self
            .ssh_to_nao()
            .arg("systemctl")
            .arg("reboot")
            .status()
            .await
            .wrap_err("failed to execute reboot ssh command")?;

        if !status.success() {
            bail!("reboot ssh command exited with {status}");
        }

        Ok(())
    }

    pub async fn upload(
        &self,
        local_directory: impl AsRef<Path>,
        delete_remaining: bool,
    ) -> Result<()> {
        let mut command = self.rsync_with_nao(true);
        command
            .arg("--keep-dirlinks")
            .arg("--copy-links")
            .arg(format!("{}/", local_directory.as_ref().display()))
            .arg(format!("{}:hulk/", self.host));

        if delete_remaining {
            command.arg("--delete").arg("--delete-excluded");
        }

        let status = command
            .status()
            .await
            .wrap_err("failed to execute rsync command")?;

        if !status.success() {
            bail!("rsync command exited with {status}");
        }

        Ok(())
    }

    pub async fn get_network_status(&self) -> Result<String> {
        let output = self
            .ssh_to_nao()
            .arg("iwctl")
            .arg("station")
            .arg("wlan0")
            .arg("show")
            .output()
            .await
            .wrap_err("failed to execute iwctl ssh command")?;

        if !output.status.success() {
            bail!("iwctl ssh command exited with {}", output.status);
        }

        String::from_utf8(output.stdout).wrap_err("failed to decode UTF-8")
    }

    pub async fn get_available_networks(&self) -> Result<String> {
        let output = self
            .ssh_to_nao()
            .arg("iwctl")
            .arg("station")
            .arg("wlan0")
            .arg("get-networks")
            .output()
            .await
            .wrap_err("failed to execute iwctl ssh command")?;

        if !output.status.success() {
            bail!("iwctl ssh command exited with {}", output.status);
        }

        String::from_utf8(output.stdout).wrap_err("failed to decode UTF-8")
    }

    pub async fn set_network(&self, network: Network) -> Result<()> {
        let command_string = [
            Network::SplA,
            Network::SplB,
            Network::SplC,
            Network::SplD,
            Network::SplE,
            Network::SplF,
        ]
        .into_iter()
        .map(|possible_network| {
            format!(
                "iwctl known-networks {possible_network} set-property AutoConnect {}",
                if network == possible_network {
                    "yes"
                } else {
                    "no"
                }
            )
        })
        .collect::<Vec<_>>()
        .join(" && ");
        let command_string = format!(
            "{command_string} && iwctl station wlan0 {}",
            match network {
                Network::None => "disconnect".to_string(),
                _ => format!("connect {network}"),
            }
        );
        let mut command = self.ssh_to_nao();
        command.arg(command_string);
        let status = command
            .status()
            .await
            .wrap_err("failed to execute iwctl ssh command")?;

        if !status.success() {
            bail!("iwctl ssh command exited with {status}");
        }

        Ok(())
    }

    pub async fn flash_image(&self, image_path: impl AsRef<Path>) -> Result<()> {
        let status = self
            .rsync_with_nao(false)
            .arg("--copy-links")
            .arg(image_path.as_ref().to_str().unwrap())
            .arg(format!("{}:/data/.image/", self.host))
            .status()
            .await
            .wrap_err("failed to execute rsync command")?;

        if !status.success() {
            bail!("failed to upload image")
        }

        self.reboot().await
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SystemctlAction {
    Disable,
    Enable,
    Restart,
    Start,
    Status,
    Stop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Network {
    None,
    SplA,
    SplB,
    SplC,
    SplD,
    SplE,
    SplF,
}

impl Display for Network {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Network::None => formatter.write_str("None"),
            Network::SplA => formatter.write_str("SPL_A"),
            Network::SplB => formatter.write_str("SPL_B"),
            Network::SplC => formatter.write_str("SPL_C"),
            Network::SplD => formatter.write_str("SPL_D"),
            Network::SplE => formatter.write_str("SPL_E"),
            Network::SplF => formatter.write_str("SPL_F"),
        }
    }
}

fn extract_version_number(input: &str) -> Option<String> {
    let lines = input.lines();
    for line in lines {
        if line.contains("VERSION_ID") {
            let Some((_, os_version)) = line.split_once('=') else { continue; };
            return Some(os_version.to_string());
        }
    }
    None
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn matches_stable_os_version() {
        let input = r#"ID=hulks-os
    NAME="HULKs-OS"
    VERSION="5.1.3 (langdale)"
    VERSION_ID=5.1.3
    PRETTY_NAME="HULKs-OS 5.1.3 (langdale)"
    DISTRO_CODENAME="langdale"#;

        let output = extract_version_number(input);
        assert_eq!(output, Some("5.1.3".to_string()));
    }
}
