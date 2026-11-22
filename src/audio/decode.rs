//! 通过 ffmpeg 解码任意音频/视频文件中的音轨。

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};

/// ffmpeg 可执行文件路径：优先环境变量 `STV_FFMPEG`，否则使用 PATH 中的 `ffmpeg`。
pub fn ffmpeg_bin() -> PathBuf {
    std::env::var_os("STV_FFMPEG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ffmpeg"))
}

/// 创建一个不弹出控制台窗口的 ffmpeg 命令。
pub fn ffmpeg_command() -> Command {
    let mut cmd = Command::new(ffmpeg_bin());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// 把文件解码为交错的 f32 PCM。
pub fn decode(path: &Path, sample_rate: u32, channels: u16) -> Result<Vec<f32>> {
    let output = ffmpeg_command()
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-vn", "-ac"])
        .arg(channels.to_string())
        .arg("-ar")
        .arg(sample_rate.to_string())
        .args(["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("无法启动 ffmpeg（{}），请确认已安装并在 PATH 中，或设置 STV_FFMPEG", ffmpeg_bin().display()))?;
    if !output.status.success() {
        bail!(
            "ffmpeg 解码失败：{}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let samples: Vec<f32> = output
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if samples.is_empty() {
        bail!("文件中没有可解码的音轨：{}", path.display());
    }
    Ok(samples)
}

/// ffmpeg 是否带有某个编码器。
pub fn has_encoder(name: &str) -> bool {
    static LIST: OnceLock<String> = OnceLock::new();
    let list = LIST.get_or_init(|| {
        ffmpeg_command()
            .args(["-hide_banner", "-encoders"])
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default()
    });
    list.split_whitespace().any(|w| w == name)
}

/// 实际试跑一次 NVENC，确认驱动可用（结果缓存）。
pub fn nvenc_works() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        has_encoder("h264_nvenc")
            && ffmpeg_command()
                .args([
                    "-v", "error", "-nostdin", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
                    "-c:v", "h264_nvenc", "-f", "null", "-",
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
    })
}
