//! 通过 ffmpeg 解码任意音频/视频文件中的音轨。

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Result};

/// ffmpeg 的下载地址，界面提示和报错都用它。
pub const FFMPEG_URL: &str = "https://ffmpeg.org/download.html";

static FOUND: OnceLock<bool> = OnceLock::new();

/// ffmpeg 可执行文件路径，依次查找：环境变量 `STV_FFMPEG`；stv 可执行文件旁边的
/// `ffmpeg/ffmpeg.exe`（发布包自带）或 `ffmpeg.exe`；PATH 中的 `ffmpeg`。
pub fn ffmpeg_bin() -> PathBuf {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        std::env::var_os("STV_FFMPEG")
            .map(PathBuf::from)
            .or_else(bundled_ffmpeg)
            .unwrap_or_else(|| PathBuf::from("ffmpeg"))
    })
    .clone()
}

fn bundled_ffmpeg() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let name = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
    [dir.join("ffmpeg").join(name), dir.join(name)]
        .into_iter()
        .find(|p| p.is_file())
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

/// 启动 ffmpeg 失败时的错误；找不到程序时说明怎么安装。
pub fn spawn_error(e: std::io::Error) -> anyhow::Error {
    let bin = ffmpeg_bin();
    if e.kind() == std::io::ErrorKind::NotFound {
        anyhow!(
            "找不到 ffmpeg（{}）：请下载 ffmpeg（{FFMPEG_URL}），把 ffmpeg.exe 放到 stv.exe 旁边或加入 PATH，也可以用环境变量 STV_FFMPEG 指向它",
            bin.display()
        )
    } else {
        anyhow!("无法启动 ffmpeg（{}）：{e}", bin.display())
    }
}

/// ffmpeg 能否运行。第一次调用会运行一次 `ffmpeg -version`，结果缓存到进程结束。
pub fn ffmpeg_found() -> bool {
    *FOUND.get_or_init(|| {
        ffmpeg_command()
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

/// [`ffmpeg_found`] 已检测过时返回结果；还没检测时返回 `None`，不阻塞。
pub fn ffmpeg_found_cached() -> Option<bool> {
    FOUND.get().copied()
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
        .map_err(spawn_error)?;
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
