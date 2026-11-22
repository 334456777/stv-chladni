//! 实时录制：把界面里的实时画面录成 mp4，可选混入同期声音（麦克风或正在播放的文件）。
//!
//! 画面：GUI 每帧按录制帧率抓取输出纹理，经后台线程写入 ffmpeg；界面帧率低于录制帧率时
//! 重复上一帧、高于时跳帧，保证视频时长与真实时间一致。
//! 声音：录制期间由音源回调写入 [`AudioTap`]，停止后存成 WAV 再与视频合成。

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};

use crate::audio::decode;
use crate::audio::live::{AudioTap, TapBuffer};
use crate::export::{pick_encoder, video_codec_args, Codec};

pub struct RecordConfig {
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub codec: Codec,
    pub quality: u32,
    /// 录入声音时音源的（采样率, 声道数）
    pub audio: Option<(u32, u16)>,
}

pub struct Recorder {
    output: PathBuf,
    video_path: PathBuf,
    fps: u32,
    size: (u32, u32),
    started: Instant,
    frames_scheduled: u64,
    with_audio: bool,
    tap: AudioTap,
    tx: Option<SyncSender<(Vec<u8>, u32)>>,
    writer: Option<JoinHandle<Result<()>>>,
}

impl Recorder {
    pub fn start(cfg: RecordConfig, tap: AudioTap, is_nvidia: bool) -> Result<Self> {
        let width = cfg.width & !1;
        let height = cfg.height & !1;
        if let Some(dir) = cfg.output.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("无法创建目录 {}", dir.display()))?;
        }
        // 带声音时先录无声视频，停止后再合成
        let video_path = if cfg.audio.is_some() {
            cfg.output.with_extension("video.tmp.mp4")
        } else {
            cfg.output.clone()
        };
        let encoder = pick_encoder(cfg.codec, is_nvidia);
        let mut child = decode::ffmpeg_command()
            .args(["-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
            .arg(format!("{width}x{height}"))
            .arg("-r")
            .arg(cfg.fps.to_string())
            .args(["-i", "pipe:0"])
            .args(video_codec_args(encoder, cfg.quality))
            .args(["-movflags", "+faststart"])
            .arg(&video_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("无法启动 ffmpeg（{}）", decode::ffmpeg_bin().display()))?;
        let mut stdin = child.stdin.take().context("ffmpeg stdin 不可用")?;
        let mut stderr = child.stderr.take().context("ffmpeg stderr 不可用")?;

        let (tx, rx) = sync_channel::<(Vec<u8>, u32)>(16);
        let writer = std::thread::Builder::new()
            .name("stv-record".into())
            .spawn(move || -> Result<()> {
                let err_thread = std::thread::spawn(move || {
                    let mut s = String::new();
                    let _ = stderr.read_to_string(&mut s);
                    s
                });
                let mut write_err = None;
                for (pixels, repeat) in rx {
                    for _ in 0..repeat {
                        if let Err(e) = stdin.write_all(&pixels) {
                            write_err = Some(e);
                            break;
                        }
                    }
                    if write_err.is_some() {
                        break;
                    }
                }
                drop(stdin);
                let status = child.wait().context("等待 ffmpeg 失败")?;
                let log = err_thread.join().unwrap_or_default();
                if let Some(e) = write_err {
                    bail!("向 ffmpeg 写入失败：{e}；{}", log.trim());
                }
                if !status.success() {
                    bail!("ffmpeg 编码失败：{}", log.trim());
                }
                Ok(())
            })
            .context("无法创建录制线程")?;

        if let Some((sr, ch)) = cfg.audio {
            tap.start(sr, ch);
        }
        Ok(Self {
            output: cfg.output,
            video_path,
            fps: cfg.fps.max(1),
            size: (width, height),
            started: Instant::now(),
            frames_scheduled: 0,
            with_audio: cfg.audio.is_some(),
            tap,
            tx: Some(tx),
            writer: Some(writer),
        })
    }

    /// 录制画面尺寸（偶数宽高）。
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn elapsed(&self) -> f32 {
        self.started.elapsed().as_secs_f32()
    }

    /// 到现在为止应补上的帧数（0 表示这一界面帧不需要抓取）。
    pub fn frames_due(&mut self) -> u32 {
        let target = (self.started.elapsed().as_secs_f64() * self.fps as f64).floor() as u64 + 1;
        let n = target.saturating_sub(self.frames_scheduled);
        self.frames_scheduled += n;
        n.min(u32::MAX as u64) as u32
    }

    /// 送一帧像素去编码，重复 `repeat` 次。编码线程已退出时返回错误。
    pub fn push(&self, pixels: Vec<u8>, repeat: u32) -> Result<()> {
        let tx = self.tx.as_ref().context("录制已停止")?;
        tx.send((pixels, repeat)).map_err(|_| anyhow!("录制编码线程已退出"))
    }

    /// 编码线程是否已经异常结束。
    pub fn failed(&self) -> bool {
        self.writer.as_ref().is_some_and(|w| w.is_finished())
    }

    /// 结束录制：立即停止收音，后台等待编码完成并合成声音，返回最终文件路径。
    pub fn finish(mut self) -> JoinHandle<Result<PathBuf>> {
        let audio = if self.with_audio { self.tap.take() } else { None };
        drop(self.tx.take());
        let writer = self.writer.take();
        std::thread::spawn(move || {
            if let Some(w) = writer {
                w.join().map_err(|_| anyhow!("录制编码线程崩溃"))??;
            }
            if !self.with_audio {
                return Ok(self.output);
            }
            match audio.filter(|a| !a.samples.is_empty()) {
                Some(buf) => {
                    let wav = self.output.with_extension("audio.tmp.wav");
                    write_wav(&wav, &buf)?;
                    let out = decode::ffmpeg_command()
                        .args(["-y", "-v", "error", "-i"])
                        .arg(&self.video_path)
                        .arg("-i")
                        .arg(&wav)
                        .args([
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                            "-shortest", "-movflags", "+faststart",
                        ])
                        .arg(&self.output)
                        .stdin(Stdio::null())
                        .output()
                        .context("无法启动 ffmpeg 合成声音")?;
                    let _ = std::fs::remove_file(&wav);
                    if !out.status.success() {
                        bail!("合成声音失败：{}", String::from_utf8_lossy(&out.stderr).trim());
                    }
                    let _ = std::fs::remove_file(&self.video_path);
                }
                // 没收到任何声音：直接保留无声视频
                None => std::fs::rename(&self.video_path, &self.output).context("保存录像失败")?,
            }
            Ok(self.output)
        })
    }
}

/// 写 16 位 PCM WAV。
fn write_wav(path: &Path, buf: &TapBuffer) -> Result<()> {
    let ch = buf.channels.max(1);
    let data_len = (buf.samples.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&ch.to_le_bytes());
    bytes.extend_from_slice(&buf.sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(buf.sample_rate * ch as u32 * 2).to_le_bytes());
    bytes.extend_from_slice(&(ch * 2).to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for s in &buf.samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).with_context(|| format!("写入 {}", path.display()))
}

/// 默认录像目录：用户"视频"文件夹下的 stv。
pub fn default_dir() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(|h| PathBuf::from(h).join("Videos").join("stv"))
        .unwrap_or_else(|| PathBuf::from("stv-recordings"))
}

/// 以当前时间命名的录像文件名。
pub fn timestamp_name() -> String {
    chrono::Local::now().format("stv_%Y%m%d_%H%M%S.mp4").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_is_valid() {
        let dir = std::env::temp_dir().join("stv_wav_test.wav");
        let buf = TapBuffer { sample_rate: 48000, channels: 2, samples: vec![0.0, 0.5, -0.5, 1.0] };
        write_wav(&dir, &buf).unwrap();
        let b = std::fs::read(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(&b[8..16], b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes([b[22], b[23]]), 2);
        assert_eq!(u32::from_le_bytes([b[24], b[25], b[26], b[27]]), 48000);
        assert_eq!(u32::from_le_bytes([b[40], b[41], b[42], b[43]]), 8);
        assert_eq!(b.len(), 44 + 8);
        assert_eq!(i16::from_le_bytes([b[46], b[47]]), 16384);
    }
}
