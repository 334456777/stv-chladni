//! 无头导出：逐帧分析音频 → GPU 模拟与渲染 → 读回像素 → 通过管道交给 ffmpeg 编码并混入原音轨。

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::analysis::{Analyzer, FFT_SIZE};
use crate::audio::{decode, live::window_centered};
use crate::gpu::{steps_for, HeadlessGpu, Simulator};
use crate::modes::ModeSet;
use crate::params::Params;

/// 分析用采样率。
const ANALYSIS_SR: u32 = 48_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Codec {
    /// 有 NVIDIA 显卡时用 NVENC，否则 libx264
    Auto,
    /// NVIDIA 硬件编码 h264_nvenc
    Nvenc,
    /// 软件编码 libx264
    X264,
}

impl Codec {
    pub fn label(self) -> &'static str {
        match self {
            Codec::Auto => "自动",
            Codec::Nvenc => "NVENC (GPU)",
            Codec::X264 => "x264 (CPU)",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExportSettings {
    pub input: PathBuf,
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub codec: Codec,
    /// 质量（CRF / CQ，越小越好，默认 18）。
    pub quality: u32,
    /// 从音频的第几秒开始。
    pub start: f32,
    /// 导出时长（秒），None 表示到结尾。
    pub duration: Option<f32>,
}

/// 导出进度（可跨线程共享）。
#[derive(Default)]
pub struct Progress {
    pub frame: AtomicU64,
    pub total: AtomicU64,
    pub cancel: AtomicBool,
    pub stage: Mutex<String>,
}

impl Progress {
    pub fn set_stage(&self, s: impl Into<String>) {
        if let Ok(mut g) = self.stage.lock() {
            *g = s.into();
        }
    }
    pub fn fraction(&self) -> f32 {
        let t = self.total.load(Ordering::Relaxed);
        if t == 0 {
            0.0
        } else {
            self.frame.load(Ordering::Relaxed) as f32 / t as f32
        }
    }
}

#[derive(Debug)]
pub struct ExportSummary {
    pub frames: u64,
    pub seconds: f32,
    pub encoder: &'static str,
    pub adapter: String,
}

/// 根据设置与显卡选择 ffmpeg 视频编码器。
pub fn pick_encoder(codec: Codec, is_nvidia: bool) -> &'static str {
    match codec {
        Codec::Nvenc => "h264_nvenc",
        Codec::X264 => "libx264",
        Codec::Auto if is_nvidia && decode::nvenc_works() => "h264_nvenc",
        Codec::Auto => "libx264",
    }
}

/// ffmpeg 视频编码参数（含像素格式）。
pub fn video_codec_args(encoder: &str, quality: u32) -> Vec<String> {
    let q = quality.clamp(0, 51).to_string();
    let args: Vec<&str> = match encoder {
        "h264_nvenc" => vec!["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", &q, "-b:v", "0"],
        _ => vec!["-c:v", "libx264", "-preset", "medium", "-crf", &q],
    };
    args.into_iter()
        .map(String::from)
        .chain(["-pix_fmt".to_string(), "yuv420p".to_string()])
        .collect()
}

pub fn export(settings: &ExportSettings, params: &Params, progress: &Progress) -> Result<ExportSummary> {
    let t0 = Instant::now();
    let mut p = params.clone();
    p.sanitize();
    let width = (settings.width.max(16) + 1) & !1;
    let height = (settings.height.max(16) + 1) & !1;
    let fps = settings.fps.clamp(1, 240);

    progress.set_stage("解码音频…");
    let mono = decode::decode(&settings.input, ANALYSIS_SR, 1)?;
    let sr = ANALYSIS_SR as f32;
    let total_secs = mono.len() as f32 / sr;
    let start = settings.start.clamp(0.0, total_secs);
    let dur = settings
        .duration
        .unwrap_or(f32::INFINITY)
        .min(total_secs - start)
        .max(0.0);
    if dur <= 0.0 {
        bail!("导出时长为 0");
    }
    let total_frames = (dur * fps as f32).ceil() as u64;
    progress.total.store(total_frames, Ordering::Relaxed);
    progress.frame.store(0, Ordering::Relaxed);

    progress.set_stage("初始化 GPU…");
    let gpu = HeadlessGpu::new()?;
    let modes = ModeSet::build(&p);
    let mut analyzer = Analyzer::new(ANALYSIS_SR, &modes, &p);
    // 沙粒初始均匀铺满整块板，视频从这里开始
    let mut sim = Simulator::new(&gpu.device, &gpu.queue, width, height, &p, false);
    let mut window = vec![0.0f32; FFT_SIZE];

    // ---- 启动 ffmpeg ----
    if settings.codec == Codec::Auto {
        progress.set_stage("检测编码器…");
    }
    let encoder = pick_encoder(settings.codec, gpu.is_nvidia);
    if let Some(dir) = settings.output.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).ok();
        }
    }
    let mut cmd = decode::ffmpeg_command();
    cmd.args(["-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
        .arg(format!("{width}x{height}"))
        .arg("-r")
        .arg(fps.to_string())
        .args(["-i", "pipe:0", "-ss"])
        .arg(format!("{start:.6}"))
        .arg("-t")
        .arg(format!("{dur:.6}"))
        .arg("-i")
        .arg(&settings.input)
        .args(["-map", "0:v:0", "-map", "1:a:0?"])
        .args(video_codec_args(encoder, settings.quality));
    cmd.args(["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart"])
    .arg(&settings.output)
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(decode::spawn_error)?;
    let mut stdin = child.stdin.take().context("ffmpeg stdin 不可用")?;
    let mut stderr = child.stderr.take().context("ffmpeg stderr 不可用")?;
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });

    // ---- 逐帧渲染 ----
    progress.set_stage(format!("渲染中（{encoder}）"));
    let dt = 1.0 / fps as f32;
    let mut acc = 0.0f32;
    let mut pixels = Vec::new();
    let mut failure: Option<anyhow::Error> = None;
    for i in 0..total_frames {
        if progress.cancel.load(Ordering::Relaxed) {
            failure = Some(anyhow::anyhow!("已取消"));
            break;
        }
        let t = start + i as f32 * dt;
        window_centered(&mono, (t * sr) as isize, &mut window);
        let frame = analyzer.process(&window, dt, &p);
        sim.set_modes(&gpu.queue, p.shape, &modes.gpu_modes(&frame.weights));
        let steps = steps_for(dt, p.speed, &mut acc);
        let mut enc = gpu
            .device
            .create_command_encoder(&eframe::egui_wgpu::wgpu::CommandEncoderDescriptor { label: Some("stv.frame") });
        sim.encode_step(&gpu.queue, &mut enc, &frame, &p, steps);
        sim.encode_render(&gpu.queue, &mut enc, &p);
        gpu.queue.submit(Some(enc.finish()));
        if let Err(e) = sim.read_pixels(&gpu.device, &gpu.queue, &mut pixels) {
            failure = Some(e);
            break;
        }
        if let Err(e) = stdin.write_all(&pixels) {
            failure = Some(anyhow::anyhow!("向 ffmpeg 写入失败：{e}"));
            break;
        }
        progress.frame.store(i + 1, Ordering::Relaxed);
    }
    drop(stdin);

    if let Some(e) = failure {
        let _ = child.kill();
        let _ = child.wait();
        let log = err_thread.join().unwrap_or_default();
        if progress.cancel.load(Ordering::Relaxed) {
            let _ = std::fs::remove_file(&settings.output);
        }
        if log.trim().is_empty() {
            return Err(e);
        }
        return Err(e.context(format!("ffmpeg：{}", log.trim())));
    }

    progress.set_stage("等待编码完成…");
    let status = child.wait().context("等待 ffmpeg 失败")?;
    let log = err_thread.join().unwrap_or_default();
    if !status.success() {
        bail!("ffmpeg 编码失败：{}", log.trim());
    }
    progress.set_stage("完成");
    Ok(ExportSummary {
        frames: total_frames,
        seconds: t0.elapsed().as_secs_f32(),
        encoder,
        adapter: gpu.adapter_name,
    })
}
