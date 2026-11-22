//! 实时音频：麦克风输入与文件播放（cpal）。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};

use super::decode;

/// 单声道环形缓冲。
pub struct Ring {
    buf: Vec<f32>,
    write: usize,
    /// 累计收到的样本数（界面据此判断麦克风有没有数据）
    total: u64,
    /// 最近一次流错误
    error: Option<String>,
}

impl Ring {
    pub fn new(cap: usize) -> Self {
        Self { buf: vec![0.0; cap], write: 0, total: 0, error: None }
    }

    fn push(&mut self, s: f32) {
        self.buf[self.write] = s;
        self.write = (self.write + 1) % self.buf.len();
        self.total += 1;
    }

    /// 复制最近的 `out.len()` 个样本（按时间顺序）。
    pub fn latest(&self, out: &mut [f32]) {
        let n = out.len().min(self.buf.len());
        let cap = self.buf.len();
        let start = (self.write + cap - n) % cap;
        for (i, o) in out.iter_mut().take(n).enumerate() {
            *o = self.buf[(start + i) % cap];
        }
    }
}

/// 录音抽头：录制期间，当前音源把实际采到 / 播出的声音按原始声道交错追加进来。
#[derive(Clone, Default)]
pub struct AudioTap(Arc<Mutex<Option<TapBuffer>>>);

pub struct TapBuffer {
    pub sample_rate: u32,
    pub channels: u16,
    /// 交错样本
    pub samples: Vec<f32>,
}

impl AudioTap {
    /// 开始收集（清空之前的内容）。
    pub fn start(&self, sample_rate: u32, channels: u16) {
        if let Ok(mut g) = self.0.lock() {
            *g = Some(TapBuffer { sample_rate, channels, samples: Vec::with_capacity(sample_rate as usize * 60) });
        }
    }

    /// 停止收集并取走已收集的声音。
    pub fn take(&self) -> Option<TapBuffer> {
        self.0.lock().ok().and_then(|mut g| g.take())
    }
}

/// 输入设备名列表。
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .map(|it| it.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default()
}

/// 麦克风 / 线路输入。
pub struct MicInput {
    _stream: cpal::Stream,
    ring: Arc<Mutex<Ring>>,
    pub sample_rate: u32,
    pub name: String,
}

impl MicInput {
    pub fn start(name: Option<&str>, tap: AudioTap) -> Result<Self> {
        let host = cpal::default_host();
        let device = match name {
            Some(n) => host
                .input_devices()?
                .find(|d| d.name().map(|x| x == n).unwrap_or(false))
                .ok_or_else(|| anyhow!("找不到输入设备：{n}"))?,
            None => host.default_input_device().context("没有默认输入设备")?,
        };
        let dev_name = device.name().unwrap_or_else(|_| "未知设备".into());
        let supported = device.default_input_config().context("读取输入设备配置失败")?;
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let ring = Arc::new(Mutex::new(Ring::new(1 << 16)));
        let r = ring.clone();
        let stream = match format {
            cpal::SampleFormat::F32 => build_input::<f32>(&device, &config, r, tap)?,
            cpal::SampleFormat::I16 => build_input::<i16>(&device, &config, r, tap)?,
            cpal::SampleFormat::U16 => build_input::<u16>(&device, &config, r, tap)?,
            cpal::SampleFormat::I32 => build_input::<i32>(&device, &config, r, tap)?,
            cpal::SampleFormat::U8 => build_input::<u8>(&device, &config, r, tap)?,
            cpal::SampleFormat::F64 => build_input::<f64>(&device, &config, r, tap)?,
            other => bail!("不支持的输入采样格式：{other:?}"),
        };
        stream.play().context("启动输入流失败")?;
        Ok(Self {
            _stream: stream,
            ring,
            sample_rate: config.sample_rate.0,
            name: dev_name,
        })
    }

    pub fn latest(&self, out: &mut [f32]) {
        if let Ok(r) = self.ring.lock() {
            r.latest(out);
        }
    }

    /// 累计收到的样本数与最近一次流错误。
    pub fn stats(&self) -> (u64, Option<String>) {
        self.ring
            .lock()
            .map(|r| (r.total, r.error.clone()))
            .unwrap_or((0, None))
    }
}

fn build_input<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    ring: Arc<Mutex<Ring>>,
    tap: AudioTap,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let ch = config.channels.max(1) as usize;
    let err_ring = ring.clone();
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let mut tap = tap.0.lock().ok();
            let mut rec = tap.as_mut().and_then(|g| g.as_mut());
            if let Ok(mut r) = ring.lock() {
                for frame in data.chunks(ch) {
                    let s: f32 = frame.iter().map(|&x| f32::from_sample(x)).sum::<f32>() / frame.len() as f32;
                    r.push(s);
                    if let Some(b) = rec.as_mut() {
                        b.samples.push(s);
                    }
                }
            }
        },
        move |e| {
            log::warn!("音频输入错误：{e}");
            if let Ok(mut r) = err_ring.lock() {
                r.error = Some(e.to_string());
            }
        },
        None,
    )?;
    Ok(stream)
}

struct PlayerShared {
    /// 交错立体声
    stereo: Vec<f32>,
    frames: usize,
    pos: AtomicUsize,
    playing: AtomicBool,
    /// 音量（f32 位模式）
    volume: AtomicU32,
}

/// 音频文件播放器：一边从扬声器播放，一边提供播放头处的分析窗口。
pub struct Player {
    _stream: cpal::Stream,
    shared: Arc<PlayerShared>,
    mono: Vec<f32>,
    pub sample_rate: u32,
    pub path: PathBuf,
}

impl Player {
    pub fn open(path: &Path, tap: AudioTap) -> Result<Self> {
        let host = cpal::default_host();
        let device = host.default_output_device().context("没有默认输出设备")?;
        let supported = device.default_output_config().context("读取输出设备配置失败")?;
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let sr = config.sample_rate.0;

        let stereo = decode::decode(path, sr, 2)?;
        let frames = stereo.len() / 2;
        let mono: Vec<f32> = stereo.as_chunks::<2>().0.iter().map(|[l, r]| 0.5 * (l + r)).collect();
        let shared = Arc::new(PlayerShared {
            stereo,
            frames,
            pos: AtomicUsize::new(0),
            playing: AtomicBool::new(false),
            volume: AtomicU32::new(1.0f32.to_bits()),
        });

        let s = shared.clone();
        let stream = match format {
            cpal::SampleFormat::F32 => build_output::<f32>(&device, &config, s, tap)?,
            cpal::SampleFormat::I16 => build_output::<i16>(&device, &config, s, tap)?,
            cpal::SampleFormat::U16 => build_output::<u16>(&device, &config, s, tap)?,
            cpal::SampleFormat::I32 => build_output::<i32>(&device, &config, s, tap)?,
            cpal::SampleFormat::U8 => build_output::<u8>(&device, &config, s, tap)?,
            cpal::SampleFormat::F64 => build_output::<f64>(&device, &config, s, tap)?,
            other => bail!("不支持的输出采样格式：{other:?}"),
        };
        stream.play().context("启动输出流失败")?;
        Ok(Self {
            _stream: stream,
            shared,
            mono,
            sample_rate: sr,
            path: path.to_path_buf(),
        })
    }

    pub fn is_playing(&self) -> bool {
        self.shared.playing.load(Ordering::Relaxed)
    }

    pub fn set_playing(&self, on: bool) {
        if on && self.shared.pos.load(Ordering::Relaxed) >= self.shared.frames {
            self.shared.pos.store(0, Ordering::Relaxed);
        }
        self.shared.playing.store(on, Ordering::Relaxed);
    }

    pub fn set_volume(&self, v: f32) {
        self.shared.volume.store(v.clamp(0.0, 2.0).to_bits(), Ordering::Relaxed);
    }

    pub fn duration(&self) -> f32 {
        self.shared.frames as f32 / self.sample_rate as f32
    }

    pub fn position(&self) -> f32 {
        self.shared.pos.load(Ordering::Relaxed) as f32 / self.sample_rate as f32
    }

    pub fn seek(&self, secs: f32) {
        let f = ((secs.max(0.0) * self.sample_rate as f32) as usize).min(self.shared.frames);
        self.shared.pos.store(f, Ordering::Relaxed);
    }

    /// 以播放头为中心取分析窗口（单声道）。
    pub fn window_at_playhead(&self, out: &mut [f32]) {
        let center = self.shared.pos.load(Ordering::Relaxed) as isize;
        window_centered(&self.mono, center, out);
    }
}

/// 以 `center` 为中心从 `src` 复制窗口，越界部分补零。
pub fn window_centered(src: &[f32], center: isize, out: &mut [f32]) {
    let n = out.len() as isize;
    let start = center - n / 2;
    for (i, o) in out.iter_mut().enumerate() {
        let j = start + i as isize;
        *o = if j >= 0 && (j as usize) < src.len() { src[j as usize] } else { 0.0 };
    }
}

fn build_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: Arc<PlayerShared>,
    tap: AudioTap,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let ch = config.channels.max(1) as usize;
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            let playing = shared.playing.load(Ordering::Relaxed);
            let vol = f32::from_bits(shared.volume.load(Ordering::Relaxed));
            let start = shared.pos.load(Ordering::Relaxed);
            let mut pos = start;
            let mut tap = tap.0.lock().ok();
            let mut rec = tap.as_mut().and_then(|g| g.as_mut());
            for frame in data.chunks_mut(ch) {
                let (l0, r0) = if playing && pos < shared.frames {
                    let i = pos * 2;
                    pos += 1;
                    (shared.stereo[i], shared.stereo[i + 1])
                } else {
                    (0.0, 0.0)
                };
                // 录音取音量调整前的原始声音（暂停时为静音，保持音画同步）
                if let Some(b) = rec.as_mut() {
                    b.samples.extend_from_slice(&[l0, r0]);
                }
                let (l, r) = (l0 * vol, r0 * vol);
                if ch == 1 {
                    frame[0] = T::from_sample(0.5 * (l + r));
                } else {
                    frame[0] = T::from_sample(l);
                    frame[1] = T::from_sample(r);
                    for s in frame.iter_mut().skip(2) {
                        *s = T::from_sample(0.0f32);
                    }
                }
            }
            if playing {
                // 若期间发生了 seek，则不覆盖新的位置
                let _ = shared
                    .pos
                    .compare_exchange(start, pos, Ordering::Relaxed, Ordering::Relaxed);
                if pos >= shared.frames {
                    shared.playing.store(false, Ordering::Relaxed);
                }
            }
        },
        |e| log::warn!("音频输出错误：{e}"),
        None,
    )?;
    Ok(stream)
}
