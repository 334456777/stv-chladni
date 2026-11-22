//! 图形界面：实时预览（麦克风 / 文件播放）+ 参数调节 + 实时录制 + 后台导出视频。

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use egui::{Color32, RichText};
use serde::{Deserialize, Serialize};

use crate::analysis::{AnalysisFrame, Analyzer, FFT_SIZE};
use crate::audio::live::{list_input_devices, AudioTap, MicInput, Player};
use crate::export::{self, Codec, ExportSettings, ExportSummary, Progress};
use crate::gpu::{steps_for, FrameGrabber, Simulator};
use crate::modes::ModeSet;
use crate::params::{Params, PlateShape, Preset};
use crate::record::{self, RecordConfig, Recorder};

const PREVIEW_MAX_SIDE: u32 = 1024;
/// 界面帧率上限（垂直同步失效时的保险）。
const MAX_UI_FPS: f32 = 165.0;
/// 窗口底部留白（逻辑像素）。
const BOTTOM_GAP: f32 = 18.0;
const RED: Color32 = Color32::from_rgb(235, 120, 110);
const GREEN: Color32 = Color32::from_rgb(140, 210, 140);
const YELLOW: Color32 = Color32::from_rgb(232, 196, 104);
const MEDIA_EXTS: &[&str] = &[
    "wav", "mp3", "flac", "ogg", "opus", "m4a", "aac", "wma", "aiff", "aif", "mp4", "mov", "mkv", "webm", "avi",
];

/// 启动参数。
#[derive(Default)]
pub struct Launch {
    pub params: Option<Params>,
    /// 启动后立即打开并播放的文件
    pub open: Option<PathBuf>,
    /// 播放音量（只对本次启动生效）
    pub volume: Option<f32>,
    /// 撒沙动画时长（只对本次启动生效）
    pub sprinkle: Option<f32>,
}

pub fn run(launch: Launch) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("stv · speak to video")
            .with_inner_size([1380.0, 900.0])
            .with_min_inner_size([960.0, 640.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            present_mode: wgpu::PresentMode::AutoVsync,
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew {
                instance_descriptor: wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::PRIMARY,
                    ..Default::default()
                },
                power_preference: wgpu::PowerPreference::HighPerformance,
                native_adapter_selector: Some(Arc::new(select_adapter)),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "stv",
        options,
        Box::new(move |cc| {
            StvApp::new(cc, launch)
                .map(|a| Box::new(a) as Box<dyn eframe::App>)
                .map_err(Into::into)
        }),
    )
    .map_err(|e| anyhow!("界面启动失败：{e}"))
}

/// 选择显卡适配器：优先独立显卡，其次优先 DX12。
///
/// 在 Windows 上，部分驱动的 Vulkan 交换链不遵守垂直同步，界面会空转到数百帧、占满主线程，
/// 导致拖动窗口明显滞后；DX12 的翻转模型交换链能正确同步到显示器刷新率。
fn select_adapter(adapters: &[wgpu::Adapter], surface: Option<&wgpu::Surface<'_>>) -> Result<wgpu::Adapter, String> {
    adapters
        .iter()
        .filter(|a| surface.is_none_or(|s| a.is_surface_supported(s)))
        .max_by_key(|a| {
            let info = a.get_info();
            let device = match info.device_type {
                wgpu::DeviceType::DiscreteGpu => 3,
                wgpu::DeviceType::IntegratedGpu => 2,
                wgpu::DeviceType::VirtualGpu => 1,
                _ => 0,
            };
            let backend = match info.backend {
                wgpu::Backend::Dx12 | wgpu::Backend::Metal => 2,
                wgpu::Backend::Vulkan => 1,
                _ => 0,
            };
            (device, backend)
        })
        .cloned()
        .ok_or_else(|| "没有可用于窗口显示的显卡适配器".to_string())
}

enum Source {
    None,
    Mic(MicInput),
    File(Player),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct ExportPrefs {
    output: String,
    width: u32,
    height: u32,
    fps: u32,
    codec: Codec,
    quality: u32,
}

impl Default for ExportPrefs {
    fn default() -> Self {
        Self {
            output: String::new(),
            width: 1080,
            height: 1080,
            fps: 30,
            codec: Codec::Auto,
            quality: 18,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct RecordPrefs {
    /// 录像保存目录（空 = 默认目录）
    dir: String,
    /// 同时录入声音
    audio: bool,
}

impl Default for RecordPrefs {
    fn default() -> Self {
        Self { dir: String::new(), audio: true }
    }
}

impl RecordPrefs {
    fn dir(&self) -> PathBuf {
        if self.dir.trim().is_empty() {
            record::default_dir()
        } else {
            PathBuf::from(self.dir.trim())
        }
    }
}

struct ExportJob {
    handle: JoinHandle<Result<ExportSummary>>,
    progress: Arc<Progress>,
    started: Instant,
    output: PathBuf,
}

struct StvApp {
    rs: egui_wgpu::RenderState,
    adapter: String,
    params: Params,
    modes: ModeSet,
    mode_key: (PlateShape, bool, u32, [u32; 4]),
    analyzer: Analyzer,
    analyzer_sr: u32,
    sim: Simulator,
    tex_id: egui::TextureId,
    sim_shape: PlateShape,

    source: Source,
    devices: Vec<String>,
    device_choice: Option<String>,
    file_path: Option<PathBuf>,
    pending_open: Option<PathBuf>,
    /// 启动后自动打开麦克风
    pending_mic: bool,
    mic_started: Instant,
    /// 麦克风数据从何时起一直是精确的零（硬件静音的典型表现）
    mic_zero_since: Option<Instant>,
    volume: f32,
    /// 命令行覆盖的（本次值, 原保存值）；界面上没改动时退出写回原保存值
    cli_volume: Option<(f32, f32)>,
    cli_sprinkle: Option<(f32, f32)>,
    window: Vec<f32>,
    last_time: Instant,
    step_acc: f32,
    frame: AnalysisFrame,
    fps: f32,
    particles_dragging: bool,

    export: ExportPrefs,
    job: Option<ExportJob>,
    export_msg: Option<Outcome>,
    status: Option<String>,

    // 实时录制
    is_nvidia: bool,
    tap: AudioTap,
    rec: RecordPrefs,
    recorder: Option<Recorder>,
    grabber: Option<FrameGrabber>,
    rec_finish: Option<JoinHandle<Result<PathBuf>>>,
    rec_msg: Option<Outcome>,
}

/// 录制或导出的结果，显示在对应区域下方。
enum Outcome {
    /// 说明文字和保存好的文件
    Saved(String, PathBuf),
    Failed(String),
}

impl Outcome {
    fn show(&self, ui: &mut egui::Ui) {
        match self {
            Outcome::Saved(text, file) => {
                ui.label(RichText::new(format!("{text}\n{}", file.display())).color(GREEN));
                if ui.button("打开所在文件夹").clicked() {
                    reveal(file);
                }
            }
            Outcome::Failed(text) => {
                ui.label(RichText::new(text).color(RED));
            }
        }
    }
}

impl StvApp {
    fn new(cc: &eframe::CreationContext<'_>, launch: Launch) -> Result<Self> {
        let rs = cc
            .wgpu_render_state
            .clone()
            .context("需要 wgpu 渲染后端")?;
        setup_style(&cc.egui_ctx);

        let mut params = launch
            .params
            .or_else(|| cc.storage.and_then(|s| eframe::get_value::<Params>(s, "stv.params")))
            .unwrap_or_default();
        // 命令行给的音量 / 撒沙时长只对本次启动生效：记下原来保存的值，退出时写回
        let stored_volume = cc
            .storage
            .and_then(|s| eframe::get_value::<f32>(s, "stv.volume"))
            .unwrap_or(1.0)
            .clamp(0.0, 2.0);
        let volume = launch.volume.map(|v| v.clamp(0.0, 2.0)).unwrap_or(stored_volume);
        let cli_volume = launch.volume.map(|_| (volume, stored_volume));
        params.sanitize();
        let stored_sprinkle = params.sprinkle_secs;
        if let Some(s) = launch.sprinkle {
            params.sprinkle_secs = s;
            params.sanitize();
        }
        let cli_sprinkle = launch.sprinkle.map(|_| (params.sprinkle_secs, stored_sprinkle));
        let export: ExportPrefs = cc
            .storage
            .and_then(|s| eframe::get_value(s, "stv.export"))
            .unwrap_or_default();
        let rec: RecordPrefs = cc
            .storage
            .and_then(|s| eframe::get_value(s, "stv.record"))
            .unwrap_or_default();

        let info = rs.adapter.get_info();
        let adapter = format!("{} · {:?}", info.name, info.backend);
        let is_nvidia = info.vendor == 0x10DE;
        // 在后台检测 ffmpeg（找不到时面板顶部提示）；有 NVIDIA 显卡时再探测 NVENC，避免第一次录制时卡顿
        std::thread::spawn(move || {
            if crate::audio::decode::ffmpeg_found() && is_nvidia {
                crate::audio::decode::nvenc_works();
            }
        });
        let modes = ModeSet::build(&params);
        let analyzer = Analyzer::new(48_000, &modes, &params);
        let (pw, ph) = preview_size(export.width, export.height);
        let sim = Simulator::new(&rs.device, &rs.queue, pw, ph, &params, true);
        let tex_id = rs
            .renderer
            .write()
            .register_native_texture(&rs.device, &sim.srgb_view(), wgpu::FilterMode::Linear);

        Ok(Self {
            adapter,
            mode_key: params.mode_key(),
            sim_shape: params.shape,
            modes,
            analyzer,
            analyzer_sr: 48_000,
            sim,
            tex_id,
            rs,
            params,
            source: Source::None,
            devices: list_input_devices(),
            device_choice: None,
            file_path: None,
            pending_mic: launch.open.is_none(),
            pending_open: launch.open,
            mic_started: Instant::now(),
            mic_zero_since: None,
            volume,
            cli_volume,
            cli_sprinkle,
            window: vec![0.0; FFT_SIZE],
            last_time: Instant::now(),
            step_acc: 0.0,
            frame: AnalysisFrame::default(),
            fps: 60.0,
            particles_dragging: false,
            export,
            job: None,
            export_msg: None,
            status: None,
            is_nvidia,
            tap: AudioTap::default(),
            rec,
            recorder: None,
            grabber: None,
            rec_finish: None,
            rec_msg: None,
        })
    }

    // ------------------------------------------------------------------ 音源

    fn start_mic(&mut self) {
        self.source = Source::None;
        match MicInput::start(self.device_choice.as_deref(), self.tap.clone()) {
            Ok(m) => {
                self.status = None;
                self.source = Source::Mic(m);
                self.mic_started = Instant::now();
            }
            Err(e) => self.status = Some(format!("麦克风启动失败：{e:#}")),
        }
    }

    fn open_file(&mut self, path: &Path) {
        let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        let path = abs.as_path();
        self.source = Source::None;
        match Player::open(path, self.tap.clone()) {
            Ok(p) => {
                p.set_volume(self.volume);
                p.set_playing(true);
                self.source = Source::File(p);
                self.file_path = Some(path.to_path_buf());
                self.export.output = default_output(path).display().to_string();
                self.status = None;
            }
            Err(e) => self.status = Some(format!("打开失败：{e:#}")),
        }
    }

    fn pick_file(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("音频 / 视频", MEDIA_EXTS)
            .add_filter("所有文件", &["*"])
            .pick_file()
        {
            self.open_file(&p);
        }
    }

    // ------------------------------------------------------------------ 模拟

    fn advance(&mut self, dt: f32) {
        if self.params.mode_key() != self.mode_key {
            self.modes = ModeSet::build(&self.params);
            self.analyzer = Analyzer::new(self.analyzer_sr, &self.modes, &self.params);
            self.mode_key = self.params.mode_key();
        }
        let (sr, live) = match &self.source {
            Source::Mic(m) => (m.sample_rate, true),
            Source::File(p) => (p.sample_rate, p.is_playing()),
            Source::None => (self.analyzer_sr, false),
        };
        if sr != self.analyzer_sr {
            self.analyzer = Analyzer::new(sr, &self.modes, &self.params);
            self.analyzer_sr = sr;
        }
        match &self.source {
            Source::Mic(m) => {
                m.latest(&mut self.window);
                let all_zero = self.window.iter().all(|&x| x == 0.0);
                match (all_zero, self.mic_zero_since) {
                    (true, None) => self.mic_zero_since = Some(Instant::now()),
                    (false, _) => self.mic_zero_since = None,
                    _ => {}
                }
            }
            Source::File(p) if live => p.window_at_playhead(&mut self.window),
            _ => self.window.fill(0.0),
        }
        self.frame = self.analyzer.process(&self.window, dt, &self.params);

        // 录制时按录制分辨率渲染，预览直接缩放显示
        let (pw, ph) = match &self.recorder {
            Some(r) => r.size(),
            None => preview_size(self.export.width, self.export.height),
        };
        if self.sim.size() != (pw, ph) {
            self.sim.resize(&self.rs.device, pw, ph);
            self.rs.renderer.write().update_egui_texture_from_wgpu_texture(
                &self.rs.device,
                &self.sim.srgb_view(),
                wgpu::FilterMode::Linear,
                self.tex_id,
            );
        }
        if (self.sim.particle_count() != self.params.particles && !self.particles_dragging)
            || self.sim_shape != self.params.shape
        {
            self.sim.reset_particles(&self.rs.device, &self.rs.queue, &self.params, true);
            self.sim_shape = self.params.shape;
        }

        let gm = self.modes.gpu_modes(&self.frame.weights);
        self.sim.set_modes(&self.rs.queue, self.params.shape, &gm);
        self.sim.advance_anim(dt);
        let steps = steps_for(dt, self.params.speed, &mut self.step_acc);
        let mut enc = self
            .rs
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stv.preview") });
        self.sim.encode_step(&self.rs.queue, &mut enc, &self.frame, &self.params, steps);
        self.sim.encode_render(&self.rs.queue, &mut enc, &self.params);
        self.rs.queue.submit(Some(enc.finish()));

        self.record_frame();
    }

    // ------------------------------------------------------------------ 录制

    fn is_recording(&self) -> bool {
        self.recorder.is_some()
    }

    fn start_recording(&mut self) {
        let audio = if self.rec.audio {
            match &self.source {
                Source::Mic(m) => Some((m.sample_rate, 1)),
                Source::File(p) => Some((p.sample_rate, 2)),
                Source::None => None,
            }
        } else {
            None
        };
        let cfg = RecordConfig {
            output: self.rec.dir().join(record::timestamp_name()),
            width: self.export.width,
            height: self.export.height,
            fps: self.export.fps,
            codec: self.export.codec,
            quality: self.export.quality,
            audio,
        };
        match Recorder::start(cfg, self.tap.clone(), self.is_nvidia) {
            Ok(r) => {
                let (w, h) = r.size();
                self.grabber = Some(FrameGrabber::new(&self.rs.device, w, h, 4));
                self.recorder = Some(r);
                self.rec_msg = None;
            }
            Err(e) => self.rec_msg = Some(Outcome::Failed(format!("开始录制失败：{e:#}"))),
        }
    }

    /// 按录制帧率抓取刚渲染好的画面，并把已读回的帧送去编码。
    fn record_frame(&mut self) {
        let Some(rec) = &mut self.recorder else { return };
        let mut ready = Vec::new();
        if let Some(g) = &mut self.grabber {
            let due = rec.frames_due();
            if due > 0 {
                g.grab(&self.rs.device, &self.rs.queue, &self.sim, due, &mut ready);
            }
            g.collect(&self.rs.device, false, &mut ready);
        }
        let mut failed = rec.failed();
        for (pixels, repeat) in ready {
            if rec.push(pixels, repeat).is_err() {
                failed = true;
                break;
            }
        }
        if failed {
            // 编码线程出错：结束录制，错误信息在保存结果里显示
            self.stop_recording();
        }
    }

    fn stop_recording(&mut self) {
        let Some(rec) = self.recorder.take() else { return };
        let mut ready = Vec::new();
        if let Some(mut g) = self.grabber.take() {
            g.collect(&self.rs.device, true, &mut ready);
        }
        for (pixels, repeat) in ready {
            if rec.push(pixels, repeat).is_err() {
                break;
            }
        }
        self.rec_finish = Some(rec.finish());
        self.rec_msg = None;
    }

    fn poll_recording(&mut self) {
        if self.rec_finish.as_ref().is_some_and(|h| h.is_finished()) {
            let h = self.rec_finish.take().unwrap();
            self.rec_msg = Some(match h.join() {
                Ok(Ok(path)) => Outcome::Saved("录像已保存".into(), path),
                Ok(Err(e)) => Outcome::Failed(format!("录制失败：{e:#}")),
                Err(_) => Outcome::Failed("录制线程崩溃".into()),
            });
        }
    }

    fn ui_record(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("录制实时画面")
            .default_open(true)
            .show(ui, |ui| {
                let recording = self.is_recording();
                ui.add_enabled_ui(!recording, |ui| {
                    ui.checkbox(&mut self.rec.audio, "同时录入声音（麦克风 / 正在播放的文件）");
                    ui.horizontal(|ui| {
                        ui.label("保存到");
                        let hint = record::default_dir().display().to_string();
                        ui.add(
                            egui::TextEdit::singleline(&mut self.rec.dir)
                                .hint_text(hint)
                                .desired_width(190.0),
                        );
                        if ui.button("…").clicked() {
                            if let Some(d) = rfd::FileDialog::new().set_directory(self.rec.dir()).pick_folder() {
                                self.rec.dir = d.display().to_string();
                            }
                        }
                    });
                });
                ui.label(
                    RichText::new(format!("{}×{}  {} fps", self.export.width, self.export.height, self.export.fps))
                        .weak(),
                );
                if let Some(rec) = &self.recorder {
                    let t = rec.elapsed();
                    ui.horizontal(|ui| {
                        if ui.button(RichText::new("■ 停止录制").strong()).clicked() {
                            self.stop_recording();
                        }
                        ui.label(RichText::new(format!("● 录制中 {}", fmt_time(t))).color(RED));
                    });
                } else if self.rec_finish.is_some() {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("正在保存录像…");
                    });
                } else if ui.button(RichText::new("● 开始录制").strong()).clicked() {
                    self.start_recording();
                }
                if let Some(msg) = &self.rec_msg {
                    msg.show(ui);
                }
            });
    }

    // ------------------------------------------------------------------ 导出

    fn start_export(&mut self) {
        let Some(input) = self.file_path.clone() else {
            self.export_msg = Some(Outcome::Failed("请先打开一个音频文件".into()));
            return;
        };
        let output = if self.export.output.trim().is_empty() {
            default_output(&input)
        } else {
            PathBuf::from(self.export.output.trim())
        };
        let settings = ExportSettings {
            input,
            output: output.clone(),
            width: self.export.width,
            height: self.export.height,
            fps: self.export.fps,
            codec: self.export.codec,
            quality: self.export.quality,
            start: 0.0,
            duration: None,
        };
        let params = self.params.clone();
        let progress = Arc::new(Progress::default());
        let pg = progress.clone();
        let handle = std::thread::Builder::new()
            .name("stv-export".into())
            .spawn(move || export::export(&settings, &params, &pg))
            .expect("无法创建导出线程");
        self.export_msg = None;
        self.job = Some(ExportJob {
            handle,
            progress,
            started: Instant::now(),
            output,
        });
    }

    fn poll_export(&mut self) {
        if self.job.as_ref().is_some_and(|j| j.handle.is_finished()) {
            let job = self.job.take().unwrap();
            self.export_msg = Some(match job.handle.join() {
                Ok(Ok(s)) => Outcome::Saved(
                    format!("已导出 {} 帧，用时 {:.1}s（{}）", s.frames, s.seconds, s.encoder),
                    job.output,
                ),
                Ok(Err(e)) => Outcome::Failed(format!("导出失败：{e:#}")),
                Err(_) => Outcome::Failed("导出线程崩溃".into()),
            });
        }
    }

    // ------------------------------------------------------------------ 界面

    fn ui_source(&mut self, ui: &mut egui::Ui) {
        ui.heading("音源");
        // 录制中不允许切换音源（声音格式会变）
        let recording = self.is_recording();
        ui.horizontal(|ui| {
            let mic_on = matches!(self.source, Source::Mic(_));
            if ui
                .add_enabled(!recording, egui::SelectableLabel::new(mic_on, "麦克风实时"))
                .clicked()
            {
                if mic_on {
                    self.source = Source::None;
                } else {
                    self.start_mic();
                }
            }
            if ui.button("重新撒沙").clicked() {
                self.sim.reset_particles(&self.rs.device, &self.rs.queue, &self.params, true);
            }
            if ui.add_enabled(!recording, egui::Button::new("打开音频 / 视频…")).clicked() {
                self.pick_file();
            }
        });
        ui.add_enabled_ui(!recording, |ui| {
            egui::ComboBox::from_label("输入设备")
                .width(200.0)
                .selected_text(self.device_choice.clone().unwrap_or_else(|| "系统默认".into()))
                .show_ui(ui, |ui| {
                    let mut changed = ui.selectable_value(&mut self.device_choice, None, "系统默认").changed();
                    for d in self.devices.clone() {
                        changed |= ui.selectable_value(&mut self.device_choice, Some(d.clone()), d).changed();
                    }
                    if changed && matches!(self.source, Source::Mic(_)) {
                        self.start_mic();
                    }
                });
        });

        match &self.source {
            Source::None => {
                ui.label(RichText::new("未选择音源：可拖入音频文件，或打开麦克风").weak());
            }
            Source::Mic(m) => {
                ui.label(format!("监听中：{}（{} Hz）", m.name, m.sample_rate));
                let (received, err) = m.stats();
                if let Some(e) = err {
                    ui.label(RichText::new(format!("输入流错误：{e}")).color(RED));
                } else if received == 0 && self.mic_started.elapsed().as_secs_f32() > 1.0 {
                    ui.label(
                        RichText::new("没有收到音频数据：请在系统设置 → 隐私和安全性 → 麦克风中允许桌面应用访问")
                            .color(RED),
                    );
                } else if self.mic_zero_since.is_some_and(|t| t.elapsed().as_secs_f32() > 1.5) {
                    ui.label(
                        RichText::new("收到的全是静音数据（数字零）：麦克风可能被静音，请检查设备上的静音键")
                            .color(RED),
                    );
                } else {
                    ui.label(RichText::new(format!("已接收 {:.1} 秒音频", received as f32 / m.sample_rate as f32)).weak());
                }
            }
            Source::File(p) => {
                let name = p
                    .path
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                ui.label(RichText::new(name).strong());
                ui.horizontal(|ui| {
                    let playing = p.is_playing();
                    if ui.button(if playing { "⏸ 暂停" } else { "▶ 播放" }).clicked() {
                        p.set_playing(!playing);
                    }
                    if ui.button("⏮ 从头").clicked() {
                        p.seek(0.0);
                        p.set_playing(true);
                    }
                    ui.label(format!("{} / {}", fmt_time(p.position()), fmt_time(p.duration())));
                });
                let mut pos = p.position();
                let dur = p.duration().max(0.01);
                let w = ui.spacing().slider_width;
                ui.spacing_mut().slider_width = ui.available_width() - 12.0;
                if ui
                    .add(egui::Slider::new(&mut pos, 0.0..=dur).show_value(false))
                    .changed()
                {
                    p.seek(pos);
                }
                ui.spacing_mut().slider_width = w;
                if ui
                    .add(egui::Slider::new(&mut self.volume, 0.0..=1.5).text("音量"))
                    .changed()
                {
                    p.set_volume(self.volume);
                }
            }
        }
    }

    fn ui_plate(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        egui::CollapsingHeader::new("振动板与模态")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for s in [PlateShape::Square, PlateShape::Circle] {
                        ui.selectable_value(&mut p.shape, s, s.label());
                    }
                    ui.checkbox(&mut p.symmetric, "仅对称模态");
                });
                ui.add(egui::Slider::new(&mut p.max_order, 4..=16).text("最高阶数"));
                ui.add(
                    egui::Slider::new(&mut p.freq_lo, 30.0..=600.0)
                        .logarithmic(true)
                        .text("最低频率 Hz"),
                );
                ui.add(
                    egui::Slider::new(&mut p.freq_hi, 800.0..=12000.0)
                        .logarithmic(true)
                        .text("最高频率 Hz"),
                );
                ui.add(egui::Slider::new(&mut p.bandwidth, 0.05..=1.0).text("共振带宽 oct"));
                ui.add(egui::Slider::new(&mut p.tilt, -3.0..=9.0).text("高频补偿 dB/oct"));
                ui.add(egui::Slider::new(&mut p.sharpness, 0.5..=6.0).text("模态锐度"));
                ui.add(egui::Slider::new(&mut p.top_modes, 1..=8).text("叠加模态数"));
                ui.add(egui::Slider::new(&mut p.smoothing, 0.0..=2.0).text("图案停留 s"));
                ui.add(egui::Slider::new(&mut p.attack, 0.0..=0.5).text("起音 s"));
                ui.add(egui::Slider::new(&mut p.release, 0.0..=2.0).text("释放 s"));
                ui.label(RichText::new(format!("模态表共 {} 个模态", self.modes.len())).weak());
            });
    }

    fn ui_sand(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("沙粒")
            .default_open(true)
            .show(ui, |ui| {
                let p = &mut self.params;
                let r = ui.add(
                    egui::Slider::new(&mut p.particles, 5_000..=1_000_000)
                        .logarithmic(true)
                        .text("沙粒数量"),
                );
                self.particles_dragging = r.dragged() || r.has_focus();
                ui.add(egui::Slider::new(&mut p.pull, 0.0..=0.3).text("聚拢力度"));
                ui.add(egui::Slider::new(&mut p.bounce, 0.0..=0.05).text("波腹弹跳"));
                ui.add(egui::Slider::new(&mut p.jitter, 0.0..=0.02).text("线条粗细"));
                ui.add(egui::Slider::new(&mut p.onset_kick, 0.0..=4.0).text("瞬态散射"));
                ui.add(egui::Slider::new(&mut p.speed, 0.1..=4.0).logarithmic(true).text("模拟速度"));
                ui.add(egui::Slider::new(&mut p.gate_db, -90.0..=-10.0).text("静音门限 dB"));
                ui.add(egui::Slider::new(&mut p.sensitivity, 0.0..=3.0).text("灵敏度"));
                ui.add(
                    egui::Slider::new(&mut p.sprinkle_secs, 0.1..=3.0)
                        .logarithmic(true)
                        .text("撒沙时长 s"),
                );
            });
    }

    fn ui_look(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        egui::CollapsingHeader::new("外观")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("风格：");
                    for pr in Preset::ALL {
                        if ui.button(pr.label()).clicked() {
                            pr.apply(p);
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("沙色");
                    ui.color_edit_button_rgb(&mut p.sand_color);
                    ui.label("背景");
                    ui.color_edit_button_rgb(&mut p.bg_color);
                });
                ui.add(egui::Slider::new(&mut p.grain_size, 0.002..=0.02).text("颗粒大小"));
                ui.add(egui::Slider::new(&mut p.plate_scale, 0.3..=1.2).text("板面比例"));
                ui.add(egui::Slider::new(&mut p.grain_variation, 0.0..=1.0).text("明暗变化"));
            });
    }

    fn ui_export(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("导出视频")
            .default_open(true)
            .show(ui, |ui| {
                if let Some(name) = self.file_path.as_ref().and_then(|p| p.file_name()) {
                    ui.add(egui::Label::new(RichText::new(name.to_string_lossy()).weak()).truncate());
                }
                ui.horizontal(|ui| {
                    ui.label("保存到");
                    ui.add(egui::TextEdit::singleline(&mut self.export.output).desired_width(190.0));
                    if ui.button("…").clicked() {
                        let mut dlg = rfd::FileDialog::new().add_filter("MP4 视频", &["mp4"]);
                        if let Some(f) = &self.file_path {
                            if let Some(dir) = f.parent() {
                                dlg = dlg.set_directory(dir);
                            }
                            dlg = dlg.set_file_name(default_output(f).file_name().unwrap().to_string_lossy());
                        }
                        if let Some(out) = dlg.save_file() {
                            self.export.output = out.display().to_string();
                        }
                    }
                });

                let e = &mut self.export;
                let presets: [(&str, u32, u32); 5] = [
                    ("1080×1080 方形", 1080, 1080),
                    ("1920×1080 横屏", 1920, 1080),
                    ("1080×1920 竖屏", 1080, 1920),
                    ("720×720", 720, 720),
                    ("2160×2160 4K", 2160, 2160),
                ];
                let current = presets
                    .iter()
                    .find(|(_, w, h)| *w == e.width && *h == e.height)
                    .map(|x| x.0.to_string())
                    .unwrap_or_else(|| format!("{}×{} 自定义", e.width, e.height));
                egui::ComboBox::from_label("分辨率")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        for (name, w, h) in presets {
                            if ui.selectable_label(e.width == w && e.height == h, name).clicked() {
                                e.width = w;
                                e.height = h;
                            }
                        }
                    });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut e.width).range(64..=7680).prefix("宽 "));
                    ui.add(egui::DragValue::new(&mut e.height).range(64..=7680).prefix("高 "));
                });
                ui.horizontal(|ui| {
                    ui.label("帧率");
                    for f in [24, 30, 60] {
                        ui.selectable_value(&mut e.fps, f, f.to_string());
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("编码");
                    for c in [Codec::Auto, Codec::Nvenc, Codec::X264] {
                        ui.selectable_value(&mut e.codec, c, c.label());
                    }
                });
                ui.add(egui::Slider::new(&mut e.quality, 10..=35).text("质量 CRF/CQ（小=清晰）"));

                if let Some(job) = &self.job {
                    let pg = &job.progress;
                    let frac = pg.fraction();
                    let done = pg.frame.load(Ordering::Relaxed);
                    let total = pg.total.load(Ordering::Relaxed);
                    let el = job.started.elapsed().as_secs_f32();
                    let eta = if frac > 0.01 { el / frac - el } else { 0.0 };
                    let stage = pg.stage.lock().map(|s| s.clone()).unwrap_or_default();
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .text(format!("{done}/{total} 帧 · 剩余 {}", fmt_time(eta)))
                            .animate(true),
                    );
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(stage).weak());
                        if ui.button("取消").clicked() {
                            pg.cancel.store(true, Ordering::Relaxed);
                        }
                    });
                } else {
                    let can = self.file_path.is_some();
                    if ui
                        .add_enabled(can, egui::Button::new(RichText::new("开始导出").strong()))
                        .clicked()
                    {
                        self.start_export();
                    }
                }
                if let Some(msg) = &self.export_msg {
                    msg.show(ui);
                }
            });
    }

    fn ui_params_file(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("参数文件")
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("保存参数…").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("JSON", &["json"])
                            .set_file_name("stv-params.json")
                            .save_file()
                        {
                            let r = serde_json::to_string_pretty(&self.params)
                                .map_err(anyhow::Error::from)
                                .and_then(|s| std::fs::write(&path, s).map_err(Into::into));
                            if let Err(e) = r {
                                self.status = Some(format!("保存失败：{e}"));
                            }
                        }
                    }
                    if ui.button("载入参数…").clicked() {
                        if let Some(path) = rfd::FileDialog::new().add_filter("JSON", &["json"]).pick_file() {
                            match load_params(&path) {
                                Ok(p) => self.params = p,
                                Err(e) => self.status = Some(format!("载入失败：{e:#}")),
                            }
                        }
                    }
                    if ui.button("恢复默认").clicked() {
                        self.params = Params::default();
                    }
                });
            });
    }

    fn spectrum_panel(&self, ui: &mut egui::Ui) {
        // 下方留出 BOTTOM_GAP 的空白，不贴着窗口底边
        let size = ui.available_size() - egui::vec2(0.0, BOTTOM_GAP);
        let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_gray(16));
        let spec = self.analyzer.display_spectrum();
        let (f0, f1) = self.analyzer.display_range();
        let sc = self.params.sand_color;
        let sand = Color32::from_rgb((sc[0] * 255.0) as u8, (sc[1] * 255.0) as u8, (sc[2] * 255.0) as u8);
        let bw = rect.width() / spec.len() as f32;
        for (i, v) in spec.iter().enumerate() {
            let h = v * (rect.height() - 18.0);
            let x = rect.left() + i as f32 * bw;
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(x + 1.0, rect.bottom() - h), egui::pos2(x + bw - 1.0, rect.bottom())),
                1.0,
                sand.gamma_multiply(0.35),
            );
        }
        // 模态激发标记
        let xf = |f: f32| rect.left() + rect.width() * ((f / f0).ln() / (f1 / f0).ln()).clamp(0.0, 1.0);
        let mut top: Vec<(usize, f32)> = self.frame.weights.iter().copied().enumerate().filter(|x| x.1 > 0.02).collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(k, w) in &top {
            let x = xf(self.modes.modes[k].freq);
            painter.line_segment(
                [egui::pos2(x, rect.bottom()), egui::pos2(x, rect.bottom() - w * (rect.height() - 18.0))],
                egui::Stroke::new(2.0_f32, sand.gamma_multiply(0.4 + 0.6 * w)),
            );
        }
        let names: Vec<String> = top
            .iter()
            .take(4)
            .map(|&(k, w)| format!("{} {:.0}%", self.modes.modes[k].name, w * 100.0))
            .collect();
        let info = format!(
            "{}  电平 {:>6.1} dB   振动 {:.2}   瞬态 {:.2}   模态 {}",
            if self.frame.active { "●" } else { "○" },
            self.frame.level_db.max(-120.0),
            self.frame.drive,
            self.frame.kick,
            if names.is_empty() { "—".into() } else { names.join("  ") }
        );
        painter.text(
            rect.left_top() + egui::vec2(8.0, 4.0),
            egui::Align2::LEFT_TOP,
            info,
            egui::FontId::monospace(12.0),
            Color32::from_gray(170),
        );
    }
}

impl eframe::App for StvApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = (now - self.last_time).as_secs_f32().clamp(1e-4, 0.1);
        self.last_time = now;
        self.fps += (1.0 / dt - self.fps) * 0.05;
        self.poll_export();
        self.poll_recording();
        if let Some(p) = self.pending_open.take() {
            self.open_file(&p);
        } else if std::mem::take(&mut self.pending_mic) {
            self.start_mic();
        }

        // 拖入文件（录制中忽略）
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if let Some(p) = dropped.first() {
            if !self.is_recording() {
                self.open_file(p);
            }
        }
        // 空格：播放 / 暂停
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) && !ctx.wants_keyboard_input() {
            if let Source::File(p) = &self.source {
                p.set_playing(!p.is_playing());
            }
        }

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("stv").size(22.0).strong());
                    ui.label(RichText::new("speak to video").weak());
                });
                if crate::audio::decode::ffmpeg_found_cached() == Some(false) {
                    ui.label(RichText::new("未找到 ffmpeg：打开音频 / 视频、录制和导出都需要它。").color(YELLOW));
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        ui.hyperlink_to("下载 ffmpeg", crate::audio::decode::FFMPEG_URL);
                        ui.label("，把 ffmpeg.exe 放到 stv.exe 旁边、加入 PATH 或设置环境变量 STV_FFMPEG，然后重启 stv。");
                    });
                }
                if let Some(s) = &self.status {
                    ui.label(RichText::new(s).color(RED));
                }
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.ui_source(ui);
                    ui.separator();
                    self.ui_record(ui);
                    self.ui_export(ui);
                    self.ui_look(ui);
                    self.ui_sand(ui);
                    self.ui_plate(ui);
                    self.ui_params_file(ui);
                    ui.add_space(BOTTOM_GAP);
                });
            });

        self.params.sanitize();
        self.advance(dt);

        egui::TopBottomPanel::bottom("spectrum")
            .exact_height(92.0 + BOTTOM_GAP)
            .show(ctx, |ui| self.spectrum_panel(ui));

        let bg = self.params.bg_color;
        let bg = Color32::from_rgb((bg[0] * 255.0) as u8, (bg[1] * 255.0) as u8, (bg[2] * 255.0) as u8);
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(bg))
            .show(ctx, |ui| {
                let avail = ui.available_rect_before_wrap();
                let (tw, th) = self.sim.size();
                let s = (avail.width() / tw as f32).min(avail.height() / th as f32);
                let rect = egui::Rect::from_center_size(avail.center(), egui::vec2(tw as f32 * s, th as f32 * s));
                ui.painter().image(
                    self.tex_id,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
                ui.painter().text(
                    avail.left_top() + egui::vec2(10.0, 8.0),
                    egui::Align2::LEFT_TOP,
                    format!(
                        "{}   {:.0} fps   {} 粒   {} {}×{}",
                        self.adapter,
                        self.fps,
                        self.sim.particle_count(),
                        if self.is_recording() { "录制" } else { "预览" },
                        tw,
                        th
                    ),
                    egui::FontId::monospace(12.0),
                    Color32::from_gray(110),
                );
                if let Some(rec) = &self.recorder {
                    // 录制指示：右上角红点 + 时长（只在界面上显示，不进录像）
                    let pos = egui::pos2(avail.right() - 14.0, avail.top() + 16.0);
                    let blink = ((rec.elapsed() * 2.0) as u32).is_multiple_of(2);
                    if blink {
                        ui.painter().circle_filled(pos - egui::vec2(96.0, 0.0), 6.0, RED);
                    }
                    ui.painter().text(
                        pos,
                        egui::Align2::RIGHT_CENTER,
                        format!("REC {}", fmt_time(rec.elapsed())),
                        egui::FontId::monospace(14.0),
                        RED,
                    );
                }
            });

        // 帧率上限：正常情况下由垂直同步限制，这里只是兜底，避免空转占满主线程
        ctx.request_repaint_after(Duration::from_secs_f32(1.0 / MAX_UI_FPS));
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        // 命令行临时覆盖的值，若界面上没再改动，就保存原来的值
        let mut params = self.params.clone();
        if let Some((cli, stored)) = self.cli_sprinkle {
            if params.sprinkle_secs == cli {
                params.sprinkle_secs = stored;
            }
        }
        let volume = match self.cli_volume {
            Some((cli, stored)) if self.volume == cli => stored,
            _ => self.volume,
        };
        eframe::set_value(storage, "stv.params", &params);
        eframe::set_value(storage, "stv.export", &self.export);
        eframe::set_value(storage, "stv.volume", &volume);
        eframe::set_value(storage, "stv.record", &self.rec);
    }

    fn on_exit(&mut self) {
        if let Some(job) = &self.job {
            job.progress.cancel.store(true, Ordering::Relaxed);
        }
        // 关闭窗口时正在录制：保存好录像再退出
        self.stop_recording();
        if let Some(h) = self.rec_finish.take() {
            let _ = h.join();
        }
    }
}

fn preview_size(w: u32, h: u32) -> (u32, u32) {
    let (w, h) = (w.max(1) as f32, h.max(1) as f32);
    let s = PREVIEW_MAX_SIDE as f32 / w.max(h);
    (((w * s).round() as u32).max(64), ((h * s).round() as u32).max(64))
}

fn default_output(input: &Path) -> PathBuf {
    let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "out".into());
    input.with_file_name(format!("{stem}_stv.mp4"))
}

/// 打开 `file` 所在的文件夹并选中它；文件已被移走时只打开文件夹。
fn reveal(file: &Path) {
    let file = std::path::absolute(file).unwrap_or_else(|_| file.to_path_buf());
    let mut cmd;
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd = std::process::Command::new("explorer");
        if file.is_file() {
            // explorer 自己解析命令行：整个参数加引号时认不出 /select，只能给路径单独加引号
            let mut arg = std::ffi::OsString::from("/select,\"");
            arg.push(file.as_os_str());
            arg.push("\"");
            cmd.raw_arg(arg);
        } else if let Some(dir) = file.parent() {
            cmd.arg(dir);
        }
    }
    #[cfg(target_os = "macos")]
    {
        cmd = std::process::Command::new("open");
        if file.is_file() {
            cmd.arg("-R").arg(&file);
        } else if let Some(dir) = file.parent() {
            cmd.arg(dir);
        }
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        cmd = std::process::Command::new("xdg-open");
        cmd.arg(file.parent().unwrap_or(Path::new(".")));
    }
    // 双击启动时 main 释放了控制台，继承来的标准句柄已失效，不显式置空 spawn 会失败（os error 50）
    cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    if let Err(e) = cmd.spawn() {
        log::warn!("无法打开文件夹 {}：{e}", file.display());
    }
}

fn fmt_time(s: f32) -> String {
    let s = s.max(0.0);
    format!("{:02}:{:04.1}", (s / 60.0) as u32, s % 60.0)
}

pub fn load_params(path: &Path) -> Result<Params> {
    let text = std::fs::read_to_string(path).with_context(|| format!("读取 {}", path.display()))?;
    let mut p: Params = serde_json::from_str(&text).context("解析参数 JSON")?;
    p.sanitize();
    Ok(p)
}

fn setup_style(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\Deng.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ];
    for path in candidates {
        if let Ok(bytes) = std::fs::read(path) {
            fonts
                .font_data
                .insert("cjk".into(), Arc::new(egui::FontData::from_owned(bytes)));
            for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts.families.entry(fam).or_default().push("cjk".into());
            }
            break;
        }
    }
    ctx.set_fonts(fonts);
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.style_mut(|s| {
        s.spacing.slider_width = 150.0;
        s.spacing.item_spacing = egui::vec2(8.0, 6.0);
    });
}
