//! stv (speak to video)：用声音驱动 GPU 沙粒模拟，生成克拉尼振动纹路的实时画面或视频。
//!
//! Windows 上 stv.exe 是窗口程序，双击不会弹出终端；发布包里的 stv.com 是同一程序的控制台版本
//! （scripts/package.ps1 只改了 PE 头的子系统），终端里输入 `stv` 时按 PATHEXT 先找到它。
#![cfg_attr(all(windows, not(test)), windows_subsystem = "windows")]

mod analysis;
mod audio;
mod export;
mod gpu;
mod gui;
mod modes;
mod params;
mod record;

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

use crate::export::{Codec, ExportSettings, Progress};
use crate::params::{Params, PlateShape, Preset};

#[derive(Parser)]
#[command(
    name = "stv",
    version,
    about = "stv (speak to video)：把声音变成沙粒在振动板上形成的克拉尼纹路",
    long_about = None,
    args_conflicts_with_subcommands = true
)]
struct Cli {
    /// 打开界面并播放这个音频 / 视频文件（把文件拖到 stv.exe 上、"打开方式"选 stv 时就是这样启动的）
    #[arg(value_parser = existing_file)]
    input: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

/// 拼错的子命令也会落到 `input` 上，不是已有文件就报错，而不是打开界面。
fn existing_file(s: &str) -> Result<PathBuf, String> {
    let p = PathBuf::from(s);
    if p.is_file() {
        Ok(p)
    } else {
        Err("既不是已有的文件，也不是子命令（见 stv --help）".into())
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// 打开图形界面（默认）
    Gui {
        /// 启动后立即打开并播放的音频 / 视频文件
        input: Option<PathBuf>,
        /// 启动时载入的参数 JSON
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// 播放音量（0 静音，1 原始音量；只对本次启动生效）
        #[arg(long)]
        volume: Option<f32>,
        /// 撒沙动画时长（秒，0.1～5；只对本次启动生效）
        #[arg(long)]
        sprinkle: Option<f32>,
    },
    /// 无头渲染：把音频（或带音轨的视频）转换成视频
    Render(RenderArgs),
    /// 输出默认参数 JSON（可修改后用 --config 载入）
    Config {
        /// 写入文件；省略则打印到标准输出
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// 列出可用的音频输入设备
    Devices,
}

#[derive(Args)]
struct RenderArgs {
    /// 输入音频 / 视频文件
    input: PathBuf,
    /// 输出 mp4 路径（默认：<输入名>_stv.mp4）
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 1080)]
    width: u32,
    #[arg(long, default_value_t = 1080)]
    height: u32,
    #[arg(long, default_value_t = 30)]
    fps: u32,
    /// 视频编码器
    #[arg(long, value_enum, default_value_t = Codec::Auto)]
    codec: Codec,
    /// 质量 CRF/CQ（越小越清晰）
    #[arg(long, default_value_t = 18)]
    quality: u32,
    /// 起始时间（秒）
    #[arg(long, default_value_t = 0.0)]
    start: f32,
    /// 导出时长（秒）
    #[arg(long)]
    duration: Option<f32>,
    /// 参数 JSON（由 `stv config` 生成或界面中保存）
    #[arg(short, long)]
    config: Option<PathBuf>,
    /// 风格预设
    #[arg(long, value_enum)]
    preset: Option<Preset>,
    /// 振动板形状（覆盖配置）
    #[arg(long, value_enum)]
    shape: Option<PlateShape>,
    /// 沙粒数量（覆盖配置）
    #[arg(long)]
    particles: Option<u32>,
    /// 随机种子（覆盖配置）
    #[arg(long)]
    seed: Option<u32>,
}

fn main() {
    attach_parent_console();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,wgpu_core=error,wgpu_hal=error,naga=warn"))
        .init();
    let cli = Cli::parse();
    let gui_mode = matches!(cli.cmd, None | Some(Cmd::Gui { .. }));
    if gui_mode {
        detach_console_if_owned();
    }
    let result = match cli.cmd {
        None => gui::run(gui::Launch {
            open: cli.input,
            ..Default::default()
        }),
        Some(Cmd::Gui { input, config, volume, sprinkle }) => config
            .as_deref()
            .map(gui::load_params)
            .transpose()
            .and_then(|params| gui::run(gui::Launch { params, open: input, volume, sprinkle })),
        Some(Cmd::Render(args)) => render(args),
        Some(Cmd::Config { output }) => write_config(output),
        Some(Cmd::Devices) => {
            for d in audio::live::list_input_devices() {
                println!("{d}");
            }
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("错误：{e:#}");
        if gui_mode {
            // 双击启动时没有终端，只能弹窗告诉用户
            rfd::MessageDialog::new()
                .set_level(rfd::MessageLevel::Error)
                .set_title("stv")
                .set_description(format!("{e:#}"))
                .show();
        }
        std::process::exit(1);
    }
}

fn render(a: RenderArgs) -> Result<()> {
    let mut params = match &a.config {
        Some(path) => gui::load_params(path)?,
        None => Params::default(),
    };
    if let Some(pr) = a.preset {
        pr.apply(&mut params);
    }
    if let Some(s) = a.shape {
        params.shape = s;
    }
    if let Some(n) = a.particles {
        params.particles = n;
    }
    if let Some(s) = a.seed {
        params.seed = s;
    }
    params.sanitize();

    let output = a.output.clone().unwrap_or_else(|| {
        let stem = a
            .input
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "out".into());
        a.input.with_file_name(format!("{stem}_stv.mp4"))
    });
    let settings = ExportSettings {
        input: a.input.clone(),
        output: output.clone(),
        width: a.width,
        height: a.height,
        fps: a.fps,
        codec: a.codec,
        quality: a.quality,
        start: a.start,
        duration: a.duration,
    };

    let progress = Arc::new(Progress::default());
    let pg = progress.clone();
    let handle = std::thread::spawn(move || export::export(&settings, &params, &pg));
    let mut last_stage = String::new();
    while !handle.is_finished() {
        std::thread::sleep(Duration::from_millis(250));
        let stage = progress.stage.lock().map(|s| s.clone()).unwrap_or_default();
        if stage != last_stage {
            if !last_stage.is_empty() {
                eprintln!();
            }
            last_stage = stage;
        }
        let done = progress.frame.load(Ordering::Relaxed);
        let total = progress.total.load(Ordering::Relaxed);
        eprint!(
            "\r{last_stage}  {done}/{total} 帧  {:5.1}%   ",
            progress.fraction() * 100.0
        );
        let _ = std::io::stderr().flush();
    }
    eprintln!();
    let summary = handle.join().map_err(|_| anyhow::anyhow!("导出线程崩溃"))??;
    println!(
        "完成：{}（{} 帧，{:.1}s，编码器 {}，GPU {}）",
        output.display(),
        summary.frames,
        summary.seconds,
        summary.encoder,
        summary.adapter
    );
    Ok(())
}

fn write_config(output: Option<PathBuf>) -> Result<()> {
    let json = serde_json::to_string_pretty(&Params::default())?;
    match output {
        Some(p) => std::fs::write(&p, json).with_context(|| format!("写入 {}", p.display()))?,
        None => println!("{json}"),
    }
    Ok(())
}

/// 在终端里直接运行窗口程序 stv.exe 时本来没有控制台，接到终端上，输出才看得见。
/// 终端不会等窗口程序结束，所以命令行用法应走 stv.com。
fn attach_parent_console() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Console::{AttachConsole, GetConsoleWindow, ATTACH_PARENT_PROCESS};
        if GetConsoleWindow().is_null() {
            AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}

/// 双击控制台版 stv.com（控制台只属于本进程）时隐藏它自带的控制台窗口；从终端启动则保留输出。
fn detach_console_if_owned() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Console::{FreeConsole, GetConsoleProcessList};
        let mut pids = [0u32; 4];
        if GetConsoleProcessList(pids.as_mut_ptr(), pids.len() as u32) == 1 {
            FreeConsole();
        }
    }
}
