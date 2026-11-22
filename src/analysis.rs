//! 音频分析：短时傅里叶变换 → 每个板模态的激发能量 → 平滑后的模态权重、振动强度与瞬态冲击。

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::modes::ModeSet;
use crate::params::Params;

pub const FFT_SIZE: usize = 4096;
/// GUI 频谱显示的频带数量。
pub const DISPLAY_BANDS: usize = 72;

/// 一帧分析结果，驱动 GPU 模拟。
#[derive(Clone, Debug, Default)]
pub struct AnalysisFrame {
    /// 每个模态的平滑权重（0..1）。
    pub weights: Vec<f32>,
    /// 振动强度（0 表示沙粒静止）。
    pub drive: f32,
    /// 瞬态冲击强度。
    pub kick: f32,
    /// 当前电平（dBFS）。
    pub level_db: f32,
    /// 是否高于静音门限。
    pub active: bool,
}

pub struct Analyzer {
    sample_rate: f32,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    window_sum: f32,
    window_sq_sum: f32,
    input: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    power: Vec<f32>,
    mags: Vec<f32>,
    smooth_mags: Vec<f32>,
    /// 每个模态：(起始频点, 高斯权重序列, 频谱倾斜补偿系数)
    filters: Vec<(usize, Vec<f32>, f32)>,
    groups: Vec<u8>,
    /// 当前对称类（对称模式下）
    group: Option<u8>,
    energies: Vec<f32>,
    targets: Vec<f32>,
    weights: Vec<f32>,
    drive: f32,
    peak_db: f32,
    kick: f32,
    display: Vec<f32>,
    display_edges: Vec<usize>,
    flux_range: (usize, usize),
}

impl Analyzer {
    pub fn new(sample_rate: u32, modes: &ModeSet, p: &Params) -> Self {
        let sample_rate = sample_rate.max(8000) as f32;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| {
                let x = i as f32 / FFT_SIZE as f32;
                0.5 - 0.5 * (std::f32::consts::TAU * x).cos()
            })
            .collect();
        let window_sum = window.iter().sum();
        let window_sq_sum = window.iter().map(|w| w * w).sum();
        let bins = FFT_SIZE / 2 + 1;
        let bin_hz = sample_rate / FFT_SIZE as f32;

        // 每个模态一个对数频率上的高斯共振滤波器。
        let sigma = (p.bandwidth / 2.355).max(0.01);
        let filters = modes
            .modes
            .iter()
            .map(|m| {
                let lo = m.freq * 2f32.powf(-3.0 * sigma);
                let hi = m.freq * 2f32.powf(3.0 * sigma);
                let b0 = ((lo / bin_hz).floor() as usize).clamp(1, bins - 1);
                let b1 = ((hi / bin_hz).ceil() as usize).clamp(b0, bins - 1);
                let w: Vec<f32> = (b0..=b1)
                    .map(|b| {
                        let f = b as f32 * bin_hz;
                        let d = (f / m.freq).log2() / sigma;
                        (-0.5 * d * d).exp()
                    })
                    .collect();
                let tilt = 10f32.powf(p.tilt * (m.freq / p.freq_lo).log2() / 10.0);
                (b0, w, tilt)
            })
            .collect();

        // 频谱显示：40 Hz ~ min(16 kHz, Nyquist) 的对数频带。
        let f_top = (sample_rate * 0.5).min(16000.0);
        let display_edges = (0..=DISPLAY_BANDS)
            .map(|i| {
                let f = 40.0 * (f_top / 40.0).powf(i as f32 / DISPLAY_BANDS as f32);
                ((f / bin_hz).round() as usize).clamp(1, bins - 1)
            })
            .collect();
        let flux_lo = ((p.freq_lo * 0.5 / bin_hz) as usize).clamp(1, bins - 1);
        let flux_hi = ((9000.0 / bin_hz) as usize).clamp(flux_lo + 1, bins);

        let k = modes.len();
        Self {
            sample_rate,
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            input: vec![0.0; FFT_SIZE],
            fft,
            window,
            window_sum,
            window_sq_sum,
            power: vec![0.0; bins],
            mags: vec![0.0; bins],
            smooth_mags: vec![0.0; bins],
            filters,
            groups: modes.modes.iter().map(|m| m.group).collect(),
            group: None,
            energies: vec![0.0; k],
            targets: vec![0.0; k],
            weights: vec![0.0; k],
            drive: 0.0,
            peak_db: -30.0,
            kick: 0.0,
            display: vec![0.0; DISPLAY_BANDS],
            display_edges,
            flux_range: (flux_lo, flux_hi),
        }
    }

    /// 最近一次分析的频谱（0..1），用于界面显示。
    pub fn display_spectrum(&self) -> &[f32] {
        &self.display
    }

    /// 频谱显示的频率范围（Hz）。
    pub fn display_range(&self) -> (f32, f32) {
        (40.0, (self.sample_rate * 0.5).min(16000.0))
    }

    /// 用第一段有声音的音频直接设定模态权重（导出时预热用）。
    pub fn prime(&mut self, samples: &[f32], p: &Params) -> AnalysisFrame {
        let mut f = self.process(samples, 10.0, p);
        self.drive = 0.0;
        self.kick = 0.0;
        f.drive = 0.0;
        f.kick = 0.0;
        f
    }

    /// 分析一段音频窗口。`samples` 取最后 FFT_SIZE 个样本（不足则前面补零），
    /// `dt` 为距上次分析的时间（秒）。
    pub fn process(&mut self, samples: &[f32], dt: f32, p: &Params) -> AnalysisFrame {
        let dt = dt.max(1e-4);
        let n = samples.len().min(FFT_SIZE);
        let pad = FFT_SIZE - n;
        self.input[..pad].fill(0.0);
        self.input[pad..].copy_from_slice(&samples[samples.len() - n..]);

        let mut sq = 0.0f32;
        for (x, w) in self.input.iter_mut().zip(&self.window) {
            *x *= w;
            sq += *x * *x;
        }
        let rms = (sq / self.window_sq_sum).sqrt();
        let level_db = 20.0 * (rms + 1e-9).log10();

        let _ = self
            .fft
            .process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch);
        let scale = 2.0 / self.window_sum;
        for ((c, pw), mg) in self.spectrum.iter().zip(&mut self.power).zip(&mut self.mags) {
            let m = c.norm() * scale;
            *mg = m;
            *pw = m * m;
        }

        // 频谱显示
        for (i, d) in self.display.iter_mut().enumerate() {
            let (b0, b1) = (self.display_edges[i], self.display_edges[i + 1].max(self.display_edges[i] + 1));
            let b1 = b1.min(self.power.len());
            let mean = self.power[b0..b1].iter().sum::<f32>() / (b1 - b0).max(1) as f32;
            let db = 10.0 * (mean + 1e-12).log10();
            let v = ((db + 90.0) / 80.0).clamp(0.0, 1.0);
            *d = if v > *d { v } else { *d + (v - *d) * (1.0 - (-dt / 0.15).exp()) };
        }

        let active = level_db > p.gate_db;

        // 模态激发能量（时间平滑，避免图案随每个音节抖动）→ 目标权重
        if active {
            let ce = smooth_coef(dt, p.smoothing);
            for (e, (b0, w, tilt)) in self.energies.iter_mut().zip(&self.filters) {
                let x: f32 = w.iter().zip(&self.power[*b0..]).map(|(a, b)| a * b).sum::<f32>() * tilt;
                *e += (x - *e) * ce;
            }
            let mut energies = self.energies.clone();
            let (imax, emax) = energies
                .iter()
                .copied()
                .enumerate()
                .fold((0, 0.0f32), |acc, (i, e)| if e > acc.1 { (i, e) } else { acc });
            if emax > 1e-14 {
                // 对称模式下只叠加与主导模态同一对称类的模态，保持四重对称；
                // 切换对称类需要明显更强（滞回），避免来回跳。
                if p.symmetric {
                    let g_new = self.groups[imax];
                    let g = match self.group {
                        Some(g) if g != g_new => {
                            let cur = energies
                                .iter()
                                .zip(&self.groups)
                                .filter(|(_, &gi)| gi == g)
                                .map(|(e, _)| *e)
                                .fold(0.0f32, f32::max);
                            if emax > cur * 1.8 {
                                g_new
                            } else {
                                g
                            }
                        }
                        _ => g_new,
                    };
                    self.group = Some(g);
                    for (e, &gi) in energies.iter_mut().zip(&self.groups) {
                        if gi != g {
                            *e = 0.0;
                        }
                    }
                } else {
                    self.group = None;
                }
                let emax = energies.iter().copied().fold(0.0f32, f32::max).max(1e-20);
                for e in &mut energies {
                    *e = (*e / emax).powf(p.sharpness);
                }
                // 仅保留最强的 top_modes 个模态
                let keep = (p.top_modes as usize).max(1);
                if energies.len() > keep {
                    let mut sorted = energies.clone();
                    sorted.sort_by(|a, b| b.total_cmp(a));
                    let thresh = sorted[keep - 1];
                    for e in &mut energies {
                        if *e < thresh {
                            *e = 0.0;
                        }
                    }
                }
                self.targets.copy_from_slice(&energies);
            }
        }

        let ca = smooth_coef(dt, p.attack);
        let cr = smooth_coef(dt, p.release);
        let c_out = smooth_coef(dt, 0.05);
        for ((w, t), &g) in self.weights.iter_mut().zip(&self.targets).zip(&self.groups) {
            let c = if *t > *w {
                ca
            } else if self.group.is_some_and(|cur| cur != g) {
                // 非当前对称类：快速淡出
                c_out
            } else {
                cr
            };
            *w += (*t - *w) * c;
        }

        // 振动强度：带自动增益的电平映射
        self.peak_db = (self.peak_db - 4.0 * dt).max(level_db).max(p.gate_db + 24.0);
        let target_drive = if active {
            let x = ((level_db - p.gate_db) / (self.peak_db - p.gate_db)).clamp(0.0, 1.0);
            p.sensitivity * x.powf(0.7)
        } else {
            0.0
        };
        let c = if target_drive > self.drive {
            smooth_coef(dt, 0.02)
        } else {
            smooth_coef(dt, 0.12)
        };
        self.drive += (target_drive - self.drive) * c;

        // 瞬态：相对平滑频谱的正向谱通量
        let (f0, f1) = self.flux_range;
        let mut flux = 0.0f32;
        let mut total = 1e-9f32;
        for b in f0..f1 {
            flux += (self.mags[b] - self.smooth_mags[b]).max(0.0);
            total += self.mags[b];
        }
        let flux = flux / total;
        let cs = smooth_coef(dt, 0.06);
        for (s, m) in self.smooth_mags.iter_mut().zip(&self.mags) {
            *s += (*m - *s) * cs;
        }
        self.kick *= (-dt / 0.12).exp();
        if active {
            let k = ((flux - 0.25) * 3.0).clamp(0.0, 1.5) * p.onset_kick;
            self.kick = self.kick.max(k);
        }

        AnalysisFrame {
            weights: self.weights.clone(),
            drive: self.drive,
            kick: self.kick,
            level_db,
            active,
        }
    }
}

fn smooth_coef(dt: f32, tau: f32) -> f32 {
    if tau <= 1e-4 {
        1.0
    } else {
        1.0 - (-dt / tau).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 * (std::f32::consts::TAU * freq * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn sine_excites_nearest_mode() {
        let p = Params::default();
        let modes = ModeSet::build(&p);
        let mut a = Analyzer::new(48000, &modes, &p);
        let target = &modes.modes[modes.len() / 2];
        let s = sine(target.freq, 48000.0, FFT_SIZE);
        let f = a.prime(&s, &p);
        let best = f
            .weights
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .unwrap()
            .0;
        assert!((modes.modes[best].freq / target.freq).log2().abs() < 0.1);
        assert!(f.active);
    }

    #[test]
    fn silence_is_inactive() {
        let p = Params::default();
        let modes = ModeSet::build(&p);
        let mut a = Analyzer::new(48000, &modes, &p);
        let f = a.process(&vec![0.0; FFT_SIZE], 1.0 / 30.0, &p);
        assert!(!f.active);
        assert_eq!(f.drive, 0.0);
        assert!(f.weights.iter().all(|&w| w == 0.0));
    }
}
