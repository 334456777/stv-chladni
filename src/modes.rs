//! 振动板的本征模态表。
//!
//! * 方形自由边板采用经典 Chladni 近似：
//!   φ(u,v) = cos(mπu)·cos(nπv) ± cos(nπu)·cos(mπv)，u,v ∈ [0,1]，本征值 ∝ m²+n²。
//! * 圆形板采用贝塞尔模态：φ(r,θ) = J_n(j_{n,s}·r)·cos(nθ)，本征值 ∝ j_{n,s}²。
//!
//! 本征值按幂律映射到 [freq_lo, freq_hi] 的频率区间，音频频谱在每个模态频率附近的能量
//! 决定该模态的激发强度。

use std::sync::OnceLock;

use crate::params::{Params, PlateShape};

/// 贝塞尔查找表：阶数 0..BESSEL_ORDERS，x ∈ [0, BESSEL_XMAX]。
pub const BESSEL_ORDERS: usize = 17;
pub const BESSEL_LUT_N: usize = 2048;
pub const BESSEL_XMAX: f32 = 64.0;
/// 一次最多送进 GPU 叠加的模态数。
pub const MAX_GPU_MODES: usize = 64;

/// 送给 GPU 的模态描述（与 WGSL 中的 `Mode` 结构一致）。
/// 方形：a=m, b=n, c=±1；圆形：a=n, b=j_{n,s}, c=角相位。w 为已归一化的权重。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMode {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub w: f32,
}

#[derive(Clone, Debug)]
pub struct Mode {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    /// 映射后的共振频率（Hz）。
    pub freq: f32,
    /// 1 / RMS(φ)，用于把各模态归一化到相同能量。
    pub inv_rms: f32,
    /// 人类可读的名字，例如 "(2,6)+" 或 "J4·2"。
    pub name: String,
    /// 对称类：同类模态叠加后 |F| 仍保持四重对称。
    /// 方形：0 = "+"（关于对角线对称），1 = "−"（反对称）；
    /// 圆形：n ≡ 0 (mod 4) 为 0，n ≡ 2 (mod 4) 为 1，奇数阶为 2。
    pub group: u8,
}

#[derive(Clone, Debug)]
pub struct ModeSet {
    pub modes: Vec<Mode>,
}

impl ModeSet {
    pub fn build(p: &Params) -> Self {
        // (a, b, c, 本征值, 名字)
        let mut raw: Vec<(f32, f32, f32, f64, String)> = Vec::new();
        let n_max = p.max_order.clamp(2, 16);
        match p.shape {
            PlateShape::Square => {
                for m in 0..=n_max {
                    for n in m..=n_max {
                        if m == 0 && n == 0 {
                            continue;
                        }
                        if p.symmetric && (m % 2 == 1 || n % 2 == 1) {
                            continue;
                        }
                        let lambda = (m * m + n * n) as f64;
                        raw.push((m as f32, n as f32, 1.0, lambda, format!("({m},{n})+")));
                        if m != n {
                            // 反对称组合的频率略高一点，使两者可区分。
                            raw.push((
                                m as f32,
                                n as f32,
                                -1.0,
                                lambda * 1.04,
                                format!("({m},{n})−"),
                            ));
                        }
                    }
                }
            }
            PlateShape::Circle => {
                let zeros = bessel_zeros();
                for n in 0..=n_max.min(BESSEL_ORDERS as u32 - 1) {
                    if p.symmetric && n % 2 == 1 {
                        continue;
                    }
                    for (s, &k) in zeros[n as usize].iter().enumerate().take(4) {
                        if k > 40.0 {
                            break;
                        }
                        raw.push((
                            n as f32,
                            k as f32,
                            0.0,
                            k * k,
                            format!("J{}·{}", n, s + 1),
                        ));
                    }
                }
            }
        }
        raw.sort_by(|x, y| x.3.total_cmp(&y.3));

        let lmin = raw.first().map(|r| r.3).unwrap_or(1.0).max(1e-6);
        let lmax = raw.last().map(|r| r.3).unwrap_or(2.0).max(lmin * 1.001);
        let f_lo = p.freq_lo as f64;
        let f_hi = (p.freq_hi as f64).max(f_lo * 1.01);
        let beta = (f_hi / f_lo).ln() / (lmax / lmin).ln();

        let modes = raw
            .into_iter()
            .map(|(a, b, c, lambda, name)| {
                let freq = (f_lo * (lambda / lmin).powf(beta)) as f32;
                let rms = mode_rms(p.shape, a, b, c);
                let group = match p.shape {
                    PlateShape::Square => u8::from(c < 0.0),
                    PlateShape::Circle => {
                        let n = a as u32;
                        if n % 2 == 1 {
                            2
                        } else {
                            ((n / 2) % 2) as u8
                        }
                    }
                };
                Mode {
                    a,
                    b,
                    c,
                    freq,
                    inv_rms: if rms > 1e-6 { 1.0 / rms } else { 0.0 },
                    name,
                    group,
                }
            })
            .collect();
        Self { modes }
    }

    pub fn len(&self) -> usize {
        self.modes.len()
    }

    /// 根据每个模态的权重生成 GPU 模态列表：剔除近零项、按权重排序、整体归一化。
    pub fn gpu_modes(&self, weights: &[f32]) -> Vec<GpuMode> {
        let mut active: Vec<(usize, f32)> = weights
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, w)| w > 1e-3)
            .collect();
        active.sort_by(|x, y| y.1.total_cmp(&x.1));
        active.truncate(MAX_GPU_MODES);
        let norm = active.iter().map(|&(_, w)| w * w).sum::<f32>().sqrt();
        if norm <= 1e-6 {
            return Vec::new();
        }
        active
            .into_iter()
            .map(|(i, w)| {
                let m = &self.modes[i];
                GpuMode {
                    a: m.a,
                    b: m.b,
                    c: m.c,
                    w: w * m.inv_rms / norm,
                }
            })
            .collect()
    }
}

/// 模态在板面上的均方根（解析式）。
///
/// * 方形：⟨cos²(kπu)⟩ = 1（k=0）或 1/2，交叉项仅在 m=n 时不为零。
/// * 圆形：∫₀¹ J_n(kr)² r dr = (1 − n²/k²)·J_n(k)²/2（k 为 J_n' 的零点），
///   角向 ⟨cos²⟩ = 1/2（n≥1）。
fn mode_rms(shape: PlateShape, a: f32, b: f32, c: f32) -> f32 {
    match shape {
        PlateShape::Square => {
            let e = |k: f32| if k == 0.0 { 1.0f64 } else { 0.5 };
            let ms = if a == b {
                (1.0 + c as f64).powi(2) * e(a) * e(a)
            } else {
                2.0 * e(a) * e(b)
            };
            ms.sqrt() as f32
        }
        PlateShape::Circle => {
            let (n, k) = (a as f64, b as f64);
            let j = bessel_j(a as u32, k);
            let radial = 0.5 * (1.0 - n * n / (k * k)) * j * j;
            let ms = if a == 0.0 { 2.0 * radial } else { radial };
            ms.sqrt() as f32
        }
    }
}

/// 第一类贝塞尔函数 J_n(x)，用贝塞尔积分 + 梯形法（对周期函数谱精度收敛）。
pub fn bessel_j(n: u32, x: f64) -> f64 {
    const N: usize = 512;
    let nf = n as f64;
    let mut sum = 0.0;
    for i in 0..=N {
        let t = std::f64::consts::PI * i as f64 / N as f64;
        let g = (nf * t - x * t.sin()).cos();
        sum += if i == 0 || i == N { 0.5 * g } else { g };
    }
    sum / N as f64
}

/// J_n 的导数：J_n'(x) = (J_{n-1}(x) − J_{n+1}(x)) / 2，J_{-1} = −J_1。
pub fn bessel_jp(n: u32, x: f64) -> f64 {
    if n == 0 {
        -bessel_j(1, x)
    } else {
        0.5 * (bessel_j(n - 1, x) - bessel_j(n + 1, x))
    }
}

/// 每个阶数 n 的前若干个正的 J_n' 零点 j'_{n,s}。
///
/// 圆板边缘自由（与真实克拉尼板一致）时，径向导数在 r=1 处为 0，
/// 因此边缘是波腹而不是节线，沙子不会被永久吸附在圆周上。
pub fn bessel_zeros() -> &'static Vec<Vec<f64>> {
    static ZEROS: OnceLock<Vec<Vec<f64>>> = OnceLock::new();
    ZEROS.get_or_init(|| {
        (0..BESSEL_ORDERS as u32)
            .map(|n| {
                let mut zs = Vec::new();
                let step = 0.05;
                let mut x0 = 0.1;
                let mut f0 = bessel_jp(n, x0);
                while x0 < BESSEL_XMAX as f64 - 1.0 && zs.len() < 6 {
                    let x1 = x0 + step;
                    let f1 = bessel_jp(n, x1);
                    if f0 == 0.0 || f0.signum() != f1.signum() {
                        let (mut lo, mut hi, mut flo) = (x0, x1, f0);
                        for _ in 0..60 {
                            let mid = 0.5 * (lo + hi);
                            let fm = bessel_jp(n, mid);
                            if fm.signum() == flo.signum() {
                                lo = mid;
                                flo = fm;
                            } else {
                                hi = mid;
                            }
                        }
                        zs.push(0.5 * (lo + hi));
                    }
                    x0 = x1;
                    f0 = f1;
                }
                zs
            })
            .collect()
    })
}

/// GPU 用的贝塞尔查找表，按 `[n * BESSEL_LUT_N + i]` 排列。
pub fn bessel_lut() -> &'static Vec<f32> {
    static LUT: OnceLock<Vec<f32>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut v = Vec::with_capacity(BESSEL_ORDERS * BESSEL_LUT_N);
        for n in 0..BESSEL_ORDERS as u32 {
            for i in 0..BESSEL_LUT_N {
                let x = i as f64 / (BESSEL_LUT_N - 1) as f64 * BESSEL_XMAX as f64;
                v.push(bessel_j(n, x) as f32);
            }
        }
        v
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_known_values() {
        assert!((bessel_j(0, 0.0) - 1.0).abs() < 1e-12);
        assert!((bessel_j(1, 1.0) - 0.440_050_585_744_933_5).abs() < 1e-12);
        assert!((bessel_j(3, 10.0) - 0.058_379_379_305_186_8).abs() < 1e-10);
    }

    #[test]
    fn bessel_derivative_zeros() {
        let z = bessel_zeros();
        // j'_{0,1}（跳过 x=0 的平凡零点）、j'_{1,1}、j'_{2,1}、j'_{1,2}
        assert!((z[0][0] - 3.831_705_970_207_51).abs() < 1e-8);
        assert!((z[1][0] - 1.841_183_780_931_68).abs() < 1e-8);
        assert!((z[2][0] - 3.054_236_928_227_14).abs() < 1e-8);
        assert!((z[1][1] - 5.331_442_773_525_03).abs() < 1e-8);
    }

    /// 数值积分验证解析 RMS。
    #[test]
    fn analytic_rms_matches_numeric() {
        use std::f64::consts::PI;
        let cases = [
            (PlateShape::Square, 2.0, 6.0, 1.0),
            (PlateShape::Square, 0.0, 4.0, -1.0),
            (PlateShape::Square, 4.0, 4.0, 1.0),
            (PlateShape::Circle, 0.0, bessel_zeros()[0][1] as f32, 0.0),
            (PlateShape::Circle, 3.0, bessel_zeros()[3][0] as f32, 0.0),
        ];
        for (shape, a, b, c) in cases {
            const G: usize = 200;
            let (mut acc, mut cnt) = (0.0f64, 0usize);
            for j in 0..G {
                for i in 0..G {
                    let x = (i as f64 + 0.5) / G as f64 * 2.0 - 1.0;
                    let y = (j as f64 + 0.5) / G as f64 * 2.0 - 1.0;
                    let v = match shape {
                        PlateShape::Square => {
                            let (u, w) = (x * 0.5 + 0.5, y * 0.5 + 0.5);
                            let (m, n) = (a as f64, b as f64);
                            (m * PI * u).cos() * (n * PI * w).cos()
                                + c as f64 * (n * PI * u).cos() * (m * PI * w).cos()
                        }
                        PlateShape::Circle => {
                            if x * x + y * y > 1.0 {
                                continue;
                            }
                            let r = (x * x + y * y).sqrt();
                            bessel_j(a as u32, b as f64 * r) * (a as f64 * y.atan2(x)).cos()
                        }
                    };
                    acc += v * v;
                    cnt += 1;
                }
            }
            let numeric = (acc / cnt as f64).sqrt() as f32;
            let analytic = mode_rms(shape, a, b, c);
            assert!(
                (numeric - analytic).abs() / analytic < 0.02,
                "{shape:?} ({a},{b},{c}): numeric {numeric} vs analytic {analytic}"
            );
        }
    }

    #[test]
    fn square_modes_are_sorted_and_normalized() {
        let p = Params::default();
        let set = ModeSet::build(&p);
        assert!(set.len() > 10);
        assert!(set.modes.windows(2).all(|w| w[0].freq <= w[1].freq));
        assert!((set.modes[0].freq - p.freq_lo).abs() < 1.0);
        assert!((set.modes.last().unwrap().freq - p.freq_hi).abs() < 2.0);
        assert!(set.modes.iter().all(|m| m.inv_rms > 0.0));
    }
}
