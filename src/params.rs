//! 全部可调参数。GUI、CLI、导出共用同一份结构，可序列化为 JSON。

use serde::{Deserialize, Serialize};

/// 振动板形状。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum PlateShape {
    /// 方形自由边板（经典克拉尼图形）
    Square,
    /// 圆形板（贝塞尔模态）
    Circle,
}

impl PlateShape {
    pub fn label(self) -> &'static str {
        match self {
            PlateShape::Square => "方形板",
            PlateShape::Circle => "圆形板",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Params {
    // ---------- 振动板 / 模态 ----------
    pub shape: PlateShape,
    /// 只使用中心对称（四重对称）的模态，图案更规整。
    pub symmetric: bool,
    /// 最高模态阶数（方形：m,n 上限；圆形：角向阶数 n 上限）。
    pub max_order: u32,
    /// 模态映射的最低 / 最高频率（Hz）。
    pub freq_lo: f32,
    pub freq_hi: f32,
    /// 每个模态的共振带宽（倍频程）。
    pub bandwidth: f32,
    /// 高频补偿（dB/倍频程），抵消人声高频衰减，让高阶模态也有机会出现。
    pub tilt: f32,
    /// 模态权重的锐化指数，越大越偏向单一主导模态。
    pub sharpness: f32,
    /// 同时参与叠加的模态数量上限。
    pub top_modes: u32,
    /// 频谱能量的平滑时间（秒），越大图案停留越久。
    pub smoothing: f32,
    /// 模态权重的起音 / 释放时间（秒）。
    pub attack: f32,
    pub release: f32,

    // ---------- 沙粒 ----------
    pub particles: u32,
    /// 向节线聚拢的力度（每步牛顿步长比例）。
    pub pull: f32,
    /// 波腹处的弹跳强度。
    pub bounce: f32,
    /// 基础抖动，决定纹路线条粗细。
    pub jitter: f32,
    /// 瞬态（辅音、鼓点）触发的散射强度。
    pub onset_kick: f32,
    /// 模拟速度倍率。
    pub speed: f32,
    /// 静音门限（dBFS），低于此值沙粒静止。
    pub gate_db: f32,
    /// 振动强度增益。
    pub sensitivity: f32,
    /// 撒沙动画时长（秒）：旧沙淡出后，新沙从振动板正上方落到板面。
    pub sprinkle_secs: f32,

    // ---------- 外观 ----------
    /// 颗粒半径（以板半宽为 1 的单位）。
    pub grain_size: f32,
    /// 板面占画面短边的比例。
    pub plate_scale: f32,
    /// 沙粒颜色（sRGB 0..1）。
    pub sand_color: [f32; 3],
    /// 背景颜色（sRGB 0..1）。
    pub bg_color: [f32; 3],
    /// 颗粒明暗随机变化幅度。
    pub grain_variation: f32,
    /// 随机种子（决定初始撒沙位置）。
    pub seed: u32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            shape: PlateShape::Square,
            symmetric: true,
            max_order: 10,
            freq_lo: 90.0,
            freq_hi: 3200.0,
            bandwidth: 0.25,
            tilt: 3.0,
            sharpness: 3.0,
            top_modes: 2,
            smoothing: 0.3,
            attack: 0.15,
            release: 0.5,

            particles: 50_000,
            pull: 0.10,
            bounce: 0.004,
            jitter: 0.003,
            onset_kick: 0.3,
            speed: 1.0,
            gate_db: -50.0,
            sensitivity: 1.0,
            sprinkle_secs: 0.5,

            grain_size: 0.0065,
            plate_scale: 0.78,
            sand_color: hex(0xECE5D5),
            bg_color: hex(0x0E0E0E),
            grain_variation: 0.25,
            seed: 7,
        }
    }
}

/// 0xRRGGBB → sRGB 浮点。
pub const fn hex(c: u32) -> [f32; 3] {
    [
        ((c >> 16) & 0xff) as f32 / 255.0,
        ((c >> 8) & 0xff) as f32 / 255.0,
        (c & 0xff) as f32 / 255.0,
    ]
}

/// 内置风格预设。
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Preset {
    /// 白沙 · 方形板（默认）
    Sand,
    /// 金沙 · 圆形板
    Gold,
    /// 冰蓝 · 方形板
    Ice,
    /// 青柠 · 圆形板
    Mint,
}

impl Preset {
    pub const ALL: [Preset; 4] = [Preset::Sand, Preset::Gold, Preset::Ice, Preset::Mint];

    pub fn label(self) -> &'static str {
        match self {
            Preset::Sand => "白沙·方",
            Preset::Gold => "金沙·圆",
            Preset::Ice => "冰蓝·方",
            Preset::Mint => "青柠·圆",
        }
    }

    /// 把预设的形状和配色套用到现有参数上（保留其他调节）。
    pub fn apply(self, p: &mut Params) {
        match self {
            Preset::Sand => {
                p.shape = PlateShape::Square;
                p.sand_color = hex(0xECE5D5);
                p.bg_color = hex(0x0E0E0E);
            }
            Preset::Gold => {
                p.shape = PlateShape::Circle;
                p.sand_color = hex(0xE0A75E);
                p.bg_color = hex(0x0D0C0B);
            }
            Preset::Ice => {
                p.shape = PlateShape::Square;
                p.sand_color = hex(0xB4CCE4);
                p.bg_color = hex(0x0B0D10);
            }
            Preset::Mint => {
                p.shape = PlateShape::Circle;
                p.sand_color = hex(0xC8E27A);
                p.bg_color = hex(0x0C0E0B);
            }
        }
    }
}

impl Params {
    /// 影响模态表的参数；它们改变时需要重建模态表与分析器。
    pub fn mode_key(&self) -> (PlateShape, bool, u32, [u32; 4]) {
        (
            self.shape,
            self.symmetric,
            self.max_order,
            [
                self.freq_lo.to_bits(),
                self.freq_hi.to_bits(),
                self.bandwidth.to_bits(),
                self.tilt.to_bits(),
            ],
        )
    }

    /// 把数值夹到安全范围内（防止手写 JSON 写出离谱值）。
    pub fn sanitize(&mut self) {
        self.max_order = self.max_order.clamp(2, 16);
        self.freq_lo = self.freq_lo.clamp(20.0, 8000.0);
        self.freq_hi = self.freq_hi.clamp(self.freq_lo * 2.0, 16000.0);
        self.bandwidth = self.bandwidth.clamp(0.02, 2.0);
        self.tilt = self.tilt.clamp(-6.0, 12.0);
        self.sharpness = self.sharpness.clamp(0.5, 8.0);
        self.top_modes = self.top_modes.clamp(1, 12);
        self.smoothing = self.smoothing.clamp(0.0, 3.0);
        self.attack = self.attack.clamp(0.0, 2.0);
        self.release = self.release.clamp(0.0, 5.0);
        self.particles = self.particles.clamp(1_000, 2_000_000);
        self.pull = self.pull.clamp(0.0, 0.5);
        self.bounce = self.bounce.clamp(0.0, 0.1);
        self.jitter = self.jitter.clamp(0.0, 0.05);
        self.onset_kick = self.onset_kick.clamp(0.0, 10.0);
        self.speed = self.speed.clamp(0.05, 8.0);
        self.gate_db = self.gate_db.clamp(-100.0, 0.0);
        self.sensitivity = self.sensitivity.clamp(0.0, 4.0);
        self.sprinkle_secs = self.sprinkle_secs.clamp(0.1, 5.0);
        self.grain_size = self.grain_size.clamp(0.0005, 0.05);
        self.plate_scale = self.plate_scale.clamp(0.1, 1.5);
        self.grain_variation = self.grain_variation.clamp(0.0, 1.0);
    }
}
