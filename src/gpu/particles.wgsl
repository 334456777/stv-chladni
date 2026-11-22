// 沙粒更新：每个沙粒沿 -F·∇F/|∇F|² 方向（牛顿步）滑向最近的节线，
// 同时在振幅大的地方（波腹）被随机弹跳，形成克拉尼纹路。
// 每个粒子：xy = 位置，z = 明暗随机量，w = 尺寸随机量。

struct Sim {
    shape: u32,
    mode_count: u32,
    count: u32,
    seed: u32,
    res: u32,
    steps: u32,
    _p0: u32,
    _p1: u32,
    drive: f32,
    pull: f32,
    bounce: f32,
    jitter: f32,
    kick: f32,
    max_step: f32,
    anim_time: f32,
    anim_fall: f32,
};

@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read_write> particles: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> field: array<f32>;
// 每粒沙开始下落的时间（撒沙动画）
@group(0) @binding(3) var<storage, read> spawn: array<f32>;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand01(s: ptr<function, u32>) -> f32 {
    *s = pcg(*s);
    return f32(*s >> 8u) / 16777216.0;
}

fn gauss2(s: ptr<function, u32>) -> vec2<f32> {
    let u1 = max(rand01(s), 1e-7);
    let u2 = rand01(s);
    let r = sqrt(-2.0 * log(u1));
    let a = 6.28318530718 * u2;
    return r * vec2<f32>(cos(a), sin(a));
}

fn texel(ix: i32, iy: i32) -> f32 {
    let r = i32(sim.res);
    let x = clamp(ix, 0, r - 1);
    let y = clamp(iy, 0, r - 1);
    return field[u32(y * r + x)];
}

fn sample_field(p: vec2<f32>) -> f32 {
    let r = f32(sim.res);
    let t = (p * 0.5 + 0.5) * (r - 1.0);
    let i = floor(t);
    let fr = t - i;
    let ix = i32(i.x);
    let iy = i32(i.y);
    let a = texel(ix, iy);
    let b = texel(ix + 1, iy);
    let c = texel(ix, iy + 1);
    let d = texel(ix + 1, iy + 1);
    return mix(mix(a, b, fr.x), mix(c, d, fr.x), fr.y);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= sim.count) {
        return;
    }
    // 撒沙动画中还没落到板上的沙粒不参与模拟
    if (sim.anim_time < spawn[idx] + sim.anim_fall) {
        return;
    }
    let q = particles[idx];
    var p = q.xy;
    var st = pcg(idx * 2654435769u + pcg(sim.seed));
    let h = 2.0 / f32(sim.res - 1u);
    let circle = sim.shape == 1u;

    for (var s = 0u; s < sim.steps; s++) {
        let f = sample_field(p);
        let gx = sample_field(p + vec2<f32>(h, 0.0)) - sample_field(p - vec2<f32>(h, 0.0));
        let gy = sample_field(p + vec2<f32>(0.0, h)) - sample_field(p - vec2<f32>(0.0, h));
        let g = vec2<f32>(gx, gy) / (2.0 * h);
        let g2 = dot(g, g);

        var mv = vec2<f32>(0.0, 0.0);
        if (g2 > 1e-8) {
            mv = -(f / g2) * g * sim.pull;
            let l = length(mv);
            if (l > sim.max_step) {
                mv *= sim.max_step / l;
            }
        }
        // 波腹弹跳 + 基础抖动 + 瞬态冲击
        let amp = max(abs(f) - 0.35, 0.0);
        let sigma = sim.bounce * min(amp, 2.5) + sim.jitter + sim.kick * 0.006;
        p += sim.drive * (mv + gauss2(&st) * sigma);

        if (circle) {
            let r = length(p);
            if (r > 1.0) {
                p *= max(2.0 - r, 0.0) / r;
            }
        } else {
            if (p.x > 1.0) { p.x = 2.0 - p.x; }
            if (p.x < -1.0) { p.x = -2.0 - p.x; }
            if (p.y > 1.0) { p.y = 2.0 - p.y; }
            if (p.y < -1.0) { p.y = -2.0 - p.y; }
            p = clamp(p, vec2<f32>(-1.0), vec2<f32>(1.0));
        }
    }
    particles[idx] = vec4<f32>(p, q.z, q.w);
}
