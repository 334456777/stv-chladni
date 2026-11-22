// 振动场计算：在 res×res 网格上叠加所有激活模态，得到板面位移场 F(x,y)。
// 坐标 p ∈ [-1,1]²。

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

struct Mode {
    a: f32,
    b: f32,
    c: f32,
    w: f32,
};

@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> modes: array<Mode>;
@group(0) @binding(2) var<storage, read_write> field: array<f32>;
@group(0) @binding(3) var<storage, read> bessel: array<f32>;

const PI: f32 = 3.14159265358979;
// 必须与 modes.rs 中的 BESSEL_LUT_N / BESSEL_XMAX 一致
const LUT_N: u32 = 2048u;
const LUT_XMAX: f32 = 64.0;

fn bessel_j(n: u32, x: f32) -> f32 {
    let t = clamp(x / LUT_XMAX, 0.0, 1.0) * f32(LUT_N - 1u);
    let i = min(u32(t), LUT_N - 2u);
    let fr = t - f32(i);
    let base = n * LUT_N;
    return mix(bessel[base + i], bessel[base + i + 1u], fr);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = sim.res;
    if (gid.x >= res || gid.y >= res) {
        return;
    }
    let p = vec2<f32>(gid.xy) / f32(res - 1u) * 2.0 - 1.0;
    var f: f32 = 0.0;
    if (sim.shape == 0u) {
        let u = p * 0.5 + 0.5;
        for (var k = 0u; k < sim.mode_count; k++) {
            let md = modes[k];
            f += md.w * (cos(md.a * PI * u.x) * cos(md.b * PI * u.y)
                + md.c * cos(md.b * PI * u.x) * cos(md.a * PI * u.y));
        }
    } else {
        let r = length(p);
        let th = atan2(p.y, p.x);
        for (var k = 0u; k < sim.mode_count; k++) {
            let md = modes[k];
            f += md.w * bessel_j(u32(md.a), md.b * r) * cos(md.a * th + md.c);
        }
    }
    field[gid.y * res + gid.x] = f;
}
