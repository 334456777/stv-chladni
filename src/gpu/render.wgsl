// 沙粒渲染：每个粒子一个实例化小方块，片元里画带软边和轻微高光的圆点。
// 输出为预乘 alpha，目标纹理直接存储 sRGB 编码值。
//
// 撒沙动画期间实例数为 old_count + count：
//   前 old_count 个是旧沙，原位很快淡出消失；
//   后 count 个是新沙，旧沙消失后从振动板正上方（垂直于板面、靠近镜头处）落到板面。
//   镜头位于板面中心正上方，按透视投影：离板面越高的沙粒离镜头越近，
//   看起来越大、越虚，位置也越偏离中心；落到板面时恢复正常大小和位置。

struct Render {
    sand: vec4<f32>,
    // 板半宽在 NDC 中的尺寸
    scale: vec2<f32>,
    // 颗粒半径在 NDC 中的尺寸
    grain: vec2<f32>,
    variation: f32,
    anim_time: f32,
    count: u32,
    old_count: u32,
    // 新沙下落时长
    anim_fall: f32,
    // 旧沙淡出时长
    out_fade: f32,
    _p0: f32,
    _p1: f32,
};

@group(0) @binding(0) var<uniform> rd: Render;
@group(0) @binding(1) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> old_particles: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> spawn: array<f32>;

// 镜头离板面的高度（以板半宽为单位）
const CAMERA_H: f32 = 3.0;
// 新沙起始高度范围
const DROP_MIN: f32 = 1.2;
const DROP_MAX: f32 = 2.2;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) shade: f32,
    @location(2) alpha: f32,
    // 0 = 清晰，1 = 最虚（离镜头近）
    @location(3) blur: f32,
};

fn hash01(v: u32) -> f32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return f32((word >> 22u) ^ word) / 4294967296.0;
}

fn hidden() -> VOut {
    var out: VOut;
    out.pos = vec4<f32>(-10.0, -10.0, 0.0, 1.0);
    return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vi];
    var q: vec4<f32>;
    var center: vec2<f32>;
    var size: f32;
    var alpha = 1.0;
    var blur = 0.0;

    if (ii < rd.old_count) {
        // 旧沙：原位很快淡出
        q = old_particles[ii];
        center = q.xy;
        size = 0.75 + 0.5 * q.w;
        let v = rd.anim_time / max(rd.out_fade, 1e-4);
        if (v >= 1.0) {
            return hidden();
        }
        alpha = 1.0 - v;
    } else {
        // 新沙：从板面上方高度 z0 加速落到板面（z = 0）
        let i = ii - rd.old_count;
        q = particles[i];
        center = q.xy;
        size = 0.75 + 0.5 * q.w;
        let u = (rd.anim_time - spawn[i]) / max(rd.anim_fall, 1e-4);
        if (u < 0.0) {
            return hidden();
        }
        if (u < 1.0) {
            let z0 = mix(DROP_MIN, DROP_MAX, hash01(i * 2654435761u + 7u));
            let z = z0 * (1.0 - u * u);
            // 透视放大倍数
            let k = CAMERA_H / (CAMERA_H - z);
            center = q.xy * k;
            size *= k;
            let near = z / DROP_MAX;
            alpha = 1.0 - 0.65 * near;
            blur = near;
        }
    }

    var out: VOut;
    out.pos = vec4<f32>(center * rd.scale + c * rd.grain * size, 0.0, 1.0);
    out.uv = c;
    out.shade = 1.0 - rd.variation * q.z;
    out.alpha = alpha;
    out.blur = blur;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    if (d > 1.0) {
        discard;
    }
    // 离镜头近的沙粒边缘更柔和，模拟失焦
    let a = (1.0 - smoothstep(0.72 - 0.6 * in.blur, 1.0, d)) * in.alpha;
    let light = mix(1.1, 0.85, d * d);
    let col = clamp(rd.sand.rgb * in.shade * light, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(col * a, a);
}
