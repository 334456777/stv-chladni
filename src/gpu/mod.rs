//! GPU 沙粒模拟器：模态场（compute）→ 沙粒更新（compute）→ 颗粒渲染（render）。
//!
//! 同一套代码既用于界面实时预览（共享 eframe 的 wgpu 设备），也用于无头导出（独立设备）。

use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use eframe::egui_wgpu::wgpu;
use wgpu::util::DeviceExt;

use crate::analysis::AnalysisFrame;
use crate::modes::{self, GpuMode, MAX_GPU_MODES};
use crate::params::{Params, PlateShape};

/// 振动场网格分辨率。
pub const FIELD_RES: u32 = 512;
/// 输出纹理格式（存储 sRGB 编码值，可直接送入 ffmpeg 的 rgba）。
pub const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// 每秒模拟步数（乘以 `Params::speed`）。
pub const STEPS_PER_SECOND: f32 = 120.0;
/// 单步最大位移（板半宽单位）。
const MAX_STEP: f32 = 0.02;

// ---- 撒沙动画 ----
// 旧沙先很快消失，随后新沙从振动板正上方（垂直于板面、靠近镜头处）落到板面上。
/// 撒沙各阶段时长（秒），由总时长 `Params::sprinkle_secs` 按比例分配。
#[derive(Clone, Copy, Debug)]
struct SprinkleTiming {
    /// 旧沙淡出消失的时长
    out_fade: f32,
    /// 新沙最早开始下落的时间（旧沙消失之后）
    new_offset: f32,
    /// 每粒新沙开始下落的随机先后
    new_jitter: f32,
    /// 新沙从板面上方落到板面的时长
    new_fall: f32,
    /// 整段动画时长
    total: f32,
}

impl SprinkleTiming {
    /// `has_old`：有旧沙要先消失（重新撒沙）；没有时（程序刚启动）只播放新沙落下。
    fn new(total: f32, has_old: bool) -> Self {
        let d = total.max(0.05);
        let out_fade = if has_old { 0.12 * d } else { 0.0 };
        let new_offset = out_fade;
        let (new_jitter, new_fall) = (0.18 * d, 0.66 * d);
        Self {
            out_fade,
            new_offset,
            new_jitter,
            new_fall,
            total: new_offset + new_jitter + new_fall + 0.04 * d,
        }
    }
}
/// 动画关闭时的时间值（所有沙粒都视为已落地）。
const ANIM_OFF: f32 = 1.0e9;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SimUniforms {
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
    /// 下落时长（未落地的沙粒不参与模拟）
    anim_fall: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderUniforms {
    sand: [f32; 4],
    scale: [f32; 2],
    grain: [f32; 2],
    variation: f32,
    anim_time: f32,
    count: u32,
    old_count: u32,
    /// 新沙下落时长
    anim_fall: f32,
    /// 旧沙淡出时长
    out_fade: f32,
    _p: [f32; 2],
}

pub struct Simulator {
    width: u32,
    height: u32,
    particle_count: u32,
    mode_count: u32,
    shape: PlateShape,
    frame: u32,
    resets: u32,

    // 撒沙动画状态
    anim_time: f32,
    old_count: u32,
    timing: SprinkleTiming,

    sim_uniform: wgpu::Buffer,
    render_uniform: wgpu::Buffer,
    modes_buf: wgpu::Buffer,
    field_buf: wgpu::Buffer,
    _bessel_buf: wgpu::Buffer,
    particles: wgpu::Buffer,
    /// 上一批沙粒（撒沙动画开头在原位淡出）
    old_particles: wgpu::Buffer,
    /// 每粒新沙开始下落的时间
    spawn: wgpu::Buffer,

    field_pipeline: wgpu::ComputePipeline,
    particle_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    particle_bgl: wgpu::BindGroupLayout,
    render_bgl: wgpu::BindGroupLayout,
    field_bg: wgpu::BindGroup,
    particle_bg: wgpu::BindGroup,
    render_bg: wgpu::BindGroup,

    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    readback: Option<wgpu::Buffer>,
}

fn uniform_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, vis: wgpu::ShaderStages, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn shader(device: &wgpu::Device, label: &str, src: &'static str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
    })
}

impl Simulator {
    /// `animate` 为真时，沙粒以撒沙动画落到板上（界面用）；导出时为假，直接均匀铺好。
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        p: &Params,
        animate: bool,
    ) -> Self {
        let cs = wgpu::ShaderStages::COMPUTE;

        let sim_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stv.sim_uniform"),
            size: std::mem::size_of::<SimUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let render_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stv.render_uniform"),
            size: std::mem::size_of::<RenderUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let modes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stv.modes"),
            size: (std::mem::size_of::<GpuMode>() * MAX_GPU_MODES) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let field_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stv.field"),
            size: (FIELD_RES * FIELD_RES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bessel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stv.bessel_lut"),
            contents: bytemuck::cast_slice(modes::bessel_lut()),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ---- 场计算 ----
        let field_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stv.field_bgl"),
            entries: &[
                uniform_entry(0, cs),
                storage_entry(1, cs, true),
                storage_entry(2, cs, false),
                storage_entry(3, cs, true),
            ],
        });
        let field_module = shader(device, "stv.field", include_str!("field.wgsl"));
        let field_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stv.field_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&field_bgl],
                push_constant_ranges: &[],
            })),
            module: &field_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let field_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stv.field_bg"),
            layout: &field_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sim_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: modes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: field_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bessel_buf.as_entire_binding() },
            ],
        });

        // ---- 沙粒更新 ----
        let particle_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stv.particle_bgl"),
            entries: &[
                uniform_entry(0, cs),
                storage_entry(1, cs, false),
                storage_entry(2, cs, true),
                storage_entry(3, cs, true),
            ],
        });
        let particle_module = shader(device, "stv.particles", include_str!("particles.wgsl"));
        let particle_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stv.particle_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&particle_bgl],
                push_constant_ranges: &[],
            })),
            module: &particle_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ---- 渲染 ----
        let vf = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let vs = wgpu::ShaderStages::VERTEX;
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stv.render_bgl"),
            entries: &[
                uniform_entry(0, vf),
                storage_entry(1, vs, true),
                storage_entry(2, vs, true),
                storage_entry(3, vs, true),
            ],
        });
        let render_module = shader(device, "stv.render", include_str!("render.wgsl"));
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stv.render_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&render_bgl],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &render_module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: OUTPUT_FORMAT,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let timing = SprinkleTiming::new(p.sprinkle_secs, false);
        let (particles, spawn) = create_particles(device, p, 0, animate.then_some(timing));
        let old_particles = empty_storage(device, "stv.old_particles");
        let particle_bg = make_particle_bg(device, &particle_bgl, &sim_uniform, &particles, &field_buf, &spawn);
        let render_bg = make_render_bg(device, &render_bgl, &render_uniform, &particles, &old_particles, &spawn);
        let (target, target_view) = create_target(device, width, height);

        let _ = queue;
        Self {
            width,
            height,
            particle_count: p.particles,
            mode_count: 0,
            shape: p.shape,
            frame: 0,
            resets: 0,
            anim_time: if animate { 0.0 } else { ANIM_OFF },
            old_count: 0,
            timing,
            sim_uniform,
            render_uniform,
            modes_buf,
            field_buf,
            _bessel_buf: bessel_buf,
            particles,
            old_particles,
            spawn,
            field_pipeline,
            particle_pipeline,
            render_pipeline,
            particle_bgl,
            render_bgl,
            field_bg,
            particle_bg,
            render_bg,
            target,
            target_view,
            readback: None,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }

    pub fn target(&self) -> &wgpu::Texture {
        &self.target
    }

    /// 重新撒沙（粒子数变化时会重建缓冲）。
    ///
    /// `animate` 为真时播放撒沙动画：旧沙先很快消失，随后新沙从振动板正上方
    /// 落到整块板面上。总时长为 `p.sprinkle_secs`。
    pub fn reset_particles(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, p: &Params, animate: bool) {
        self.resets = self.resets.wrapping_add(1);
        self.timing = SprinkleTiming::new(p.sprinkle_secs, true);
        let (particles, spawn) = create_particles(device, p, self.resets, animate.then_some(self.timing));

        if animate {
            // 把当前沙粒拷贝一份，用来画开头淡出的旧沙
            let size = self.particle_count as u64 * 16;
            let old = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stv.old_particles"),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stv.reset") });
            enc.copy_buffer_to_buffer(&self.particles, 0, &old, 0, size);
            queue.submit(Some(enc.finish()));
            self.old_particles = old;
            self.old_count = self.particle_count;
            self.anim_time = 0.0;
        } else {
            self.old_count = 0;
            self.anim_time = ANIM_OFF;
        }

        self.particles = particles;
        self.spawn = spawn;
        self.particle_count = p.particles;
        self.shape = p.shape;
        self.particle_bg = make_particle_bg(
            device,
            &self.particle_bgl,
            &self.sim_uniform,
            &self.particles,
            &self.field_buf,
            &self.spawn,
        );
        self.render_bg = make_render_bg(
            device,
            &self.render_bgl,
            &self.render_uniform,
            &self.particles,
            &self.old_particles,
            &self.spawn,
        );
    }

    /// 推进撒沙动画的时钟。
    pub fn advance_anim(&mut self, dt: f32) {
        if self.anim_time < ANIM_OFF {
            self.anim_time += dt;
            if self.anim_time > self.timing.total {
                self.anim_time = ANIM_OFF;
                self.old_count = 0;
            }
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if (width, height) == (self.width, self.height) {
            return;
        }
        let (t, v) = create_target(device, width, height);
        self.target = t;
        self.target_view = v;
        self.width = width;
        self.height = height;
        self.readback = None;
    }

    /// 上传当前激活的模态。
    pub fn set_modes(&mut self, queue: &wgpu::Queue, shape: PlateShape, modes: &[GpuMode]) {
        let n = modes.len().min(MAX_GPU_MODES);
        if n > 0 {
            queue.write_buffer(&self.modes_buf, 0, bytemuck::cast_slice(&modes[..n]));
        }
        self.mode_count = n as u32;
        self.shape = shape;
    }

    /// 记录一帧的模拟命令：场计算 + `steps` 步沙粒更新。
    pub fn encode_step(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &AnalysisFrame,
        p: &Params,
        steps: u32,
    ) {
        self.frame = self.frame.wrapping_add(1);
        let u = SimUniforms {
            shape: match self.shape {
                PlateShape::Square => 0,
                PlateShape::Circle => 1,
            },
            mode_count: self.mode_count,
            count: self.particle_count,
            seed: self.frame.wrapping_mul(0x9E37_79B9) ^ p.seed,
            res: FIELD_RES,
            steps,
            drive: frame.drive,
            pull: p.pull,
            bounce: p.bounce,
            jitter: p.jitter,
            kick: frame.kick,
            max_step: MAX_STEP,
            anim_time: self.anim_time,
            anim_fall: self.timing.new_fall,
            ..Default::default()
        };
        queue.write_buffer(&self.sim_uniform, 0, bytemuck::bytes_of(&u));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("stv.sim_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.field_pipeline);
        pass.set_bind_group(0, &self.field_bg, &[]);
        pass.dispatch_workgroups(FIELD_RES.div_ceil(8), FIELD_RES.div_ceil(8), 1);
        if steps > 0 && frame.drive > 0.0 {
            pass.set_pipeline(&self.particle_pipeline);
            pass.set_bind_group(0, &self.particle_bg, &[]);
            pass.dispatch_workgroups(self.particle_count.div_ceil(256), 1, 1);
        }
    }

    /// 记录渲染命令，把沙粒画到输出纹理上。
    pub fn encode_render(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, p: &Params) {
        let min_side = self.width.min(self.height) as f32;
        let half_px = p.plate_scale * min_side * 0.5;
        let scale = [half_px / (self.width as f32 * 0.5), half_px / (self.height as f32 * 0.5)];
        let animating = self.anim_time < ANIM_OFF;
        let old_count = if animating { self.old_count } else { 0 };
        let u = RenderUniforms {
            sand: [p.sand_color[0], p.sand_color[1], p.sand_color[2], 1.0],
            scale,
            grain: [scale[0] * p.grain_size, scale[1] * p.grain_size],
            variation: p.grain_variation,
            anim_time: self.anim_time,
            count: self.particle_count,
            old_count,
            anim_fall: self.timing.new_fall,
            out_fade: self.timing.out_fade,
            _p: [0.0; 2],
        };
        queue.write_buffer(&self.render_uniform, 0, bytemuck::bytes_of(&u));

        let bg = p.bg_color;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stv.render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: bg[0] as f64,
                        g: bg[1] as f64,
                        b: bg[2] as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_bg, &[]);
        pass.draw(0..6, 0..self.particle_count + old_count);
    }

    /// 输出纹理的 sRGB 视图（供 egui 正确显示）。
    pub fn srgb_view(&self) -> wgpu::TextureView {
        self.target.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            ..Default::default()
        })
    }

    /// 读回输出纹理为紧密排列的 RGBA8 像素（阻塞等待 GPU）。
    pub fn read_pixels(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, out: &mut Vec<u8>) -> Result<()> {
        let (unpadded, padded) = row_pitch(self.width);
        let size = (padded * self.height) as u64;
        let buf = self.readback.get_or_insert_with(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stv.readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stv.readback") });
        copy_texture(&mut enc, &self.target, buf, self.width, self.height, padded);
        queue.submit(Some(enc.finish()));

        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().context("GPU 读回通道断开")?.context("GPU 读回映射失败")?;
        {
            let data = slice.get_mapped_range();
            unpad_rows(&data, unpadded, padded, self.height, out);
        }
        buf.unmap();
        Ok(())
    }
}

/// 一行像素的实际字节数与对齐后的字节数。
fn row_pitch(width: u32) -> (u32, u32) {
    let unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    (unpadded, unpadded.div_ceil(align) * align)
}

fn copy_texture(
    enc: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    buf: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded: u32,
) {
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
}

fn unpad_rows(data: &[u8], unpadded: u32, padded: u32, height: u32, out: &mut Vec<u8>) {
    out.clear();
    out.reserve((unpadded * height) as usize);
    for row in data.chunks_exact(padded as usize) {
        out.extend_from_slice(&row[..unpadded as usize]);
    }
}

const SLOT_PENDING: u8 = 0;
const SLOT_READY: u8 = 1;
const SLOT_FAILED: u8 = 2;

struct GrabSlot {
    buf: wgpu::Buffer,
    state: Arc<AtomicU8>,
    repeat: u32,
    busy: bool,
}

/// 不阻塞界面的帧读回：输出纹理拷贝进轮转的暂存缓冲，映射完成后再取走（录制用）。
pub struct FrameGrabber {
    width: u32,
    height: u32,
    slots: Vec<GrabSlot>,
    pending: VecDeque<usize>,
}

impl FrameGrabber {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, slots: usize) -> Self {
        let (_, padded) = row_pitch(width);
        let slots = (0..slots.max(2))
            .map(|_| GrabSlot {
                buf: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("stv.grab"),
                    size: (padded * height) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(SLOT_PENDING)),
                repeat: 0,
                busy: false,
            })
            .collect();
        Self { width, height, slots, pending: VecDeque::new() }
    }

    /// 抓取当前输出纹理；该帧在视频里重复 `repeat` 次。已完成的帧追加到 `out`（像素, 重复次数）。
    pub fn grab(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sim: &Simulator,
        repeat: u32,
        out: &mut Vec<(Vec<u8>, u32)>,
    ) {
        if sim.size() != (self.width, self.height) {
            return;
        }
        let idx = match self.slots.iter().position(|s| !s.busy) {
            Some(i) => i,
            None => {
                // 暂存缓冲都在等 GPU：阻塞取走最旧的一帧
                self.collect(device, true, out);
                match self.slots.iter().position(|s| !s.busy) {
                    Some(i) => i,
                    None => return,
                }
            }
        };
        let (_, padded) = row_pitch(self.width);
        let slot = &mut self.slots[idx];
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stv.grab") });
        copy_texture(&mut enc, sim.target(), &slot.buf, self.width, self.height, padded);
        queue.submit(Some(enc.finish()));
        slot.state.store(SLOT_PENDING, Ordering::Release);
        let state = slot.state.clone();
        slot.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            state.store(if r.is_ok() { SLOT_READY } else { SLOT_FAILED }, Ordering::Release);
        });
        slot.repeat = repeat;
        slot.busy = true;
        self.pending.push_back(idx);
    }

    /// 取走已经映射完成的帧（按抓取顺序）。`block` 为真时先等待 GPU 完成全部工作。
    pub fn collect(&mut self, device: &wgpu::Device, block: bool, out: &mut Vec<(Vec<u8>, u32)>) {
        device.poll(if block { wgpu::Maintain::Wait } else { wgpu::Maintain::Poll });
        let (unpadded, padded) = row_pitch(self.width);
        while let Some(&i) = self.pending.front() {
            let slot = &mut self.slots[i];
            let st = slot.state.load(Ordering::Acquire);
            if st == SLOT_PENDING {
                break;
            }
            self.pending.pop_front();
            if st == SLOT_READY {
                let mut pixels = Vec::new();
                {
                    let data = slot.buf.slice(..).get_mapped_range();
                    unpad_rows(&data, unpadded, padded, self.height, &mut pixels);
                }
                slot.buf.unmap();
                out.push((pixels, slot.repeat));
            }
            slot.busy = false;
        }
    }
}

/// 根据当前帧时长计算本帧的模拟步数（带小数累积）。
pub fn steps_for(dt: f32, speed: f32, acc: &mut f32) -> u32 {
    *acc += dt.max(0.0) * STEPS_PER_SECOND * speed;
    let n = acc.floor().clamp(0.0, 64.0);
    *acc -= acc.floor();
    n as u32
}

/// 生成均匀分布的新沙粒，以及每粒开始下落的时间。
/// `timing` 为 `Some` 时播放撒沙动画，按其中的时间安排每粒新沙开始下落的时刻。
fn create_particles(
    device: &wgpu::Device,
    p: &Params,
    reset: u32,
    timing: Option<SprinkleTiming>,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let mut rng = XorShift::new((p.seed as u64 ^ 0x5eed_5a4d).wrapping_add((reset as u64) << 32));
    let n = p.particles as usize;
    let mut data = Vec::with_capacity(n * 4);
    let mut spawn = Vec::with_capacity(n);
    for _ in 0..n {
        let (x, y) = loop {
            let x = rng.next_f32() * 2.0 - 1.0;
            let y = rng.next_f32() * 2.0 - 1.0;
            if p.shape == PlateShape::Square || x * x + y * y < 1.0 {
                break (x, y);
            }
        };
        data.extend_from_slice(&[x, y, rng.next_f32(), rng.next_f32()]);
        // 动画关闭时把下落时间设在很久以前，视为已落地
        spawn.push(match timing {
            Some(t) => t.new_offset + t.new_jitter * rng.next_f32(),
            None => -ANIM_OFF,
        });
    }
    let particles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stv.particles"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });
    let spawn = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stv.spawn"),
        contents: bytemuck::cast_slice(&spawn),
        usage: wgpu::BufferUsages::STORAGE,
    });
    (particles, spawn)
}

fn empty_storage(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_particle_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sim: &wgpu::Buffer,
    particles: &wgpu::Buffer,
    field: &wgpu::Buffer,
    spawn: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stv.particle_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: sim.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: particles.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: field.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: spawn.as_entire_binding() },
        ],
    })
}

fn make_render_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    particles: &wgpu::Buffer,
    old: &wgpu::Buffer,
    spawn: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stv.render_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: particles.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: old.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: spawn.as_entire_binding() },
        ],
    })
}

fn create_target(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stv.output"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OUTPUT_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// 按偏好顺序逐个在适配器上创建设备，返回第一个成功的：独立显卡优先，同一块显卡上 DX12 / Metal 优先于 Vulkan
/// （DX12 优先的原因见 `gui::select_adapter`）。
///
/// 某块显卡或某个后端的驱动出错时换下一个。wgpu 把 DX12 驱动返回的大多数错误都报成 "Parent device is lost"，
/// 具体原因只写在错误日志里。全部失败时返回每个适配器各自的错误。
pub fn open_first_device(
    adapters: impl IntoIterator<Item = wgpu::Adapter>,
    descriptor: impl Fn(&wgpu::Adapter) -> wgpu::DeviceDescriptor<'static>,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue), String> {
    let mut adapters: Vec<_> = adapters.into_iter().collect();
    adapters.sort_by_key(|a| {
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
        std::cmp::Reverse((device, backend))
    });
    if adapters.is_empty() {
        return Err("没有可用的显卡适配器".into());
    }
    let mut failures = Vec::new();
    for adapter in adapters {
        let info = adapter.get_info();
        match pollster::block_on(adapter.request_device(&descriptor(&adapter), None)) {
            Ok((device, queue)) => return Ok((adapter, device, queue)),
            Err(e) => {
                log::warn!("{} · {:?} 无法创建 GPU 设备：{e}", info.name, info.backend);
                failures.push(format!("{} · {:?}：{e}", info.name, info.backend));
            }
        }
    }
    Err(format!("所有显卡适配器都无法创建 GPU 设备：\n{}", failures.join("\n")))
}

/// 无头 GPU 上下文（导出视频用）。
pub struct HeadlessGpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub is_nvidia: bool,
}

impl HeadlessGpu {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY);
        let (adapter, device, queue) = open_first_device(adapters, |adapter| wgpu::DeviceDescriptor {
            label: Some("stv.headless"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::Performance,
        })
        .map_err(anyhow::Error::msg)
        .context("创建 GPU 设备失败")?;
        let info = adapter.get_info();
        Ok(Self {
            device,
            queue,
            adapter_name: format!("{} ({:?})", info.name, info.backend),
            is_nvidia: info.vendor == 0x10DE,
        })
    }
}

/// 简单可复现的伪随机数。
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
