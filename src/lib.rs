use bytemuck::Pod;
use std::marker::PhantomData;
use wgpu::util::DeviceExt;

mod camera;
pub use camera::{Camera3d, Projection3d};

pub struct VertexShader;
pub struct FragmentShader;
pub struct ComputeShader;

pub struct Shader<Stage> {
    source: &'static str,
    entry_point: &'static str,
    stage: PhantomData<Stage>,
}

impl<Stage> Shader<Stage> {
    pub(crate) const fn from_generated(source: &'static str, entry_point: &'static str) -> Self {
        Self {
            source,
            entry_point,
            stage: PhantomData,
        }
    }

    pub fn source(&self) -> &'static str {
        self.source
    }

    pub fn entry_point(&self) -> &'static str {
        self.entry_point
    }
}

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

pub trait Vertex: Pod + 'static {
    const ATTRIBUTES: &'static [wgpu::VertexAttribute];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: Self::ATTRIBUTES,
        }
    }
}

pub struct TriangleRenderer<V: Vertex> {
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    vertex_count: u32,
    _vertex: PhantomData<V>,
}

impl<V: Vertex> TriangleRenderer<V> {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        vertex_shader: Shader<VertexShader>,
        fragment_shader: Shader<FragmentShader>,
        vertices: &[V],
    ) -> Self {
        assert!(!vertices.is_empty(), "triangle renderer needs vertex data");
        assert!(
            vertices.len() % 3 == 0,
            "triangle list vertex count must be divisible by 3"
        );
        let vertex_count = vertices.len() as u32;
        let shader_source = format!("{}\n{}", vertex_shader.source(), fragment_shader.source());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("triangle pipeline layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let buffers = [Some(V::layout())];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("triangle pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(vertex_shader.entry_point()),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fragment_shader.entry_point()),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("triangle vertices"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            pipeline,
            vertices,
            vertex_count,
            _vertex: PhantomData,
        }
    }

    pub fn draw<'pass>(&'pass self, pass: &mut wgpu::RenderPass<'pass>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, bytemuck::Zeroable)]
pub struct PositionColor {
    pub position: [f32; 2],
    pub color: [f32; 3],
}

impl Vertex for PositionColor {
    const ATTRIBUTES: &'static [wgpu::VertexAttribute] =
        &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3];
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, bytemuck::Zeroable)]
pub struct Position3Color {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex for Position3Color {
    const ATTRIBUTES: &'static [wgpu::VertexAttribute] =
        &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
}

/// A ray-marched SDF sphere rendered with a full-screen triangle.
pub struct SphereRenderer {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    settings_buffer: wgpu::Buffer,
    settings: SdfSphereSettings,
}

/// Editable controls for the ray-marched sphere and its simple lighting.
#[derive(Clone, Copy, Debug)]
pub struct SdfSphereSettings {
    pub center: [f32; 3],
    pub radius: f32,
    pub max_steps: u32,
    pub hit_epsilon: f32,
    pub max_distance: f32,
    pub color: [f32; 3],
    pub ambient: f32,
    pub light_direction: [f32; 3],
    pub diffuse: f32,
    pub accent_color: [f32; 3],
    pub pattern_strength: f32,
    pub pattern_frequency: [f32; 2],
    pub pattern_threshold: f32,
    pub pattern_softness: f32,
    pub spin_speed_radians_per_second: f32,
}

impl Default for SdfSphereSettings {
    fn default() -> Self {
        Self {
            center: [0.0; 3],
            radius: 1.0,
            max_steps: 128,
            hit_epsilon: 0.001,
            max_distance: 100.0,
            color: [0.18, 0.48, 1.0],
            ambient: 0.22,
            light_direction: [0.4, 0.7, 1.0],
            diffuse: 0.78,
            accent_color: [0.25, 0.85, 0.65],
            pattern_strength: 0.75,
            pattern_frequency: [4.0, 3.0],
            pattern_threshold: 0.25,
            pattern_softness: 0.1,
            spin_speed_radians_per_second: 0.7,
        }
    }
}

impl SdfSphereSettings {
    fn gpu_data(self, rotation_radians: f32) -> [[f32; 4]; 6] {
        [
            [self.center[0], self.center[1], self.center[2], self.radius],
            [
                self.max_steps as f32,
                self.hit_epsilon,
                self.max_distance,
                rotation_radians,
            ],
            [self.color[0], self.color[1], self.color[2], self.ambient],
            [
                self.light_direction[0],
                self.light_direction[1],
                self.light_direction[2],
                self.diffuse,
            ],
            [
                self.accent_color[0],
                self.accent_color[1],
                self.accent_color[2],
                self.pattern_strength,
            ],
            [
                self.pattern_frequency[0],
                self.pattern_frequency[1],
                self.pattern_threshold,
                self.pattern_softness,
            ],
        ]
    }
}

impl SphereRenderer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        vertex_shader: Shader<VertexShader>,
        fragment_shader: Shader<FragmentShader>,
        settings: SdfSphereSettings,
    ) -> Self {
        let shader_source = format!("{}\n{}", vertex_shader.source(), fragment_shader.source());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sphere shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera inverse view projection matrix"),
            size: std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let settings_data = settings.gpu_data(0.0);
        let settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SDF sphere settings"),
            contents: bytemuck::bytes_of(&settings_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: settings_buffer.as_entire_binding(),
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sphere pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let buffers = [];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sphere pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(vertex_shader.entry_point()),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fragment_shader.entry_point()),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            camera_buffer,
            camera_bind_group,
            settings_buffer,
            settings,
        }
    }

    pub fn set_camera(&self, queue: &wgpu::Queue, camera: Camera3d, projection: Projection3d) {
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&camera.inverse_view_projection(projection)),
        );
    }

    pub fn set_settings(&mut self, queue: &wgpu::Queue, settings: SdfSphereSettings) {
        self.settings = settings;
        queue.write_buffer(
            &self.settings_buffer,
            0,
            bytemuck::bytes_of(&settings.gpu_data(0.0)),
        );
    }

    pub fn set_rotation(&self, queue: &wgpu::Queue, rotation_radians: f32) {
        queue.write_buffer(
            &self.settings_buffer,
            0,
            bytemuck::bytes_of(&self.settings.gpu_data(rotation_radians)),
        );
    }

    pub fn draw<'pass>(&'pass self, pass: &mut wgpu::RenderPass<'pass>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
