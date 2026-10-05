use bytemuck::Pod;
use std::marker::PhantomData;
use wgpu::util::DeviceExt;

pub use glam::{Mat4, Vec3};

mod camera;
pub use camera::{Camera3d, Projection3d};
mod terrain;
pub use terrain::{SphereRenderer, SphereTerrainSettings, TerrainColorRange};

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
