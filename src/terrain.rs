use crate::{Camera3d, Vertex, shaders};
use bytemuck::{Pod, Zeroable};
use std::time::Duration;
use wgpu::util::DeviceExt;

pub const TERRAIN_SAMPLE_COUNT: u32 = 100;
const COMPUTE_WORKGROUP_SIZE: u32 = 64;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TerrainVertex {
    position: [f32; 3],
    _position_padding: f32,
    normal: [f32; 3],
    displacement: f32,
}

impl crate::Vertex for TerrainVertex {
    const ATTRIBUTES: &'static [wgpu::VertexAttribute] = &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 16,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32,
            offset: 28,
            shader_location: 2,
        },
    ];
}

/// Geometry and motion settings for the procedural terrain sphere.
#[derive(Clone, Copy, Debug)]
pub struct SphereTerrainSettings {
    pub center: glam::Vec3,
    pub radius: f32,
    pub terrain_seed: u32,
    pub terrain_amplitude: f32,
    pub terrain_smoothing: f32,
    pub animation_speed: f32,
    pub latitude_segments: u32,
    pub longitude_segments: u32,
}

impl Default for SphereTerrainSettings {
    fn default() -> Self {
        Self {
            center: glam::Vec3::ZERO,
            radius: 1.0,
            terrain_seed: 42,
            terrain_amplitude: 0.25,
            terrain_smoothing: 0.01,
            animation_speed: 1.0,
            latitude_segments: 64,
            longitude_segments: 128,
        }
    }
}

/// Fragment color ramp keyed by distance from the undisplaced sphere.
#[derive(Clone, Copy, Debug)]
pub struct TerrainColorRange {
    pub minimum_displacement: f32,
    pub maximum_displacement: f32,
    pub low_color: glam::Vec3,
    pub high_color: glam::Vec3,
    pub ambient: f32,
    pub diffuse: f32,
    pub light_direction: glam::Vec3,
}

impl Default for TerrainColorRange {
    fn default() -> Self {
        Self {
            minimum_displacement: 0.0,
            maximum_displacement: 0.15,
            low_color: glam::Vec3::new(0.1, 0.1, 0.1),
            high_color: glam::Vec3::ONE,
            ambient: 0.22,
            diffuse: 0.78,
            light_direction: glam::Vec3::new(0.4, 0.7, 1.0),
        }
    }
}

impl TerrainColorRange {
    fn gpu_data(self) -> [[f32; 4]; 4] {
        let low = self.low_color.to_array();
        let high = self.high_color.to_array();
        let light = self.light_direction.to_array();
        [
            [
                self.minimum_displacement,
                self.maximum_displacement,
                self.ambient,
                self.diffuse,
            ],
            [low[0], low[1], low[2], 0.0],
            [high[0], high[1], high[2], 0.0],
            [light[0], light[1], light[2], 0.0],
        ]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AnimationUniform {
    seed: u32,
    elapsed_seconds: f32,
    speed: f32,
    _padding: f32,
}

/// GPU-driven sphere terrain. Compute passes animate sample points and rewrite
/// displaced vertices before the indexed render pass.
pub struct SphereRenderer {
    sample_pipeline: wgpu::ComputePipeline,
    vertex_compute_pipeline: wgpu::ComputePipeline,
    sample_bind_group: wgpu::BindGroup,
    vertex_compute_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    color_buffer: wgpu::Buffer,
    color_bind_group: wgpu::BindGroup,
    animation_buffer: wgpu::Buffer,
    vertices: wgpu::Buffer,
    vertex_count: u32,
    indices: wgpu::Buffer,
    index_count: u32,
    settings: SphereTerrainSettings,
}

impl SphereRenderer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        settings: SphereTerrainSettings,
        color_range: TerrainColorRange,
    ) -> Self {
        assert!(settings.latitude_segments >= 2);
        assert!(settings.longitude_segments >= 3);
        assert!(color_range.maximum_displacement > color_range.minimum_displacement);

        let (base_directions, indices) = create_sphere_topology(settings);
        let vertex_count = base_directions.len() as u32;
        let index_count = indices.len() as u32;
        let base_directions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("base sphere directions"),
            contents: bytemuck::cast_slice(&base_directions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("computed terrain vertices"),
            size: (vertex_count as u64) * std::mem::size_of::<TerrainVertex>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere triangle indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let terrain_samples = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("animated terrain samples"),
            size: (TERRAIN_SAMPLE_COUNT as u64) * std::mem::size_of::<[f32; 4]>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let animation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain animation parameters"),
            contents: bytemuck::bytes_of(&AnimationUniform {
                seed: settings.terrain_seed,
                elapsed_seconds: 0.0,
                speed: settings.animation_speed,
                _padding: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let terrain_parameters = terrain_gpu_data(settings, vertex_count);
        let terrain_parameters_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain compute parameters"),
                contents: bytemuck::bytes_of(&terrain_parameters),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let sample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain sample compute layout"),
            entries: &[
                storage_entry(0, false),
                uniform_entry(1, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let sample_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain sample compute bindings"),
            layout: &sample_layout,
            entries: &[
                buffer_entry(0, &terrain_samples),
                buffer_entry(1, &animation_buffer),
            ],
        });
        let sample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain sample compute pipeline layout"),
                bind_group_layouts: &[Some(&sample_layout)],
                immediate_size: 0,
            });
        let sample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain sample compute shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::terrain_samples_comp.source().into()),
        });
        let sample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain sample compute pipeline"),
            layout: Some(&sample_pipeline_layout),
            module: &sample_shader,
            entry_point: Some(shaders::terrain_samples_comp.entry_point()),
            compilation_options: Default::default(),
            cache: None,
        });

        let vertex_compute_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain vertex compute layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, true),
                    uniform_entry(3, wgpu::ShaderStages::COMPUTE),
                    uniform_entry(4, wgpu::ShaderStages::COMPUTE),
                ],
            });
        let vertex_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain vertex compute bindings"),
            layout: &vertex_compute_layout,
            entries: &[
                buffer_entry(0, &base_directions),
                buffer_entry(1, &vertices),
                buffer_entry(2, &terrain_samples),
                buffer_entry(3, &terrain_parameters_buffer),
                buffer_entry(4, &animation_buffer),
            ],
        });
        let vertex_compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain vertex compute pipeline layout"),
                bind_group_layouts: &[Some(&vertex_compute_layout)],
                immediate_size: 0,
            });
        let vertex_compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain vertex compute shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::terrain_vertices_comp.source().into()),
        });
        let vertex_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("terrain vertex compute pipeline"),
                layout: Some(&vertex_compute_pipeline_layout),
                module: &vertex_compute_shader,
                entry_point: Some(shaders::terrain_vertices_comp.entry_point()),
                compilation_options: Default::default(),
                cache: None,
            });

        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain camera layout"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera view projection matrix"),
            size: std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain camera bindings"),
            layout: &camera_layout,
            entries: &[buffer_entry(0, &camera_buffer)],
        });
        let color_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain color range layout"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                storage_entry_visible(1, true, wgpu::ShaderStages::FRAGMENT),
                uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                uniform_entry(3, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let color_data = color_range.gpu_data();
        let color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain displacement color range"),
            contents: bytemuck::bytes_of(&color_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let color_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain color range bindings"),
            layout: &color_layout,
            entries: &[
                buffer_entry(0, &color_buffer),
                buffer_entry(1, &terrain_samples),
                buffer_entry(2, &terrain_parameters_buffer),
                buffer_entry(3, &animation_buffer),
            ],
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain render pipeline layout"),
                bind_group_layouts: &[Some(&camera_layout), Some(&color_layout)],
                immediate_size: 0,
            });
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain vertex shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::mesh3d_vert.source().into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain fragment shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::terrain_frag.source().into()),
        });
        let vertex_layout = [Some(TerrainVertex::layout())];
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("animated terrain sphere pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some(shaders::mesh3d_vert.entry_point()),
                compilation_options: Default::default(),
                buffers: &vertex_layout,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some(shaders::terrain_frag.entry_point()),
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
            sample_pipeline,
            vertex_compute_pipeline,
            sample_bind_group,
            vertex_compute_bind_group,
            render_pipeline,
            camera_buffer,
            camera_bind_group,
            color_buffer,
            color_bind_group,
            animation_buffer,
            vertices,
            vertex_count,
            indices,
            index_count,
            settings,
        }
    }

    pub fn set_camera(
        &self,
        queue: &wgpu::Queue,
        camera: Camera3d,
        projection: crate::Projection3d,
    ) {
        let matrix = camera.view_projection(projection).to_cols_array_2d();
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&matrix));
    }

    pub fn set_color_range(&self, queue: &wgpu::Queue, settings: TerrainColorRange) {
        assert!(settings.maximum_displacement > settings.minimum_displacement);
        queue.write_buffer(
            &self.color_buffer,
            0,
            bytemuck::bytes_of(&settings.gpu_data()),
        );
    }

    pub fn update(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        elapsed: Duration,
    ) {
        let animation = AnimationUniform {
            seed: self.settings.terrain_seed,
            elapsed_seconds: elapsed.as_secs_f32(),
            speed: self.settings.animation_speed,
            _padding: 0.0,
        };
        queue.write_buffer(&self.animation_buffer, 0, bytemuck::bytes_of(&animation));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("generate animated terrain samples"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sample_pipeline);
            pass.set_bind_group(0, &self.sample_bind_group, &[]);
            pass.dispatch_workgroups(TERRAIN_SAMPLE_COUNT.div_ceil(COMPUTE_WORKGROUP_SIZE), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("displace terrain vertices"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.vertex_compute_pipeline);
            pass.set_bind_group(0, &self.vertex_compute_bind_group, &[]);
            pass.dispatch_workgroups(self.vertex_count.div_ceil(COMPUTE_WORKGROUP_SIZE), 1, 1);
        }
    }

    pub fn draw<'pass>(&'pass self, pass: &mut wgpu::RenderPass<'pass>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_bind_group(1, &self.color_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    storage_entry_visible(binding, read_only, wgpu::ShaderStages::COMPUTE)
}

fn storage_entry_visible(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_sphere_topology(settings: SphereTerrainSettings) -> (Vec<[f32; 4]>, Vec<u32>) {
    let latitude_segments = settings.latitude_segments;
    let longitude_segments = settings.longitude_segments;
    let grid_width = longitude_segments + 1;
    let mut directions = Vec::with_capacity((latitude_segments + 1) as usize * grid_width as usize);
    for latitude in 0..=latitude_segments {
        let phi = std::f32::consts::PI * (latitude as f32 / latitude_segments as f32 - 0.5);
        for longitude in 0..=longitude_segments {
            let theta = std::f32::consts::TAU * longitude as f32 / longitude_segments as f32;
            let direction =
                glam::Vec3::new(phi.cos() * theta.cos(), phi.sin(), phi.cos() * theta.sin());
            directions.push([direction.x, direction.y, direction.z, 0.0]);
        }
    }
    let mut indices = Vec::with_capacity((latitude_segments * longitude_segments * 6) as usize);
    for latitude in 0..latitude_segments {
        for longitude in 0..longitude_segments {
            let p00 = latitude * grid_width + longitude;
            let p10 = (latitude + 1) * grid_width + longitude;
            let p01 = p00 + 1;
            let p11 = p10 + 1;
            indices.extend_from_slice(&[p00, p10, p01, p01, p10, p11]);
        }
    }
    (directions, indices)
}

fn terrain_gpu_data(settings: SphereTerrainSettings, vertex_count: u32) -> [[f32; 4]; 3] {
    let center = settings.center.to_array();
    [
        [center[0], center[1], center[2], settings.radius],
        [
            settings.terrain_amplitude,
            settings.terrain_smoothing,
            settings.latitude_segments as f32,
            settings.longitude_segments as f32,
        ],
        [vertex_count as f32, 0.0, 0.0, 0.0],
    ]
}
