use glam::{Quat, Vec3};
use planet_renderer::{
    Camera3d, PositionColor, Projection3d, SphereRenderer, SphereTerrainSettings,
    TerrainColorRange, TriangleRenderer, shaders,
};
use std::{sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const CAMERA_ZOOM_STEP: f32 = 0.25;
const CAMERA_ROTATION_STEP: f32 = std::f32::consts::PI / 90.0;
const CAMERA_SMOOTHING: f32 = 7.0;
const MIN_CAMERA_DISTANCE: f32 = 1.4;
const MAX_CAMERA_DISTANCE: f32 = 8.0;
const FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);

struct RenderState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    triangle: TriangleRenderer<PositionColor>,
    sphere: SphereRenderer,
    depth: wgpu::Texture,
    started_at: Instant,
    last_frame: Instant,
    camera: Camera3d,
    camera_orbit_direction: Vec3,
    camera_distance: f32,
    target_camera_distance: f32,
    camera_yaw: f32,
    target_camera_yaw: f32,
    projection: Projection3d,
    show_sphere: bool,
}

struct App {
    window: Option<Arc<Window>>,
    state: Arc<std::sync::Mutex<Option<RenderState>>>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            state: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    async fn initialize(window: Arc<Window>) -> RenderState {
        eprintln!("initializing wgpu");
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("surface creation failed");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("no compatible GPU adapter found");
        eprintln!("adapter ready");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("device request failed");
        eprintln!("device ready");
        device.on_uncaptured_error(Arc::new(|error| eprintln!("wgpu error: {error}")));
        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(capabilities.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: capabilities.present_modes[0],
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let depth = create_depth(&device, config.width, config.height);
        let vertices = [
            PositionColor {
                position: [0.0, 0.65],
                color: [1.0, 0.15, 0.15],
            },
            PositionColor {
                position: [-0.65, -0.55],
                color: [0.15, 1.0, 0.15],
            },
            PositionColor {
                position: [0.65, -0.55],
                color: [0.15, 0.3, 1.0],
            },
        ];
        let triangle = TriangleRenderer::new(
            &device,
            format,
            shaders::triangle_vert,
            shaders::triangle_frag,
            &vertices,
        );
        let sphere = SphereRenderer::new(
            &device,
            format,
            SphereTerrainSettings::default(),
            TerrainColorRange::default(),
        );
        eprintln!("pipeline ready");
        let camera = Camera3d::default();
        let camera_orbit_direction = (camera.eye - camera.look_at).normalize();
        let camera_distance = camera.eye.distance(camera.look_at);
        RenderState {
            surface,
            device,
            queue,
            config,
            triangle,
            sphere,
            depth,
            started_at: Instant::now(),
            last_frame: Instant::now(),
            camera,
            camera_orbit_direction,
            camera_distance,
            target_camera_distance: camera_distance,
            camera_yaw: 0.0,
            target_camera_yaw: 0.0,
            projection: Projection3d::default(),
            show_sphere: true,
        }
    }

    fn render(&mut self) {
        if self.window.is_none() {
            return;
        }
        let mut state_guard = self.state.lock().expect("render state lock poisoned");
        let Some(state) = state_guard.as_mut() else {
            return;
        };
        let now = Instant::now();
        let delta_seconds = now.duration_since(state.last_frame).as_secs_f32().min(0.1);
        state.last_frame = now;
        let blend = 1.0 - (-CAMERA_SMOOTHING * delta_seconds).exp();
        state.camera_distance += (state.target_camera_distance - state.camera_distance) * blend;
        let yaw_delta = (state.target_camera_yaw - state.camera_yaw + std::f32::consts::PI)
            .rem_euclid(std::f32::consts::TAU)
            - std::f32::consts::PI;
        state.camera_yaw += yaw_delta * blend;
        let rotation = Quat::from_axis_angle(state.camera.up.normalize(), state.camera_yaw);
        let camera_direction = rotation * state.camera_orbit_direction;
        state.camera.eye = state.camera.look_at + camera_direction * state.camera_distance;
        let frame = match state.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                state.surface.configure(&state.device, &state.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let depth_view = state.depth.create_view(&Default::default());
        if state.show_sphere {
            state
                .sphere
                .set_camera(&state.queue, state.camera, state.projection);
        }
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("triangle frame encoder"),
            });
        if state.show_sphere {
            state
                .sphere
                .update(&state.queue, &mut encoder, state.started_at.elapsed());
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("triangle render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: if state.show_sphere {
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    })
                } else {
                    None
                },
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if state.show_sphere {
                let side = state.config.width.min(state.config.height) as f32;
                let x = (state.config.width as f32 - side) * 0.5;
                let y = (state.config.height as f32 - side) * 0.5;
                pass.set_viewport(x, y, side, side, 0.0, 1.0);
            }
            if state.show_sphere {
                state.sphere.draw(&mut pass);
            } else {
                state.triangle.draw(&mut pass);
            }
        }
        state.queue.submit([encoder.finish()]);
        state.queue.present(frame);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Gradient triangle"))
                .expect("window creation failed"),
        );
        #[cfg(not(target_arch = "wasm32"))]
        {
            *self.state.lock().expect("render state lock poisoned") =
                Some(pollster::block_on(Self::initialize(window.clone())));
        }
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen_futures::spawn_local;
            let shared_init = self.state.clone();
            let window_init = window.clone();
            let window_ready = window.clone();
            spawn_local(async move {
                *shared_init.lock().expect("render state lock poisoned") =
                    Some(Self::initialize(window_init).await);
                window_ready.request_redraw();
            });
        }
        self.window = Some(window.clone());
        #[cfg(not(target_arch = "wasm32"))]
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = self
                    .state
                    .lock()
                    .expect("render state lock poisoned")
                    .as_mut()
                {
                    state.config.width = size.width.max(1);
                    state.config.height = size.height.max(1);
                    state.surface.configure(&state.device, &state.config);
                    state.depth =
                        create_depth(&state.device, state.config.width, state.config.height);
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                match event.logical_key {
                    Key::Named(NamedKey::Space) => {
                        if let Some(state) = self
                            .state
                            .lock()
                            .expect("render state lock poisoned")
                            .as_mut()
                        {
                            state.show_sphere = !state.show_sphere;
                        }
                        self.render();
                    }
                    Key::Named(key @ (NamedKey::ArrowUp | NamedKey::ArrowDown)) => {
                        if let Some(state) = self
                            .state
                            .lock()
                            .expect("render state lock poisoned")
                            .as_mut()
                        {
                            let zoom_delta = match key {
                                NamedKey::ArrowUp => CAMERA_ZOOM_STEP,
                                _ => -CAMERA_ZOOM_STEP,
                            };
                            state.target_camera_distance = (state.target_camera_distance
                                + zoom_delta)
                                .clamp(MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE);
                        }
                    }
                    Key::Named(key @ (NamedKey::ArrowRight | NamedKey::ArrowLeft)) => {
                        if let Some(state) = self
                            .state
                            .lock()
                            .expect("render state lock poisoned")
                            .as_mut()
                        {
                            let rotation_delta = match key {
                                NamedKey::ArrowLeft => -CAMERA_ROTATION_STEP,
                                _ => CAMERA_ROTATION_STEP,
                            };
                            state.target_camera_yaw += rotation_delta;
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + FRAME_INTERVAL));
        self.render();
    }
}

fn create_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sphere depth texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let event_loop = EventLoop::new().expect("event loop creation failed");
    event_loop
        .run_app(&mut App::new())
        .expect("event loop failed");
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
fn main() {
    use winit::platform::web::EventLoopExtWebSys;
    let event_loop = EventLoop::new().expect("event loop creation failed");
    event_loop.spawn_app(App::new());
}
