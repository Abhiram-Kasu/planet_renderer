use planet_renderer::{
    Camera3d, PositionColor, Projection3d, SdfSphereSettings, SphereRenderer, TriangleRenderer,
    shaders,
};
use std::{sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct RenderState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    triangle: TriangleRenderer<PositionColor>,
    sphere: SphereRenderer,
    sphere_settings: SdfSphereSettings,
    started_at: Instant,
    camera: Camera3d,
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
        let sphere_settings = SdfSphereSettings::default();
        let sphere = SphereRenderer::new(
            &device,
            format,
            shaders::sdf_sphere_vert,
            shaders::sdf_sphere_frag,
            sphere_settings,
        );
        eprintln!("pipeline ready");
        RenderState {
            surface,
            device,
            queue,
            config,
            triangle,
            sphere,
            sphere_settings,
            started_at: Instant::now(),
            camera: Camera3d::default(),
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
        if state.show_sphere {
            state
                .sphere
                .set_camera(&state.queue, state.camera, state.projection);
            let rotation = state.started_at.elapsed().as_secs_f32()
                * state.sphere_settings.spin_speed_radians_per_second;
            state.sphere.set_rotation(&state.queue, rotation);
        }
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("triangle frame encoder"),
            });
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
                depth_stencil_attachment: None,
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
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state.is_pressed()
                    && event.logical_key
                        == winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) =>
            {
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
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.render();
    }
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
