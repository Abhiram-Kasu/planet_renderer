use crate::{Mat4, Vec3};

/// Camera pose for 3D rendering. `up` controls roll and should not be parallel
/// to the direction from `eye` to `look_at`.
#[derive(Clone, Copy, Debug)]
pub struct Camera3d {
    pub eye: Vec3,
    pub look_at: Vec3,
    pub up: Vec3,
}

impl Default for Camera3d {
    fn default() -> Self {
        Self {
            eye: Vec3::new(0.0, 0.0, 3.0),
            look_at: Vec3::ZERO,
            up: Vec3::Y,
        }
    }
}

impl Camera3d {
    pub fn view_projection(self, projection: Projection3d) -> Mat4 {
        self.view_projection_matrix(projection)
    }

    pub fn inverse_view_projection(self, projection: Projection3d) -> Mat4 {
        self.view_projection_matrix(projection).inverse()
    }

    fn view_projection_matrix(self, projection: Projection3d) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.look_at, self.up);
        projection.matrix() * view
    }
}

/// Perspective projection parameters (right-handed, zero-to-one depth).
#[derive(Clone, Copy, Debug)]
pub struct Projection3d {
    pub vertical_fov_radians: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Projection3d {
    fn default() -> Self {
        Self {
            vertical_fov_radians: 60.0_f32.to_radians(),
            aspect_ratio: 1.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Projection3d {
    fn matrix(self) -> Mat4 {
        Mat4::perspective_rh(
            self.vertical_fov_radians,
            self.aspect_ratio,
            self.near,
            self.far,
        )
    }
}
