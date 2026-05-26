use bevy::{
    pbr::Material,
    prelude::*,
    render::alpha::AlphaMode,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef},
};

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct WaterMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub terrainmap: Handle<Image>,
}

impl Material for WaterMaterial {
    fn vertex_shader() -> ShaderRef {
        "water.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "water.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}
