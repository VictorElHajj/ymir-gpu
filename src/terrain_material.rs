use bevy::{
    pbr::Material,
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef},
};

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct TerrainMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub terrainmap: Handle<Image>,
}

impl Material for TerrainMaterial {
    fn vertex_shader() -> ShaderRef {
        "terrain.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "terrain.wgsl".into()
    }
}
