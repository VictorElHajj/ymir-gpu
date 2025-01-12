use bevy::{
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef},
    sprite::Material2d,
};

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct MapMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub heightmap: Handle<Image>,
    #[texture(3)]
    #[sampler(4)]
    pub flowmap: Handle<Image>,
    #[texture(5)]
    #[sampler(6)]
    pub velocitymap: Handle<Image>,
}

impl Material2d for MapMaterial {
    fn fragment_shader() -> ShaderRef {
        "map.wgsl".into()
    }
}
