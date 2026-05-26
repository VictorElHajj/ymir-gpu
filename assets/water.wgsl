#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}

@group(2) @binding(0) var heightmap: texture_2d<f32>;
@group(2) @binding(1) var heightmap_sampler: sampler;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) water_depth: f32,
};

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.uv;
    let sample = textureSampleLevel(heightmap, heightmap_sampler, in.uv, 0.0);
    let terrain_height = sample.r;
    let water = sample.g;
    out.water_depth = water;
    // Offset 2 world units above terrain to prevent z-fighting
    out.clip_position = mesh_position_local_to_clip(
        get_world_from_local(in.instance_index),
        vec4(in.position.x, terrain_height + water, in.position.z, 1.0),
    );
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.water_depth < 0.0001 {
        discard;
    }
    let alpha = min(0.85, in.water_depth * 200.0);
    let shallow = vec3(0.30, 0.65, 0.95);
    let deep    = vec3(0.05, 0.25, 0.65);
    let color = mix(shallow, deep, min(1.0, in.water_depth * 20.0));
    return vec4(color, alpha);
}
