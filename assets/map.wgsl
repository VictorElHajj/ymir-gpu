
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@group(2) @binding(0) var heightmap: texture_2d<f32>;
@group(2) @binding(1) var heightmap_sampler: sampler;

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(heightmap, heightmap_sampler, input.uv);
    let height = sample.r;
    let water = sample.g;
    return with_water(height, water);
}

fn with_water(height: f32, water: f32) -> vec4<f32> {
    let water_color = vec3(0., 0., 1.);
    let height_color = vec3(height);
    return vec4((1. - water) * height_color + water * water_color, 1.);
}