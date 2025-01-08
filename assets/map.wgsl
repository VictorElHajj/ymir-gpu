
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
    return textureSample(heightmap, heightmap_sampler, input.uv);
}
