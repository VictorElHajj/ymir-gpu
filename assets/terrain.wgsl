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
};

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.uv;
    let height = textureSampleLevel(heightmap, heightmap_sampler, in.uv, 0.0).r;
    out.clip_position = mesh_position_local_to_clip(
        get_world_from_local(in.instance_index),
        vec4(in.position.x, height, in.position.z, 1.0),
    );
    return out;
}

const HEIGHT_SCALE: f32 = 1000.0;
const TEX_DX: f32 = 1.0 / 4096.0;
const TEX_DZ: f32 = 1.0 / 2048.0;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let height = textureSample(heightmap, heightmap_sampler, in.uv).r;

    // Compute surface normal from neighbouring heights for shading
    let h_r = textureSample(heightmap, heightmap_sampler, in.uv + vec2(TEX_DX, 0.0)).r;
    let h_l = textureSample(heightmap, heightmap_sampler, in.uv - vec2(TEX_DX, 0.0)).r;
    let h_u = textureSample(heightmap, heightmap_sampler, in.uv + vec2(0.0, TEX_DZ)).r;
    let h_d = textureSample(heightmap, heightmap_sampler, in.uv - vec2(0.0, TEX_DZ)).r;
    // XZ world distances between pixels are 1 unit; height differences are in world units already
    let normal = normalize(vec3(h_l - h_r, 2.0, h_u - h_d));

    let light_dir = normalize(vec3(1.0, 2.0, 0.5));
    let ambient = 0.3;
    let diffuse = max(0.0, dot(normal, light_dir));
    let lighting = ambient + (1.0 - ambient) * diffuse;

    // Height-based colour bands
    let t = height / HEIGHT_SCALE;
    let sand  = vec3(0.76, 0.70, 0.50);
    let grass = vec3(0.38, 0.55, 0.28);
    let rock  = vec3(0.55, 0.50, 0.45);
    let snow  = vec3(0.95, 0.95, 1.00);

    var color: vec3<f32>;
    if t < 0.2 {
        color = mix(sand,  grass, t / 0.2);
    } else if t < 0.6 {
        color = mix(grass, rock,  (t - 0.2) / 0.4);
    } else {
        color = mix(rock,  snow,  (t - 0.6) / 0.4);
    }

    return vec4(color * lighting, 1.0);
}
