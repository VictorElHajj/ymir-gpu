
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@group(2) @binding(0) var heightmap: texture_2d<f32>;
@group(2) @binding(1) var heightmap_sampler: sampler;

const MaxTemp = 40;
const MinTemp = -40;
const PI = 3.14159;

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(heightmap, heightmap_sampler, input.uv);
    let height = sample.r;
    let water = sample.g;

    // Range from -40 to +30, with 40 at equator
    let temperature: f32 = (1.0 - abs(input.uv.y - 0.5) * 2.0) * 70. - 40.;
    // Hadley cells
    let wind_y_velocity: f32 = sin(2 * PI * input.uv.y * 3);
    // Coriolis effect
    let wind_x_velocity: f32 = sin(2 * PI * abs(input.uv.y-0.5) * 3);
    return with_water(height, water);
}

fn with_water(height: f32, water: f32) -> vec4<f32> {
    let water_color = vec3(0., 0., 0.5);
    let height_color = vec3(height);
    let adjusted_water = min(0.5, max(0., water));
    return vec4((1. - adjusted_water) * height_color + adjusted_water * water_color, 1.);
}

fn show_wind(wind: f32) -> vec4<f32> {
    if wind < 0.0 {
        return vec4(abs(wind), 0., 0., 1.);
    }
    else {
        return vec4(0., abs(wind), 0., 1.);
    }
}

fn show_temperature(temperature: f32) -> vec4<f32> {
    if temperature > 15 {
        return mix(vec4(1.0, 0.0, 0.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0), (30.0 - temperature) / 15.0);
    }
    if temperature > 0 {
        return mix(vec4(1.0, 1.0, .0, 1.0), vec4(0.0, 0.5, 0.5, 1.0), (15 - temperature) / 15.1);
    }
    if temperature > -20 {
        return mix(vec4(0.0, 0.5, 0.5, 1.0), vec4(0.0, 0.0, 0.5, 1.0), (0 - temperature) / 20.0);
    }
    if temperature > -40 {
        return mix(vec4(0.0, 0.0, 0.5, 1.0), vec4(0.5, 0.0, 0.5, 1.0), (-20 - temperature) / 20.0);
    }
    return vec4(0.);
}