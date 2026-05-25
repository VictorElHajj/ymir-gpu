@group(0) @binding(0) var in_terrainmap: texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var in_flowmap: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var in_velocitymap: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var out_terrainmap: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var out_flowmap: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var out_velocitymap: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(1, 1, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
}

// Texture parameters
const Width = 4096;
const Height = 2048;


// Simulation parameters
const TimeStep = 0.1; // Setting this too high will break the simulation, too low and it will be very slow.
const FlowPipeCrossSectionArea = 1.; // Should probably always be one when pixel:grid is 1:1
const Gravity = 9.; // Higher values increase water flow (and therefore velocity, which affects carrying capacity etc)
const SedimentCapacity = 0.1; // Multipler on how much sediment the water can hold
const DissolvingRate = .9; // [0, 1], where 0 means no dissolving and 1 means terrain will always be dissolved to reach capacity
const DepositionRate = .9; // [0, 1], where 0 means no despositing and 1 means sediment will always be dropped to reach capacity
const Evaporation = 0.8; // [0, 1] The percent of water evaporated each update.
const Precipitation = 0.01; // [0, 1] Units of water added each update in all locations
const MinTiltAngle = 0.261799388;

// Based on https://inria.hal.science/inria-00402079/document with side lengths and pipe length set to 1 and removed from calculations
// TODO: When adapting the map to be a sphere projected onto a square the side lengths should added and set to the length between poitns on the sphere

// terrinamp: r = height, g = water, b = sediment 
// flowmap: r = left, g = right, b = top, a = bottom
// velocitymap: r = x velocity, g = y velocity

// Water added from rain
@compute @workgroup_size(1, 1, 1)
fn _1_precipitation(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    var terrain = textureLoad(in_terrainmap, location);
    terrain.g += Precipitation * TimeStep;
    
    textureStore(out_terrainmap, location, terrain);
}

// Flux map, each location stores the water output to left/right/top/bottom location
@compute @workgroup_size(1, 1, 1)
fn _2_1_outflow_flux(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let location_top = location + vec2(0, 1);
    let location_bottom = location + vec2(0, -1);
    // left and right loop around
    var left_pos = location.x - 1;
    if (left_pos < 0) {
        left_pos = Width-1;
    }
    var right_pos = location.x + 1;
    if (right_pos >= Width) {
        right_pos = 0;
    }
    let location_left = vec2(left_pos, location.y);
    let location_right = vec2(right_pos, location.y);

    let terrain = textureLoad(in_terrainmap, location);
    let terrain_left = textureLoad(in_terrainmap, location_left);
    let terrain_right = textureLoad(in_terrainmap, location_right);
    let terrain_top = textureLoad(in_terrainmap, location_top);
    let terrain_bottom = textureLoad(in_terrainmap, location_bottom);

    let ground_height = terrain.r;
    let water = terrain.g;

    var flow = textureLoad(in_flowmap, location);
    let ground_height_left = terrain_left.r;
    let water_left = terrain_left.g;
    let flow_left = max(0., flow.r + TimeStep * FlowPipeCrossSectionArea * Gravity * (ground_height + water - ground_height_left - water_left));

    let ground_height_right = terrain_right.r;
    let water_right = terrain_right.g;
    let flow_right = max(0., flow.g + TimeStep * FlowPipeCrossSectionArea * Gravity * (ground_height + water - ground_height_right - water_right));

    let ground_height_top = terrain_top.r;
    let water_top = terrain_top.g;
    // Boundary condition top
    var flow_top = 0.0;
    if (location_top.y < Height) {
        flow_top = max(0., flow.b + TimeStep * FlowPipeCrossSectionArea * Gravity * (ground_height + water - ground_height_top - water_top));
    }

    let ground_height_bottom = terrain_bottom.r;
    let water_bottom = terrain_bottom.g;
    var flow_bottom = 0.0;
    // Boundary condition top
    if (location_bottom.y >= 0) {
        flow_bottom = max(0., flow.a + TimeStep * FlowPipeCrossSectionArea * Gravity * (ground_height + water - ground_height_bottom - water_bottom));
    }

    // New flow divided by K to ensure outflow is never more than current water amount
    var downscale = 1.0;
    let total_flow = flow_left + flow_right + flow_top + flow_bottom;
    if (total_flow > 0.0) {
        downscale = min(1., water / ((flow_left + flow_right + flow_top + flow_bottom) * TimeStep));
    }
    textureStore(out_flowmap, location, vec4(flow_left, flow_right, flow_top, flow_bottom) * downscale);
    textureStore(out_terrainmap, location, terrain);
}

// Use updated flux map to update water height
@compute @workgroup_size(1, 1, 1)
fn _2_2_water_height(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let location_top = location + vec2(0, 1);
    let location_bottom = location + vec2(0, -1);
    // left and right loop around
    var left_pos = location.x - 1;
    if (left_pos < 0) {
        left_pos = Width-1;
    }
    var right_pos = location.x + 1;
    if (right_pos >= Width) {
        right_pos = 0;
    }
    let location_left = vec2(left_pos, location.y);
    let location_right = vec2(right_pos, location.y);

    let flow_left = textureLoad(in_flowmap, location_left);
    let flow_right = textureLoad(in_flowmap, location_right);
    // Boundary condition top
    var flow_top = vec4(0.0);
    if (location_top.y < Height) {
        flow_top = textureLoad(in_flowmap, location_top);
    }
    // Boundary condition bottom
    var flow_bottom = vec4(0.0);
    if (location_bottom.y >= 0) {
        flow_bottom = textureLoad(in_flowmap, location_bottom);
    }

    // Update the water height
    var terrain = textureLoad(in_terrainmap, location);
    let flow = textureLoad(in_flowmap, location);
    terrain.g += TimeStep * (flow_left.g + flow_right.r + flow_bottom.b + flow_top.a - (flow.r + flow.g + flow.b + flow.a));
    textureStore(out_terrainmap, location, terrain);
    textureStore(out_flowmap, location, flow);
}

// Use updated water height to update velocity field map
@compute @workgroup_size(1, 1, 1)
fn _2_3_velocity_field(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let location_top = location + vec2(0, 1);
    let location_bottom = location + vec2(0, -1);
    // left and right loop around
    var left_pos = location.x - 1;
    if (left_pos < 0) {
        left_pos = Width-1;
    }
    var right_pos = location.x + 1;
    if (right_pos >= Width) {
        right_pos = 0;
    }
    let location_left = vec2(left_pos, location.y);
    let location_right = vec2(right_pos, location.y);

    let flow = textureLoad(in_flowmap, location);
    let flow_left = textureLoad(in_flowmap, location_left);
    let flow_right = textureLoad(in_flowmap, location_right);
    // Boundary condition top
    var flow_top = vec4(0.0);
    if (location_top.y < Height) {
        flow_top = textureLoad(in_flowmap, location_top);
    }
    // Boundary condition bottom
    var flow_bottom = vec4(0.0);
    if (location_bottom.y >= 0) {
        flow_bottom = textureLoad(in_flowmap, location_bottom);
    }

    // Input from the left minus output to the left plus output to the right minus input from the right
    let velocity_x = (flow_left.g - flow.r + flow.g - flow_right.r) / 2;
    // Input from the top minus output to the top plus output to the bottom minus input from the bottom
    // l r t b
    let velocity_y = (flow_top.a - flow.b + flow.a - flow_bottom.b) / 2;

    var terrain = textureLoad(in_terrainmap, location);
    textureStore(out_terrainmap, location, terrain);
    textureStore(out_flowmap, location, flow);
    textureStore(out_velocitymap, location, vec4(velocity_x, velocity_y, 0.0, 0.0));
}

// Calculate carrying capacity using graidnet and velocity, compare with held sediment and
// erode or dispose
@compute @workgroup_size(1, 1, 1)
fn _3_erosion_deposition(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var terrain = textureLoad(in_terrainmap, location);
    let flow = textureLoad(in_flowmap, location);
    let velocity = textureLoad(in_velocitymap, location);

    // TODO: Add optional minimum carrying capacity for flat terrains
    let carrying_capacity = SedimentCapacity * sin(min(MinTiltAngle, tilt_angle(location, in_terrainmap))) * length(velocity);
    var suspended_sediment= terrain.b;
    var terrain_height = terrain.r;

    if carrying_capacity > suspended_sediment {
        // Some soil is dissolved and added to suspended sediment
        // Terrain cannot go below 0
        let dissolved_terrain = min(DissolvingRate * (carrying_capacity - suspended_sediment), terrain.r);
        terrain_height -=  dissolved_terrain;
        suspended_sediment += dissolved_terrain;
    } else {
        // Some soil is removed from the suspended sediment
        let dropped_sediment = DepositionRate * (suspended_sediment - carrying_capacity);
        terrain_height +=  dropped_sediment;
        suspended_sediment -= dropped_sediment;
    }

    terrain.r = terrain_height;
    terrain.b = suspended_sediment;
    textureStore(out_terrainmap, location, terrain);
    textureStore(out_flowmap, location, flow);
    textureStore(out_velocitymap, location, velocity);
}

@compute @workgroup_size(1, 1, 1)
fn _4_sediment_transport(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var terrain = textureLoad(in_terrainmap, location);
    let flow = textureLoad(in_flowmap, location);
    let velocity = textureLoad(in_velocitymap, location);

    let new_sediment = bilinear_sample(in_terrainmap, vec2(f32(location.x), f32(location.y)) - velocity.xy * TimeStep).b;
    terrain.b = new_sediment;

    textureStore(out_terrainmap, location, terrain);
    textureStore(out_flowmap, location, flow);
    textureStore(out_velocitymap, location, velocity);
}

@compute @workgroup_size(1, 1, 1)
fn _5_evaporation(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    var terrainmap = textureLoad(in_terrainmap, location);
    var flowmap = textureLoad(in_flowmap, location);
    var velocity = textureLoad(in_velocitymap, location);

    var water = terrainmap.g;
    water *= 1 - Evaporation * TimeStep;
    terrainmap.g = water;
    textureStore(out_terrainmap, location, terrainmap);

    // All stages need to write all textures so that swapping can work properly
    textureStore(out_flowmap, location, flowmap);
    textureStore(out_velocitymap, location, velocity);
}

fn tilt_angle(location: vec2<i32>, heightmap: texture_storage_2d<rgba32float, read>) -> f32 {
    let location_top = location + vec2(0, 1);
    let location_bottom = location + vec2(0, -1);
    // left and right loop around
    var left_pos = location.x - 1;
    if (left_pos < 0) {
        left_pos = Width-1;
    }
    var right_pos = location.x + 1;
    if (right_pos >= Width) {
        right_pos = 0;
    }
    let location_left = vec2(left_pos, location.y);
    let location_right = vec2(right_pos, location.y);

    let height_left = textureLoad(heightmap, location_left).r;
    let height_right = textureLoad(heightmap, location_right).r;
    let height_bottom = textureLoad(heightmap, location_bottom).r;
    let height_top = textureLoad(heightmap, location_top).r;

    // Convert to gradient vectors in x and y directions
    let dx = vec3<f32>(1.0, 0.0, height_right - height_left);
    let dy = vec3<f32>(0.0, 1.0, height_top - height_bottom);

    // Get surface normal from cross product of gradients
    let normal = normalize(cross(dy, dx));

    // Compare with up vector (z-up assumed; for y-up use vec3(0.0, 1.0, 0.0))
    let up = vec3<f32>(0.0, 0.0, 1.0);
    let angle = acos(clamp(dot(normal, up), -1.0, 1.0));

    return angle;
}

fn bilinear_sample(
    texture: texture_storage_2d<rgba32float, read>,
    location: vec2<f32>, // Pixel space
) -> vec4<f32> {
    let base_coords: vec2<i32> = vec2<i32>(floor(location));
    let fractional_offset: vec2<f32> = fract(location);

    let bottom_left     = textureLoad(texture, base_coords);
    let bottom_right    = textureLoad(texture, base_coords + vec2<i32>(1, 0));
    let top_left        = textureLoad(texture, base_coords + vec2<i32>(0, 1));
    let top_right       = textureLoad(texture, base_coords + vec2<i32>(1, 1));

    // Interpolate along X first (left to right)
    let lower_row_interp = mix(bottom_left, bottom_right, fractional_offset.x);
    let upper_row_interp = mix(top_left, top_right, fractional_offset.x);

    // Then interpolate along Y (bottom to top)
    let final_color = mix(lower_row_interp, upper_row_interp, fractional_offset.y);

    return final_color;
}
