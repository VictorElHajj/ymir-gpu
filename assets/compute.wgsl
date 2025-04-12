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
const TimeStep = 0.1;
// Should probably always be one when pixel:grid is 1:1
const FlowPipeCrossSectionArea = 1.;
const Gravity = 9.82;

// Based on https://inria.hal.science/inria-00402079/document with side lengths and pipe length set to 1 and removed from calculations

// Water added from rain
@compute @workgroup_size(1, 1, 1)
fn _1_precipitation(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    var terrainmap = textureLoad(in_terrainmap, location);

    var water = terrainmap.g;
    water += 0.0;
    terrainmap.g = water;
    textureStore(out_terrainmap, location, terrainmap);

    // All stages need to write all textures so that swapping can work properly
    var flowmap = textureLoad(in_flowmap, location);
    var velocity = textureLoad(in_velocitymap, location);
    textureStore(out_flowmap, location, flowmap);
    textureStore(out_velocitymap, location, velocity);
}

// Update flux map
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
    let terrainmap = textureLoad(in_terrainmap, location);
    let terrainmap_left = textureLoad(in_terrainmap, location_left);
    let terrainmap_right = textureLoad(in_terrainmap, location_right);
    let terrainmap_top = textureLoad(in_terrainmap, location_top);
    let terrainmap_bottom = textureLoad(in_terrainmap, location_bottom);

    let ground_height = terrainmap.r;
    let water = terrainmap.g;

    var flow = textureLoad(in_flowmap, location);
    let ground_height_left = terrainmap_left.r;
    let water_left = terrainmap_left.g;
    let flow_out_left = max(0., flow.r + TimeStep * FlowPipeCrossSectionArea * Gravity * (ground_height + water - ground_height_left - water_left));

    let ground_height_right = terrainmap_right.r;
    let water_right = terrainmap_right.g;
    let flow_out_right = max(0., flow.g + TimeStep * FlowPipeCrossSectionArea * Gravity * (ground_height + water - ground_height_right - water_right));

    let ground_height_top = terrainmap_top.r;
    let water_top = terrainmap_top.g;
    // Boundary condition top
    var flow_out_top = 0.0;
    if (location_top.y < Height) {
        flow_out_top = max(0., flow.b + TimeStep * FlowPipeCrossSectionArea * Gravity * (ground_height + water - ground_height_top - water_top));
    }

    let ground_height_bottom = terrainmap_bottom.r;
    let water_bottom = terrainmap_bottom.g;
    var flow_out_bottom = 0.0;
    // Boundary condition top
    if (location_bottom.y >= 0) {
        flow_out_bottom = max(0., flow.a + TimeStep * FlowPipeCrossSectionArea * Gravity * (ground_height + water - ground_height_bottom - water_bottom));
    }

    // New flow divided by K to ensure outflow is never more than current water amount
    let downscale = min(1., water / ((0.) * TimeStep));
    textureStore(out_flowmap, location, vec4(flow_out_left, flow_out_right, flow_out_top, flow_out_bottom) / downscale);

    let velocity = textureLoad(in_velocitymap, location);
    textureStore(out_terrainmap, location, terrainmap);
    textureStore(out_velocitymap, location, velocity);
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

    let flow_out = textureLoad(in_flowmap, location);
    let flow_out_left = textureLoad(in_flowmap, location_left);
    let flow_out_right = textureLoad(in_flowmap, location_right);
    // Boundary condition top
    var flow_out_top = vec4(0.0);
    if (location_top.y < Height) {
        flow_out_top = textureLoad(in_flowmap, location_top);
    }
    // Boundary condition bottom
    var flow_out_bottom = vec4(0.0);
    if (location_bottom.y >= 0) {
        flow_out_bottom = textureLoad(in_flowmap, location_bottom);
    }

    var terrainmap = textureLoad(in_terrainmap, location);

    // Water
    terrainmap.g += TimeStep * (flow_out_left.g + flow_out_right.r + flow_out_bottom.b + flow_out_top.a - (flow_out.r + flow_out.g + flow_out.b + flow_out.a));
    textureStore(out_terrainmap, location, terrainmap);

    let velocity = textureLoad(in_velocitymap, location);
    textureStore(out_flowmap, location, flow_out);
    textureStore(out_velocitymap, location, velocity);
}

// Use updated water height to update velocity field map
@compute @workgroup_size(1, 1, 1)
fn _2_3_velocity_field(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var terrainmap = textureLoad(in_terrainmap, location);
    var flowmap = textureLoad(in_flowmap, location);
    var velocity = textureLoad(in_velocitymap, location);
    textureStore(out_terrainmap, location, terrainmap);
    textureStore(out_flowmap, location, flowmap);
    textureStore(out_velocitymap, location, velocity);
}

@compute @workgroup_size(1, 1, 1)
fn _3_erosion_deposition(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var terrainmap = textureLoad(in_terrainmap, location);
    var flowmap = textureLoad(in_flowmap, location);
    var velocity = textureLoad(in_velocitymap, location);
    textureStore(out_terrainmap, location, terrainmap);
    textureStore(out_flowmap, location, flowmap);
    textureStore(out_velocitymap, location, velocity);
}

@compute @workgroup_size(1, 1, 1)
fn _4_sediment_transport(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var terrainmap = textureLoad(in_terrainmap, location);
    var flowmap = textureLoad(in_flowmap, location);
    var velocity = textureLoad(in_velocitymap, location);
    textureStore(out_terrainmap, location, terrainmap);
    textureStore(out_flowmap, location, flowmap);
    textureStore(out_velocitymap, location, velocity);
}

@compute @workgroup_size(1, 1, 1)
fn _5_evaporation(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    var terrainmap = textureLoad(in_terrainmap, location);

    var water = terrainmap.g;
    water *= 1.0;
    terrainmap.g = water;
    textureStore(out_terrainmap, location, terrainmap);

    // All stages need to write all textures so that swapping can work properly
    var flowmap = textureLoad(in_flowmap, location);
    var velocity = textureLoad(in_velocitymap, location);
    textureStore(out_flowmap, location, flowmap);
    textureStore(out_velocitymap, location, velocity);
}


@compute @workgroup_size(1, 1, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    // let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    // let terrainmap = textureLoad(in_terrainmap, location);
    // var ground_height = terrainmap.r;
    // var water = terrainmap.g;
    // var sediment = terrainmap.b;

    // // top and bottom will require bounds checks later..
    // let location_top = location + vec2(0, 1);
    // let location_bottom = location + vec2(0, -1);
    // // left and right loop around
    // var left_pos = location.x - 1;
    // if (left_pos < 0) {
    //     left_pos = Width-1;
    // }
    // var right_pos = location.x + 1;
    // if (right_pos >= Width) {
    //     right_pos = 0;
    // }
    // let location_left = vec2(left_pos, location.y);
    // let location_right = vec2(right_pos, location.y);

    // // 1. Water increases due to rainfall
    // water += precipetation(location);

    // // 2. Update flow, water level and velocity
    // let flow_out = flow_out(location);
    // textureStore(out_flowmap, location, flow_out);
    // let flow_out_left = flow_out(location_left);
    // let flow_out_right = flow_out(location_right);
    // // Boundary condition top
    // var flow_out_top = vec4(0.0);
    // if (location_top.y < Height) {
    //     flow_out_top = flow_out(location_top);
    // }
    // // Boundary condition bottom
    // var flow_out_bottom = vec4(0.0);
    // if (location_bottom.y >= 0) {
    //     flow_out_bottom = flow_out(location_bottom);
    // }
    // water += TimeStep * (flow_out_left.g + flow_out_right.r + flow_out_bottom.b + flow_out_top.a - (flow_out.r + flow_out.g + flow_out.b + flow_out.a));

    // // 4. Sediment transportation

    // // 5. Water Evaporation
    // water *= evaporation();

    // textureStore(out_terrainmap, location, vec4(ground_height, water, sediment, 1.0));
}