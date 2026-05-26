@group(0) @binding(0) var in_terrainmap: texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var in_flowmap: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var in_velocitymap: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var out_terrainmap: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var out_flowmap: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var out_velocitymap: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
}

// Texture parameters
const Width = 4096;
const Height = 2048;

// Simulation parameters
const TimeStep = 0.00125; // Setting this too high will break the simulation, too low and it will be very slow.
const FlowPipeCrossSectionArea = 1.; // Should probably always be one when pixel:grid is 1:1
const Gravity = 2000.; // Higher values increase water flow (and therefore velocity, which affects carrying capacity etc)
const SedimentCapacity = 0.05; // Multipler on how much sediment the water can hold
const DissolvingRate = .4; // [0, 1], where 0 means no dissolving and 1 means terrain will always be dissolved to reach capacity
const DepositionRate = .9; // [0, 1], where 0 means no despositing and 1 means sediment will always be dropped to reach capacity
const Evaporation = 0.2; // [0, 1] The percent of water evaporated each update.
const Precipitation = 0.05; // [0, 1] Units of water added each update in all locations
const MinTiltAngle = 0.00001; // Minimum tilt angle floor for carrying capacity (~0.057°)
const Friction = 0.99; // Damps wave oscillations each step so gravity-driven flow dominates
const AngleOfRepose = 35.0; // Angle of repose in degrees, slopes steeper than this collapse
const ThermalErosionRate = 0.5; // Fraction of excess slope transferred per step [0, 1]

// Based on https://inria.hal.science/inria-00402079/document with side lengths and pipe length set to 1 and removed from calculations
// TODO: When adapting the map to be a sphere projected onto a square the side lengths should be added and set to the length between points on the sphere

// terrainmap channels: height, water, sediment, tilt
// flowmap channels: left, right, top, bottom
// velocitymap channels: x, y, -, -

struct Terrain { height: f32, water: f32, sediment: f32, tilt: f32 }
struct Flow { left: f32, right: f32, top: f32, bottom: f32 }
struct Velocity { x: f32, y: f32 }

fn load_terrain(loc: vec2<i32>) -> Terrain {
    let v = textureLoad(in_terrainmap, loc);
    return Terrain(v.x, v.y, v.z, v.w);
}
fn store_terrain(loc: vec2<i32>, t: Terrain) {
    textureStore(out_terrainmap, loc, vec4(t.height, t.water, t.sediment, t.tilt));
}

fn load_flow(loc: vec2<i32>) -> Flow {
    let v = textureLoad(in_flowmap, loc);
    return Flow(v.x, v.y, v.z, v.w);
}
fn store_flow(loc: vec2<i32>, f: Flow) {
    textureStore(out_flowmap, loc, vec4(f.left, f.right, f.top, f.bottom));
}

fn load_velocity(loc: vec2<i32>) -> Velocity {
    let v = textureLoad(in_velocitymap, loc);
    return Velocity(v.x, v.y);
}
fn store_velocity(loc: vec2<i32>, vel: Velocity) {
    textureStore(out_velocitymap, loc, vec4(vel.x, vel.y, 0.0, 0.0));
}

// Water added from rain
@compute @workgroup_size(8, 8, 1)
fn _1_precipitation(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    var terrain = load_terrain(location);
    terrain.water += Precipitation * TimeStep;

    store_terrain(location, terrain);
    store_flow(location, load_flow(location));
    store_velocity(location, load_velocity(location));
}

// Flux map: each location stores the water output to left/right/top/bottom neighbor
@compute @workgroup_size(8, 8, 1)
fn _2_1_outflow_flux(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    var left_pos = location.x - 1;
    if (left_pos < 0) { left_pos = Width - 1; }
    var right_pos = location.x + 1;
    if (right_pos >= Width) { right_pos = 0; }
    let location_left   = vec2(left_pos,  location.y);
    let location_right  = vec2(right_pos, location.y);
    let location_top    = location + vec2(0,  1);
    let location_bottom = location + vec2(0, -1);

    let terrain       = load_terrain(location);
    let terrain_left  = load_terrain(location_left);
    let terrain_right = load_terrain(location_right);
    let flow = load_flow(location);

    let surface       = terrain.height + terrain.water;
    let surface_left  = terrain_left.height  + terrain_left.water;
    let surface_right = terrain_right.height + terrain_right.water;

    let new_flow_left  = max(0., flow.left  * Friction + TimeStep * FlowPipeCrossSectionArea * Gravity * (surface - surface_left));
    let new_flow_right = max(0., flow.right * Friction + TimeStep * FlowPipeCrossSectionArea * Gravity * (surface - surface_right));

    var new_flow_top = 0.0;
    if (location_top.y < Height) {
        let terrain_top = load_terrain(location_top);
        let surface_top = terrain_top.height + terrain_top.water;
        new_flow_top = max(0., flow.top * Friction + TimeStep * FlowPipeCrossSectionArea * Gravity * (surface - surface_top));
    }

    var new_flow_bottom = 0.0;
    if (location_bottom.y >= 0) {
        let terrain_bottom = load_terrain(location_bottom);
        let surface_bottom = terrain_bottom.height + terrain_bottom.water;
        new_flow_bottom = max(0., flow.bottom * Friction + TimeStep * FlowPipeCrossSectionArea * Gravity * (surface - surface_bottom));
    }

    // Scale down outflow so it never exceeds available water
    let total_flow = new_flow_left + new_flow_right + new_flow_top + new_flow_bottom;
    var downscale = 1.0;
    if (total_flow > 0.0) {
        downscale = min(1., terrain.water / (total_flow * TimeStep));
    }

    store_flow(location, Flow(new_flow_left * downscale, new_flow_right * downscale, new_flow_top * downscale, new_flow_bottom * downscale));
    store_terrain(location, terrain);
    store_velocity(location, load_velocity(location));
}

// Use updated flux map to update water height
@compute @workgroup_size(8, 8, 1)
fn _2_2_water_height(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    var left_pos = location.x - 1;
    if (left_pos < 0) { left_pos = Width - 1; }
    var right_pos = location.x + 1;
    if (right_pos >= Width) { right_pos = 0; }
    let location_left   = vec2(left_pos,  location.y);
    let location_right  = vec2(right_pos, location.y);
    let location_top    = location + vec2(0,  1);
    let location_bottom = location + vec2(0, -1);

    let flow_left  = load_flow(location_left);
    let flow_right = load_flow(location_right);
    var flow_top    = Flow(0.0, 0.0, 0.0, 0.0);
    if (location_top.y < Height) { flow_top = load_flow(location_top); }
    var flow_bottom = Flow(0.0, 0.0, 0.0, 0.0);
    if (location_bottom.y >= 0) { flow_bottom = load_flow(location_bottom); }

    var terrain = load_terrain(location);
    let flow    = load_flow(location);

    // Inflow: right-flowing from left, left-flowing from right, top-flowing from below, bottom-flowing from above
    terrain.water += TimeStep * (
        flow_left.right + flow_right.left + flow_bottom.top + flow_top.bottom
        - (flow.left + flow.right + flow.top + flow.bottom)
    );
    if (terrain.water < 1e-5) { terrain.water = 0.0; }

    store_terrain(location, terrain);
    store_flow(location, flow);
    store_velocity(location, load_velocity(location));
}

// Use updated water height to update velocity field map
@compute @workgroup_size(8, 8, 1)
fn _2_3_velocity_field(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    var left_pos = location.x - 1;
    if (left_pos < 0) { left_pos = Width - 1; }
    var right_pos = location.x + 1;
    if (right_pos >= Width) { right_pos = 0; }
    let location_left   = vec2(left_pos,  location.y);
    let location_right  = vec2(right_pos, location.y);
    let location_top    = location + vec2(0,  1);
    let location_bottom = location + vec2(0, -1);

    let flow       = load_flow(location);
    let flow_left  = load_flow(location_left);
    let flow_right = load_flow(location_right);
    var flow_top    = Flow(0.0, 0.0, 0.0, 0.0);
    if (location_top.y < Height) { flow_top = load_flow(location_top); }
    var flow_bottom = Flow(0.0, 0.0, 0.0, 0.0);
    if (location_bottom.y >= 0) { flow_bottom = load_flow(location_bottom); }

    let terrain     = load_terrain(location);
    let water_depth = max(terrain.water, 0.01);
    // Net rightward flux through left and right faces, divided by water depth (paper eqs. 8-9)
    let velocity_x = (flow_left.right - flow.left + flow.right - flow_right.left) / (2.0 * water_depth);
    // Net upward flux through bottom and top faces, divided by water depth (paper eqs. 8-9)
    let velocity_y = (flow_bottom.top - flow.bottom + flow.top - flow_top.bottom) / (2.0 * water_depth);

    store_terrain(location, terrain);
    store_flow(location, flow);
    if (terrain.water < 1e-5) {
        store_velocity(location, Velocity(0.0, 0.0));
    } else {
        store_velocity(location, Velocity(velocity_x, velocity_y));
    }
}

// Calculate carrying capacity using gradient and velocity, compare with held sediment and erode or deposit
@compute @workgroup_size(8, 8, 1)
fn _3_erosion_deposition(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var terrain = load_terrain(location);
    let flow    = load_flow(location);
    let vel     = load_velocity(location);

    // TODO: Add optional minimum carrying capacity for flat terrains
    let tilt = tilt_angle(location);
    let carrying_capacity = SedimentCapacity * sin(max(MinTiltAngle, tilt)) * length(vec2(vel.x, vel.y) * TimeStep);
    terrain.tilt = sin(tilt);

    if carrying_capacity > terrain.sediment {
        let dissolved = min(DissolvingRate * (carrying_capacity - terrain.sediment), terrain.height);
        terrain.height   -= dissolved;
        terrain.sediment += dissolved;
    } else {
        let dropped = DepositionRate * (terrain.sediment - carrying_capacity);
        terrain.height   += dropped;
        terrain.sediment -= dropped;
    }

    store_terrain(location, terrain);
    store_flow(location, flow);
    store_velocity(location, vel);
}

@compute @workgroup_size(8, 8, 1)
fn _4_sediment_transport(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var terrain = load_terrain(location);
    let flow    = load_flow(location);
    let vel     = load_velocity(location);

    let sampled = bilinear_sample_terrain(vec2(f32(location.x), f32(location.y)) - vec2(vel.x, vel.y) * TimeStep);
    terrain.sediment = sampled.sediment;

    store_terrain(location, terrain);
    store_flow(location, flow);
    store_velocity(location, vel);
}

@compute @workgroup_size(8, 8, 1)
fn _5_evaporation(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    var terrain = load_terrain(location);
    terrain.water *= 1.0 - Evaporation * TimeStep;
    if (location.y <= 0 || location.y >= Height - 1) {
        terrain.water = 0.0;
    }

    store_terrain(location, terrain);
    store_flow(location, load_flow(location));
    store_velocity(location, load_velocity(location));
}

fn tilt_angle(location: vec2<i32>) -> f32 {
    let x = location.x;
    let y = location.y;

    var left_x = x - 1;
    if (left_x < 0) { left_x = Width - 1; }
    var right_x = x + 1;
    if (right_x >= Width) { right_x = 0; }

    let h_center = load_terrain(location).height;
    let h_left   = load_terrain(vec2<i32>(left_x,  y)).height;
    let h_right  = load_terrain(vec2<i32>(right_x, y)).height;
    var h_up   = h_center;
    if (y + 1 < Height) { h_up   = load_terrain(vec2<i32>(x, y + 1)).height; }
    var h_down = h_center;
    if (y - 1 >= 0)     { h_down = load_terrain(vec2<i32>(x, y - 1)).height; }

    let dz_dx = (h_right - h_left) * 0.5;
    let dz_dy = (h_up - h_down) * 0.5;

    return atan(sqrt(dz_dx * dz_dx + dz_dy * dz_dy));
}

// Slope collapse: steep walls shed material downhill until slope is below AngleOfRepose.
// Each cell computes net gain/loss from all 4 neighbours using the frozen in_ state,
// so there are no write conflicts in the parallel dispatch.
@compute @workgroup_size(8, 8, 1)
fn _6_thermal_erosion(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    var terrain = load_terrain(location);
    let h = terrain.height;

    var left_x = location.x - 1;
    if (left_x < 0) { left_x = Width - 1; }
    var right_x = location.x + 1;
    if (right_x >= Width) { right_x = 0; }

    let h_left  = load_terrain(vec2<i32>(left_x,      location.y    )).height;
    let h_right = load_terrain(vec2<i32>(right_x,     location.y    )).height;
    let h_up    = select(h, load_terrain(vec2<i32>(location.x, location.y + 1)).height, location.y + 1 < Height);
    let h_down  = select(h, load_terrain(vec2<i32>(location.x, location.y - 1)).height, location.y - 1 >= 0);

    // Convert angle of repose to a height-difference-per-pixel threshold: tan(angle)
    // Pixel spacing is 1 world unit, so slope = dh/1 = dh, and atan(dh) is the actual angle.
    let talus_slope = tan(radians(AngleOfRepose));

    var delta = 0.0;
    let neighbors = array<f32, 4>(h_left, h_right, h_up, h_down);
    for (var i = 0; i < 4; i++) {
        let dh = h - neighbors[i];
        if (dh > talus_slope) {
            // We are steeper than stable; shed half the excess (neighbour's pass sheds the other half)
            delta -= ThermalErosionRate * (dh - talus_slope) * 0.5;
        } else if (dh < -talus_slope) {
            delta += ThermalErosionRate * (-dh - talus_slope) * 0.5;
        }
    }

    terrain.height = max(0.0, h + delta);
    store_terrain(location, terrain);
    store_flow(location, load_flow(location));
    store_velocity(location, load_velocity(location));
}

fn bilinear_sample_terrain(location: vec2<f32>) -> Terrain {
    let base = vec2<i32>(floor(location));
    let frac = fract(location);

    let bl = textureLoad(in_terrainmap, base);
    let br = textureLoad(in_terrainmap, base + vec2<i32>(1, 0));
    let tl = textureLoad(in_terrainmap, base + vec2<i32>(0, 1));
    let tr = textureLoad(in_terrainmap, base + vec2<i32>(1, 1));

    let lower = mix(bl, br, frac.x);
    let upper = mix(tl, tr, frac.x);
    let v = mix(lower, upper, frac.y);

    return Terrain(v.x, v.y, v.z, v.w);
}
