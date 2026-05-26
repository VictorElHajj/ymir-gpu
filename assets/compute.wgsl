@group(0) @binding(0) var in_terrainmap:        texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var in_flowmap_cardinal:  texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var in_flowmap_diagonal:  texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var in_velocitymap:       texture_storage_2d<rgba32float, read>;
@group(0) @binding(4) var out_terrainmap:       texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var out_flowmap_cardinal: texture_storage_2d<rgba32float, write>;
@group(0) @binding(6) var out_flowmap_diagonal: texture_storage_2d<rgba32float, write>;
@group(0) @binding(7) var out_velocitymap:      texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
}

// Texture parameters
const Width  = 2048;
const Height = 1024;

// Simulation parameters
const TimeStep                 = 0.00225; // Setting this too high will break the simulation, too low and it will be very slow.
const FlowPipeCrossSectionArea = 1.;     // Should probably always be one when pixel:grid is 1:1
const Gravity                  = 2000.;  // Higher values increase water flow (and therefore velocity, which affects carrying capacity etc)
const SedimentCapacity         = 0.2;   // Multipler on how much sediment the water can hold
const DissolvingRate           = .4;     // [0, 1], where 0 means no dissolving and 1 means terrain will always be dissolved to reach capacity
const DepositionRate           = .6;     // [0, 1], where 0 means no despositing and 1 means sediment will always be dropped to reach capacity
const Evaporation              = 0.05;    // [0, 1] The percent of water evaporated each update.
const Precipitation            = 0.02;   // [0, 1] Units of water added each update in all locations
const MinTiltAngle             = 0.001; // Minimum tilt angle floor for carrying capacity
const Friction                 = 0.99;   // Damps wave oscillations each step so gravity-driven flow dominates
const AngleOfRepose            = 35.0;   // Angle of repose in degrees, slopes steeper than this collapse
const ThermalErosionRate       = 0.5;    // Fraction of excess slope transferred per step [0, 1]
// Diagonal pipes are sqrt(2) longer, so the gravity-driven acceleration (g*Δh/l) is reduced by 1/sqrt(2).
// Used for outflow flux (paper eq. 2) and velocity projection (paper eqs. 8-9).
const DiagonalScale = 0.7071067811865476;

// Based on https://inria.hal.science/inria-00402079/document with side lengths and pipe length set to 1 and removed from calculations
// TODO: When adapting the map to be a sphere projected onto a square the side lengths should be added and set to the length between points on the sphere

// terrainmap channels:          height, water, sediment, tilt
// flowmap_cardinal channels:    left, right, top, bottom
// flowmap_diagonal channels:    top_left, top_right, bottom_left, bottom_right
// velocitymap channels:         x, y, -, -
// All four edges are draining boundaries (no outflow, water zeroed in evaporation step).

struct Terrain  { height: f32, water: f32, sediment: f32, tilt: f32 }
struct Flow     { left: f32, right: f32, top: f32, bottom: f32, top_left: f32, top_right: f32, bottom_left: f32, bottom_right: f32 }
struct Velocity { x: f32, y: f32 }
struct CellState { terrain: Terrain, flow: Flow, vel: Velocity }

fn load_cell(loc: vec2<i32>) -> CellState {
    let tv = textureLoad(in_terrainmap,       loc);
    let fc = textureLoad(in_flowmap_cardinal, loc);
    let fd = textureLoad(in_flowmap_diagonal, loc);
    let vv = textureLoad(in_velocitymap,      loc);
    return CellState(
        Terrain(tv.x, tv.y, tv.z, tv.w),
        Flow(fc.x, fc.y, fc.z, fc.w, fd.x, fd.y, fd.z, fd.w),
        Velocity(vv.x, vv.y),
    );
}

fn store_cell(loc: vec2<i32>, s: CellState) {
    textureStore(out_terrainmap,       loc, vec4(s.terrain.height, s.terrain.water, s.terrain.sediment, s.terrain.tilt));
    textureStore(out_flowmap_cardinal, loc, vec4(s.flow.left, s.flow.right, s.flow.top, s.flow.bottom));
    textureStore(out_flowmap_diagonal, loc, vec4(s.flow.top_left, s.flow.top_right, s.flow.bottom_left, s.flow.bottom_right));
    textureStore(out_velocitymap,      loc, vec4(s.vel.x, s.vel.y, 0.0, 0.0));
}

// Water added from rain
@compute @workgroup_size(8, 8, 1)
fn _1_precipitation(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var cell = load_cell(location);
    cell.terrain.water += Precipitation * TimeStep;
    store_cell(location, cell);
}

// Flux map: each location stores the water outflow to all 8 neighbors
@compute @workgroup_size(8, 8, 1)
fn _2_1_outflow_flux(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let left_x   = location.x - 1;
    let right_x  = location.x + 1;
    let top_y    = location.y + 1;
    let bottom_y = location.y - 1;

    // Boundary flags for all four edges — no outflow permitted past the grid edge
    let left_ok   = left_x   >= 0;
    let right_ok  = right_x  <  Width;
    let top_ok    = top_y    <  Height;
    let bottom_ok = bottom_y >= 0;

    let cell    = load_cell(location);
    let surface = cell.terrain.height + cell.terrain.water;

    // Terrain surface for all 8 neighbors (OOB reads return 0 via WGSL guarantee)
    let t_left         = load_cell(vec2(left_x,       location.y)).terrain;
    let t_right        = load_cell(vec2(right_x,      location.y)).terrain;
    let t_top          = load_cell(vec2(location.x,   top_y     )).terrain;
    let t_bottom       = load_cell(vec2(location.x,   bottom_y  )).terrain;
    let t_top_left     = load_cell(vec2(left_x,        top_y    )).terrain;
    let t_top_right    = load_cell(vec2(right_x,       top_y    )).terrain;
    let t_bottom_left  = load_cell(vec2(left_x,        bottom_y )).terrain;
    let t_bottom_right = load_cell(vec2(right_x,       bottom_y )).terrain;

    let s_left         = t_left.height         + t_left.water;
    let s_right        = t_right.height        + t_right.water;
    let s_top          = t_top.height          + t_top.water;
    let s_bottom       = t_bottom.height       + t_bottom.water;
    let s_top_left     = t_top_left.height     + t_top_left.water;
    let s_top_right    = t_top_right.height    + t_top_right.water;
    let s_bottom_left  = t_bottom_left.height  + t_bottom_left.water;
    let s_bottom_right = t_bottom_right.height + t_bottom_right.water;

    let f    = cell.flow;
    let g_dt = TimeStep * FlowPipeCrossSectionArea * Gravity;

    // Cardinal outflows (paper eq. 2, l=1); masked to 0 at grid edges
    let new_left   = select(0., max(0., f.left   * Friction + g_dt * (surface - s_left)),   left_ok);
    let new_right  = select(0., max(0., f.right  * Friction + g_dt * (surface - s_right)),  right_ok);
    let new_top    = select(0., max(0., f.top    * Friction + g_dt * (surface - s_top)),    top_ok);
    let new_bottom = select(0., max(0., f.bottom * Friction + g_dt * (surface - s_bottom)), bottom_ok);
    // Diagonal outflows (paper eq. 2, l=sqrt(2)); masked at any adjacent edge
    let new_top_left     = select(0., max(0., f.top_left     * Friction + g_dt * DiagonalScale * (surface - s_top_left)),     top_ok && left_ok);
    let new_top_right    = select(0., max(0., f.top_right    * Friction + g_dt * DiagonalScale * (surface - s_top_right)),    top_ok && right_ok);
    let new_bottom_left  = select(0., max(0., f.bottom_left  * Friction + g_dt * DiagonalScale * (surface - s_bottom_left)),  bottom_ok && left_ok);
    let new_bottom_right = select(0., max(0., f.bottom_right * Friction + g_dt * DiagonalScale * (surface - s_bottom_right)), bottom_ok && right_ok);

    // Scale down all outflows so total never exceeds available water (paper eq. 4)
    let total = new_left + new_right + new_top + new_bottom
              + new_top_left + new_top_right + new_bottom_left + new_bottom_right;
    var k = 1.0;
    if (total > 0.0) { k = min(1., cell.terrain.water / (total * TimeStep)); }

    var out = cell;
    out.flow = Flow(
        new_left * k, new_right * k, new_top * k, new_bottom * k,
        new_top_left * k, new_top_right * k, new_bottom_left * k, new_bottom_right * k,
    );
    store_cell(location, out);
}

// Use updated flux map to update water height
@compute @workgroup_size(8, 8, 1)
fn _2_2_water_height(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let left_x   = location.x - 1;
    let right_x  = location.x + 1;
    let top_y    = location.y + 1;
    let bottom_y = location.y - 1;

    // Neighbor flows — OOB reads return 0, which is correct: no inflow from outside the grid
    let f_left         = load_cell(vec2(left_x,      location.y)).flow;
    let f_right        = load_cell(vec2(right_x,     location.y)).flow;
    let f_top          = load_cell(vec2(location.x,  top_y     )).flow;
    let f_bottom       = load_cell(vec2(location.x,  bottom_y  )).flow;
    let f_top_left     = load_cell(vec2(left_x,       top_y    )).flow;
    let f_top_right    = load_cell(vec2(right_x,      top_y    )).flow;
    let f_bottom_left  = load_cell(vec2(left_x,       bottom_y )).flow;
    let f_bottom_right = load_cell(vec2(right_x,      bottom_y )).flow;

    var cell = load_cell(location);
    let f = cell.flow;

    // Paper eq. 6–7: net volume change = dt * (all inflows − all outflows)
    // Inflow from direction D = neighbor's flow in the opposite direction (pointing toward this cell)
    cell.terrain.water += TimeStep * (
        f_left.right        + f_right.left       + f_bottom.top          + f_top.bottom
        + f_top_left.bottom_right + f_top_right.bottom_left
        + f_bottom_left.top_right + f_bottom_right.top_left
        - (f.left + f.right + f.top + f.bottom
           + f.top_left + f.top_right + f.bottom_left + f.bottom_right)
    );
    if (cell.terrain.water < 1e-5) { cell.terrain.water = 0.0; }

    store_cell(location, cell);
}

// Use updated water height to update velocity field
@compute @workgroup_size(8, 8, 1)
fn _2_3_velocity_field(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let left_x   = location.x - 1;
    let right_x  = location.x + 1;
    let top_y    = location.y + 1;
    let bottom_y = location.y - 1;

    let f_left         = load_cell(vec2(left_x,      location.y)).flow;
    let f_right        = load_cell(vec2(right_x,     location.y)).flow;
    let f_top          = load_cell(vec2(location.x,  top_y     )).flow;
    let f_bottom       = load_cell(vec2(location.x,  bottom_y  )).flow;
    let f_top_left     = load_cell(vec2(left_x,       top_y    )).flow;
    let f_top_right    = load_cell(vec2(right_x,      top_y    )).flow;
    let f_bottom_left  = load_cell(vec2(left_x,       bottom_y )).flow;
    let f_bottom_right = load_cell(vec2(right_x,      bottom_y )).flow;

    var cell = load_cell(location);
    let f = cell.flow;
    let water_depth = max(cell.terrain.water, 0.01);

    // Paper eqs. 8-9, extended to 8 pipes.
    // Each diagonal pipe contributes its x/y projection (cos45° = sin45° = DiagonalScale) to velocity.
    let vel_x = (
        // Cardinal: net rightward flux across left/right faces
        f_left.right - f.left + f.right - f_right.left
        // Diagonal: x-component (±1/√2) of each diagonal pipe's flux
        + (f_top_left.bottom_right + f_bottom_left.top_right - f_top_right.bottom_left - f_bottom_right.top_left
           + f.top_right + f.bottom_right - f.top_left - f.bottom_left) * DiagonalScale
    ) / (2.0 * water_depth);

    let vel_y = (
        // Cardinal: net upward flux across bottom/top faces
        f_bottom.top - f.bottom + f.top - f_top.bottom
        // Diagonal: y-component (±1/√2) of each diagonal pipe's flux
        + (f_bottom_left.top_right + f_bottom_right.top_left - f_top_left.bottom_right - f_top_right.bottom_left
           + f.top_left + f.top_right - f.bottom_left - f.bottom_right) * DiagonalScale
    ) / (2.0 * water_depth);

    if (cell.terrain.water < 1e-5) {
        cell.vel = Velocity(0.0, 0.0);
    } else {
        cell.vel = Velocity(vel_x, vel_y);
    }
    store_cell(location, cell);
}

// Calculate carrying capacity using gradient and velocity, compare with held sediment and erode or deposit
@compute @workgroup_size(8, 8, 1)
fn _3_erosion_deposition(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var cell = load_cell(location);

    // TODO: Add optional minimum carrying capacity for flat terrains
    let tilt = tilt_angle(location);
    let carrying_capacity = SedimentCapacity * sin(max(MinTiltAngle, tilt)) * length(vec2(cell.vel.x, cell.vel.y) * TimeStep);
    cell.terrain.tilt = sin(tilt);

    if carrying_capacity > cell.terrain.sediment {
        let dissolved = min(DissolvingRate * (carrying_capacity - cell.terrain.sediment), cell.terrain.height);
        cell.terrain.height   -= dissolved;
        cell.terrain.sediment += dissolved;
    } else {
        let dropped = DepositionRate * (cell.terrain.sediment - carrying_capacity);
        cell.terrain.height   += dropped;
        cell.terrain.sediment -= dropped;
    }

    store_cell(location, cell);
}

@compute @workgroup_size(8, 8, 1)
fn _4_sediment_transport(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var cell = load_cell(location);

    let sampled = bilinear_sample_terrain(vec2(f32(location.x), f32(location.y)) - vec2(cell.vel.x, cell.vel.y) * TimeStep);
    cell.terrain.sediment = sampled.sediment;

    store_cell(location, cell);
}

@compute @workgroup_size(8, 8, 1)
fn _5_evaporation(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var cell = load_cell(location);

    cell.terrain.water *= 1.0 - Evaporation * TimeStep;
    // Drain all four edges so water exits the simulation rather than accumulating at boundaries
    if (location.x <= 0 || location.x >= Width - 1 || location.y <= 0 || location.y >= Height - 1) {
        cell.terrain.water = 0.0;
    }

    store_cell(location, cell);
}

fn tilt_angle(location: vec2<i32>) -> f32 {
    let x = location.x;
    let y = location.y;

    let left_x  = x - 1;
    let right_x = x + 1;

    let h_center = load_cell(location).terrain.height;
    // At the grid edge, use h_center so the gradient is zero rather than wrapping
    let h_left  = select(h_center, load_cell(vec2<i32>(left_x,  y)).terrain.height, left_x  >= 0);
    let h_right = select(h_center, load_cell(vec2<i32>(right_x, y)).terrain.height, right_x <  Width);
    var h_up   = h_center;
    if (y + 1 < Height) { h_up   = load_cell(vec2<i32>(x, y + 1)).terrain.height; }
    var h_down = h_center;
    if (y - 1 >= 0)     { h_down = load_cell(vec2<i32>(x, y - 1)).terrain.height; }

    let dz_dx = (h_right - h_left) * 0.5;
    let dz_dy = (h_up - h_down) * 0.5;

    return atan(sqrt(dz_dx * dz_dx + dz_dy * dz_dy));
}

// Slope collapse: steep walls shed material downhill until slope is below AngleOfRepose.
// Each cell computes net gain/loss from all 8 neighbours using the frozen in_ state,
// so there are no write conflicts in the parallel dispatch.
@compute @workgroup_size(8, 8, 1)
fn _6_thermal_erosion(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    var cell = load_cell(location);
    let h = cell.terrain.height;

    let left_x  = location.x - 1;
    let right_x = location.x + 1;
    let left_ok  = left_x  >= 0;
    let right_ok = right_x <  Width;

    // Cardinal neighbors (distance 1); use h at grid edge so no artificial slope
    let h_left  = select(h, load_cell(vec2<i32>(left_x,      location.y    )).terrain.height, left_ok);
    let h_right = select(h, load_cell(vec2<i32>(right_x,     location.y    )).terrain.height, right_ok);
    let h_up    = select(h, load_cell(vec2<i32>(location.x, location.y + 1)).terrain.height, location.y + 1 < Height);
    let h_down  = select(h, load_cell(vec2<i32>(location.x, location.y - 1)).terrain.height, location.y - 1 >= 0);
    // Diagonal neighbors (distance sqrt(2)); masked at any adjacent edge
    let h_top_left     = select(h, load_cell(vec2<i32>(left_x,  location.y + 1)).terrain.height, location.y + 1 < Height && left_ok);
    let h_top_right    = select(h, load_cell(vec2<i32>(right_x, location.y + 1)).terrain.height, location.y + 1 < Height && right_ok);
    let h_bottom_left  = select(h, load_cell(vec2<i32>(left_x,  location.y - 1)).terrain.height, location.y - 1 >= 0     && left_ok);
    let h_bottom_right = select(h, load_cell(vec2<i32>(right_x, location.y - 1)).terrain.height, location.y - 1 >= 0     && right_ok);

    // Convert angle of repose to height-difference threshold at each distance.
    // Cardinal spacing = 1, diagonal spacing = sqrt(2), so diagonal threshold scales up.
    let talus_cardinal = tan(radians(AngleOfRepose));
    let talus_diagonal = talus_cardinal * sqrt(2.0);

    var delta = 0.0;

    let cardinal_neighbors = array<f32, 4>(h_left, h_right, h_up, h_down);
    for (var i = 0; i < 4; i++) {
        let dh = h - cardinal_neighbors[i];
        if (dh > talus_cardinal) {
            delta -= ThermalErosionRate * (dh - talus_cardinal) * 0.5;
        } else if (dh < -talus_cardinal) {
            delta += ThermalErosionRate * (-dh - talus_cardinal) * 0.5;
        }
    }

    let diagonal_neighbors = array<f32, 4>(h_top_left, h_top_right, h_bottom_left, h_bottom_right);
    for (var i = 0; i < 4; i++) {
        let dh = h - diagonal_neighbors[i];
        if (dh > talus_diagonal) {
            delta -= ThermalErosionRate * (dh - talus_diagonal) * 0.5;
        } else if (dh < -talus_diagonal) {
            delta += ThermalErosionRate * (-dh - talus_diagonal) * 0.5;
        }
    }

    cell.terrain.height = max(0.0, h + delta);
    store_cell(location, cell);
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
    let v     = mix(lower, upper, frac.y);

    return Terrain(v.x, v.y, v.z, v.w);
}
