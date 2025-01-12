@group(0) @binding(0) var in_heightmap: texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var in_flowmap: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var in_velocitymap: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var out_heightmap: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var out_flowmap: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var out_velocitymap: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(1, 1, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
}

@compute @workgroup_size(1, 1, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let location_left = location + vec2(-1, 0);
    let location_top = location + vec2(0, 1);
    let location_right = location + vec2(1, 0);
    let location_bottom = location + vec2(1, 0);
    let heightmap = textureLoad(in_heightmap, location);
    let heightmap_left = textureLoad(in_heightmap, location_left);
    let heightmap_top = textureLoad(in_heightmap, location_top);
    let heightmap_right = textureLoad(in_heightmap, location_right);
    let heightmap_bottom = textureLoad(in_heightmap, location_bottom);
    textureStore(out_heightmap, location, heightmap);

    // let n_alive = count_alive(location);

    // var alive: bool;
    // if (n_alive == 3) {
    //     alive = true;
    // } else if (n_alive == 2) {
    //     let currently_alive = is_alive(location, 0, 0);
    //     alive = bool(currently_alive);
    // } else {
    //     alive = false;
    // }
    // let color = vec4<f32>(f32(alive));

    // textureStore(output, location, color);
}