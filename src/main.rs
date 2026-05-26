use bevy::{
    asset::RenderAssetUsages,
    pbr::MaterialPlugin,
    prelude::*,
    render::{
        mesh::Indices,
        render_resource::{
            Extent3d, PrimitiveTopology, TextureDimension, TextureFormat, TextureUsages,
        },
    },
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use compute::{ComputeTerrainImages, TerrainComputePlugin};
use noise::{Fbm, NoiseFn, Perlin};
use terrain_material::TerrainMaterial;
use water_material::WaterMaterial;
mod compute;
mod terrain_material;
mod water_material;

const WINDOW_WIDTH: u32 = 2048;
const WINDOW_HEIGHT: u32 = 1024;
const TERRAINMAP_WIDTH: u32 = 4096;
const TERRAINMAP_HEIGHT: u32 = 2048;
const TERRAIN_HEIGHT_SCALE: f32 = 200.0;

// Toggle: true = procedural FBM noise, false = load perlin.png
const USE_FBM: bool = true;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Ymir GPU".into(),
                        resolution: (WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32).into(),
                        present_mode: bevy::window::PresentMode::Immediate,
                        ..Default::default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            MaterialPlugin::<TerrainMaterial>::default(),
            MaterialPlugin::<WaterMaterial>::default(),
            TerrainComputePlugin,
            PanOrbitCameraPlugin,
        ))
        .init_state::<LoadingState>()
        .init_resource::<HeightMapTextureHandle>()
        .add_systems(OnEnter(LoadingState::Loading), load_terrainmap)
        .add_systems(
            Update,
            check_for_loaded.run_if(in_state(LoadingState::Loading)),
        )
        .add_systems(OnEnter(LoadingState::Loaded), setup_sim)
        .run();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, States, Default)]
pub enum LoadingState {
    #[default]
    Loading,
    Loaded,
}

#[derive(Resource, Default)]
struct HeightMapTextureHandle(Handle<Image>);

fn load_terrainmap(asset_server: Res<AssetServer>, mut handle: ResMut<HeightMapTextureHandle>) {
    if !USE_FBM {
        handle.0 = asset_server.load("perlin.png");
    }
}

fn check_for_loaded(
    handle: Res<HeightMapTextureHandle>,
    images: Res<Assets<Image>>,
    mut next_state: ResMut<NextState<LoadingState>>,
) {
    if USE_FBM || images.get(&handle.0).is_some() {
        next_state.set(LoadingState::Loaded);
    }
}

fn generate_heightmap() -> Vec<f32> {
    let fbm = Fbm::<Perlin>::new(42);
    let scale = 8.0 / TERRAINMAP_WIDTH as f64;
    let mut heights = vec![0f32; (TERRAINMAP_WIDTH * TERRAINMAP_HEIGHT) as usize];
    let mut min_h = f64::MAX;
    let mut max_h = f64::MIN;
    for y in 0..TERRAINMAP_HEIGHT {
        for x in 0..TERRAINMAP_WIDTH {
            let h = fbm.get([x as f64 * scale, y as f64 * scale]);
            heights[(y * TERRAINMAP_WIDTH + x) as usize] = h as f32;
            if h < min_h {
                min_h = h;
            }
            if h > max_h {
                max_h = h;
            }
        }
    }
    let range = (max_h - min_h) as f32;
    for h in &mut heights {
        *h = (*h - min_h as f32) / range;
    }
    heights
}

fn create_terrain_mesh() -> Mesh {
    let w = TERRAINMAP_WIDTH;
    let h = TERRAINMAP_HEIGHT;
    let x_quads = w - 1;
    let z_quads = h - 1;

    let vert_count = (w * h) as usize;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vert_count);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(vert_count);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(vert_count);
    let mut indices: Vec<u32> = Vec::with_capacity((x_quads * z_quads * 6) as usize);

    for z in 0..h {
        for x in 0..w {
            // World XZ: pixel (0,0) → (-W/2, -H/2), pixel (W-1,H-1) → (W/2-1, H/2-1)
            positions.push([x as f32 - w as f32 / 2.0, 0.0, z as f32 - h as f32 / 2.0]);
            normals.push([0.0, 1.0, 0.0]);
            // UV samples the centre of each texel
            uvs.push([(x as f32 + 0.5) / w as f32, (z as f32 + 0.5) / h as f32]);
        }
    }

    for z in 0..z_quads {
        for x in 0..x_quads {
            let tl = z * w + x;
            let tr = tl + 1;
            let bl = tl + w;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn setup_sim(
    mut commands: Commands,
    handle: Res<HeightMapTextureHandle>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
    mut water_materials: ResMut<Assets<WaterMaterial>>,
) {
    let raw_heights: Vec<f32> = if USE_FBM {
        generate_heightmap()
    } else {
        let heightmap = images.get(&handle.0).expect("Heightmap not found");
        (0..TERRAINMAP_HEIGHT)
            .flat_map(|y| (0..TERRAINMAP_WIDTH).map(move |x| (x, y)))
            .map(|(x, y)| heightmap.get_color_at(x, y).unwrap().to_linear().red)
            .collect()
    };

    let mut terrainmap = Image::new_fill(
        Extent3d {
            width: TERRAINMAP_WIDTH,
            height: TERRAINMAP_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        [0.0f32, 0.0, 0.0, 0.0].map(f32::to_le_bytes).as_flattened(),
        TextureFormat::Rgba32Float,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    terrainmap.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    for y in 0..TERRAINMAP_HEIGHT {
        for x in 0..TERRAINMAP_WIDTH {
            let h = raw_heights[(y * TERRAINMAP_WIDTH + x) as usize] * TERRAIN_HEIGHT_SCALE;
            terrainmap
                .set_color_at(x, y, Color::linear_rgba(h, 0.0, 0.0, 0.0))
                .ok();
        }
    }
    let terrainmap_a = images.add(terrainmap.clone());
    let terrainmap_b = images.add(terrainmap);

    let zero_map = Image::new_fill(
        Extent3d {
            width: TERRAINMAP_WIDTH,
            height: TERRAINMAP_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        [0.0f32, 0.0, 0.0, 0.0].map(f32::to_le_bytes).as_flattened(),
        TextureFormat::Rgba32Float,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    let storage_usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let mut flowmap_cardinal = zero_map.clone();
    flowmap_cardinal.texture_descriptor.usage = storage_usage;
    let flowmap_cardinal_a = images.add(flowmap_cardinal.clone());
    let flowmap_cardinal_b = images.add(flowmap_cardinal);

    let mut flowmap_diagonal = zero_map.clone();
    flowmap_diagonal.texture_descriptor.usage = storage_usage;
    let flowmap_diagonal_a = images.add(flowmap_diagonal.clone());
    let flowmap_diagonal_b = images.add(flowmap_diagonal);

    let mut velocitymap = zero_map;
    velocitymap.texture_descriptor.usage = storage_usage;
    let velocitymap_a = images.add(velocitymap.clone());
    let velocitymap_b = images.add(velocitymap);

    let grid = meshes.add(create_terrain_mesh());

    commands.spawn((
        Mesh3d(grid.clone()),
        MeshMaterial3d(terrain_materials.add(TerrainMaterial {
            terrainmap: terrainmap_a.clone(),
        })),
    ));

    commands.spawn((
        Mesh3d(grid),
        MeshMaterial3d(water_materials.add(WaterMaterial {
            terrainmap: terrainmap_a.clone(),
        })),
    ));

    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_3,
            near: 1.0,
            far: 100_000.0,
            ..default()
        }),
        Transform::from_xyz(0.0, 2000.0, 3000.0).looking_at(Vec3::new(0.0, 200.0, 0.0), Vec3::Y),
        PanOrbitCamera::default(),
    ));

    commands.insert_resource(ComputeTerrainImages {
        terrainmap_a,
        flowmap_cardinal_a,
        flowmap_diagonal_a,
        velocitymap_a,
        terrainmap_b,
        flowmap_cardinal_b,
        flowmap_diagonal_b,
        velocitymap_b,
    });
}
