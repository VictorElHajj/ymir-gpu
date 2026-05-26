use bevy::{
    asset::RenderAssetUsages,
    math::vec3,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    sprite::Material2dPlugin,
};
use compute::{ComputeTerrainImages, TerrainComputePlugin};
use map_material::MapMaterial;
use noise::{Fbm, NoiseFn, Perlin};
mod compute;
mod map_material;

const WINDOW_WIDTH: u32 = 2048;
const WINDOW_HEIGHT: u32 = 1024;
const TERRAINMAP_WIDTH: u32 = 4096;
const TERRAINMAP_HEIGHT: u32 = 2048;

// Toggle: true = procedural FBM noise, false = load perlin2.png
const USE_FBM: bool = false;

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
            Material2dPlugin::<MapMaterial>::default(),
            TerrainComputePlugin,
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

fn load_terrainmap(
    asset_server: Res<AssetServer>,
    mut height_map_texture_handle: ResMut<HeightMapTextureHandle>,
) {
    if !USE_FBM {
        height_map_texture_handle.0 = asset_server.load("perlin2.png");
    }
}

fn check_for_loaded(
    height_map_texture_handle: Res<HeightMapTextureHandle>,
    images: Res<Assets<Image>>,
    mut next_state: ResMut<NextState<LoadingState>>,
) {
    if USE_FBM || images.get(&height_map_texture_handle.0).is_some() {
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
            if h < min_h { min_h = h; }
            if h > max_h { max_h = h; }
        }
    }

    let range = (max_h - min_h) as f32;
    for h in &mut heights {
        *h = (*h - min_h as f32) / range;
    }
    heights
}

fn setup_sim(
    mut commands: Commands,
    height_map_texture_handle: Res<HeightMapTextureHandle>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<MapMaterial>>,
) {
    let heights: Vec<f32> = if USE_FBM {
        generate_heightmap()
    } else {
        let heightmap = images
            .get(&height_map_texture_handle.0)
            .expect("Heightmap not found with handle");
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
            let h = heights[(y * TERRAINMAP_WIDTH + x) as usize];
            terrainmap
                .set_color_at(x, y, Color::linear_rgba(h, 0.0, 0.0, 0.0))
                .ok();
        }
    }
    let terrainmap_a = images.add(terrainmap.clone());
    let terrainmap_b = images.add(terrainmap);

    let mut flowmap = Image::new_fill(
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
    flowmap.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let flowmap_a = images.add(flowmap.clone());
    let flowmap_b = images.add(flowmap);

    let mut velocitymap = Image::new_fill(
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
    velocitymap.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let velocitymap_a = images.add(velocitymap.clone());
    let velocitymap_b = images.add(velocitymap);

    let quad_handle = meshes.add(Rectangle::new(2., 1.));
    let material_handle = materials.add(MapMaterial {
        terrainmap: terrainmap_a.clone(),
        flowmap: flowmap_a.clone(),
        velocitymap: velocitymap_a.clone(),
    });
    commands.insert_resource(ComputeTerrainImages {
        terrainmap_a,
        flowmap_a,
        velocitymap_a,
        terrainmap_b,
        flowmap_b,
        velocitymap_b,
    });
    commands.spawn((
        Mesh2d(quad_handle),
        MeshMaterial2d(material_handle),
        Transform::from_xyz(0.0, 0.0, 0.0).with_scale(vec3(1024., 1024., 1.)),
    ));
    commands.spawn(Camera2d);
}
