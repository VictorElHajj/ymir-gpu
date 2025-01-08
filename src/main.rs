use bevy::{asset, math::vec3, prelude::*, sprite::Material2dPlugin, window::WindowMode};
use map::MapMaterial;
mod map;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Ymir GPU".into(),
                        resolution: (2048., 1024.).into(),
                        present_mode: bevy::window::PresentMode::Immediate,
                        ..Default::default()
                    }),
                    ..default()
                })
                .build(),
        )
        .add_plugins(Material2dPlugin::<MapMaterial>::default())
        .add_systems(Startup, setup)
        .run();
}

// Load height map, create material and set up fullscreen quad
fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<MapMaterial>>,
) {
    let heightmap: Handle<Image> = asset_server.load("heightmap.png");
    let quad_handle = meshes.add(Rectangle::new(2., 1.));
    let material_handle = materials.add(MapMaterial { heightmap });
    commands.spawn((
        Mesh2d(quad_handle.clone()),
        MeshMaterial2d(material_handle),
        Transform::from_xyz(0.0, 0.0, 0.0).with_scale(vec3(1024., 1024., 1.)),
    ));
    commands.spawn(Camera2d);
}
