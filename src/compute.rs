use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{binding_types::texture_storage_2d, *},
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
    },
};
use std::borrow::Cow;

use crate::{TERRAINMAP_HEIGHT, TERRAINMAP_WIDTH};

const SHADER_ASSET_PATH: &str = "compute.wgsl";
const SIZE: (u32, u32) = (TERRAINMAP_WIDTH, TERRAINMAP_HEIGHT);
const WORKGROUP_SIZE: u32 = 1;

pub struct TerrainComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct TerrainComptueLabel;

impl Plugin for TerrainComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the image resource from the main world into the render world
        // for operation on by the compute shader
        app.add_plugins(ExtractResourcePlugin::<ComputeTerrainImages>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_bind_group
                .in_set(RenderSet::PrepareBindGroups)
                .run_if(
                    not(resource_exists::<ComputeTerrainBindGroups>)
                        .and(resource_exists::<ComputeTerrainImages>),
                ),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(TerrainComptueLabel, TerrainCompute::default());
        render_graph.add_node_edge(TerrainComptueLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ComputeTerrainPipeline>();
    }
}

#[derive(Resource, Clone, ExtractResource)]
pub struct ComputeTerrainImages {
    pub terrainmap_a: Handle<Image>,
    pub flowmap_a: Handle<Image>,
    pub velocitymap_a: Handle<Image>,
    pub terrainmap_b: Handle<Image>,
    pub flowmap_b: Handle<Image>,
    pub velocitymap_b: Handle<Image>,
}

#[derive(Resource)]
struct ComputeTerrainBindGroups([BindGroup; 2]);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<ComputeTerrainPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    comptue_terrain_images: Res<ComputeTerrainImages>,
    render_device: Res<RenderDevice>,
) {
    let terrainmap_a = gpu_images
        .get(&comptue_terrain_images.terrainmap_a)
        .unwrap();
    let flowmap_a = gpu_images.get(&comptue_terrain_images.flowmap_a).unwrap();
    let velocitymap_a = gpu_images
        .get(&comptue_terrain_images.velocitymap_a)
        .unwrap();
    let terrainmap_b = gpu_images
        .get(&comptue_terrain_images.terrainmap_b)
        .unwrap();
    let flowmap_b = gpu_images.get(&comptue_terrain_images.flowmap_b).unwrap();
    let velocitymap_b = gpu_images
        .get(&comptue_terrain_images.velocitymap_b)
        .unwrap();
    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &terrainmap_a.texture_view,
            &flowmap_a.texture_view,
            &velocitymap_a.texture_view,
            &terrainmap_b.texture_view,
            &flowmap_b.texture_view,
            &velocitymap_b.texture_view,
        )),
    );
    let bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &terrainmap_b.texture_view,
            &flowmap_b.texture_view,
            &velocitymap_b.texture_view,
            &terrainmap_a.texture_view,
            &flowmap_a.texture_view,
            &velocitymap_a.texture_view,
        )),
    );
    commands.insert_resource(ComputeTerrainBindGroups([bind_group_0, bind_group_1]));
}

#[derive(Resource)]
struct ComputeTerrainPipeline {
    texture_bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for ComputeTerrainPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let texture_bind_group_layout = render_device.create_bind_group_layout(
            "ComputeTerrainImages",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );
        let shader = world.load_asset(SHADER_ASSET_PATH);
        let pipeline_cache = world.resource::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("init"),
            zero_initialize_workgroup_memory: false,
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("update"),
            zero_initialize_workgroup_memory: false,
        });

        ComputeTerrainPipeline {
            texture_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

enum ComputeTerrainState {
    Loading,
    Init,
    Update(usize),
}

struct TerrainCompute {
    state: ComputeTerrainState,
}

impl Default for TerrainCompute {
    fn default() -> Self {
        Self {
            state: ComputeTerrainState::Loading,
        }
    }
}

impl render_graph::Node for TerrainCompute {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<ComputeTerrainPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            ComputeTerrainState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = ComputeTerrainState::Init;
                    }
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {}
                }
            }
            ComputeTerrainState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = ComputeTerrainState::Update(1);
                }
            }
            ComputeTerrainState::Update(0) => {
                self.state = ComputeTerrainState::Update(1);
            }
            ComputeTerrainState::Update(1) => {
                self.state = ComputeTerrainState::Update(0);
            }
            ComputeTerrainState::Update(_) => unreachable!(),
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_groups = &world.get_resource::<ComputeTerrainBindGroups>();
        if let Some(bind_groups) = bind_groups {
            let pipeline_cache = world.resource::<PipelineCache>();
            let pipeline = world.resource::<ComputeTerrainPipeline>();

            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());

            // select the pipeline based on the current state
            match self.state {
                ComputeTerrainState::Loading => {}
                ComputeTerrainState::Init => {
                    let init_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.init_pipeline)
                        .unwrap();
                    pass.set_bind_group(0, &bind_groups.0[0], &[]);
                    pass.set_pipeline(init_pipeline);
                    pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
                }
                ComputeTerrainState::Update(index) => {
                    if let Some(update_pipeline) =
                        pipeline_cache.get_compute_pipeline(pipeline.update_pipeline)
                    {
                        pass.set_bind_group(0, &bind_groups.0[index], &[]);
                        pass.set_pipeline(update_pipeline);
                        pass.dispatch_workgroups(
                            SIZE.0 / WORKGROUP_SIZE,
                            SIZE.1 / WORKGROUP_SIZE,
                            1,
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
