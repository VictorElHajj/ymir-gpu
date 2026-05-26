# Ymir-GPU

This is a implementation of *Fast Hydraulic Erosion Simulation and Visualization on
GPU* by Xing Mei, Philippe Decaudin and Bao-Gang Hu. 
https://inria.hal.science/inria-00402079/document

This implementation is written in Rust using Bevy and performs a hydraulic erosion simulation using multiple pass computer shaders.

I planned to expand this to include some basic initial map generation simulating tectonic plates and then adding a wind, rain and climate / biome simulation step following the hydraulic erosion step.

### TODO and notes to self

1. Grid based GPU implementation of hydraulic erosion https://inria.hal.science/inria-00402079/document

- Loop around sides
- Preserve total amount of water (evaporation is added to rain) to maintain oceans

2. Hardcoded prevailing winds
3. Evoporation from 1. moves according to 2. Only dropping water depending on air pressure / terrain (rain shadows)
4. Moisture map

Biome map using Whittaker biome map