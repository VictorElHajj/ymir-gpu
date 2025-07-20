This is a implementation of the following paper using rust and bevy: Grid based GPU implementation of hydraulic erosion https://inria.hal.science/inria-00402079/document

Future TODOs:
- Loop around sides
- Preserve total amount of water (evaporation is added to rain) to maintain oceans

2. Hardcoded prevailing winds
3. Evoporation from 1. moves according to 2. Only dropping water depending on air pressure / terrain (rain shadows)
4. Moisture map

Biome map using Whittaker biome map

The implemented paper has several issues which can be fixed by using this paper in stead: https://old.cescg.org/CESCG-2011/papers/TUBudapest-Jako-Balazs.pdf
