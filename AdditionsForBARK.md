# Overview of STRIVE Extension Files, Classes and Functions
#### Tools for working with OpenDrive maps, INTERACTION data and BARK simulator in STRIVE environment


## I. For using map and scenario data (input) with STRIVE model and scenario generation:
**`XodrMapEnv`** (adapted from `MapEnv`): class

in: `~/STRIVE/src/datasets/xodr_map_env.py`

- Creates binary rasterized map image of OpenDrive map to be used with STRIVE model.
The map image consists of four layers: drivable area, road dividers, lane dividers and carpark area. (The carpark area is set to 0 by default.)

- **`RasterizePolygon (polygon, x_grid, y_grid, grid_size=None)`**: Creates rasterized image of BARK polygon (Polygon2d) based on given raster grid.

- **`RasterizeLine (line, x_grid, y_grid, grid_size=None)`**: Creates rasterized image of BARK line (Line2d) based on given raster grid.

- **`BresenhamLine(x0, y0, x1, y1)`**: Rasterizes line between two points (x0, y0) and (x1, y1) using Bresenham line algorithm.

- **`PlotRasterMap(x_grid, y_grid, raster_map, grid_size, layer_name)`**: Plots binary image of one r- asterized map layer.


**InteractionDataset** (adapted from `NuScenesDataset`): subclass of `Dataset` class

in: `~/STRIVE/src/datasets/interaction_dataset.py`

- Creates dataset by taking BARK scenarios (generated from INTERACTION data) and map environment as input and storing selected information about ego and other agents and the map in dataset.

- **`compile_bark_scenarios (bark_scenario_list)`**: Extracts information about ego and other agent (including trajectories) and map from scenario list and returns them in dictionaries.


## II.  For using scenarios from STRIVE with BARK framework:

**`StriveDatasetScenarioGeneration`** (adapted from `InteractionDatasetScenarioGeneration`): subclass of `ScenarioGeneration` class

in: `~/STRIVE/src/utils/strive_dataset_processing/strive_dataset_scenario_generation.py`

- Creates a list of BARK scenarios from (adversarial) scenarios generated using the STRIVE model.
uses `StriveDatasetReader` to extract agents and their tracks (instead of the `InteractionDatasetReader` used in `InteractionDatasetScenarioGeneration`)

**`StriveDatasetReader`** (adapted from `InteractionDatasetReader`): class

in: `~/STRIVE/src/utils/strive_dataset_processing/strive_dataset_reader.py`

- Provides  functions for extracting agents and their tracks from STRIVE scenarios.

- **`TrackFromTrackfile(filename, track_id, start_time, end_time)`**: Returns track of specific agent by reading trackfile. (Calls read_tracks function from strive_scenario_decomposer to process STRIVE scenario file.)

- **`AgentFromTrackfile(track_params, param_server, scenario_track_info, agent_id, goal_def)`**: Returns BARK agent based on track information and agent ID.

**`strive_scenario_decomposer`** (adapted from `data_utils`): Python file
Provides functions for extracting track for specific agent from STRIVE scenario file.

in: `~/STRIVE/src/utils/strive_dataset_processing/strive_scenario_decomposer.py`

- **`read_tracks(filename, start_ts=0, end_ts=0) (adapted from read_tracks)`**: Generates dictionary containing tracks for each agent in STRIVE scenario.


## III.  For running adversarial scenario generation in STRIVE with INTERACTION data:

**`adv_scenario_gen_bark.py`** (adapted from `adv_scenario_gen.py`)

in: `~/STRIVE/src/adv_scenario_gen_bark.py`

- in `main()`: set variables for running scenario generation

- for creating `XodrMapEnv`: `map_data_path`, `mname`, `pix_per_m`, `plt`

- for creating `InteractionDataset`: `npast`, `nfuture`, `param_filename`

- **`param_filename`**: path to file (json) to load data parameters from 
(by default: `STRIVE/src/utils/interaction_dataset/interaction_tracks_05.json`)

- **`interaction_tracks_005.json`** (adapted from `examples/params/interaction_example.json`): defines map (xodr) and trackfile (csv) to be used

    map: `~/STRIVE/data/interaction/DR_DEU_Merging_MT_v01_shifted.xodr`

    track: `~/STRIVE/data/interaction/vehicle_tracks_005_short.csv`

- **`vehicle_tracks_005_short.csv`**: excerpt from INTERACTION dataset file `vehicle_tracks_005.csv` (in folder `interaction_dataset/DR_DEU_Merging_MT/tracks`) with reduced number of agents and **aligned timesteps(!)** (needed for corresponding indices in trajectories for all agents)

        track_id: 1 to 5 (5 agents)

        timestamp_ms: 36200 to 41000 ms (49 steps)


**adv_gen_bark.cfg** (adapted from `adv_gen_replay.cfg`):

in: `~/STRIVE/configs/adv_gen_bark.cfg`

- configuration file to be used with `adv_scenario_gen_bark.py`

- sets paths, options and parameters for scenario generation and optimization

- parameters copied from config for STRIVE replay planner


### Steps for adversarial scenario generation in STRIVE:
1. Activate STRIVE environment using:
```
conda activate nv_strive_env
```
2. Run python script with config file and STRIVE model:
```
python src/adv_scenario_gen_bark.py --config ./configs/adv_gen_bark.cfg --ckpt ./model_ckpt/traffic_model.pth
```
**Outcome**: Generates and optimizes adversarial scenarios and saves the resulting STRIVE scenarios in output folder `~/STRIVE/out`.


## IV.  For running example using BARK scenario generated using STRIVE:

**`strive_example.py`** (adapted from `examples/interaction_dataset.py`): 

in: `~/STRIVE/src/strive_example.py`

- in `main()`: set param_filename for loading map and tracks from json file

- by default: `~/STRIVE/src/utils/strive_dataset_processing/interaction_data_adv.json`

- **`interaction_data_adv.json`** (adapted from `examples/params/interaction_example.json`): defines map (xodr) and trackfile (json) to be used. (Note that the trackfile here is the STRIVE scenario json file which contains all the scenario data including the tracks of all agents.)

    map: `~/STRIVE/data/interaction/DR_DEU_Merging_MT_v01_shifted.xodr`

    track: `~/STRIVE/data/strive/vehicle_tracks_005_adv.json`

- **`vehicle_tracks_005_adv.json`**: adverse scenario generated using STRIVE based on INTERACTION data (track) file `vehicle_tracks_005_short.csv` (see also **Section III**)


### Steps for loading STRIVE scenario into BARK environment: 
1. Activate STRIVE environment using:
```
conda activate nv_strive_env
```
2. Run python script after adapting parameter file (sets map and STRIVE scenario trackfile):
```
python src/strive_example.py
```
**Outcome**: Generates BARK scenario based on STRIVE scenario file (json) and matching map (xodr) and visualizes scenario as short video sequence using `VideoRender` class provided in BARK.