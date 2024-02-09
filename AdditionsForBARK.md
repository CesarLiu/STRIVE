# Overview of STRIVE Extension Files, Classes and Functions
#### Tools for working with OpenDrive maps, INTERACTION data and BARK simulator in STRIVE environment


## I. For using map and scenario data (input) with STRIVE model and scenario generation:
**`XodrMapEnv`** (adapted from `MapEnv`): class

in: `~/STRIVE/src/datasets/xodr_map_env.py`

- Creates binary rasterized map image of OpenDrive map to be used with STRIVE model.
The map image consists of four layers: drivable area, road dividers, lane dividers and carpark area. (The carpark area is set to 0 by default.)

- **`RasterizePolygon (polygon, x_grid, y_grid, grid_size=None)`**: Creates rasterized image of BARK polygon (Polygon2d) based on given raster grid.

- **`RasterizeLine (line, x_grid, y_grid, grid_size=None)`**: Creates rasterized image of BARK line (Line2d) based on given raster grid.

- **`BresenhamLine (x0, y0, x1, y1)`**: Rasterizes line between two points (x0, y0) and (x1, y1) using Bresenham line algorithm.

- **`PlotRasterMap (x_grid, y_grid, raster_map, grid_size, layer_name)`**: Plots binary image of one rasterized map layer.


**`InteractionDataset`** (adapted from `NuScenesDataset`): subclass of `Dataset` class

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

- **`TrackFromTrackfile (filename, track_id, start_time, end_time)`**: Returns track of specific agent by reading trackfile. (Calls read_tracks function from strive_scenario_decomposer to process STRIVE scenario file.)

- **`AgentFromTrackfile (track_params, param_server, scenario_track_info, agent_id, goal_def)`**: Returns BARK agent based on track information and agent ID.

**`strive_scenario_decomposer`** (adapted from `data_utils`): Python file
Provides functions for extracting track for specific agent from STRIVE scenario file.

in: `~/STRIVE/src/utils/strive_dataset_processing/strive_scenario_decomposer.py`

- **`read_tracks (filename, start_ts=0, end_ts=0) (adapted from read_tracks)`**: Generates dictionary containing tracks for each agent in STRIVE scenario.


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

- **`vehicle_tracks_005_short.csv`**: excerpt from INTERACTION dataset file `vehicle_tracks_005.csv` (in folder `interaction_dataset/DR_DEU_Merging_MT/tracks`) with reduced number of agents and **aligned timesteps(!)** (needed for matching indexing in trajectories for all agents)

        track_id: 1 to 5 (5 agents)

        timestamp_ms: 36200 to 41000 ms (49 steps)


**`adv_gen_bark.cfg`** (adapted from `adv_gen_replay.cfg`):

in: `~/STRIVE/configs/adv_gen_bark.cfg`

- configuration file to be used with `adv_scenario_gen_bark.py`

- sets paths, options and parameters for scenario generation and optimization

- parameters copied from config file for STRIVE replay planner `~STRIVE/configs/adv_gen_replay.cfg`


### Steps for adversarial scenario generation in STRIVE:
1. Activate STRIVE environment using:
```
conda activate nv_strive_env
```
2. Run python script with config file and STRIVE model:
```
python src/adv_scenario_gen_bark.py --config ./configs/adv_gen_bark.cfg --ckpt ./model_ckpt/traffic_model.pth
```
**Outcome**: Generates and optimizes adversarial scenarios and saves the resulting STRIVE scenarios (as json files) in output folder `~/STRIVE/out`.


## IV.  For running example using BARK scenario generated using STRIVE:

**`strive_example.py`** (adapted from `examples/interaction_dataset.py`): 

in: `~/STRIVE/src/strive_example.py`

- in `main()`: set `param_filename` for loading map and tracks from json file

    param file: `~/STRIVE/src/utils/strive_dataset_processing/interaction_data_adv.json`)

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


## Open tasks, future work and improvements

The following advancements could be made to extend and improve the additions to STRIVE explained above.
The suggestions are listed in their order of relevance for the output of the adversarial scenario generation in STRIVE.

### 1. Generating more scenarios from INTERACTION trackfile

When creating `InteractionDataset` in STRIVE, use `InteractionDatasetScenarioGenerationFull` (BARK class) to **generate multiple BARK scenarios** from INTERACTION input data.
- Create list of input scenarios from given INTERACTION trackfile (i.e., one scenario for each of the `N` agents).
- Iterate over all `N` BARK scenarios in `scenario_list` (e.g., each agent is set as ego agent once) and perform adversarial optimization for each one.
- For storing agent information in `scene2info` dictionary, adapt scenario name `sname` and set correct ego agent `ego_idx` for each scenario. (**Note** that the `ego_idx` may not be the same as the agent ID of the ego vehicle depending on the order and number of `track_ids` contained in the input trackfile.)

### 2. Creating lane graph to use with rule-based planner
When creating `XodrMapEnv` in STRIVE, create **lane graph of the OpenDrive map** in addition to the rasterized images of each layer.
- When using rule-based planner (i.e., `cfg.planner = 'hardcode'`) STRIVE requires a lane graph representation of the input map. (The planner type is defined in the config file.)
- The required structure of the lane graph can be derived from `nutils.process_lanegraph (...)` function.
- When creating `XodrMapEnv` and using the rule-based planner, the boolean `load_lanegraph` is set to `True` and the `process_lanegraph` function is called.
- To generate the lane graph with `process_lanegraph`, the resolution of the lane graph `lanegraph_res_meters` and `lanegraph_eps` should be defined. (Otherwise, default values are used.)

### 3. Extending OpenDrive map environment

When creating `XodrMapEnv` in STRIVE, create rasterized images for **all of the maps** stored in the given map directory and included in `map_list`.
- Note that all maps should be padded to the same size `(max_H, max_W)` (as in original STRIVE `MapEnv` class).
- All rasterized maps are stored in `xodr_raster` with shape `(M, 4, max_h, max_w)` and `M`: number of maps.