# Theory-Informed Human Mobility Modeling with LLM-Derived Behavioral Priors

![Cover Image](assets/fig1.jpg)

## Environment Setup

To set up the environment, install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Parameter Setup

Configure the selected city's inputs and outputs in `config.yaml` as described under [City-specific Configuration](#city-specific-configuration).

## Population and Agent Initialization

The initialization pipeline generates TIMA agent and CBG profiles for U.S. cities covered by the source datasets, using compatible ACS demographic distributions and CBG-level context from the urban sustainability dataset. Precomputed NYC profiles are available in `agent_initialization/`. Follow the instructions below to initialize a selected city.

### Download the population data

Download these two files using the source links below:

| Source | File to download |
|---|---|
| [SafeGraph Open Census Data](https://docs.safegraph.com/docs/open-census-data), **2019 5-year ACS** | `safegraph_open_census_data_2019.tar.gz` |
| [Urban sustainability dataset, Figshare version 1](https://doi.org/10.6084/m9.figshare.23936787.v1) | `Data_A Satellite Imagery Dataset for Long-Term Sustainable Development in United States Cities.zip` |

The pipeline reads the required tables directly from the archives and saves the selected city's inputs.

### Initialize from the downloaded files

Run the following command from the repository root, replacing the two source paths with your download locations:

```bash
python initialize_agents.py \
  --acs-source "/path/to/safegraph_open_census_data_2019.tar.gz" \
  --urban-source "/path/to/Data_A Satellite Imagery Dataset for Long-Term Sustainable Development in United States Cities.zip" \
  --city-name "New York city" \
  --city-key NYC \
  --year 2019 \
  --output-dir generated/NYC
```

`--output-dir` specifies where to save the results. `generated/NYC` is an example directory; you can choose another path.

The pipeline selects the city using the urban dataset's `City Name` field and obtains its `CBG Code` values. It then matches those identifiers to Open Census's `census_block_group` field to extract the corresponding demographic records. The selected urban context records are filtered by city, CBG, and year and joined to the demographic inputs.

### Initialize another city

Keep the same downloaded archives and change the city selection and output names. For example, replace the last four arguments above with:

```bash
  --city-name "Orlando city" \
  --city-key Orlando \
  --year 2019 \
  --output-dir generated/Orlando
```

`--city-name` must exactly match a city represented in the urban dataset. `--city-key` controls output filenames. The defaults are 10 agents per CBG and seed 42; these can be changed with `--agents-per-cbg` and `--seed`.

### Use the included NYC population tables

To initialize NYC from the small processed tables already included in the repository:

```bash
python initialize_agents.py \
  --census-table data/population/NYC/cbg_extracted_data.csv \
  --context-table data/population/NYC/cbg_context.csv \
  --city-key NYC \
  --output-dir generated/NYC
```

### Use the generated profiles in TIMA

For NYC, the pipeline produces `agent_profiles_NYC.json` and `cbg_profiles_NYC.json` in the chosen output directory. Update the corresponding entries under `paths` in `config.yaml`:

```yaml
paths:
  agent_profiles: generated/NYC/agent_profiles_NYC.json
  cbg_profiles: generated/NYC/cbg_profiles_NYC.json
  cbg_geo_data: generated/NYC/cbg_geographic_data.csv
```

The geographic file above is produced by the archive-based command. When using the included NYC tables, set `cbg_geo_data` to `data/population/NYC/cbg_geographic_data.csv` instead. Keep the other configuration entries, supply the corresponding city-specific simulation inputs, and continue with behavioral inference below.

See [population initialization instructions](agent_initialization/README.md) for the four modules, optional CBG selection, and complete output list. The archive-based initialization has been tested for NYC and Orlando.

## City-specific Configuration

After initialization, update `config.yaml` with the selected city's profiles, geographic data, POIs, and weekly visit data. Behavioral inference, simulation, and evaluation all read this configuration from the repository root.

For example, after initializing Orlando from the downloaded archives, update the following entries in the existing configuration. Replace the example POI and visit-data paths with your corresponding files, and retain the other settings:

```yaml
simulation:
  city_name: "Orlando"
  enable_region_filter: false

paths:
  agent_profiles: "generated/Orlando/agent_profiles_Orlando.json"
  cbg_profiles: "generated/Orlando/cbg_profiles_Orlando.json"
  cbg_geo_data: "generated/Orlando/cbg_geographic_data.csv"
  poi_data_pattern: "/path/to/Orlando_core_poi.csv"
  weekly_patterns: "/path/to/Orlando_weekly_patterns.csv"
  policy_functions: "generated/Orlando/generated_behavioral_rules.json"
  output_dir: "TIMA_simulation_output/Orlando"
  output_filename: "agent_movements_llm.jsonl"
```

The input paths determine which city's records are loaded. `simulation.city_name` records the city label. Use POI and CBG identifiers that match across the profiles, geographic records, POI data, and weekly visit data. The initialization step prepares population and CBG inputs; supply the corresponding POI and visit files separately. The included dummy files provide these inputs for the NYC demonstration.

Set `simulation.d_max_km` to the value appropriate for the selected city's local opportunity density, following the method described in the manuscript. To simulate a CBG subset, set `simulation.enable_region_filter` to `true` and specify its prefix in `simulation.valid_cbg_prefix`.

Use a separate `paths.policy_functions` file for each city's behavioral inference and a separate `paths.output_dir` for its simulation results. Then complete behavioral inference, simulation, and evaluation in that order using the commands below.

Evaluation reads the selected city's reference visits from `paths.weekly_patterns` and its simulated trajectories from `paths.output_dir` and `paths.output_filename`. Keep the same city configuration for both stages. The additional home-panel input for Trip Distance KL and POI Proportion KL is described under [Running Evaluation Metrics](#2-running-evaluation-metrics).

## LLM-based Behavioral Inference

Configure the LLM provider and model in `config.yaml` and the corresponding API key under `api_keys` in your local `secrets.yaml`. To infer profile-conditioned behavioral parameters and rules, run:

```bash
python query_action_code.py
```

This infers POI-category preferences, socioeconomic affinity weights, and exploration probabilities for each unique demographic profile. The generated behavioral rules are saved to the path specified by `paths.policy_functions` in `config.yaml` and reused during trajectory simulation.

## TIMA Simulation

After completing behavioral inference:

1. Complete [City-specific Configuration](#city-specific-configuration) and set `paths.policy_functions` to the behavioral-rule file generated for that city.

2. Run the simulation:

```bash
python TIMA_simulation.py
```

This generates trajectories using the configured simulation parameters and inferred behavioral rules. Results are saved under `paths.output_dir` using `paths.output_filename` from `config.yaml`.

## Dummy Data for Testing
The full SafeGraph mobility data are subject to licensing restrictions and are not redistributed in this repository. The provided dummy files are derived from source data by replacing POI identifiers and perturbing coordinates:

- **`data/core_poi/NYC_core_poi_dummy.csv`**: Contains randomized business IDs and jittered coordinates (approx. ±2km offset).
- **`data/Weekly_patterns/..._dummy.csv`**: Uses the corresponding replacement POI IDs to maintain consistency between the files.

With the dependencies, LLM credentials, and file paths configured, the supplied population inputs and dummy files support the initialization, behavioral-inference, and trajectory-simulation pipeline, producing JSONL outputs with the schema documented below. Users with authorized access to the corresponding SafeGraph datasets can obtain the files under their applicable data-use agreement, replace the dummy inputs, and update the paths in `config.yaml` to run the pipeline with those data.

## Model Output & Evaluation

### 1. Simulator Output Format
The simulator generates human mobility trajectories in `JSONL` (JSON Lines) format. Each line represents a single movement decision made by an agent. For privacy and demonstration purposes, POI IDs are masked with dummy identifiers.

**Example Output (`agent_movements_llm.jsonl`):**

```json
{"agent_id": "0", "home_cbg": "360470405001", "time_step": 1, "poi_id": "sg:dummy_poi_0001", "poi_cbg": "360050119002", "category": "Arts, Entertainment, and Recreation", "action": "explore", "dist_km": 15.7696}
{"agent_id": "1", "home_cbg": "360470405001", "time_step": 1, "poi_id": "sg:dummy_poi_0002", "poi_cbg": "360610076002", "category": "Accommodation and Food Services", "action": "explore", "dist_km": 10.016}
{"agent_id": "2", "home_cbg": "360470405001", "time_step": 1, "poi_id": "sg:dummy_poi_0003", "poi_cbg": "360610219001", "category": "Others", "action": "explore", "dist_km": 15.701}
```

*   `agent_id`: Unique identifier for the generative agent.
*   `home_cbg`: The residential Census Block Group of the agent.
*   `action`: Whether the agent is exploring a new location (`explore`) or returning to a familiar one (`return`).
*   **`dist_km`**: The **step distance** from the agent's previous location to the current destination.

> **Note on Distance Calculation:** While the simulator outputs step-by-step distances, the evaluation script (`TIMA_analyze.py`) will recalculate all movements as **home-based distances** (from the agent's home CBG centroid to the POI) to ensure a consistent comparison with SafeGraph ground truth data.


### 2. Running Evaluation Metrics
`TIMA_analyze.py` calculates macroscopic alignment, fundamental mobility laws, and mobility-mediated social metrics.

**Prerequisites:**
Use the same [city-specific configuration](#city-specific-configuration) as the simulation. Set `paths.weekly_patterns` to the reference visit data and `paths.output_dir` and `paths.output_filename` to the simulation results to evaluate. The script reads the corresponding POI, geographic, profile, and behavioral-rule files from this configuration.

For Trip Distance KL and POI Proportion KL, add `home_panel_summary` under `paths`, pointing to the matching week's home-panel CSV with `census_block_group` and `number_devices_residing` columns. If it is not supplied, these two metrics are reported as unavailable (`NaN`).

**Metrics Calculated:**

*   **Table 1 (Macro Alignment):** Trip Distance KL, OD Flow CPC (Tract level), Visitation Density MSE, POI Proportion KL, and Stratified OD Fidelity.
*   **Figure 3 (Mobility Laws):** Zipf’s Law RMSE, Radius of Gyration (Median & KL), and Explorer/Returner dichotomy ($k^*$ and MAE).
*   **Social Metrics:** Experienced Segregation (S-index) and Home-stay rates.
*   **Agent Parameters:** Statistical distribution (Mean/SD) of inferred Exploration Probability ($P_u$), Semantic Interest ($w_{u,k}$), and Socio-economic Affinity ($A_{u,c}$).

**Execution:**

```bash
python TIMA_analyze.py
```
