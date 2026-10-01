# Population and agent initialization

The pipeline initializes agent and CBG profiles for U.S. cities covered by the source datasets, using compatible ACS and urban CBG records. Precomputed NYC profiles are available in `agent_initialization/`. The four modules implement the population-initialization sequence:

1. `extract_cbg_data.py` selects ACS demographic marginals and integrates home-CBG context.
2. `generate_agents_profile.py` applies IPF and samples demographic profiles.
3. `citizen_initialization.py` joins the samples to home-CBG context.
4. `cbg_sample_agent.py` aggregates the sampled profiles into the CBG-level inputs used by TIMA.

Run `python -m agent_initialization.initialize_agents` from the repository root to execute the sequence. Initialization runs locally using pandas and NumPy from the repository's requirements.

## Download the source data

- **SafeGraph Open Census, 2019 ACS 5-year estimates:** download `safegraph_open_census_data_2019.tar.gz` from the [Open Census documentation and download page](https://docs.safegraph.com/docs/open-census-data).
- **Urban sustainability data:** download `Data_A Satellite Imagery Dataset for Long-Term Sustainable Development in United States Cities.zip` from [Liu et al. (2023), Figshare, version 1](https://doi.org/10.6084/m9.figshare.23936787.v1). The related [Scientific Data paper](https://doi.org/10.1038/s41597-023-02576-3) describes the dataset.

The extractor reads the selected city's records from these archives or extracted directories.

The NYC tables under `data/population/NYC` are selected and processed from these releases.

Home-CBG income and education context are classified into city-specific Low, Medium, and High categories, stored as `home_cbg_income_level` and `home_cbg_edu`.

## Run directly from the downloaded archives

```bash
python -m agent_initialization.initialize_agents \
  --acs-source "/path/to/safegraph_open_census_data_2019.tar.gz" \
  --urban-source "/path/to/Data_A Satellite Imagery Dataset for Long-Term Sustainable Development in United States Cities.zip" \
  --city-name "New York city" \
  --city-key NYC \
  --year 2019 \
  --output-dir TIMA_model_input/NYC
```

`--output-dir` specifies where the results are saved; `TIMA_model_input/NYC` is an example directory and can be replaced with your preferred path.

The pipeline matches `--city-name` exactly against `City Name` in the urban dataset's lookup table and obtains the corresponding `CBG Code` values. Those identifiers are matched to `census_block_group` in the ACS tables. Urban context records are selected using city, CBG, and year, then joined by CBG identifier.

For another city represented in the urban dataset, change `--city-name`, `--city-key`, and `--output-dir`, for example `"Orlando city"`, `Orlando`, and `TIMA_model_input/Orlando`. The key names output files; the city name determines which records are selected. An optional `--cbg-list /path/to/cbgs.csv` selects CBGs within the named city. This CSV must contain `census_block_group`, `CBG Code`, or `CBG`. CBG identifiers are read as strings to preserve leading zeroes.

Use matching ACS and urban-data years and compatible CBG geographies. The required ACS variables are checked when reading the tables; this pipeline has been validated against the supplied 2019 release.

## Run from the included NYC input tables

```bash
python -m agent_initialization.initialize_agents \
  --census-table data/population/NYC/cbg_extracted_data.csv \
  --context-table data/population/NYC/cbg_context.csv \
  --city-key NYC \
  --output-dir TIMA_model_input/NYC
```

By default, the pipeline allocates an average of 10 agents per CBG in proportion to resident population, with at least one agent per CBG. Use `--agents-per-cbg` to change the average, `--allocation uniform` for equal counts, and `--seed` to set the sampling seed.

Agent demographics are sampled from ACS distributions using iterative proportional fitting (IPF). When running `generate_agents_profile.py` directly, supply the population context CSV with `--context`.

## Outputs and connection to TIMA

When reading the downloaded `.tar.gz` and `.zip` files, the pipeline first saves the selected city's `cbg_list.csv`, `cbg_extracted_data.csv`, `cbg_context.csv`, `cbg_geographic_data.csv`, and `cbg_boundaries.csv`. The boundary CSV contains the selected city's and year's WKT polygons from `Basic_Geographic_Statistics_CBG.csv`. Whether starting from the downloaded files or the included NYC tables, it generates:

- `<city-key>_agent_census.csv`
- `agent_profiles_<city-key>.json`
- `cbg_profiles_<city-key>.json`

For a new initialization, set these paths in `config.yaml`:

```yaml
paths:
  agent_profiles: TIMA_model_input/NYC/agent_profiles_NYC.json
  cbg_profiles: TIMA_model_input/NYC/cbg_profiles_NYC.json
  cbg_geo_data: data/population/NYC/cbg_geographic_data.csv
```

When starting from the downloaded files, use `TIMA_model_input/NYC/cbg_geographic_data.csv` instead. Keep the remaining configuration entries and provide the appropriate POI data, inferred behavioral rules, and other simulation inputs. Then follow the repository's behavioral-inference and simulation instructions. These scripts prepare population inputs; reproducing the manuscript's mobility results also requires the corresponding empirical data, LLM outputs, and experiment configuration.

Precomputed population-proportional NYC JSON files are available under `agent_initialization/` for the demonstration pipeline. Regenerate behavioral outputs when changing profile inputs; do not pair old trajectories with a newly allocated population.

## Extract CBG boundaries

To extract boundaries using only the urban sustainability ZIP:

```bash
python -m agent_initialization.extract_cbg_data \
  --boundaries-only \
  --urban-source "/path/to/Data_A Satellite Imagery Dataset for Long-Term Sustainable Development in United States Cities.zip" \
  --city-name "New York city" \
  --year 2019 \
  --output-dir TIMA_model_input/NYC
```

This writes `cbg_boundaries.csv`. Set `paths.cbg_boundaries` to this file and `evaluation.boundary_cbg_column` to `CBG Code`. The full archive-based initialization command already produces this file.

## Situational context

Population profiles are shared across routine and context-specific runs. To add a scenario, copy `contexts/Manhattan-pandemic/`, which contains `context.json`, `interest.txt`, `preference.txt`, and `dynamics.txt`.

Edit these fields in `context.json`:

- `name`: the scenario name used in output paths.
- `text`: the scenario description inserted as `{COMMON_SCENARIO_TEXT}`.
- `openings`: opening instructions for the three tasks.
- `category_hints`: six POI-category hints in category order; entries may be empty strings.
- `cbg_prefix`: optional home-CBG and destination-POI region filter. `"36061"` selects Manhattan; an empty, null, or omitted value uses all configured city inputs.

The adjacent text files contain the task instructions. Select the JSON through `situational_context.file` in the root `config.yaml`, then run from the repository root:

```bash
python query_behavior_priors.py
python TIMA_simulation.py
```

Inference re-estimates POI-category preferences, income- and race-based affinity weights, exploration probabilities, and the distance-decay multiplier for the selected context. The behavioral priors are saved under `contexts/<name>/` beside the configured prior file and loaded during simulation. Keep the same context selected for both commands. Use a new name when changing a scenario with saved priors. Set `situational_context.file` to `null` for routine runs.
