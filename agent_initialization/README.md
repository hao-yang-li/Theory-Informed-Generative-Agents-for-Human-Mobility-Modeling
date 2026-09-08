# Population and agent initialization

The pipeline initializes agent and CBG profiles for U.S. cities covered by the source datasets, using compatible ACS and urban CBG records. Precomputed NYC profiles are available in `agent_initialization/`. The four modules implement the population-initialization sequence:

1. `extract_cbg_data.py` selects ACS demographic marginals and integrates home-CBG context.
2. `generate_agents_profile.py` applies IPF and samples demographic profiles.
3. `citizen_initialization.py` joins the samples to home-CBG context.
4. `cbg_sample_agent.py` aggregates the sampled profiles into the CBG-level inputs used by TIMA.

Run `initialize_agents.py` from the repository root to execute the sequence. Initialization runs locally using pandas and NumPy from the repository's requirements.

## Download the source data

- **SafeGraph Open Census, 2019 ACS 5-year estimates:** download `safegraph_open_census_data_2019.tar.gz` from the [Open Census documentation and download page](https://docs.safegraph.com/docs/open-census-data).
- **Urban sustainability data:** download `Data_A Satellite Imagery Dataset for Long-Term Sustainable Development in United States Cities.zip` from [Liu et al. (2023), Figshare, version 1](https://doi.org/10.6084/m9.figshare.23936787.v1). The related [Scientific Data paper](https://doi.org/10.1038/s41597-023-02576-3) describes the dataset.

The extractor streams selected CSV members from these archives and saves the selected city tables. Gzip decompression traverses the archive to reach later tables. Extracted source directories are also accepted.

The NYC tables under `data/population/NYC` are selected and processed from these releases.

## Run directly from the downloaded archives

```bash
python initialize_agents.py \
  --acs-source "/path/to/safegraph_open_census_data_2019.tar.gz" \
  --urban-source "/path/to/Data_A Satellite Imagery Dataset for Long-Term Sustainable Development in United States Cities.zip" \
  --city-name "New York city" \
  --city-key NYC \
  --year 2019 \
  --output-dir generated/NYC
```

`--output-dir` specifies where the results are saved; `generated/NYC` is an example directory and can be replaced with your preferred path.

The pipeline matches `--city-name` exactly against `City Name` in the urban dataset's lookup table and obtains the corresponding `CBG Code` values. Those identifiers are matched to `census_block_group` in the ACS tables. Urban context records are selected using city, CBG, and year, then joined by CBG identifier.

For another city represented in the urban dataset, change `--city-name`, `--city-key`, and `--output-dir`, for example `"Orlando city"`, `Orlando`, and `generated/Orlando`. The key names output files; the city name determines which records are selected. An optional `--cbg-list /path/to/cbgs.csv` selects CBGs within the named city. This CSV must contain `census_block_group`, `CBG Code`, or `CBG`. CBG identifiers are read as strings to preserve leading zeroes.

Use matching ACS and urban-data years and compatible CBG geographies. The required ACS variables are checked when reading the tables; this pipeline has been validated against the supplied 2019 release.

## Run from the included NYC input tables

```bash
python initialize_agents.py \
  --census-table data/population/NYC/cbg_extracted_data.csv \
  --context-table data/population/NYC/cbg_context.csv \
  --city-key NYC \
  --output-dir generated/NYC
```

The defaults are 10 agents per CBG and seed 42. Change them with `--agents-per-cbg` and `--seed`. The algorithm constructs sex-by-age, race, and sex-by-industry marginals, uses up to 50 IPF iterations with tolerance `1e-4`, and samples without replacement from the joint demographic combinations. If local marginals are invalid, it uses the selected city's aggregate marginals. The same supplied seed is used for each CBG's sampling call.

## Outputs and connection to TIMA

When reading the downloaded `.tar.gz` and `.zip` files, the pipeline first saves the selected city's `cbg_list.csv`, `cbg_extracted_data.csv`, `cbg_context.csv`, and `cbg_geographic_data.csv`. Whether starting from the downloaded files or the included NYC tables, it generates:

- `<city-key>_agent_census.csv`
- `agent_profiles_<city-key>.json`
- `cbg_profiles_<city-key>.json`

For a new initialization, set these paths in `config.yaml`:

```yaml
paths:
  agent_profiles: generated/NYC/agent_profiles_NYC.json
  cbg_profiles: generated/NYC/cbg_profiles_NYC.json
  cbg_geo_data: data/population/NYC/cbg_geographic_data.csv
```

When starting from the downloaded files, use `generated/NYC/cbg_geographic_data.csv` instead. Keep the remaining configuration entries and provide the appropriate POI data, inferred behavioral rules, and other simulation inputs. Then follow the repository's behavioral-inference and simulation instructions. These scripts prepare population inputs; reproducing the manuscript's mobility results also requires the corresponding empirical data, LLM outputs, and experiment configuration.

Precomputed NYC JSON files are available under `agent_initialization/` for the demonstration pipeline.

## Tests

```bash
python -m unittest discover -s tests -p 'test_agent_initialization.py' -v
```

Tests use small generated fixtures in temporary directories and remove them after execution. They cover archive and directory reading, the two pipeline modes, deterministic sampling, CBG IDs, missing inputs, and profile output structure.
