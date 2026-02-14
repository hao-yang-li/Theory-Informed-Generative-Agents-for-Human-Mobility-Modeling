# Theory-Informed Generative Agents for Human Mobility Modeling

![Cover Image](assets/fig1.jpg)

## Environment Setup

To set up the environment, install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Parameter Setup

Before running the simulation, you need to modify the file paths in `config.yaml` to match your local directory structure. Update the paths in the `paths` section of `config.yaml` to point to your local data directories and files.

## Policy Function Generation

To generate policy function code, run:

```bash
python query_action_code.py
```

This will generate the policy function code that will be used in the simulation.

## TIMA Simulation

After generating the policy function code:

1. Add the path to the generated policy function code file to the `policy_functions` variable in `config.yaml` under the `paths` section.

2. Run the simulation:

```bash
python TIMA_simulation.py
```

This will generate the simulation results based on the configured parameters and policy functions in the `TIMA_simulation_output` folder.

## Dummy Data for Testing
To comply with data privacy policies of SafeGraph, the Points of Interest (POI) data provided in this repository are **synthetic dummy files**:
- **`data/core_poi/NYC_core_poi_dummy.csv`**: Contains randomized business IDs and jittered coordinates (approx. ±2km offset).
- **`data/Weekly_patterns/..._dummy.csv`**: Contains mapped synthetic IDs to maintain simulation consistency.

These files allow the code to run out-of-the-box. To use real SafeGraph data, replace these with your own licensed files and update the `config.yaml`.