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
We provide a comprehensive evaluation script `tima_metrics_evaluator.py` to replicate the results presented in the paper (Table 1, Figures 3-5, and Supplementary Table 10). The script calculates macroscopic alignment, fundamental mobility laws, and mobility-mediated social metrics.

**Prerequisites:**
Ensure all paths to ground truth data and simulation results are correctly configured in `config.yaml`.

**Metrics Calculated:**
*   **Table 1 (Macro Alignment):** Trip Distance KL, OD Flow CPC (Tract level), Visitation Density MSE, POI Proportion KL, and Stratified OD Fidelity.
*   **Figure 3 (Mobility Laws):** Zipf’s Law RMSE, Radius of Gyration (Median & KL), and Explorer/Returner dichotomy ($k^*$ and MAE).
*   **Social Metrics:** Experienced Segregation (S-index) and Home-stay rates.
*   **Agent Parameters:** Statistical distribution (Mean/SD) of inferred Exploration Probability ($P_u$), Semantic Interest ($w_{u,k}$), and Socio-economic Affinity ($A_{u,c}$).

**Execution:**
```bash
python TIMA_analyze.py
```