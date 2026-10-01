
import os
import json
import warnings
import argparse
import ast
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
from agent_initialization.demographics import income_level, education_level
from evaluation.evaluation_laws import rank_frequency_rmse
from evaluation.evaluation_segregation import income_rank_correlation
from evaluation.evaluation_kl import calculate_kl, RevisedKLMetrics
from evaluation.evaluation_ks import visits_per_location_ks
from evaluation.evaluation_segregation import experienced_segregation

# ==============================================================================
# 1. Constants & Utils
# ==============================================================================

POI_CATEGORIES = [
    'Wholesale & Retail Trade, Transportation and Warehousing',  # 0
    'Others',  # 1
    'Educational Services',  # 2
    'Health Care and Social Assistance',  # 3
    'Arts, Entertainment, and Recreation',  # 4
    'Accommodation and Food Services'  # 5
]

# Mapping based on first 2 digits of NAICS code
NAICS_TO_CATEGORY_MAP = {
    42: POI_CATEGORIES[0], 44: POI_CATEGORIES[0], 45: POI_CATEGORIES[0],
    48: POI_CATEGORIES[0], 49: POI_CATEGORIES[0],
    61: POI_CATEGORIES[2],
    62: POI_CATEGORIES[3],
    71: POI_CATEGORIES[4],
    72: POI_CATEGORIES[5]
}


def load_config(config_path="config.yaml"):
    from situational_context import configure_context
    with open(config_path, "r", encoding="utf-8") as f:
        return configure_context(yaml.safe_load(f))


def get_tract_id(cbg_series):
    return cbg_series.astype(str).str.slice(0, 11)


def haversine(lat1, lon1, lat2, lon2):
    if isinstance(lat1, pd.Series):
        mask = lat1.notna() & lat2.notna()
        d = np.zeros(len(lat1))
        d[:] = np.nan
        if mask.sum() == 0: return d

        R = 6371.0
        phi1, phi2 = np.radians(lat1[mask].astype(float)), np.radians(lat2[mask].astype(float))
        dphi = phi2 - phi1
        dlambda = np.radians(lon2[mask].astype(float) - lon1[mask].astype(float))
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        d[mask] = 2 * R * np.arcsin(np.sqrt(a))
        return d
    else:
        # Scalar version
        if any(x is None or np.isnan(x) for x in [lat1, lon1, lat2, lon2]): return np.nan
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 2 * R * asin(sqrt(a))


# ==============================================================================
# 2. Attribute Helper Functions
# ==============================================================================

def get_dominant_attribute(profile, attribute, ranges_dict):
    if attribute == 'race':
        dist = profile.get("race_distribution", {})
    elif attribute == 'sex':
        dist = profile.get("sex_distribution", {})
    elif attribute == 'age_group':
        dist = profile.get("age_distribution", {})
    elif attribute == 'industry':
        counts = profile.get("industry_counts", {})
        return max(counts, key=counts.get) if counts else None
    elif attribute == 'education':  # Mapped to education_level
        return education_level(profile)
    elif attribute == 'income':  # Mapped to income_level
        return income_level(profile)
    else:
        return None
    # For dict based distributions
    return max(dist, key=dist.get) if dist else None


# ==============================================================================
# 3. Data Processing
# ==============================================================================

class DataProcessor:
    def __init__(self, config):
        self.paths = config['paths']
        from situational_context import context_cbg_prefix
        self.cbg_prefix = context_cbg_prefix(config)
        self.cbg_centroids = {}
        self.poi_coords = {}
        self.poi_categories = {}
        self.ranges_dict = {}
        self.evaluation = config.get('evaluation', {})
        self.devices = None

    def load_metadata(self):
        print(">>> Loading Metadata (Geo, POI, Ranges)...")
        # 1. Ranges for discretization
        if os.path.exists(self.paths['ranges']):
            with open(self.paths['ranges'], 'r') as f:
                self.ranges_dict = json.load(f)

        # 2. Geo Data
        df_geo = pd.read_csv(self.paths['cbg_geo_data'], dtype={'CBG Code': str, 'census_block_group': str})
        df_geo = df_geo.rename(columns={'census_block_group': 'CBG Code',
                                        'latitude': 'Latitude', 'longitude': 'Longitude'})
        if 'Year' in df_geo:
            df_geo = df_geo[df_geo['Year'] == self.evaluation.get('year', 2019)]
        df_geo = df_geo.drop_duplicates('CBG Code')
        if self.cbg_prefix:
            df_geo = df_geo[df_geo['CBG Code'].str.startswith(self.cbg_prefix, na=False)]
        for _, row in df_geo.iterrows():
            cbg = row['CBG Code']
            # Simple fallback for Centroid WKT or Lat/Lon cols
            if 'Latitude' in row and 'Longitude' in row:
                self.cbg_centroids[cbg] = (float(row['Latitude']), float(row['Longitude']))
            elif 'Centroid' in row:
                parts = row['Centroid'].replace("POINT (", "").replace(")", "").split()
                self.cbg_centroids[cbg] = (float(parts[1]), float(parts[0]))

        # 3. POI Data (Core POI)
        df_poi = pd.read_csv(self.paths['poi_data_pattern'], dtype={'safegraph_place_id': str, 'poi_id': str})
        if 'safegraph_place_id' not in df_poi:
            df_poi = df_poi.rename(columns={'poi_id': 'safegraph_place_id'})
        df_poi = df_poi.drop_duplicates('safegraph_place_id')
        for _, row in df_poi.iterrows():
            pid = row['safegraph_place_id']
            self.poi_coords[pid] = (float(row['latitude']), float(row['longitude']))

            # NAICS Mapping Logic
            naics_str = str(row.get('naics_code', ''))
            if len(naics_str) >= 2 and naics_str[:2].isdigit():
                prefix = int(naics_str[:2])
                self.poi_categories[pid] = NAICS_TO_CATEGORY_MAP.get(prefix, 'Others')
            else:
                self.poi_categories[pid] = 'Others'

        panel_path = self.paths.get('home_panel_summary')
        if panel_path:
            panel = pd.read_csv(panel_path, dtype={'census_block_group': str},
                                usecols=['census_block_group', 'number_devices_residing'])
            self.devices = panel.drop_duplicates('census_block_group').set_index(
                'census_block_group')['number_devices_residing'].to_dict()

    def process_real_data(self):
        print(">>> Processing Real Data...")
        df_raw = pd.read_csv(self.paths['weekly_patterns'], dtype={'safegraph_place_id': str, 'poi_id': str, 'poi_cbg': str})
        if 'safegraph_place_id' not in df_raw:
            df_raw = df_raw.rename(columns={'poi_id': 'safegraph_place_id'})
        self.poi_to_cbg = df_raw.dropna(subset=['poi_cbg']).drop_duplicates(
            'safegraph_place_id').set_index('safegraph_place_id')['poi_cbg'].to_dict()
        records = []

        for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Expanding Visits"):
            pid = row['safegraph_place_id']
            # Fallback if POI category not in Core file (use 'Others')
            cat = self.poi_categories.get(pid, 'Others')
            poi_cbg = str(row['poi_cbg'])
            if self.cbg_prefix and not poi_cbg.startswith(self.cbg_prefix):
                continue

            # Get POI Coords
            if pid in self.poi_coords:
                p_lat, p_lon = self.poi_coords[pid]
            else:
                continue  # Skip POIs with unavailable coordinates.

            try:
                visits = json.loads(row['visitor_home_cbgs'])
                for home_cbg, cnt in visits.items():
                    if home_cbg in self.cbg_centroids:
                        h_lat, h_lon = self.cbg_centroids[home_cbg]
                        dist = haversine(h_lat, h_lon, p_lat, p_lon)

                        records.append({
                            'home_cbg': home_cbg,
                            'poi_cbg': poi_cbg,
                            'poi_id': pid,
                            'category': cat,
                            'count': cnt,
                            'dist_km': dist
                        })
            except:
                continue

        return pd.DataFrame(records)

    def process_sim_data(self):
        print(">>> Processing Sim Data (LLM)...")
        sim_path = os.path.join(self.paths['output_dir'], self.paths['output_filename'])
        records = []

        with open(sim_path, 'r') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    home = str(row['home_cbg'])
                    pid = str(row['poi_id'])

                    # Recalculate Distance (Consistency)
                    dist = np.nan
                    if pid == 'Home':
                        dist = 0.0
                    elif home in self.cbg_centroids and pid in self.poi_coords:
                        h_lat, h_lon = self.cbg_centroids[home]
                        p_lat, p_lon = self.poi_coords[pid]
                        dist = haversine(h_lat, h_lon, p_lat, p_lon)

                    destination = (home if pid == 'Home' else
                                   row.get('poi_cbg') or self.poi_to_cbg.get(pid)
                                   or row.get('current_cbg_of_agent'))
                    category = row.get('category') or row.get('poi_category')
                    if self.cbg_prefix and not (home.startswith(self.cbg_prefix) and str(destination).startswith(self.cbg_prefix)):
                        continue
                    if category not in POI_CATEGORIES:
                        category = self.poi_categories.get(pid, 'Others')
                    records.append({
                        'agent_id': str(row['agent_id']),
                        'home_cbg': home,
                        'poi_cbg': str(destination) if destination is not None else '',
                        'poi_id': pid,
                        'category': category,
                        'count': 1,
                        'dist_km': dist
                    })
                except:
                    continue
        return pd.DataFrame(records)

    def load_profiles(self):
        print(">>> Loading Profiles...")
        # Sim Agents
        with open(self.paths['agent_profiles'], 'r') as f:
            raw = json.load(f)
            self.agent_map = {str(a['id']): a for a in (raw if isinstance(raw, list) else raw.values())}

        # Real CBGs
        with open(self.paths['cbg_profiles'], 'r') as f:
            raw = json.load(f)
            self.cbg_map = {str(c['census_block_group']): c for c in (raw if isinstance(raw, list) else raw.values())}

        if self.cbg_prefix:
            self.agent_map = {k: p for k, p in self.agent_map.items() if str(p['CBG']).startswith(self.cbg_prefix)}
            self.cbg_map = {k: p for k, p in self.cbg_map.items() if k.startswith(self.cbg_prefix)}
        return self.agent_map, self.cbg_map, self.ranges_dict


# ==============================================================================
# 4. Evaluator
# ==============================================================================

def returner_fraction(frame, cbg_coords, poi_coords, k_values):
    profiles = []
    for cbg, group in frame.groupby('home_cbg', sort=False):
        home = cbg_coords.get(str(cbg))
        if home is None:
            continue
        visits = []
        for pid, count in group.groupby('poi_id', sort=False)['count'].sum().items():
            point = poi_coords.get(str(pid))
            if point is None or not np.all(np.isfinite(point)) or count <= 0:
                continue
            distance = haversine(*home, *point)
            if not np.isfinite(distance) or distance > 500:
                continue
            visits.append((point, float(count)))
        if not visits:
            continue

        def weighted_radius(subset):
            total = sum(count for _, count in subset)
            latitude = sum(point[0] * count for point, count in subset) / total
            longitude = sum(point[1] * count for point, count in subset) / total
            return sqrt(sum(haversine(latitude, longitude, *point)**2 * count
                            for point, count in subset) / total)

        total_rg = weighted_radius(visits)
        visits.sort(key=lambda item: item[1], reverse=True)
        flags = []
        # Once k includes all visited POIs, the subset radius is unchanged.
        subset_radii = {}
        for k in k_values:
            size = min(int(k), len(visits))
            if size not in subset_radii:
                subset_radii[size] = weighted_radius(visits[:size])
            flags.append(total_rg > 0 and subset_radii[size] > total_rg / 2)
        profiles.append(flags)
    if not profiles:
        warnings.warn('Explorer/returner analysis requires valid home-CBG visits and POI coordinates.')
        return np.full(len(k_values), np.nan)
    return np.mean(profiles, axis=0)


def returner_transition(k_values, fraction_values):
    x, y = np.asarray(k_values), np.asarray(fraction_values)
    if not len(y) or not np.all(np.isfinite(y)):
        return float('nan')
    index = np.searchsorted(y, 0.5)
    if index == 0:
        return float(x[0])
    if index >= len(y):
        return float(x[-1])
    x0, x1 = np.log10(x[index - 1]), np.log10(x[index])
    y0, y1 = y[index - 1], y[index]
    return float(10 ** (x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)))


class MobilityEvaluator:
    def __init__(self, df_real, df_sim, agent_map, cbg_map, ranges_dict,
                 cbg_coords=None, poi_coords=None, devices=None, sampling_threshold=0.0):
        self.real = df_real
        self.sim = df_sim
        self.agent_map = agent_map
        self.cbg_map = cbg_map
        self.ranges_dict = ranges_dict
        self.kl_metrics = RevisedKLMetrics(
            df_real, df_sim, agent_map, cbg_map, POI_CATEGORIES,
            cbg_coords or {}, poi_coords or {}, devices, sampling_threshold)

        # Pre-calc Tracts
        for df in [self.real, self.sim]:
            df['home_tract'] = get_tract_id(df['home_cbg'])
            df['poi_tract'] = get_tract_id(df['poi_cbg'])

    # --- Table 1 Metrics ---

    def metric_trip_distance_kl(self):
        return self.kl_metrics.trip_distance()

    def metric_od_flow_cpc(self):
        # Tract Level
        tables = self.kl_metrics.spatial_inputs()
        if tables is None:
            return float('nan')
        real, sim = tables
        gr = real.groupby(['home_tract', 'poi_tract'])['weight'].sum()
        gs = sim.groupby(['home_tract', 'poi_tract'])['weight'].sum()

        df = pd.DataFrame({'r': gr, 's': gs}).fillna(0)
        r, s = df['r'].values, df['s'].values
        if r.sum() == 0 or s.sum() == 0: return float('nan')
        return np.sum(np.minimum(r / r.sum(), s / s.sum()))

    def metric_visits_per_location_ks(self):
        tables = self.kl_metrics.spatial_inputs()
        if tables is None:
            return float('nan')
        locations = sorted(self.kl_metrics.valid_cbgs)
        if not locations:
            return float('nan')
        vectors = [frame.groupby('poi_cbg')['weight'].sum().reindex(
            locations, fill_value=0).to_numpy() for frame in tables]
        return visits_per_location_ks(*vectors)

    def metric_poi_proportion_kl(self):
        return self.kl_metrics.poi_proportion()

    def metric_stratified_od_fidelity(self):
        from evaluation.evaluation_stratified import stratified_cpc
        tables = self.kl_metrics.spatial_inputs()
        if tables is None:
            return float('nan')
        return stratified_cpc(*tables, self.cbg_map, self.agent_map)

    # --- Fundamental Laws ---

    def analyze_explorer_returner(self, tables=None):
        k_values = np.arange(1, 51)
        real, sim = tables if tables is not None else (self.real, self.sim)
        real_curve = returner_fraction(real, self.kl_metrics.cbg_coords,
                                       self.kl_metrics.poi_coords, k_values)
        sim_curve = returner_fraction(sim, self.kl_metrics.cbg_coords,
                                      self.kl_metrics.poi_coords, k_values)
        return {
            'k_values': k_values,
            'empirical_fraction': real_curve,
            'simulated_fraction': sim_curve,
            'empirical_k_star': returner_transition(k_values, real_curve),
            'simulated_k_star': returner_transition(k_values, sim_curve),
            'mae': float(np.mean(np.abs(sim_curve - real_curve))),
        }

    def analyze_fundamental_laws(self):
        tables = self.kl_metrics._table1_inputs()
        zipf_rmse = (rank_frequency_rmse(*tables, self.kl_metrics.poi_coords)
                     if tables is not None else float('nan'))

        # 2. Rg (CBG-aggregated visit-weighted activity centers)
        rg_median, rg_kl = self.kl_metrics.radius_of_gyration()

        # 3. Law 3 (empirical and simulated home-CBG activity profiles)
        if tables is not None:
            returners = self.analyze_explorer_returner(tables)
            k_star, mae = returners['simulated_k_star'], returners['mae']
        else:
            k_star = mae = float('nan')

        return zipf_rmse, rg_median, rg_kl, k_star, mae

    # --- Social Segregation ---

    def metric_experienced_segregation(self):
        tables = self.kl_metrics.spatial_inputs()
        if tables is None:
            return float('nan')
        sim = tables[1]
        return experienced_segregation(sim, self.cbg_map)

    def metric_home_stay_rate(self):
        tables = self.kl_metrics.spatial_inputs()
        if tables is None:
            return float('nan')
        sim = tables[1]
        total = sim.weight.sum()
        return float(sim.loc[sim.home_cbg == sim.poi_cbg, 'weight'].sum() / total) if total > 0 else float('nan')

    def metric_income_rank_correlation(self):
        tables = self.kl_metrics.spatial_inputs()
        return income_rank_correlation(tables[1], self.cbg_map) if tables is not None else float('nan')


# ==============================================================================
# 5. Agent Parameters
# ==============================================================================

def literal_policy_parameters(code):
    functions = [node for node in ast.parse(code).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'policy_function']
    if len(functions) != 1:
        raise ValueError('Expected one policy_function definition.')
    values = {}

    def resolve(node):
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Dict):
            return {resolve(k): resolve(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, (ast.List, ast.Tuple)):
            return [resolve(item) for item in node.elts]
        return ast.literal_eval(node)

    for node in functions[0].body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            value = resolve(node.value)
            for target in node.targets:
                values[target.id] = value
        elif isinstance(node, ast.Return):
            result = resolve(node.value)
            if isinstance(result, dict):
                return result
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                return dict(scores=result[0], cbg_preferences=result[1])
            raise ValueError('Unsupported policy return format.')
        else:
            raise ValueError('Policy parameters require computation; summary omitted.')
    raise ValueError('No constant policy return value.')


def analyze_params(config):
    path = config['paths']['policy_functions']
    if not os.path.exists(path): return None
    with open(path, 'r') as f:
        pols = json.load(f)

    pus, ws, a_inc, a_race = [], [], [], []
    if not isinstance(pols, (list, dict)):
        warnings.warn('Policy parameter summary unavailable: expected a list or mapping.')
        return None
    for row in (pols if isinstance(pols, list) else pols.values()):
        try:
            p = literal_policy_parameters(row['policy_function_code']) if 'policy_function_code' in row else row
            probs = p.get('probs', p.get('exploration_probs', []))
            scores = p.get('scores', p.get('interest_scores', []))
            prefs = p.get('cbg_preferences', {})
            arrays = [np.asarray(v, dtype=float) for v in
                      [probs, scores, list(prefs.get('income', {}).values()), list(prefs.get('race', {}).values())]]
            if any(a.ndim != 1 or not np.isfinite(a).all() for a in arrays):
                raise ValueError('Parameter vectors must be finite one-dimensional arrays.')
            for target, values, statistic in zip([pus, ws, a_inc, a_race], arrays,
                                                  [np.mean, np.std, np.mean, np.mean]):
                if values.size:
                    target.append(float(statistic(values)))
        except (SyntaxError, ValueError, TypeError, KeyError, AttributeError) as exc:
            warnings.warn(f'Policy parameter summary skipped an unsupported policy: {exc}')
    if not any([pus, ws, a_inc, a_race]):
        warnings.warn('Policy parameter summary unavailable: no constant parameter vectors.')
        return None

    def summarize(values, statistic=np.mean):
        return float(statistic(values)) if values else float('nan')

    return {
        'Pu_Mean': summarize(pus),
        'w_SD': summarize(ws),
        'A_Inc_Mean': summarize(a_inc), 'A_Inc_SD': summarize(a_inc, np.std),
        'A_Race_Mean': summarize(a_race), 'A_Race_SD': summarize(a_race, np.std)
    }


# ==============================================================================
# 6. Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate mobility distributions, activity patterns, and social metrics.')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output', help='Optional metrics JSON output path')
    args = parser.parse_args()
    config = load_config(args.config)
    proc = DataProcessor(config)
    proc.load_metadata()

    df_real = proc.process_real_data()
    df_sim = proc.process_sim_data()
    agent_map, cbg_map, ranges = proc.load_profiles()

    ev = MobilityEvaluator(df_real, df_sim, agent_map, cbg_map, ranges,
                           proc.cbg_centroids, proc.poi_coords, proc.devices,
                           proc.evaluation.get('sampling_threshold', 0.0))

    print("\n" + "=" * 50)
    print("Table 1: Macro-Regularity Alignment")
    print("=" * 50)
    print(f"(a) Trip Distance (KL):      {ev.metric_trip_distance_kl():.3f}")
    print(f"(b) OD Flow (CPC, Tract):    {ev.metric_od_flow_cpc():.3f}")
    print(f"(c) Visits per location (KS):{ev.metric_visits_per_location_ks():.3f}")
    print(f"(d) POI Proportion (KL):     {ev.metric_poi_proportion_kl():.3f}")
    print(f"(e) Stratified OD (CPC):     {ev.metric_stratified_od_fidelity():.3f}")

    print("\n" + "=" * 50)
    print("Figure 3: Fundamental Laws")
    print("=" * 50)
    z_rmse, rg_med, rg_kl, k_star, law3_mae = ev.analyze_fundamental_laws()
    print(f"Law 1 (Zipf):      RMSE = {z_rmse:.3f}")
    print(f"Law 2 (Rg):        Median = {rg_med:.2f} km, KL = {rg_kl:.3f}")
    print(f"Law 3 (Expl/Ret):  k* = {k_star:.2f}, MAE = {law3_mae:.3f}")

    print("\n" + "=" * 50)
    print("Figure 4/5: Social & Behavioral")
    print("=" * 50)
    segregation = ev.metric_experienced_segregation()
    print(f"Experienced Segregation (S): {segregation:.3f}")
    print(f"Home Stay Rate:              {ev.metric_home_stay_rate():.3f}")
    print(f"Income Rank Correlation:     {ev.metric_income_rank_correlation():.3f}")

    metrics = dict(trip_distance_kl=ev.metric_trip_distance_kl(),
        od_flow_cpc_tract=ev.metric_od_flow_cpc(),
        visits_per_location_ks=ev.metric_visits_per_location_ks(),
        poi_proportion_kl=ev.metric_poi_proportion_kl(),
        stratified_od_cpc=ev.metric_stratified_od_fidelity(),
        rank_frequency_rmse=z_rmse, radius_of_gyration_median_km=rg_med,
        radius_of_gyration_kl=rg_kl, returner_k_star=k_star, returner_mae=law3_mae,
        experienced_segregation=segregation,
        home_stay_rate=ev.metric_home_stay_rate(),
        income_rank_correlation=ev.metric_income_rank_correlation())
    boundaries = config['paths'].get('cbg_boundaries')
    if boundaries and proc.devices is not None:
        try:
            from evaluation.evaluation_hotspots import hotspot_statistics, load_boundaries
            geo = load_boundaries(boundaries,
                proc.evaluation.get('boundary_cbg_column', 'GEOID'),
                proc.evaluation.get('year', 2019))
            tables = ev.kl_metrics.spatial_inputs()
            flows = {name: frame.groupby(['home_cbg', 'poi_cbg']).weight.sum().to_dict()
                     for name, frame in zip(['Real Data', 'TIMA'], tables)}
            hot, metadata = hotspot_statistics(geo, flows,
                {c: income_level(p) for c, p in cbg_map.items()}, ev.kl_metrics.valid_cbgs)
            metrics['hotspots'] = metadata
            target = proc.evaluation.get('hotspot_output')
            if target:
                hot.to_csv(target, index=False)
            print('Hotspot overlap:', metadata['groups'])
        except ImportError as exc:
            warnings.warn(f'Hotspots unavailable: install geopandas, libpysal and esda ({exc}).')
    else:
        warnings.warn('Hotspots unavailable: set paths.cbg_boundaries and paths.home_panel_summary.')
    if args.output:
        def finite_json(value):
            if isinstance(value, dict):
                return {key: finite_json(item) for key, item in value.items()}
            if isinstance(value, list):
                return [finite_json(item) for item in value]
            return None if isinstance(value, float) and not np.isfinite(value) else value
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(finite_json(metrics), handle, indent=2, allow_nan=False)

    print("\n" + "=" * 50)
    print("Supp. Table 10: Agent Parameters")
    print("=" * 50)
    p = analyze_params(config)
    if p:
        print(f"Exploration Prob (Pu, Mean): {p['Pu_Mean']:.3f}")
        print(f"Interest Score (w, SD):      {p['w_SD']:.3f}")
        print(f"Affinity Income [Mean(SD)]:  {p['A_Inc_Mean']:.3f} ({p['A_Inc_SD']:.3f})")
        print(f"Affinity Race [Mean(SD)]:    {p['A_Race_Mean']:.3f} ({p['A_Race_SD']:.3f})")
