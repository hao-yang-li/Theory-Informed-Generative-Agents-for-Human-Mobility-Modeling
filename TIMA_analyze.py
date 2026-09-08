# -*- coding: utf-8 -*-
"""
Evaluate mobility distributions, activity patterns, and social metrics.
"""

import os
import json
import warnings
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
from scipy.stats import entropy, linregress
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error
from evaluation_kl import calculate_kl, RevisedKLMetrics

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
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_tract_id(cbg_series):
    return cbg_series.astype(str).str.slice(0, 11)


def haversine(lat1, lon1, lat2, lon2):
    """Vectorized Haversine Distance (km)"""
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

def discretize_value(value, ranges_dict, category_name):
    if pd.isna(value): return 'Medium'

    if isinstance(value, str):
        v_lower = value.strip().lower()
        if v_lower in ['low', 'medium', 'high']:
            return v_lower.capitalize()

    try:
        val_float = float(value)
    except (ValueError, TypeError):
        return 'Medium'

    category_ranges = ranges_dict.get(category_name.lower(), {})
    if not category_ranges: return 'Medium'

    for level, (low, high) in category_ranges.items():
        if low <= val_float < high:
            return level.capitalize()

    return 'High'  # Assign values above the configured bounds to High.


def get_dominant_attribute(profile, attribute, ranges_dict):
    """Extract dominant attribute from CBG profile."""
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
        return discretize_value(profile.get("home_cbg_edu", np.nan), ranges_dict, "education")
    elif attribute == 'income':  # Mapped to income_level
        return discretize_value(profile.get("home_cbg_income", np.nan), ranges_dict, "income")
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

            # Get POI Coords
            if pid in self.poi_coords:
                p_lat, p_lon = self.poi_coords[pid]
            else:
                continue  # Skip POIs with unavailable coordinates.

            try:
                # Decode home-CBG visit counts from JSON.
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

        return self.agent_map, self.cbg_map, self.ranges_dict


# ==============================================================================
# 4. Evaluator
# ==============================================================================

def returner_fraction(frame, cbg_coords, poi_coords, k_values):
    """Fraction of home-CBG activity profiles satisfying r_g(k) > r_g / 2."""
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
    """Interpolate the 0.5 crossing in log10(k)."""
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
        gr = self.real.groupby(['home_tract', 'poi_tract'])['count'].sum()
        gs = self.sim.groupby(['home_tract', 'poi_tract'])['count'].sum()

        df = pd.DataFrame({'r': gr, 's': gs}).fillna(0)
        r, s = df['r'].values, df['s'].values
        if r.sum() == 0 or s.sum() == 0: return 0.0
        return np.sum(np.minimum(r / r.sum(), s / s.sum()))

    def metric_visitation_density_mse(self):
        # CBG Level (Log-Norm)
        gr = self.real.groupby('poi_cbg')['count'].sum()
        gs = self.sim.groupby('poi_cbg')['count'].sum()

        locs = sorted(list(set(gr.index) | set(gs.index)))
        vr = gr.reindex(locs, fill_value=0).values
        vs = gs.reindex(locs, fill_value=0).values

        lr = np.log1p(vr);
        nr = lr / lr.max()
        ls = np.log1p(vs);
        ns = ls / ls.max()
        return mean_squared_error(nr, ns)

    def metric_poi_proportion_kl(self):
        return self.kl_metrics.poi_proportion()

    def metric_stratified_od_fidelity(self):
        """
        Stratified CPC over 6 Dimensions:
        Income, Education, Race, Industry, Sex, Age Group.
        """
        dims = ['income', 'education', 'race', 'industry', 'sex', 'age_group']
        all_cpcs = []

        # 1. Attribute Injection
        # For Real Data (Map CBG -> Attribute)
        for dim in dims:
            self.real[dim] = self.real['home_cbg'].apply(
                lambda c: get_dominant_attribute(self.cbg_map.get(c, {}), dim, self.ranges_dict)
            )

        # For Sim Data (Map Agent -> Attribute)


        def get_agent_attr(aid, dim):
            prof = self.agent_map.get(str(aid), {})
            if dim == 'income':
                return discretize_value(prof.get('home_cbg_income'), self.ranges_dict, 'income')
            if dim == 'education':
                return discretize_value(prof.get('home_cbg_edu'), self.ranges_dict, 'education')
            # For others, use raw value (e.g. sex, race)
            return prof.get(dim)

        # Pre-compute agent attributes dataframe to speed up map
        agent_df = pd.DataFrame.from_dict(self.agent_map, orient='index')
        # Discretize the contextual attributes.
        agent_df['income'] = agent_df['home_cbg_income'].apply(
            lambda x: discretize_value(x, self.ranges_dict, 'income'))
        agent_df['education'] = agent_df['home_cbg_edu'].apply(
            lambda x: discretize_value(x, self.ranges_dict, 'education'))

        # Merge attributes to Sim Data
        self.sim = self.sim.merge(agent_df[dims], left_on='agent_id', right_index=True, how='left')

        # 2. Stratified Calculation
        for dim in dims:
            # Find common valid groups
            groups = set(self.real[dim].dropna().unique()) & set(self.sim[dim].dropna().unique())

            for grp in groups:
                sub_r = self.real[self.real[dim] == grp]
                sub_s = self.sim[self.sim[dim] == grp]

                if sub_r.empty or sub_s.empty: continue

                # CPC at CBG Level (Flow)
                fr = sub_r.groupby(['home_cbg', 'poi_cbg'])['count'].sum()
                fs = sub_s.groupby(['home_cbg', 'poi_cbg'])['count'].sum()

                df = pd.DataFrame({'r': fr, 's': fs}).fillna(0)
                r, s = df['r'].values, df['s'].values
                if r.sum() > 0 and s.sum() > 0:
                    cpc = np.sum(np.minimum(r / r.sum(), s / s.sum()))
                    all_cpcs.append(cpc)

        return np.mean(all_cpcs) if all_cpcs else 0.0

    # --- Fundamental Laws ---

    def analyze_explorer_returner(self):
        k_values = np.arange(1, 51)
        real_curve = returner_fraction(self.real, self.kl_metrics.cbg_coords,
                                       self.kl_metrics.poi_coords, k_values)
        sim_curve = returner_fraction(self.sim, self.kl_metrics.cbg_coords,
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
        # 1. Zipf (RMSE)
        fr = self.real.groupby('poi_id')['count'].sum().sort_values(ascending=False).values
        fs = self.sim.groupby('poi_id')['count'].sum().sort_values(ascending=False).values
        k = 50
        yr = np.log((fr[:k] / fr.sum()) + 1e-10)
        ys = np.log((fs[:k] / fs.sum()) + 1e-10)
        zipf_rmse = np.sqrt(mean_squared_error(yr, ys))

        # 2. Rg (CBG-aggregated visit-weighted activity centers)
        rg_median, rg_kl = self.kl_metrics.radius_of_gyration()

        # 3. Law 3 (empirical and simulated home-CBG activity profiles)
        returners = self.analyze_explorer_returner()
        k_star, mae = returners['simulated_k_star'], returners['mae']

        return zipf_rmse, rg_median, rg_kl, k_star, mae

    # --- Social Segregation ---

    def metric_experienced_segregation(self):
        # 1. Rank
        data = []
        for c, p in self.cbg_map.items():
            inc = p.get('home_cbg_income')
            if inc: data.append({'cbg': c, 'inc': inc})
        df_inc = pd.DataFrame(data).sort_values('inc')
        df_inc['rank'] = df_inc['inc'].rank()

        cbg_idx = {c: i for i, c in enumerate(df_inc['cbg'])}
        N = len(df_inc)

        # 2. D Matrix
        R = df_inc['rank'].values[:, np.newaxis]
        Dist = np.abs(R - R.T)
        D = np.zeros((N, N))
        for i in range(N):
            row = Dist[i, :]
            cnt = np.searchsorted(np.sort(row), row, side='left')
            D[i, :] = (cnt + 0.5) / (N - 1)

        # 3. V Matrix
        df = self.sim[self.sim['home_cbg'].isin(cbg_idx) & self.sim['poi_cbg'].isin(cbg_idx)].copy()
        df['u'] = df['home_cbg'].map(cbg_idx)
        df['l'] = df['poi_cbg'].map(cbg_idx)

        flow = df.groupby(['l', 'u'])['count'].sum().reset_index()
        V = coo_matrix((flow['count'], (flow['l'], flow['u'])), shape=(N, N)).toarray()

        # 4. S Index
        row_sums = V.sum(axis=1, keepdims=True) + 1e-10
        P = V / row_sums
        E = P @ D.T

        return 1.0 - (np.sum(V * E) / np.sum(V))

    def metric_home_stay_rate(self):
        return self.sim[self.sim['home_cbg'] == self.sim['poi_cbg']]['count'].sum() / self.sim['count'].sum()


# ==============================================================================
# 5. Agent Parameters
# ==============================================================================

def analyze_params(config):
    path = config['paths']['policy_functions']
    if not os.path.exists(path): return None
    with open(path, 'r') as f:
        pols = json.load(f)

    pus, ws, a_inc, a_race = [], [], [], []
    for _, p in pols.items():
        if 'exploration_probs' in p: pus.append(np.mean(p['exploration_probs']))
        if 'interest_scores' in p: ws.append(np.std(p['interest_scores']))
        if 'cbg_preferences' in p:
            inc = list(p['cbg_preferences'].get('income', {}).values())
            race = list(p['cbg_preferences'].get('race', {}).values())
            if inc: a_inc.append(np.mean(inc))
            if race: a_race.append(np.mean(race))

    return {
        'Pu_Mean': np.mean(pus),
        'w_SD': np.mean(ws),
        'A_Inc_Mean': np.mean(a_inc), 'A_Inc_SD': np.std(a_inc),
        'A_Race_Mean': np.mean(a_race), 'A_Race_SD': np.std(a_race)
    }


# ==============================================================================
# 6. Main
# ==============================================================================

if __name__ == "__main__":
    config = load_config("config.yaml")
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
    print(f"(c) Visitation Density (MSE):{ev.metric_visitation_density_mse():.3f}")
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
    print(f"Experienced Segregation (S): {ev.metric_experienced_segregation():.3f}")
    print(f"Home Stay Rate:              {ev.metric_home_stay_rate():.3f}")

    print("\n" + "=" * 50)
    print("Supp. Table 10: Agent Parameters")
    print("=" * 50)
    p = analyze_params(config)
    if p:
        print(f"Exploration Prob (Pu, Mean): {p['Pu_Mean']:.3f}")
        print(f"Interest Score (w, SD):      {p['w_SD']:.3f}")
        print(f"Affinity Income [Mean(SD)]:  {p['A_Inc_Mean']:.3f} ({p['A_Inc_SD']:.3f})")
        print(f"Affinity Race [Mean(SD)]:    {p['A_Race_Mean']:.3f} ({p['A_Race_SD']:.3f})")
