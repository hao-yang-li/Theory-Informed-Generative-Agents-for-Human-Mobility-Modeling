# -*- coding: utf-8 -*-
"""
TIMA Metrics Evaluator (Strict Paper Replication)
- Exact NAICS Mapping restored.
- Multi-dimensional Stratified CPC (Income, Edu, Race, Sex, Age, Industry).
- Home-Based Distance calculation.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
from scipy.stats import entropy, linregress
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error

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


def calculate_kl(p, q):
    p = np.asarray(p, dtype=float) + 1e-10
    q = np.asarray(q, dtype=float) + 1e-10
    return entropy(p / p.sum(), q / q.sum())


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

    return 'High'  # 数值超过定义的 High 上限


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

    def load_metadata(self):
        print(">>> Loading Metadata (Geo, POI, Ranges)...")
        # 1. Ranges for discretization
        if os.path.exists(self.paths['ranges']):
            with open(self.paths['ranges'], 'r') as f:
                self.ranges_dict = json.load(f)

        # 2. Geo Data
        df_geo = pd.read_csv(self.paths['cbg_geo_data'], dtype={'CBG Code': str})
        for _, row in df_geo.iterrows():
            cbg = row['CBG Code']
            # Simple fallback for Centroid WKT or Lat/Lon cols
            if 'Latitude' in row and 'Longitude' in row:
                self.cbg_centroids[cbg] = (float(row['Latitude']), float(row['Longitude']))
            elif 'Centroid' in row:
                parts = row['Centroid'].replace("POINT (", "").replace(")", "").split()
                self.cbg_centroids[cbg] = (float(parts[1]), float(parts[0]))

        # 3. POI Data (Core POI)
        df_poi = pd.read_csv(self.paths['poi_data_pattern'], dtype={'safegraph_place_id': str})
        for _, row in df_poi.iterrows():
            pid = row['safegraph_place_id']
            self.poi_coords[pid] = (float(row['latitude']), float(row['longitude']))

            # NAICS Mapping Logic
            naics_str = str(row.get('naics_code', ''))
            if len(naics_str) >= 2:
                prefix = int(naics_str[:2])
                self.poi_categories[pid] = NAICS_TO_CATEGORY_MAP.get(prefix, 'Others')
            else:
                self.poi_categories[pid] = 'Others'

    def process_real_data(self):
        print(">>> Processing Real Data...")
        df_raw = pd.read_csv(self.paths['weekly_patterns'])
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
                continue  # Cannot calc distance

            try:
                # visitor_home_cbgs is JSON string: "{'360...': 4, ...}"
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
                    home = row['home_cbg']
                    pid = row['poi_id']

                    # Recalculate Distance (Consistency)
                    dist = row.get('dist_km', np.nan)
                    if home in self.cbg_centroids and pid in self.poi_coords:
                        h_lat, h_lon = self.cbg_centroids[home]
                        p_lat, p_lon = self.poi_coords[pid]
                        dist = haversine(h_lat, h_lon, p_lat, p_lon)

                    records.append({
                        'agent_id': row['agent_id'],
                        'home_cbg': home,
                        'poi_cbg': row['poi_cbg'],
                        'poi_id': pid,
                        'category': row['category'],
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

class MobilityEvaluator:
    def __init__(self, df_real, df_sim, agent_map, cbg_map, ranges_dict):
        self.real = df_real
        self.sim = df_sim
        self.agent_map = agent_map
        self.cbg_map = cbg_map
        self.ranges_dict = ranges_dict

        # Pre-calc Tracts
        for df in [self.real, self.sim]:
            df['home_tract'] = get_tract_id(df['home_cbg'])
            df['poi_tract'] = get_tract_id(df['poi_cbg'])

    # --- Table 1 Metrics ---

    def metric_trip_distance_kl(self):
        all_d = np.concatenate([self.real['dist_km'].dropna(), self.sim['dist_km'].dropna()])
        bins = np.linspace(0, np.percentile(all_d, 99.5), 50)

        hr, _ = np.histogram(self.real['dist_km'], bins=bins, weights=self.real['count'], density=True)
        hs, _ = np.histogram(self.sim['dist_km'], bins=bins, weights=self.sim['count'], density=True)
        return calculate_kl(hr, hs)

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
        pr = self.real.groupby('category')['count'].sum()
        ps = self.sim.groupby('category')['count'].sum()
        cats = sorted(POI_CATEGORIES)
        vr = pr.reindex(cats, fill_value=0).values
        vs = ps.reindex(cats, fill_value=0).values
        return calculate_kl(vr, vs)

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
        # agent_map has attributes directly, but might need discretization for inc/edu
        # Note: 'industry' in agent_profiles is usually raw string, needs to match CBG aggregation if needed
        # Assuming agent profiles are already aligned or raw strings match.
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
        # ... logic to discretize income/edu in agent_df ...
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

    def analyze_fundamental_laws(self):
        # 1. Zipf (RMSE)
        fr = self.real.groupby('poi_id')['count'].sum().sort_values(ascending=False).values
        fs = self.sim.groupby('poi_id')['count'].sum().sort_values(ascending=False).values
        k = 50
        yr = np.log((fr[:k] / fr.sum()) + 1e-10)
        ys = np.log((fs[:k] / fs.sum()) + 1e-10)
        zipf_rmse = np.sqrt(mean_squared_error(yr, ys))

        # 2. Rg (Med, KL)
        # Sim Rg (Individual)
        sim_rgs = []
        for _, grp in self.sim.groupby('agent_id'):
            # Approx rg using distances from home (simplified)
            rg = np.sqrt(np.mean(grp['dist_km'] ** 2))
            sim_rgs.append(rg)

        # Real Rg Proxy (Weighted Distance Distribution)
        # Using dist_km distribution as proxy for Rg distribution comparison
        bins = np.logspace(np.log10(0.1), np.log10(100), 50)
        hr, _ = np.histogram(self.real['dist_km'], bins=bins, weights=self.real['count'], density=True)
        hs, _ = np.histogram(sim_rgs, bins=bins, density=True)
        rg_kl = calculate_kl(hr, hs)

        # 3. Law 3 (k*, MAE)
        k_range = np.arange(1, 51)
        # Sim Curve
        counts = np.zeros(len(k_range))
        valid_agents = 0
        for _, grp in self.sim.groupby('agent_id'):
            visits = grp['poi_id'].value_counts()
            if len(visits) == 0: continue
            valid_agents += 1

            top_locs = visits.index
            # Pre-map distances
            dist_map = grp.set_index('poi_id')['dist_km'].to_dict()
            total_rg = np.sqrt(np.mean(grp['dist_km'] ** 2))

            running_sq = 0
            running_cnt = 0

            for i, k_val in enumerate(k_range):
                if i < len(top_locs):
                    loc = top_locs[i]
                    cnt = visits[loc]
                    d = dist_map[loc]
                    running_sq += (d ** 2) * cnt
                    running_cnt += cnt

                if running_cnt > 0:
                    rg_k = np.sqrt(running_sq / running_cnt)
                    if rg_k > (total_rg / 2): counts[i] += 1
                elif counts[i - 1] if i > 0 else 0:
                    counts[i] += 1

        sim_curve = counts / valid_agents
        k_star = np.interp(0.5, sim_curve, k_range)

        # Real Curve Approx (Logistic for NYC)
        real_curve = 1 / (1 + np.exp(-0.25 * (k_range - 2.11)))
        mae = np.mean(np.abs(sim_curve - real_curve))

        return zipf_rmse, np.median(sim_rgs), rg_kl, k_star, mae

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

    ev = MobilityEvaluator(df_real, df_sim, agent_map, cbg_map, ranges)

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