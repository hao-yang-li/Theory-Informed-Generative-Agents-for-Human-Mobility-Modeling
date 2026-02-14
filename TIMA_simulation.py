"""
TIMA Large-scale Simulator.
Implements a theory-informed generative agent framework where agents
follow physical laws (EPR) parameterized by LLM-generated logic.
"""

import os
import json
import pandas as pd
import numpy as np
import random
from collections import defaultdict, Counter, namedtuple
import glob
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
import types
import time
import yaml
import concurrent.futures
import shutil
import traceback
import multiprocessing as mp

POI_CATEGORIES = [
    'Wholesale & Retail Trade, Transportation and Warehousing',
    'Others',
    'Educational Services',
    'Health Care and Social Assistance',
    'Arts, Entertainment, and Recreation',
    'Accommodation and Food Services'
]

NAICS_TO_CATEGORY_MAP = {
    42: POI_CATEGORIES[0], 44: POI_CATEGORIES[0], 45: POI_CATEGORIES[0], 48: POI_CATEGORIES[0], 49: POI_CATEGORIES[0],
    61: POI_CATEGORIES[2], 62: POI_CATEGORIES[3], 71: POI_CATEGORIES[4], 72: POI_CATEGORIES[5],
}

PoiInfo = namedtuple('PoiInfo', ['poi_id', 'latitude', 'longitude', 'category', 'cbg'])

def load_config():
    with open("config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def wait_for_stable_json_file(file_path, stability_period=120, check_interval=10):
    print(f"Waiting for stable JSON file at: {file_path}")
    print(f"Will proceed once the file has been unmodified for {stability_period} seconds.")

    last_mod_time = None
    last_size = None

    while True:
        if not os.path.exists(file_path):
            print(f"...file not found. Checking again in {check_interval}s.")
            time.sleep(check_interval)
            continue

        try:
            current_mod_time = os.path.getmtime(file_path)
            current_size = os.path.getsize(file_path)

            if current_mod_time != last_mod_time or current_size != last_size:
                print(f"...file is being updated (size: {current_size} bytes). Resetting stability timer.")
                last_mod_time = current_mod_time
                last_size = current_size
                time.sleep(check_interval)
                continue

            idle_time = time.time() - last_mod_time
            if idle_time >= stability_period:
                print(f"...file has been stable for {int(idle_time)}s. Verifying JSON integrity...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)

                print("...JSON is valid and stable. Proceeding with simulation.")
                return

            else:
                print(f"...file is stable for {int(idle_time)}s. Waiting for {stability_period}s threshold...")
                time.sleep(check_interval)

        except json.JSONDecodeError:
            print(f"...file is stable but JSON is invalid (likely incomplete). Waiting for write to finish...")
            last_mod_time = None
            last_size = None
            time.sleep(check_interval)
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying in {check_interval}s.")
            time.sleep(check_interval)


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * 6371


def discretize_value(value, ranges_dict, category_name):
    if isinstance(value, str):
        val_lower = value.lower()
        if 'low' in val_lower: return 'Low'
        if 'high' in val_lower: return 'High'
        return 'Medium'
    try:
        value = float(value)
    except:
        return 'Medium'

    ranges = ranges_dict.get(category_name.lower(), {})
    if not ranges: return 'Medium'

    for level, (low, high) in ranges.items():
        if low <= value < high: return level.capitalize()

    vals = []
    for r in ranges.values(): vals.extend(list(r))
    if not vals: return 'Medium'
    if value < min(vals): return 'Low'
    return 'High'


def get_agent_type_key(agent_profile, ranges_dict):
    features = [str(agent_profile.get(k)) for k in ["sex", "age_group", "race", "industry"]]
    inc = discretize_value(agent_profile.get("home_cbg_income", 0), ranges_dict, "income")
    edu = discretize_value(agent_profile.get("home_cbg_edu", "Unk"), ranges_dict, "education")
    return tuple(features + [edu, inc])

class Agent:
    def __init__(self, agent_profile, policy_code, cbg_centroids, global_cbg_attrs, config):
        self.agent_id = agent_profile['id']
        self.home_cbg = agent_profile['CBG']
        self.profile = agent_profile
        self.policy_code = policy_code
        self.global_cbg_attrs = global_cbg_attrs

        self.d_max = config['simulation']['d_max_km']
        self.base_alpha = config['simulation']['alpha']
        self.enable_dynamic_alpha = config['simulation']['enable_dynamic_alpha']

        home_coords = cbg_centroids.get(self.home_cbg, {'latitude': 0, 'longitude': 0})
        self.current_lat = home_coords['latitude']
        self.current_lon = home_coords['longitude']
        self.current_cbg = self.home_cbg

        self.visited_pois = Counter()
        self.policy_func = None

        self.cached_scores = None
        self.cached_cbg_prefs = None
        self.cached_probs = None
        self.cached_alpha_mult = 1.0

    def compile_policy(self):
        if self.cached_scores is not None: return

        try:
            module = types.ModuleType("policy_module")
            exec(self.policy_code, module.__dict__)

            result = module.policy_function()

            if isinstance(result, dict):
                self.cached_scores = result.get('scores', [0.1] * 6)
                self.cached_cbg_prefs = result.get('cbg_preferences', {})
                self.cached_probs = result.get('probs', None)
                self.cached_alpha_mult = result.get('alpha_multiplier', None)

                if self.cached_probs is None:
                    self.cached_probs = result.get('exploration_probs')

            elif isinstance(result, (tuple, list)) and len(result) >= 2:
                self.cached_scores = result[0]
                self.cached_cbg_prefs = result[1]
                self.cached_probs = None
                self.cached_alpha_mult = 1.0

            else:
                raise ValueError("Unknown return format")

        except Exception as e:
            print(f"Policy compilation error for agent {self.agent_id}: {e}")
            self.cached_scores = [0.1] * 6
            self.cached_cbg_prefs = {'income': {'Medium': 1}, 'education': {'Medium': 1}}
            self.cached_probs = None
            self.cached_alpha_mult = 1.0

    def get_cbg_weight(self, target_cbg_id):
        if target_cbg_id not in self.global_cbg_attrs:
            return 1.0

        target_attrs = self.global_cbg_attrs[target_cbg_id]
        target_inc = target_attrs['income']
        target_race = target_attrs['race']

        prefs = self.cached_cbg_prefs

        w_inc = prefs.get('income', {}).get(target_inc, 1.0)
        w_race = prefs.get('race', {}).get(target_race, 1.0)
        return w_inc * w_race

    def decide_action(self):
        """
        Determines whether the agent explores a new location or returns to a familiar one.
        """
        if self.cached_scores is None: self.compile_policy()

        S = len(self.visited_pois)

        if self.cached_probs and len(self.cached_probs) == 6:
            if S < 5:
                idx = 0
            elif S < 10:
                idx = 1
            elif S < 15:
                idx = 2
            elif S < 20:
                idx = 3
            elif S < 25:
                idx = 4
            else:
                idx = 5

            p_explore = self.cached_probs[idx]

        # fallback if error
        else:
            print("no LLM EPR, fallback")
            if S == 0:
                p_explore = 1.0
            else:
                p_explore = 0.6 * (S ** (-0.21))

        p_explore = max(0.0, min(1.0, p_explore))

        return 'explore' if random.random() < p_explore else 'return'

    def explore(self, all_pois_data):
        """
        Executes a two-stage destination selection process:
        1. Category Intent: Based on semantic interest and local opportunities.
        2. Specific POI: Based on personalized distance friction and CBG affinity.
        """

        if self.cached_scores is None: self.compile_policy()

        nearby_counts = defaultdict(int)
        global_pois = defaultdict(list)

        for poi in all_pois_data.values():
            dist = haversine_distance(self.current_lat, self.current_lon, poi.latitude, poi.longitude)

            global_pois[poi.category].append((poi, dist))

            if dist <= self.d_max:
                nearby_counts[poi.category] += 1

        if not global_pois: return None

        # --- Stage 1: Category Selection ---
        cat_attractions = {}
        for idx, cat in enumerate(POI_CATEGORIES):
            score = self.cached_scores[idx]
            count = nearby_counts.get(cat, 0)
            opportunity = max(count, 0.1)
            cat_attractions[cat] = score * opportunity

        total_attr = sum(cat_attractions.values())
        if total_attr == 0: return None

        cats, probs = zip(*[(k, v / total_attr) for k, v in cat_attractions.items()])
        target_cat = np.random.choice(cats, p=probs)

        # --- Stage 2: POI Selection ---
        candidates_with_dist = global_pois[target_cat]
        if not candidates_with_dist: return None

        poi_gravities = {}
        poi_ids = []

        if self.cached_alpha_mult is None: #normal case, no dynamic alpha
            personal_alpha = self.base_alpha
        else:
            if self.enable_dynamic_alpha:
                personal_alpha = self.base_alpha * self.cached_alpha_mult
            else:
                personal_alpha = self.base_alpha

        # Gravity = CBG_Weight / Distance^Alpha
        for poi, dist in candidates_with_dist:
            dist_safe = max(0.01, dist)

            cbg_w = self.get_cbg_weight(poi.cbg)

            # Gravity Formula
            g = cbg_w / (dist_safe ** personal_alpha)

            poi_ids.append(poi.poi_id)
            poi_gravities[poi.poi_id] = g

        total_g = sum(poi_gravities.values())
        if total_g == 0: return random.choice(poi_ids)

        p_weights = [poi_gravities[pid] / total_g for pid in poi_ids]

        return np.random.choice(poi_ids, p=p_weights)

    def preferential_return(self):
        if not self.visited_pois: return None
        pois = list(self.visited_pois.keys())
        counts = list(self.visited_pois.values())
        total = sum(counts)
        return np.random.choice(pois, p=[c / total for c in counts])

    def move_to(self, poi_id, all_pois_data, step, file_handle):
        if not poi_id: return
        poi = all_pois_data[poi_id]
        dist = haversine_distance(self.current_lat, self.current_lon, poi.latitude, poi.longitude)
        action = 'return' if poi_id in self.visited_pois else 'explore'

        self.current_lat, self.current_lon = poi.latitude, poi.longitude
        self.current_cbg = poi.cbg
        self.visited_pois[poi_id] += 1

        record = {
            "agent_id": self.agent_id,
            "home_cbg": self.home_cbg,
            "time_step": step,
            "poi_id": poi_id,
            "poi_cbg": poi.cbg,
            "category": poi.category,
            "action": action,
            "dist_km": round(dist, 4)
        }

        file_handle.write(json.dumps(record) + "\n")


def get_robust_category(naics_raw):
    try:
        val = float(str(naics_raw))
        code_str = str(int(val))

        if len(code_str) >= 2:
            prefix = int(code_str[:2])
            return NAICS_TO_CATEGORY_MAP.get(prefix, 'Others')
    except (ValueError, TypeError):
        pass
    return 'Others'


def load_and_preprocess(config):
    print("Loading data...")
    paths = config['paths']
    sim_conf = config['simulation']

    ranges = json.load(open(paths['ranges']))
    ranges_dict = {}
    for cat, levels in ranges.items():
        if isinstance(levels, dict):
            ranges_dict[cat] = {l: (float(r[0]), float(r[1])) for l, r in levels.items()}

    all_agent_profiles = json.load(open(paths['agent_profiles']))

    if sim_conf['enable_region_filter']:
        prefix = str(sim_conf['valid_cbg_prefix'])
        agent_profiles = [p for p in all_agent_profiles if str(p['CBG']).startswith(prefix)]
        print(f"Filtered Agents: {len(agent_profiles)} (Region Prefix: {prefix})")
    else:
        agent_profiles = all_agent_profiles

    cbg_profiles = json.load(open(paths['cbg_profiles']))

    policy_functions_path = paths['policy_functions']

    wait_for_stable_json_file(policy_functions_path, stability_period=120)

    print(f"Loading policy functions from: {policy_functions_path}")
    policy_funcs = json.load(open(policy_functions_path))

    print("Preprocessing CBG Attributes...")
    global_cbg_attrs = {}
    for prof in cbg_profiles:
        cbg_id = prof['census_block_group']

        if sim_conf['enable_region_filter'] and not str(cbg_id).startswith(str(sim_conf['valid_cbg_prefix'])):
            continue

        race_dist = prof.get('race_distribution', {})
        dominant_race = max(race_dist, key=race_dist.get) if race_dist else 'Other'

        inc_level = discretize_value(prof.get('home_cbg_income', 0), ranges_dict, 'income')
        edu_level = discretize_value(prof.get('home_cbg_edu', 'Unk'), ranges_dict, 'education')

        global_cbg_attrs[cbg_id] = {
            'income': inc_level,
            'education': edu_level,
            'race': dominant_race
        }

    active_cbgs = set()
    for p in agent_profiles:
        raw_cbg = str(p['CBG']).split('.')[0].strip()
        active_cbgs.add(raw_cbg)
    print(f"Active CBGs count: {len(active_cbgs)}")

    print("Loading Geo Data...")
    try:
        geo_df = pd.read_csv(paths['cbg_geo_data'], dtype={'census_block_group': str})

        geo_df = geo_df[geo_df['census_block_group'].isin(active_cbgs)]

        centroids = geo_df.set_index('census_block_group')[['latitude', 'longitude']].astype(float).to_dict('index')
        print(f"Loaded {len(centroids)} CBG centroids.")
    except Exception as e:
        print(f"Error loading Geo Data: {e}")
        exit()

    print("Loading POIs with Map Filtering...")

    try:
        patterns = pd.read_csv(paths['weekly_patterns'], usecols=['poi_id', 'poi_cbg'], dtype=str)
        patterns.dropna(inplace=True)
        poi_cbg_map = dict(zip(patterns['poi_id'], patterns['poi_cbg']))
        print(f"Loaded POI-CBG mapping with {len(poi_cbg_map)} entries.")
    except Exception as e:
        print(f"Error loading Weekly Patterns: {e}")
        exit()

    all_pois = {}
    files_scanned = 0

    poi_files = glob.glob(paths['poi_data_pattern'])
    for f_path in poi_files:
        files_scanned += 1
        df = pd.read_csv(f_path,
                         dtype={'poi_id': str, 'naics_code': str, 'latitude': float, 'longitude': float})

        for _, row in df.iterrows():
            pid = str(row['poi_id']).strip()

            cbg = poi_cbg_map.get(pid)
            if not cbg or cbg not in active_cbgs:
                continue

            if sim_conf['enable_region_filter'] and not str(cbg).startswith(str(sim_conf['valid_cbg_prefix'])):
                continue

            try:
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                if pd.isna(lat) or pd.isna(lon): continue

                cat = get_robust_category(row['naics_code'])

                all_pois[pid] = PoiInfo(pid, lat, lon, cat, cbg)
            except:
                continue

    print(f"Scanned {files_scanned} files. Loaded {len(all_pois)} valid POIs.")

    cat_counts = Counter(p.category for p in all_pois.values())
    print("Loaded POI Category Distribution:", dict(cat_counts))

    if len(all_pois) == 0:
        print("!!! CRITICAL: No POIs loaded. Check file paths and CBG ID matching.")
        exit()

    return agent_profiles, policy_funcs, ranges_dict, centroids, global_cbg_attrs, all_pois


def worker_simulation(worker_id, agent_chunk, all_pois, steps, output_dir, queue):
    temp_file = os.path.join(output_dir, f"worker_{worker_id}.jsonl")

    for agent in agent_chunk:
        agent.compile_policy()

    batch_size = 100
    count = 0

    with open(temp_file, 'w') as f:
        for t in range(1, steps + 1):
            for agent in agent_chunk:
                act = agent.decide_action()
                target_id = None
                if act == 'explore':
                    target_id = agent.explore(all_pois)
                else:
                    target_id = agent.preferential_return()

                agent.move_to(target_id, all_pois, t, f)

                count += 1
                if count >= batch_size:
                    queue.put((worker_id, count))
                    count = 0

    if count > 0: queue.put((worker_id, count))
    return temp_file

def main():
    config = load_config()
    paths = config['paths']
    sim_conf = config['simulation']
    out_dir = paths['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    seed = sim_conf['seed']
    random.seed(seed)
    np.random.seed(seed)

    data = load_and_preprocess(config)
    agent_profs, policies, ranges, centroids, global_cbg_attrs, all_pois = data

    policy_map = {tuple(p['agent_type_key']): p['policy_function_code'] for p in policies}
    agents = []

    print("Creating Agents...")
    for prof in agent_profs:
        type_key = get_agent_type_key(prof, ranges)
        code = policy_map.get(type_key)
        if code:
            agents.append(Agent(prof, code, centroids, global_cbg_attrs, config))

    if not agents:
        print("Error: No agents created. Check matching between profiles and policies.")
        return

    num_workers = sim_conf['num_workers']
    steps = sim_conf['num_steps']
    print(f"Starting simulation with {num_workers} workers...")

    chunk_size = len(agents) // num_workers + 1
    chunks = [agents[i:i + chunk_size] for i in range(0, len(agents), chunk_size)]
    chunks = [c for c in chunks if c]
    real_num_workers = len(chunks)
    print(f"Actual workers needed: {real_num_workers}")

    manager = mp.Manager()
    queue = manager.Queue()

    bars = [tqdm(total=len(c) * steps, desc=f"W{i}", position=i) for i, c in enumerate(chunks)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=real_num_workers) as exc:
        futures = []
        for i, chunk in enumerate(chunks):
            futures.append(exc.submit(worker_simulation, i, chunk, all_pois, steps, out_dir, queue))

        finished = 0
        while finished < real_num_workers:
            while not queue.empty():
                try:
                    wid, n = queue.get_nowait()
                    if wid < len(bars):
                        bars[wid].update(n)
                except Exception:
                    break

            finished = sum(1 for f in futures if f.done())
            time.sleep(0.5)

        while not queue.empty():
            try:
                wid, n = queue.get_nowait()
                if wid < len(bars):
                    bars[wid].update(n)
            except Exception:
                break

        # 关闭进度条
        for bar in bars:
            bar.close()

        print("\nCollecting worker results...")
        res_files = []
        for i, f in enumerate(futures):
            try:
                result_path = f.result()
                res_files.append(result_path)
            except Exception as e:
                print(f"\n[!!! CRITICAL ERROR] Worker {i} Failed!")
                print(f"Error Message: {e}")
                print("--- Worker Stack Trace ---")
                traceback.print_exc()
                print("--------------------------\n")

    if not res_files:
        print("No output files generated successfully. Simulation failed.")
        return

    print(f"\nMerging {len(res_files)} files...")

    try:
        out_dir_str = str(out_dir)
        filename_str = str(paths['output_filename'])

        abs_output_path = os.path.abspath(os.path.join(out_dir_str, filename_str))

        if os.name == 'nt':
            abs_output_path = abs_output_path.replace('/', '\\')

            if not abs_output_path.startswith('\\\\?\\'):
                final_output_path = '\\\\?\\' + abs_output_path
            else:
                final_output_path = abs_output_path
        else:
            final_output_path = abs_output_path

        print(f"Target Output Path (Long Path Aware): {final_output_path}")

        with open(final_output_path, 'w', encoding='utf-8') as f_out:
            for tf in res_files:
                if tf and os.path.exists(tf):
                    try:
                        with open(tf, 'r', encoding='utf-8') as f_in:
                            shutil.copyfileobj(f_in, f_out)
                        os.remove(tf)
                    except Exception as e:
                        print(f"Error reading/merging temp file {tf}: {e}")
                else:
                    print(f"Warning: Temp file missing: {tf}")

        print(f"Done! Final output saved successfully.")

    except Exception as e:
        print(f"Error during final file merge: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()