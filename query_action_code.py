import json
import os
import time
import re
import concurrent.futures
import textwrap
import random
from collections import defaultdict
import ast
import yaml
from tqdm import tqdm
from utils import LLM

POI_CATEGORIES = [
    'Wholesale & Retail Trade, Transportation and Warehousing',
    'Others',
    'Educational Services',
    'Health Care and Social Assistance',
    'Arts, Entertainment, and Recreation',
    'Accommodation and Food Services'
]

# Agent's persona, home cbg's income and education will be added later from home cbg's profile
AGENT_TYPE_FEATURES_FROM_AGENT = ["sex", "age_group", "race", "industry"]


def detect_variable_name(code_snippet, expected_type):
    try:
        tree = ast.parse(code_snippet)
        for node in reversed(tree.body):
            if isinstance(node, ast.Assign):
                targets = node.targets
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    var_name = targets[0].id

                    if expected_type == 'list' and isinstance(node.value, ast.List):
                        return var_name
                    elif expected_type == 'dict' and isinstance(node.value, ast.Dict):
                        return var_name
    except:
        pass
    return None

def extract_code_block(text):
    match_py = re.search(r'```python\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match_py:
        return match_py.group(1).strip()

    match_generic = re.search(r'```\s*(.*?)```', text, re.DOTALL)
    if match_generic:
        return match_generic.group(1).strip()

    lines = text.strip().split('\n')
    clean_lines = [l for l in lines if not l.strip().startswith('Here is') and not l.strip().startswith('Sure')]
    return '\n'.join(clean_lines).strip()


def combine_codes_to_function(code_interest, code_pref, code_dynamics):
    clean_1 = extract_code_block(code_interest)
    clean_2 = extract_code_block(code_pref)
    clean_3 = extract_code_block(code_dynamics)

    var_name_scores = detect_variable_name(clean_1, 'list') or 'scores'

    var_name_prefs = detect_variable_name(clean_2, 'dict') or 'cbg_preferences'

    var_name_probs = detect_variable_name(clean_3, 'list') or 'exploration_probs'

    indented_1 = textwrap.indent(clean_1, '    ')
    indented_2 = textwrap.indent(clean_2, '    ')
    indented_3 = textwrap.indent(clean_3, '    ')

    final_code = f"""def policy_function():
    # --- Part 1: Intrinsic Interest Scores ---
{indented_1}

    # --- Part 2: Neighborhood Context Preferences ---
{indented_2}

    # --- Part 3: Mobility Dynamics ---
{indented_3}

    return {{
        "scores": {var_name_scores},
        "cbg_preferences": {var_name_prefs},
        "probs": {var_name_probs}
    }}"""

    return final_code

def get_poi_index_from_industry_key(industry_key: str) -> int:
    key_lower = industry_key.lower()
    if any(word in key_lower for word in ['wholesale', 'retail', 'transportation', 'warehousing']):
        return 0
    elif any(word in key_lower for word in ['educational', 'health', 'social assistance']):
        if 'health' in key_lower:
            return 3
        else:
            return 2
    elif 'arts' in key_lower and ('entertainment' in key_lower or 'recreation' in key_lower):
        if 'accommodation' in key_lower or 'food' in key_lower:
            return 5
        else:
            return 4
    elif 'accommodation' in key_lower or 'food services' in key_lower:
        return 5
    elif any(word in key_lower for word in ['arts', 'entertainment', 'recreation']):
        return 4
    else:
        return 1

def normalize_poi_vector(poi_vector):
    total = sum(poi_vector)
    if total == 0:
        return [1.0 / len(poi_vector) for _ in poi_vector]
    return [count / total for count in poi_vector]

def create_poi_distribution_from_industry_counts(industry_counts):
    poi_vector = [0.0] * 6
    for industry_key, count in industry_counts.items():
        poi_index = get_poi_index_from_industry_key(industry_key)
        poi_vector[poi_index] += count
    return normalize_poi_vector(poi_vector)

def load_cbg_profiles_to_dict(cbg_profiles_path):
    with open(cbg_profiles_path, 'r') as f:
        cbg_profiles_list = json.load(f)

    cbg_profiles_dict = {}
    for cbg_profile in cbg_profiles_list:
        cbg_id = cbg_profile["census_block_group"]
        poi_probs = create_poi_distribution_from_industry_counts(cbg_profile["industry_counts"])
        cbg_profiles_dict[cbg_id] = {
            "home_cbg_edu": cbg_profile["home_cbg_edu"],
            "home_cbg_income": cbg_profile["home_cbg_income"],
            "home_cbg_poi_probs": poi_probs,
        }
    return cbg_profiles_dict

def load_ranges(ranges_path):
    with open(ranges_path, 'r') as f:
        return json.load(f)

def discretize_value(value, ranges_dict, category_name):
    if isinstance(value, str):
        value_lower = value.lower()
        if 'low' in value_lower:
            return 'Low'
        elif 'medium' in value_lower:
            return 'Medium'
        elif 'high' in value_lower:
            return 'High'
        else:
            return 'Medium'

    try:
        value = float(value)
    except (ValueError, TypeError):
        return 'Medium'

    category_ranges = ranges_dict.get(category_name.lower(), {})
    if not category_ranges: return 'Medium'

    for level, (low, high) in category_ranges.items():
        if low <= value < high: return level.capitalize()

    min_low = min(r[0] for r in category_ranges.values())
    max_high = max(r[1] for r in category_ranges.values())
    if value < min_low:
        return 'Low'
    elif value >= max_high:
        return 'High'
    else:
        return 'Medium'

def get_agent_type_key(agent_profile_data, ranges_dict):
    agent_features = [str(agent_profile_data.get(f, 'unknown')) for f in AGENT_TYPE_FEATURES_FROM_AGENT]
    raw_income = agent_profile_data.get("home_cbg_income", 0)
    raw_edu = agent_profile_data.get("home_cbg_edu", "Unknown")

    discretized_income = discretize_value(raw_income, ranges_dict, "income")
    discretized_edu = discretize_value(raw_edu, ranges_dict, "education")

    cbg_features = [discretized_edu, discretized_income]
    type_key = tuple(agent_features + cbg_features)
    return type_key

def clean_code_string(code):
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"): code = code[:-3]
    return code.strip()

def recover_broken_json(file_path):
    if not os.path.exists(file_path): return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("Warning: JSON file corrupted. Attempting recovery...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content: return []

        last_brace_index = content.rfind('}')
        if last_brace_index == -1: return []

        fixed_content = content[:last_brace_index + 1]
        if not fixed_content.strip().startswith('['): fixed_content = '[' + fixed_content
        if not fixed_content.strip().endswith(']'): fixed_content = fixed_content + ']'

        data = json.loads(fixed_content)
        print(f"Recovered {len(data)} records.")
        return data
    except Exception as e:
        print(f"Failed to recover: {e}")
        return []

def generate_interest_prompt(agent_profile_data, home_cbg_poi_probs):
    """Task 1: POI Category interest score"""
    agent_sex = agent_profile_data["sex"]
    agent_age_group = agent_profile_data["age_group"]
    agent_race = agent_profile_data["race"]
    agent_industry = agent_profile_data["industry"]
    home_cbg_edu_level = agent_profile_data.get("discretized_home_cbg_edu", "Unknown")
    home_cbg_income_level = agent_profile_data.get("discretized_home_cbg_income", "Unknown")

    prompt = f"""
You are a resident living in a city, your task is to define your mobility behavior by writing a Python code snippet.

---
**Your Resident Profile**
- **Your Sex:** {agent_sex}
- **Your Age Group:** {agent_age_group}
- **Your Race:** {agent_race}
- **Your Job Sector:** {agent_industry}
- **Your Home Neighborhood's General Education Level:** {home_cbg_edu_level}
- **Your Home Neighborhood's General Income Level:** {home_cbg_income_level}
- **Your Home Neighborhood's Vibe (Context Only):** {home_cbg_poi_probs}
  * (Note: This shows what is currently physically around you. It sets the context, but **DO NOT** simply give high scores to POI types just because they are abundant nearby. Output your **intrinsic** interests based on your age, job, etc.)

---
**Your Task: Define Interest Scores**
Write a Python code snippet to define a list named `scores` containing 6 floats (0.0 to 1.0).

The POI types correspond to the list indices:
0: '{POI_CATEGORIES[0]}'
1: '{POI_CATEGORIES[1]}'
2: '{POI_CATEGORIES[2]}'
3: '{POI_CATEGORIES[3]}'
4: '{POI_CATEGORIES[4]}'
5: '{POI_CATEGORIES[5]}'

**Requirements:**
1. Define a list `scores = [...]`.
2. Add comments explaining your logic based on your profile.

**Output Format Constraint:**
Return ONLY the Python code block wrapped in ```python ... ```.

**Example:**
```python
# As a <demographic> person...
scores = [<float>, <float>, <float>, <float>, <float>, <float>]
```
"""
    return prompt

def generate_preference_prompt(agent_profile_data, home_cbg_poi_probs):
    """Task 2: CBG Socio-economic Preference"""
    agent_sex = agent_profile_data["sex"]
    agent_age_group = agent_profile_data["age_group"]
    agent_race = agent_profile_data["race"]
    agent_industry = agent_profile_data["industry"]
    home_cbg_edu_level = agent_profile_data.get("discretized_home_cbg_edu", "Unknown")
    home_cbg_income_level = agent_profile_data.get("discretized_home_cbg_income", "Unknown")

    prompt = f"""
You are a resident living in a city. Your task is to define your neighborhood preferences.

---
**Your Resident Profile**
- **Your Sex:** {agent_sex}
- **Your Age Group:** {agent_age_group}
- **Your Race:** {agent_race}
- **Your Job Sector:** {agent_industry}
- **Your Home Neighborhood's General Education Level:** {home_cbg_edu_level}
- **Your Home Neighborhood's General Income Level:** {home_cbg_income_level}
- **Your Home Neighborhood's Vibe (Context Only):** {home_cbg_poi_probs}
  * (Note: This shows what is currently physically around you. It sets the context, but **DO NOT** simply give high scores to POI types just because they are abundant nearby. Output your **intrinsic** interests based on your age, job, etc.)

---
The POI types correspond to the list indices:
0: '{POI_CATEGORIES[0]}'
1: '{POI_CATEGORIES[1]}'
2: '{POI_CATEGORIES[2]}'
3: '{POI_CATEGORIES[3]}'
4: '{POI_CATEGORIES[4]}'
5: '{POI_CATEGORIES[5]}'

---
**Your Task: Define CBG Preferences**
Write a Python code snippet to define a dictionary named `cbg_preferences`.

**Requirements:**
1. The dictionary MUST have keys 'income' and 'race'.
2. 'income' maps 'High', 'Medium', 'Low' to a score (<float> 0.5 to 1.5).
3. 'race' maps 'White', 'Black', 'Other' to a score (<float> 0.5 to 1.5).
4. **1.0 is neutral**. Higher means preference (homophily), lower means avoidance.
5. Add comments explaining your logic.

**Output Format Constraint:**
Return ONLY the Python code block wrapped in ```python ... ```.

**Example:**
```python
# Based on my profile...
cbg_preferences = {{
    'income': {{'High': <float>, 'Medium': <float>, 'Low': <float>}},
    'race': {{'White': <float>, 'Black': <float>, 'Other': <float>}}
}}
```
"""
    return prompt


def generate_dynamics_prompt(agent_profile_data, home_cbg_poi_probs):
    """Task 3: exploration probs"""
    agent_sex = agent_profile_data["sex"]
    agent_age_group = agent_profile_data["age_group"]
    agent_race = agent_profile_data["race"]
    agent_industry = agent_profile_data["industry"]
    home_cbg_edu_level = agent_profile_data.get("discretized_home_cbg_edu", "Unknown")
    home_cbg_income_level = agent_profile_data.get("discretized_home_cbg_income", "Unknown")

    prompt = f"""
You are a resident living in a city. Your task is to define your mobility behavior by writing a Python code snippet.

**Context: Mobility Reproduce**
Imagine you are living in a city and you need to decide:
How likely you are to **Explore** NEW places vs. **Return** to places you have already visited.

**Definitions:**
- **Explore:** Choosing to visit a brand new place you haven't been to before.
- **Return:** Choosing to revisit a place you have been to before (Preferential Return). You are more likely to return to places you visit frequently.

---
**Your Resident Profile**
- **Your Sex:** {agent_sex}
- **Your Age Group:** {agent_age_group}
- **Your Race:** {agent_race}
- **Your Job Sector:** {agent_industry}
- **Your Home Neighborhood's General Education Level:** {home_cbg_edu_level}
- **Your Home Neighborhood's General Income Level:** {home_cbg_income_level}
- **Your Home Neighborhood's Vibe (Context Only):** {home_cbg_poi_probs}
  * (Note: This shows what is currently physically around you. It sets the context, but **DO NOT** simply give high scores to POI types just because they are abundant nearby. Output your **intrinsic** interests based on your age, job, etc.)

---
The POI types correspond to the list indices:
0: '{POI_CATEGORIES[0]}'
1: '{POI_CATEGORIES[1]}'
2: '{POI_CATEGORIES[2]}'
3: '{POI_CATEGORIES[3]}'
4: '{POI_CATEGORIES[4]}'
5: '{POI_CATEGORIES[5]}'


---
**Your Task: Define Mobility Dynamics**
Write a Python code snippet to define `exploration_probs`.

**Requirements:**

1.  **`exploration_probs` (List of 6 floats):**
    - A list of 6 numbers between 0.0 and 1.0.
    - This represents your probability of choosing to **EXPLORE** a new place based on how many unique places ($S$) you have *already* visited this week.
    - As $S$ increases (you know more places), you typically tend to **Return** more (so exploration prob decreases).
    - Provide probabilities for these specific distinct visit counts:
        - **S = 0:** (Start of the week, you know nowhere. Usually high.)
        - **S = 5:** (You know 5 places.)
        - **S = 10:** (You know 10 places.)
        - **S = 15:** (You know 15 places.)
        - **S = 20:** (You know 20 places.)
        - **S >= 25:** (You know 25+ places. Routine is likely established.)

2.  **Add Comments:** Include brief Python comments (`#`) to explain your thinking based on your profile.

**Output Format Constraint:**
Return ONLY the Python code block wrapped in ```python ... ``` containing the definitions of the two variables.

**Example:**
```python
# Exploration probabilities based on visited count (S)
# S=0, S=5, S=10, S=15, S=20, S>=25
exploration_probs = [<float>, <float>, <float>, <float>, <float>, <float>] 
```
"""
    return prompt

def validate_generated_code(code_string, agent_type_key_str):
    """validate the generated code"""
    if "def policy_function():" not in code_string: return False

    try:
        compiled_code = compile(code_string, '<string>', 'exec')
        local_namespace = {}
        exec(compiled_code, {}, local_namespace)
        func = local_namespace.get('policy_function')

        if not callable(func): return False

        result = func()

        if not isinstance(result, dict):
            # print(f"Validation failed: Result is not a dict. Type: {type(result)}")
            return False

        required_keys = ["scores", "cbg_preferences", "probs"]
        if not all(k in result for k in required_keys):
            # print(f"Validation failed: Missing keys. Found: {list(result.keys())}")
            return False

        scores = result["scores"]
        if not isinstance(scores, list) or len(scores) != 6: return False
        if not all(isinstance(s, (int, float)) and 0 <= s <= 1 for s in scores): return False

        cbg_prefs = result["cbg_preferences"]
        if not isinstance(cbg_prefs, dict): return False

        for key in ['income', 'race']:
            if key not in cbg_prefs: return False
            sub_dict = cbg_prefs[key]
            if not isinstance(sub_dict, dict): return False
            expected_levels = ['High', 'Medium', 'Low'] if key == 'income' else ['White', 'Black', 'Other']
            for level in expected_levels:
                if level not in sub_dict: return False
                val = sub_dict[level]
                if not (isinstance(val, (int, float)) and 0.5 <= val <= 1.5):
                    # print(f"Validation failed: {key}-{level} value {val} out of range [0.5, 1.5]")
                    return False

        probs = result["probs"]
        if not isinstance(probs, list) or len(probs) != 6: return False
        if not all(isinstance(p, (int, float)) and 0 <= p <= 1 for p in probs): return False

    except Exception as e:
        # print(f"Validation execution error: {e}")
        return False

    return True


def generate_single_agent_policy_split(type_key, profile_data, home_poi_probs, llm_client, temperature):
    type_key_str = str(type_key)

    prompt_interest = generate_interest_prompt(profile_data, home_poi_probs)
    prompt_pref = generate_preference_prompt(profile_data, home_poi_probs)
    prompt_dynamics = generate_dynamics_prompt(profile_data, home_poi_probs)

    retry_count = 0

    while True:
        try:
            # === Call 1: Interest Scores ===
            resp1 = llm_client.generate(prompt=prompt_interest, response_format={"type": "text"}, temperature=temperature)

            # === Call 2: CBG Preferences ===
            resp2 = llm_client.generate(prompt=prompt_pref, response_format={"type": "text"}, temperature=temperature)

            # === Call 3: Mobility Dynamics ===
            resp3 = llm_client.generate(prompt=prompt_dynamics, response_format={"type": "text"}, temperature=temperature)

            final_code = combine_codes_to_function(resp1, resp2, resp3)

            # === Validation ===
            if validate_generated_code(final_code, type_key_str):
                # 成功
                return {
                    "agent_type_key": list(type_key),
                    "policy_function_code": final_code
                }
            else:
                retry_count += 1
                if retry_count % 5 == 0:
                    print(f"\n[Validation Failed] Agent: {type_key_str} (Attempt {retry_count})")
                    print(final_code)

        except Exception as e:
            retry_count += 1
            print(f"Error for {type_key_str}: {e}. Retrying... (Attempt {retry_count})")
            time.sleep(1)

def load_config_secrets():
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if os.path.exists("secrets.yaml"):
        with open("secrets.yaml", 'r', encoding='utf-8') as f:
            secrets = yaml.safe_load(f)
    else:
        secrets = {}
    return config, secrets


def main():
    config, secrets = load_config_secrets()

    # 1. Setup Parameters from Config
    llm_conf = config['llm_generation']
    paths = config['paths']

    MODEL_NAME = llm_conf['model_name']
    PLATFORM = llm_conf['platform']
    TEM = llm_conf['temperature']
    NUM_WORKERS = config['simulation']['num_workers']

    # 2. Get API Key
    api_key = secrets.get('api_keys', {}).get(PLATFORM, "")
    if not api_key:
        raise ValueError(f"API Key for {PLATFORM} not found in secrets.yaml")

    # 3. Setup Paths
    agent_profiles_path = paths['agent_profiles']
    cbg_profiles_path = paths['cbg_profiles']
    ranges_path = paths['ranges']
    output_file_path = paths['policy_functions']

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    print("Loading data...")
    ranges_dict_raw = load_ranges(ranges_path)
    ranges_dict = {}

    print("Loading CBG profiles for Vibe context...")
    cbg_profiles_dict = load_cbg_profiles_to_dict(cbg_profiles_path)

    for category, levels in ranges_dict_raw.items():
        if isinstance(levels, dict):
            processed_levels = {}
            for level_name, range_list in levels.items():
                if isinstance(range_list, list) and len(range_list) >= 2:
                    processed_levels[level_name] = (float(range_list[0]), float(range_list[1]))
            ranges_dict[category] = processed_levels

    with open(agent_profiles_path, 'r') as f:
        agent_profiles_list = json.load(f)

    llm_client = LLM(
        model_name=MODEL_NAME,
        platform=PLATFORM,
        api_key=api_key
    )

    print("Grouping agents by type...")

    agent_groups = defaultdict(list)

    for agent_profile in agent_profiles_list:
        agent_profile["discretized_home_cbg_income"] = discretize_value(agent_profile.get("home_cbg_income", 0),
                                                                        ranges_dict, "income")
        agent_profile["discretized_home_cbg_edu"] = discretize_value(agent_profile.get("home_cbg_edu", "Unknown"),
                                                                     ranges_dict, "education")
        type_key = get_agent_type_key(agent_profile, ranges_dict)
        agent_groups[type_key].append(agent_profile)

    print(f"Total unique agent types identified: {len(agent_groups)}")
    random.seed(config['simulation']['seed'])

    agent_type_to_profile_data = {}
    all_type_keys_ordered = []

    for type_key, profiles in agent_groups.items():
        selected_profile = random.choice(profiles)
        agent_type_to_profile_data[type_key] = selected_profile
        all_type_keys_ordered.append(type_key)

    total_types_needed = len(all_type_keys_ordered)

    existing_data = recover_broken_json(output_file_path)
    completed_keys_set = {tuple(item["agent_type_key"]) for item in existing_data if "agent_type_key" in item}

    data_to_write_initial = []
    if existing_data:
        print(f"\n[Progress Check] Found {len(completed_keys_set)}/{total_types_needed} existing records.")
        user_input = input("Do you want to (c)ontinue or (r)estart? [c/r]: ").lower().strip()
        if user_input == 'c':
            data_to_write_initial = existing_data
        else:
            completed_keys_set = set()

    if len(completed_keys_set) >= total_types_needed and data_to_write_initial:
        print("All agent types have already been generated! Exiting.")
        return

    tasks = []
    for type_key in all_type_keys_ordered:
        if type_key not in completed_keys_set:
            profile_data = agent_type_to_profile_data[type_key]
            home_cbg_id = profile_data['CBG']
            home_poi_probs = cbg_profiles_dict.get(home_cbg_id, {}).get('home_cbg_poi_probs', [])
            tasks.append((type_key, profile_data, home_poi_probs, llm_client, TEM))

    print(f"Starting Split-Generation for {len(tasks)} pending types with {NUM_WORKERS} workers...")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('[\n')
            is_first_item = True

            if data_to_write_initial:
                for item in data_to_write_initial:
                    if not is_first_item: f.write(',\n')
                    json.dump(item, f, indent=4)
                    is_first_item = False

            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                future_to_key = {
                    executor.submit(generate_single_agent_policy_split, t_key, p_data, probs, client, TEM): t_key
                    for t_key, p_data, probs, client, TEM in tasks
                }

                for future in tqdm(concurrent.futures.as_completed(future_to_key), total=len(tasks), desc="Generating"):
                    try:
                        result_obj = future.result()
                        if result_obj:
                            if not is_first_item: f.write(',\n')
                            json.dump(result_obj, f, indent=4)
                            f.flush()
                            is_first_item = False
                    except Exception as e:
                        print(f"Critical Worker Error: {e}")

            f.write('\n]')
        print(f"\nJob Done. Saved to {output_file_path}")

    except KeyboardInterrupt:
        print("\nProcess interrupted. Closing file safely...")
        with open(output_file_path, 'a', encoding='utf-8') as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 2:
                f.write('\n]')
    except Exception as e:
        print(f"\nCritical Error in Main: {e}")


if __name__ == "__main__":
    main()