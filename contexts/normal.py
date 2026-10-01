from agent_initialization.demographics import education_level, income_level

POI_CATEGORIES = [
    'Wholesale & Retail Trade, Transportation and Warehousing',
    'Others',
    'Educational Services',
    'Health Care and Social Assistance',
    'Arts, Entertainment, and Recreation',
    'Accommodation and Food Services'
]

def generate_interest_prompt(agent_profile_data, home_cbg_poi_probs):
    agent_sex = agent_profile_data["sex"]
    agent_age_group = agent_profile_data["age_group"]
    agent_race = agent_profile_data["race"]
    agent_industry = agent_profile_data["industry"]
    home_cbg_edu_level = education_level(agent_profile_data)
    home_cbg_income_level = income_level(agent_profile_data)

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
    agent_sex = agent_profile_data["sex"]
    agent_age_group = agent_profile_data["age_group"]
    agent_race = agent_profile_data["race"]
    agent_industry = agent_profile_data["industry"]
    home_cbg_edu_level = education_level(agent_profile_data)
    home_cbg_income_level = income_level(agent_profile_data)

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
    agent_sex = agent_profile_data["sex"]
    agent_age_group = agent_profile_data["age_group"]
    agent_race = agent_profile_data["race"]
    agent_industry = agent_profile_data["industry"]
    home_cbg_edu_level = education_level(agent_profile_data)
    home_cbg_income_level = income_level(agent_profile_data)

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
Return ONLY the Python code block wrapped in ```python ... ``` containing the definition of `exploration_probs`.

**Example:**
```python
# Exploration probabilities based on visited count (S)
# S=0, S=5, S=10, S=15, S=20, S>=25
exploration_probs = [<float>, <float>, <float>, <float>, <float>, <float>]
```
"""
    return prompt
