import pandas as pd
import numpy as np
import os
from itertools import product
import traceback

# Demographic dimensions.
SEXES = ['Male', 'Female']
# Age groups.
AGE_GROUPS = ['Under 18 years', '18 to 60 years', 'Over 60 years']
# Race groups.
RACES = ['White', 'Black', 'Other']
# Industry categories.
INDUSTRIES = [
    'Agriculture forestry fishing and hunting and mining',
    'Construction', 'Manufacturing', 'Wholesale trade', 'Retail trade',
    'Transportation and warehousing and utilities', 'Information',
    'Finance and insurance and real estate and rental and leasing',
    'Professional scientific and management and administrative and waste management services',
    'Educational services and health care and social assistance',
    'Arts entertainment and recreation and accommodation and food services',
    'Other services except public administration', 'Public administration'
]

# Map ACS age fields to age groups.
AGE_GROUP_MAPPING = {
    # Under 18 years
    'B01001e3': 'Under 18 years',  # Under 5 years
    'B01001e4': 'Under 18 years',  # 5 to 9 years
    'B01001e5': 'Under 18 years',  # 10 to 14 years
    'B01001e6': 'Under 18 years',  # 15 to 17 years
    # 18 to 60 years
    'B01001e7': '18 to 60 years',  # 18 and 19 years
    'B01001e8': '18 to 60 years',  # 20 years
    'B01001e9': '18 to 60 years',  # 21 years
    'B01001e10': '18 to 60 years',  # 22 to 24 years
    'B01001e11': '18 to 60 years',  # 25 to 29 years
    'B01001e12': '18 to 60 years',  # 30 to 34 years
    'B01001e13': '18 to 60 years',  # 35 to 39 years
    'B01001e14': '18 to 60 years',  # 40 to 44 years
    'B01001e15': '18 to 60 years',  # 45 to 49 years
    'B01001e16': '18 to 60 years',  # 50 to 54 years
    'B01001e17': '18 to 60 years',  # 55 to 59 years
    'B01001e18': 'Over 60 years',  # 60 and 61 years
    'B01001e19': 'Over 60 years',  # 62 to 64 years
    'B01001e20': 'Over 60 years',  # 65 and 66 years
    'B01001e21': 'Over 60 years',  # 67 to 69 years
    'B01001e22': 'Over 60 years',  # 70 to 74 years
    'B01001e23': 'Over 60 years',  # 75 to 79 years
    'B01001e24': 'Over 60 years',  # 80 to 84 years
    'B01001e25': 'Over 60 years',  # 85 years and over
    # Female mappings
    'B01001e27': 'Under 18 years',  # Under 5 years
    'B01001e28': 'Under 18 years',  # 5 to 9 years
    'B01001e29': 'Under 18 years',  # 10 to 14 years
    'B01001e30': 'Under 18 years',  # 15 to 17 years
    'B01001e31': '18 to 60 years',  # 18 and 19 years
    'B01001e32': '18 to 60 years',  # 20 years
    'B01001e33': '18 to 60 years',  # 21 years
    'B01001e34': '18 to 60 years',  # 22 to 24 years
    'B01001e35': '18 to 60 years',  # 25 to 29 years
    'B01001e36': '18 to 60 years',  # 30 to 34 years
    'B01001e37': '18 to 60 years',  # 35 to 39 years
    'B01001e38': '18 to 60 years',  # 40 to 44 years
    'B01001e39': '18 to 60 years',  # 45 to 49 years
    'B01001e40': '18 to 60 years',  # 50 to 54 years
    'B01001e41': '18 to 60 years',  # 55 to 59 years
    'B01001e42': 'Over 60 years',  # 60 and 61 years
    'B01001e43': 'Over 60 years',  # 62 to 64 years
    'B01001e44': 'Over 60 years',  # 65 and 66 years
    'B01001e45': 'Over 60 years',  # 67 to 69 years
    'B01001e46': 'Over 60 years',  # 70 to 74 years
    'B01001e47': 'Over 60 years',  # 75 to 79 years
    'B01001e48': 'Over 60 years',  # 80 to 84 years
    'B01001e49': 'Over 60 years',  # 85 years and over
}

# ACS field mappings.
# ACS fields used for age aggregation.
ORIGINAL_SEX_AGE_FIELDS = [
    # Male
    'B01001e3', 'B01001e4', 'B01001e5', 'B01001e6', 'B01001e7',
    'B01001e8', 'B01001e9', 'B01001e10', 'B01001e11', 'B01001e12',
    'B01001e13', 'B01001e14', 'B01001e15', 'B01001e16', 'B01001e17',
    'B01001e18', 'B01001e19', 'B01001e20', 'B01001e21', 'B01001e22',
    'B01001e23', 'B01001e24', 'B01001e25',
    # Female
    'B01001e27', 'B01001e28', 'B01001e29', 'B01001e30', 'B01001e31',
    'B01001e32', 'B01001e33', 'B01001e34', 'B01001e35', 'B01001e36',
    'B01001e37', 'B01001e38', 'B01001e39', 'B01001e40', 'B01001e41',
    'B01001e42', 'B01001e43', 'B01001e44', 'B01001e45', 'B01001e46',
    'B01001e47', 'B01001e48', 'B01001e49'
]

# Aggregate ACS race counts into three categories.
RACE_MAPPING = {
    'White': ['B02001e2'],  # White alone
    'Black': ['B02001e3'],  # Black or African American alone
    'Other': [  # Combine the remaining race categories as Other.
        'B02001e4',  # American Indian and Alaska Native alone
        'B02001e5',  # Asian alone
        'B02001e6',  # Native Hawaiian and Other Pacific Islander alone
        'B02001e7'  # Some other race alone
    ]
}

INDUSTRY_MAPPING = {
    'Agriculture forestry fishing and hunting and mining': ('C24030e3', 'C24030e30'),
    'Construction': ('C24030e6', 'C24030e33'),
    'Manufacturing': ('C24030e7', 'C24030e34'),
    'Wholesale trade': ('C24030e8', 'C24030e35'),
    'Retail trade': ('C24030e9', 'C24030e36'),
    'Transportation and warehousing and utilities': ('C24030e10', 'C24030e37'),
    'Information': ('C24030e13', 'C24030e40'),
    'Finance and insurance and real estate and rental and leasing': ('C24030e14', 'C24030e41'),
    'Professional scientific and management and administrative and waste management services': (
    'C24030e17', 'C24030e44'),
    'Educational services and health care and social assistance': ('C24030e21', 'C24030e48'),
    'Arts entertainment and recreation and accommodation and food services': ('C24030e24', 'C24030e51'),
    'Other services except public administration': ('C24030e27', 'C24030e54'),
    'Public administration': ('C24030e28', 'C24030e55')
}


def create_initial_distribution():
    """Create a four-dimensional joint demographic distribution."""
    # Build joint demographic combinations.
    combinations = list(product(SEXES, AGE_GROUPS, RACES, INDUSTRIES))
    df = pd.DataFrame(combinations, columns=['sex', 'age_group', 'race', 'industry'])
    df['count'] = 1.0  # Initialize uniform counts.
    return df


def aggregate_age_data(row):
    """Aggregate ACS age counts into demographic age groups."""
    aggregated = {}

    # Sum counts for each sex and age group.
    for field in ORIGINAL_SEX_AGE_FIELDS:
        age_group = AGE_GROUP_MAPPING.get(field)
        if age_group is None:
            continue

        # Identify sex from the ACS field number.
        if field.startswith('B01001e') and '27' <= field[-2:] <= '49':  # Fields e27-e49 contain female counts.
            sex = 'Female'
        else:
            sex = 'Male'  # Fields e3-e25 contain male counts.

        key = (sex, age_group)

        value = row.get(field, 0)
        if pd.isna(value) or value is None:
            value = 0


        if key in aggregated:
            aggregated[key] += float(value)
        else:
            aggregated[key] = float(value)

    return aggregated


def prepare_marginals(row):
    """Prepare demographic marginals for one CBG."""

    # Sex-by-age marginals.
    age_agg = aggregate_age_data(row)
    sex_age_list = []
    for (sex, age_group), count in age_agg.items():
        # Clip counts to nonnegative values.
        safe_count = max(0.0, float(count))
        sex_age_list.append({'sex': sex, 'age_group': age_group, 'count': safe_count})
    sex_age_df = pd.DataFrame(sex_age_list)

    # Race marginals.
    race_list = []
    for race_category, original_fields in RACE_MAPPING.items():
        total_count = 0
        for field in original_fields:
            val = row.get(field, 0)
            if not (pd.isna(val) or val is None):
                total_count += max(0.0, float(val))
        race_list.append({'race': race_category, 'count': total_count})
    race_df = pd.DataFrame(race_list)

    # 3. sex × industry
    industry_list = []
    for industry, (male_col, female_col) in INDUSTRY_MAPPING.items():
        male_val = row.get(male_col, 0)
        female_val = row.get(female_col, 0)
        if pd.isna(male_val) or male_val is None:
            male_val = 0
        if pd.isna(female_val) or female_val is None:
            female_val = 0
        # Clip counts to nonnegative values.
        safe_male_val = max(0.0, float(male_val))
        safe_female_val = max(0.0, float(female_val))
        industry_list.append({'sex': 'Male', 'industry': industry, 'count': safe_male_val})
        industry_list.append({'sex': 'Female', 'industry': industry, 'count': safe_female_val})
    sex_industry_df = pd.DataFrame(industry_list)

    return sex_age_df, race_df, sex_industry_df


def validate_marginals(sex_age_df, race_df, sex_industry_df, cbg_code):
    """Validate marginal totals and count values."""
    try:
        # Require positive marginal totals.
        total_sex_age = sex_age_df['count'].sum()
        total_race = race_df['count'].sum()
        total_sex_industry = sex_industry_df['count'].sum()

        if total_sex_age <= 0:
            print(f"  Warning: Sex-Age marginal total is zero or negative for CBG {cbg_code}")
            return False
        if total_race <= 0:
            print(f"  Warning: Race marginal total is zero or negative for CBG {cbg_code}")
            return False
        if total_sex_industry <= 0:
            print(f"  Warning: Sex-Industry marginal total is zero or negative for CBG {cbg_code}")
            return False

        # Check that marginal counts are finite.
        if sex_age_df['count'].isna().any() or np.isinf(sex_age_df['count']).any():
            print(f"  Warning: Sex-Age marginal contains NaN or Inf for CBG {cbg_code}")
            return False
        if race_df['count'].isna().any() or np.isinf(race_df['count']).any():
            print(f"  Warning: Race marginal contains NaN or Inf for CBG {cbg_code}")
            return False
        if sex_industry_df['count'].isna().any() or np.isinf(sex_industry_df['count']).any():
            print(f"  Warning: Sex-Industry marginal contains NaN or Inf for CBG {cbg_code}")
            return False

        return True
    except Exception as e:
        print(f"  Error during marginal validation for CBG {cbg_code}: {e}")
        return False


def simple_ipf(joint_df, marginals_list, max_iterations=50, tolerance=1e-4):
    """Fit a joint count distribution to the supplied demographic marginals."""
    # Copy the joint distribution for iterative fitting.
    current_df = joint_df.copy()

    # Convert counts to numeric values.
    current_df['count'] = pd.to_numeric(current_df['count'], errors='coerce').fillna(1.0)
    current_df['count'] = np.clip(current_df['count'], 1e-10, None)  # Apply a positive numerical floor.

    for iteration in range(max_iterations):
        old_counts = current_df['count'].copy()
        max_change = 0.0

        # Fit each target marginal.
        for i, marginal in enumerate(marginals_list):
            dims = marginal['dims']
            target_df = marginal['df']

            # Convert target counts to nonnegative numeric values.
            target_df['count'] = pd.to_numeric(target_df['count'], errors='coerce').fillna(0.0)
            target_df['count'] = np.clip(target_df['count'], 0.0, None)

            # Aggregate current counts along the target dimensions.
            try:
                current_marginal = current_df.groupby(dims)['count'].sum().reset_index()
                current_marginal.rename(columns={'count': 'current_count'}, inplace=True)
            except Exception as e:
                print(f"    Error computing current marginal for dims {dims}: {e}")
                continue

            # Align current and target marginals.
            try:
                # Include combinations from both marginals.
                merged = pd.merge(current_marginal, target_df, on=dims, how='outer')
                merged['count'] = merged['count'].fillna(0.0)  # Fill absent target combinations with zero.
                merged['current_count'] = merged['current_count'].fillna(1e-10)  # Apply a positive floor to absent current combinations.

                # Calculate marginal adjustment ratios.

                merged['ratio'] = np.where(
                    merged['current_count'] > 1e-10,
                    merged['count'] / merged['current_count'],
                    1.0
                )
                # Use a neutral ratio for infinite values.
                merged['ratio'] = np.where(np.isinf(merged['ratio']), 1.0, merged['ratio'])
                # Use a neutral ratio for missing values.
                merged['ratio'] = merged['ratio'].fillna(1.0)

            except Exception as e:
                print(f"    Error merging or calculating ratio for dims {dims}: {e}")
                continue

            # Apply adjustment ratios to the joint distribution.
            try:
                # Index ratios by demographic combination.
                ratio_map = merged.set_index(dims)['ratio']


                current_df['ratio'] = current_df.set_index(dims).index.map(ratio_map).fillna(1.0)


                current_df['count'] = current_df['count'] * current_df['ratio']

                # Remove the temporary ratio column.
                current_df.drop('ratio', axis=1, inplace=True, errors='ignore')

                # Apply a positive floor to fitted counts.
                current_df['count'] = np.clip(current_df['count'], 1e-10, None)

            except Exception as e:
                print(f"    Error applying ratio for dims {dims}: {e}")

                continue

        # Check convergence using the maximum relative change.
        try:
            current_counts_array = current_df['count'].values
            old_counts_array = old_counts.values

            relative_changes = np.abs((current_counts_array - old_counts_array) / (old_counts_array + 1e-10))
            max_diff = np.max(relative_changes)

            if max_diff < tolerance:
                print(f"    IPF converged after {iteration + 1} iterations (max relative change: {max_diff:.2e})")
                break
            elif iteration == max_iterations - 1:
                print(f"    IPF stopped after {max_iterations} iterations (max relative change: {max_diff:.2e})")

        except Exception as e:
            print(f"    Error checking convergence: {e}")
            break

    # Normalize finite counts.
    current_df['count'] = pd.to_numeric(current_df['count'], errors='coerce').fillna(1e-10)
    current_df['count'] = np.clip(current_df['count'], 1e-10, None)

    return current_df


# City-level marginal distributions.
def compute_overall_marginals(df):
    print("Computing overall marginals from all CBGs...")
    all_sex_age = []
    all_races = []
    all_sex_industry = []

    for _, row in df.iterrows():
        try:
            sex_age_df, race_df, sex_industry_df = prepare_marginals(row)
            all_sex_age.append(sex_age_df)
            all_races.append(race_df)
            all_sex_industry.append(sex_industry_df)
        except Exception as e:
            print(f"Warning: Failed to compute marginals for CBG {row['census_block_group']}: {e}")

    # Combine CBG marginals.
    overall_sex_age = pd.concat(all_sex_age).groupby(['sex', 'age_group'])['count'].sum().reset_index()
    overall_race = pd.concat(all_races).groupby('race')['count'].sum().reset_index()
    overall_sex_industry = pd.concat(all_sex_industry).groupby(['sex', 'industry'])['count'].sum().reset_index()

    return overall_sex_age, overall_race, overall_sex_industry


def generate_agents_for_cbg(cbg_code, row, num_agents=10,
                            fallback_sex_age=None,
                            fallback_race=None,
                            fallback_sex_industry=None, seed=42):
    """Sample agents for one CBG using fitted demographic probabilities."""
    try:
        print(f"  Generating agents for CBG {cbg_code}...")

        # Check CBG input data.
        total_population_fields = [row.get(col, 0) for col in ORIGINAL_SEX_AGE_FIELDS]
        total_population = sum([val for val in total_population_fields if not (pd.isna(val) or val is None)])
        if total_population <= 0:
            print(f"    Warning: No valid population data for CBG {cbg_code}")
            # Use city-level fallback marginals.
            if fallback_sex_age is not None and fallback_race is not None and fallback_sex_industry is not None:
                print(f"    Using fallback marginals for CBG {cbg_code}")
                sex_age_df, race_df, sex_industry_df = fallback_sex_age, fallback_race, fallback_sex_industry
            else:
                return []
        else:

            sex_age_df, race_df, sex_industry_df = prepare_marginals(row)

        # Validate the CBG marginals.
        if not validate_marginals(sex_age_df, race_df, sex_industry_df, cbg_code):
            print(f"    Warning: Invalid marginals for CBG {cbg_code}, trying fallback...")
            if fallback_sex_age is not None:
                sex_age_df, race_df, sex_industry_df = fallback_sex_age, fallback_race, fallback_sex_industry
            else:
                return []

        print(
            f"    Prepared marginals - SexAge: {len(sex_age_df)}, Race: {len(race_df)}, SexIndustry: {len(sex_industry_df)}")

        # Prepare IPF inputs.
        marginals = [
            {'dims': ['sex', 'age_group'], 'df': sex_age_df},
            {'dims': ['race'], 'df': race_df},
            {'dims': ['sex', 'industry'], 'df': sex_industry_df}
        ]

        # Fit the joint distribution with IPF.
        print(f"    Running IPF...")
        df_adjusted = simple_ipf(create_initial_distribution(), marginals)
        print(f"    IPF completed.")

        # Normalize fitted counts into sampling probabilities.
        total = df_adjusted['count'].sum()
        if not np.isfinite(total) or total <= 1e-6:  # Check the fitted total.
            print(f"    Warning: IPF resulted in invalid total ({total}) for CBG {cbg_code}")
            # Retry IPF with city-level marginals.
            if fallback_sex_age is not None:
                print(f"    Trying fallback IPF for CBG {cbg_code}...")
                df_adjusted = simple_ipf(create_initial_distribution(), [
                    {'dims': ['sex', 'age_group'], 'df': fallback_sex_age},
                    {'dims': ['race'], 'df': fallback_race},
                    {'dims': ['sex', 'industry'], 'df': fallback_sex_industry}
                ])
                total = df_adjusted['count'].sum()
                if not np.isfinite(total) or total <= 1e-6:
                    print(f"    Warning: Fallback IPF also failed for CBG {cbg_code}")
                    return []
            else:
                return []

        df_adjusted['prob'] = df_adjusted['count'] / total

        # Check that sampling probabilities are finite.
        if df_adjusted['prob'].isna().any() or not np.isfinite(df_adjusted['prob']).all():
            print(f"    Warning: Invalid probabilities calculated for CBG {cbg_code}")
            return []

        # Sample demographic combinations without replacement.
        samples = df_adjusted.sample(n=num_agents, weights='prob', replace=False, random_state=seed)[
            ['sex', 'age_group', 'race', 'industry']]
        samples['census_block_group'] = cbg_code

        print(f"    Successfully generated {len(samples)} agents for CBG {cbg_code}.")
        return samples.to_dict('records')

    except Exception as e:
        print(f"Error processing CBG {cbg_code}: {e}")
        traceback.print_exc()  # Print the exception traceback.
        return []


def generate_agents(census, num_agents=10, seed=42):
    """Generate agents using IPF, fallback marginals, and per-CBG sampling."""
    import contextlib
    import io
    if num_agents < 1:
        raise ValueError("agents_per_cbg must be positive.")
    if census.empty:
        raise ValueError("Census table is empty.")
    required = set(ORIGINAL_SEX_AGE_FIELDS)
    required.update(field for fields in RACE_MAPPING.values() for field in fields)
    required.update(field for fields in INDUSTRY_MAPPING.values() for field in fields)
    missing = required - set(census.columns)
    if missing:
        raise ValueError(f"Missing ACS marginal columns: {sorted(missing)}")
    if census['census_block_group'].duplicated().any():
        raise ValueError("Census table contains duplicate CBGs.")
    with contextlib.redirect_stdout(io.StringIO()):
        fallback = compute_overall_marginals(census)
    all_agents = []
    for i, (_, row) in enumerate(census.iterrows(), 1):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            agents = generate_agents_for_cbg(
                row['census_block_group'], row, num_agents,
                fallback_sex_age=fallback[0], fallback_race=fallback[1],
                fallback_sex_industry=fallback[2], seed=seed)
        if len(agents) != num_agents:
            raise ValueError(f"Unable to generate {num_agents} agents for "
                             f"{row['census_block_group']}: {log.getvalue()}")
        all_agents.extend(agents)
        if i % 100 == 0 or i == len(census):
            print(f"Initialized {i}/{len(census)} CBGs", flush=True)
    return pd.DataFrame(all_agents)


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Generate agents from extracted ACS marginals.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--agents-per-cbg", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    census = pd.read_csv(args.input, dtype={'census_block_group': str})
    agents = generate_agents(census, args.agents_per_cbg, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    agents.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
