"""Aggregate sampled agents into the CBG profiles consumed by TIMA."""

import argparse
from collections import Counter

import pandas as pd

try:
    from .citizen_initialization import build_agent_profiles, save_profiles
except ImportError:
    from citizen_initialization import build_agent_profiles, save_profiles


def build_cbg_profiles(agents, context, agents_per_cbg=10):
    profiles = build_agent_profiles(agents, context, agents_per_cbg)
    frame = pd.DataFrame(profiles)
    result = []
    for cbg, group in frame.groupby("CBG", sort=True):
        profile = {"census_block_group": cbg}
        for field, target in (("sex", "sex_distribution"), ("age_group", "age_distribution"),
                              ("race", "race_distribution")):
            profile[target] = (group[field].value_counts() / len(group)).to_dict()
        profile["industry_counts"] = dict(Counter(group["industry"]))
        first = group.to_dict("records")[0]
        for field in ("City", "home_cbg_income", "home_cbg_population", "home_cbg_edu"):
            value = first[field]
            profile[field] = None if pd.isna(value) else value
        profile["home_cbg_population"] = int(profile["home_cbg_population"])
        result.append(profile)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--agents-per-cbg", type=int, default=10)
    args = parser.parse_args()
    agents = pd.read_csv(args.agents, dtype={"census_block_group": str})
    context = pd.read_csv(args.context, dtype={"census_block_group": str})
    save_profiles(build_cbg_profiles(agents, context, args.agents_per_cbg), args.output)


if __name__ == "__main__":
    main()
