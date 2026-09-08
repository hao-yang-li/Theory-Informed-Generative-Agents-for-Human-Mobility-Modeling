"""Join sampled demographic attributes to home-CBG context for TIMA."""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .extract_cbg_data import CBG, normalize_cbgs, unique_cbgs
    from .generate_agents_profile import SEXES, AGE_GROUPS, RACES, INDUSTRIES
except ImportError:
    from extract_cbg_data import CBG, normalize_cbgs, unique_cbgs
    from generate_agents_profile import SEXES, AGE_GROUPS, RACES, INDUSTRIES

DEMOGRAPHICS = ["sex", "age_group", "race", "industry"]
CONTEXT = ["City", "home_cbg_income", "home_cbg_edu", "home_cbg_population"]


def validate_agents(agents, agents_per_cbg=10):
    agents = agents[[*DEMOGRAPHICS, CBG]].copy()
    agents[CBG] = normalize_cbgs(agents[CBG])
    for column, choices in zip(DEMOGRAPHICS, (SEXES, AGE_GROUPS, RACES, INDUSTRIES)):
        if not agents[column].isin(choices).all():
            raise ValueError(f"Invalid demographic values in {column}.")
    if not agents.groupby(CBG).size().eq(agents_per_cbg).all():
        raise ValueError(f"Expected exactly {agents_per_cbg} agents per CBG.")
    if agents.empty:
        raise ValueError("No agents were supplied.")
    return agents


def build_agent_profiles(agents, context, agents_per_cbg=10):
    agents = validate_agents(agents, agents_per_cbg)
    context = unique_cbgs(context)[[CBG, *CONTEXT]]
    if set(agents[CBG]) - set(context[CBG]):
        raise ValueError("Some agents have no home-CBG context.")
    # Preserve within-CBG sample order and use deterministic CBG ordering for IDs.
    agents = agents.sort_values(CBG, kind="stable").reset_index(drop=True)
    combined = agents.merge(context, on=CBG, how="left", validate="many_to_one", sort=False)
    profiles = []
    for i, row in enumerate(combined.to_dict("records")):
        profile = {"id": str(i), "CBG": row[CBG]}
        profile.update({k: row[k] for k in DEMOGRAPHICS})
        profile.update({k: row[k] for k in CONTEXT})
        if pd.isna(profile["home_cbg_population"]):
            raise ValueError(f"Missing population for {row[CBG]}.")
        profile["home_cbg_population"] = int(profile["home_cbg_population"])
        if pd.isna(profile["home_cbg_income"]):
            profile["home_cbg_income"] = None
        if pd.isna(profile["home_cbg_edu"]):
            profile["home_cbg_edu"] = "Medium"
        profiles.append(profile)
    return profiles


def save_profiles(profiles, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(profiles, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", required=True, help="Sampled agent census CSV")
    parser.add_argument("--context", required=True, help="Integrated CBG context CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--agents-per-cbg", type=int, default=10)
    args = parser.parse_args()
    agents = pd.read_csv(args.agents, dtype={CBG: str})
    context = pd.read_csv(args.context, dtype={CBG: str})
    save_profiles(build_agent_profiles(agents, context, args.agents_per_cbg), args.output)


if __name__ == "__main__":
    main()
