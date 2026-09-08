"""Initialize TIMA agents from downloaded archives or prepared city tables."""

import argparse
from pathlib import Path

import pandas as pd

from agent_initialization.extract_cbg_data import CBG, normalize_cbgs, prepare_inputs
from agent_initialization.generate_agents_profile import generate_agents
from agent_initialization.citizen_initialization import build_agent_profiles, save_profiles
from agent_initialization.cbg_sample_agent import build_cbg_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acs-source", help="Open Census TAR.GZ or extracted directory")
    parser.add_argument("--urban-source", help="Urban sustainability ZIP or extracted directory")
    parser.add_argument("--city-name", help="Exact dataset name, e.g. New York city")
    parser.add_argument("--city-key", required=True, help="Output filename suffix, e.g. NYC")
    parser.add_argument("--cbg-list", help="Optional CSV selecting CBGs within the named city")
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--census-table", help="Previously extracted ACS CSV")
    parser.add_argument("--context-table", help="Previously integrated CBG context CSV")
    parser.add_argument("--agents-per-cbg", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not args.city_key or any(c in args.city_key for c in "/\\") or args.city_key in (".", ".."):
        parser.error("city-key must be a filename suffix, not a path.")
    if args.agents_per_cbg < 1:
        parser.error("agents-per-cbg must be positive.")
    prepared = args.census_table is not None or args.context_table is not None
    if prepared:
        if not (args.census_table and args.context_table):
            parser.error("Supply both --census-table and --context-table.")
        if args.acs_source or args.urban_source or args.cbg_list:
            parser.error("Choose either raw sources or prepared tables.")
        census = pd.read_csv(args.census_table, dtype={CBG: str})
        context = pd.read_csv(args.context_table, dtype={CBG: str})
        census[CBG] = normalize_cbgs(census[CBG])
        context[CBG] = normalize_cbgs(context[CBG])
    else:
        if not (args.acs_source and args.urban_source and args.city_name):
            parser.error("Raw mode requires --acs-source, --urban-source, and --city-name.")
        census, context = prepare_inputs(args.acs_source, args.urban_source, args.city_name,
                                         args.output_dir, args.year, args.cbg_list)
    if set(census[CBG]) != set(context[CBG]):
        raise ValueError("Census and context tables must cover the same CBGs.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    agents = generate_agents(census, args.agents_per_cbg, args.seed)
    agents.to_csv(output / f"{args.city_key}_agent_census.csv", index=False)
    agent_path = output / f"agent_profiles_{args.city_key}.json"
    cbg_path = output / f"cbg_profiles_{args.city_key}.json"
    save_profiles(build_agent_profiles(agents, context, args.agents_per_cbg), agent_path)
    save_profiles(build_cbg_profiles(agents, context, args.agents_per_cbg), cbg_path)
    print(f"Created {len(agents)} agents in {len(census)} CBGs.")
    print("Set these entries under paths in config.yaml:")
    print(f"  agent_profiles: {agent_path.resolve()}")
    print(f"  cbg_profiles: {cbg_path.resolve()}")
    if not prepared:
        print(f"  cbg_geo_data: {(output / 'cbg_geographic_data.csv').resolve()}")


if __name__ == "__main__":
    main()
