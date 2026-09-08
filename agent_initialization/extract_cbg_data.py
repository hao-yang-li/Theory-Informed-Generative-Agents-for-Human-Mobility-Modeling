"""Select ACS marginals and urban context directly from downloaded archives."""

import argparse
import io
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import pandas as pd


CBG = "census_block_group"
ACS_COLUMNS = {
    "cbg_b01.csv": [f"B01001e{i}" for i in (*range(3, 26), *range(27, 50))],
    "cbg_b02.csv": [f"B02001e{i}" for i in range(2, 8)],
    "cbg_c24.csv": [f"C24030e{i}" for i in (
        3, 6, 7, 8, 9, 10, 13, 14, 17, 21, 24, 27, 28,
        30, 33, 34, 35, 36, 37, 40, 41, 44, 48, 51, 54, 55)],
}
GEO_COLUMNS = [CBG, "amount_land", "amount_water", "latitude", "longitude"]
URBAN_FILES = {
    "lookup": "Geographic_Lookup_Table_Between_City_CBG.csv",
    "income": "Indicators_for_SDG_1_CBG_level.csv",
    "education": "Indicators_for_SDG_4_CBG_level.csv",
    "population": "Basic_Geographic_Statistics_CBG.csv",
}
EDU_COLUMNS = ["Population With A Bachelor's Degree",
               "Population With A Master's Degree", "Population With A Doctorate"]


class _ForwardReader(io.RawIOBase):
    """Expose a forward-only TAR member to pandas on Python 3.9 and later."""

    def __init__(self, stream):
        self.stream = stream

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.stream.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def normalize_cbgs(values):
    values = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if not values.str.fullmatch(r"\d{1,12}").all():
        raise ValueError("CBG identifiers must contain at most 12 digits.")
    return values.str.zfill(12)


def unique_cbgs(frame, key=CBG):
    frame = frame.copy()
    frame[key] = normalize_cbgs(frame[key])
    if frame[key].duplicated().any():
        raise ValueError(f"Duplicate CBG rows in {key}; select one city and year.")
    return frame


def load_cbg_list(path):
    frame = pd.read_csv(path, dtype=str)
    key = next((k for k in (CBG, "CBG Code", "CBG") if k in frame), None)
    if key is None:
        raise ValueError("CBG list needs census_block_group, CBG Code, or CBG.")
    return unique_cbgs(frame.rename(columns={key: CBG}))[[CBG]]


def _select_csv(stream, columns, key, cbgs=None, city_name=None, year=None):
    parts = []
    for chunk in pd.read_csv(stream, usecols=columns, dtype={key: str}, chunksize=10000):
        chunk[key] = normalize_cbgs(chunk[key])
        if city_name is not None:
            chunk = chunk[chunk["City Name"].eq(city_name)]
        if year is not None:
            chunk = chunk[pd.to_numeric(chunk["Year"], errors="coerce").eq(year)]
        if cbgs is not None:
            chunk = chunk[chunk[key].isin(cbgs)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        raise ValueError("No matching rows; check city name, CBG list, and year.")
    return unique_cbgs(pd.concat(parts, ignore_index=True), key)


def _find_member(names, basename):
    matches = [n for n in names if Path(n).name == basename and "__MACOSX" not in n]
    if len(matches) != 1:
        raise ValueError(f"Expected one {basename}, found {len(matches)}.")
    return matches[0]


def read_urban_table(source, basename, columns, **filters):
    source = Path(source)
    if source.is_dir():
        path = _find_member([str(p) for p in source.rglob(basename)], basename)
        with open(path, "rb") as stream:
            return _select_csv(stream, columns, "CBG Code", **filters)
    with zipfile.ZipFile(source) as archive:
        member = _find_member(archive.namelist(), basename)
        with archive.open(member) as stream:
            return _select_csv(stream, columns, "CBG Code", **filters)


def extract_cbg_data(cbg_list, census_source):
    """Read only the three required ACS tables and geographic metadata.

    TAR.GZ members are streamed in one pass. Extracted source directories are
    also accepted.
    """
    selected = set(cbg_list[CBG])
    required = {**ACS_COLUMNS, "cbg_geographic_data.csv": GEO_COLUMNS[1:]}
    tables = {}

    def read_one(name, stream):
        print(f"Reading {name} ...", flush=True)
        tables[name] = _select_csv(stream, [CBG, *required[name]], CBG, cbgs=selected)

    source = Path(census_source)
    if source.is_dir():
        for name in required:
            path = _find_member([str(p) for p in source.rglob(name)], name)
            with open(path, "rb") as stream:
                read_one(name, stream)
    else:
        with tarfile.open(source, "r|gz") as archive:
            for member in archive:
                name = Path(member.name).name
                if member.isfile() and name in required and "__MACOSX" not in member.name:
                    if name in tables:
                        raise ValueError(f"Duplicate archive member: {name}")
                    with archive.extractfile(member) as stream:
                        with io.BufferedReader(_ForwardReader(stream)) as reader:
                            read_one(name, reader)
                    if len(tables) == len(required):
                        break
    missing = set(required) - set(tables)
    if missing:
        raise ValueError(f"Missing Open Census tables: {sorted(missing)}")
    census = cbg_list.copy()
    for name in ACS_COLUMNS:
        table = tables[name]
        if selected - set(table[CBG]):
            raise ValueError(f"{name} is missing {len(selected - set(table[CBG]))} requested CBGs.")
        census = census.merge(table, on=CBG, how="left", validate="one_to_one")
    geo = cbg_list.merge(tables["cbg_geographic_data.csv"], on=CBG,
                         how="left", validate="one_to_one")
    if geo[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Some requested CBGs have no coordinates.")
    return census, geo


def prepare_context(urban_source, city_name, year=2019, cbg_list=None):
    """Assemble home-CBG contextual attributes from the urban tables."""
    if cbg_list is None:
        lookup = read_urban_table(urban_source, URBAN_FILES["lookup"],
                                 ["City Name", "CBG Code"], city_name=city_name)
        cbg_list = lookup.rename(columns={"CBG Code": CBG})[[CBG]]
    selected = set(cbg_list[CBG])
    filters = dict(cbgs=selected, city_name=city_name, year=year)
    income = read_urban_table(urban_source, URBAN_FILES["income"],
                             ["City Name", "CBG Code", "Year", "Median Household Income"],
                             **filters)
    population = read_urban_table(urban_source, URBAN_FILES["population"],
                                 ["City Name", "CBG Code", "Year", "Population"], **filters)
    education = read_urban_table(urban_source, URBAN_FILES["education"],
                                ["City Name", "CBG Code", "Year", *EDU_COLUMNS], **filters)
    for name, table in (("income", income), ("population", population), ("education", education)):
        if selected - set(table["CBG Code"]):
            raise ValueError(f"{name} is missing requested CBGs for {city_name}, {year}.")

    # Sum degree counts, sort descending, and assign quintiles.
    education["High_Edu_Pop"] = education[EDU_COLUMNS].sum(axis=1)
    education = education.sort_values("High_Edu_Pop", ascending=False).reset_index(drop=True)
    n = len(education)
    cuts = [int(n * q) for q in (0.2, 0.4, 0.6, 0.8)]
    education["home_cbg_edu"] = np.select(
        [education.index < cuts[0], education.index < cuts[1],
         education.index < cuts[2], education.index < cuts[3]],
        ["Highest", "High", "Medium", "Low"], default="Lowest")
    context = cbg_list.rename(columns={CBG: "CBG Code"}).merge(
        income, on="CBG Code", how="left", validate="one_to_one")
    context = context.merge(population[["CBG Code", "Population"]], on="CBG Code",
                            how="left", validate="one_to_one")
    context = context.merge(education[["CBG Code", "home_cbg_edu"]], on="CBG Code",
                            how="left", validate="one_to_one")
    context = context.rename(columns={"CBG Code": CBG, "City Name": "City",
        "Median Household Income": "home_cbg_income", "Population": "home_cbg_population"})
    context["home_cbg_population"] = pd.to_numeric(context["home_cbg_population"], errors="raise")
    if context["home_cbg_population"].isna().any():
        raise ValueError("Population is missing for some CBGs.")
    context["home_cbg_population"] = context["home_cbg_population"].astype(int)
    return cbg_list, context[[CBG, "City", "Year", "home_cbg_income",
                              "home_cbg_population", "home_cbg_edu"]]


def prepare_inputs(acs_source, urban_source, city_name, output_dir, year=2019, cbg_list_path=None):
    cbgs = load_cbg_list(cbg_list_path) if cbg_list_path else None
    cbgs, context = prepare_context(urban_source, city_name, year, cbgs)
    census, geo = extract_cbg_data(cbgs, acs_source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("cbg_list.csv", cbgs), ("cbg_context.csv", context),
                        ("cbg_extracted_data.csv", census), ("cbg_geographic_data.csv", geo)):
        frame.to_csv(output / name, index=False)
    return census, context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acs-source", required=True, help="Open Census TAR.GZ or extracted directory")
    parser.add_argument("--urban-source", required=True, help="Urban sustainability ZIP or extracted directory")
    parser.add_argument("--city-name", required=True, help="Exact dataset name, e.g. New York city")
    parser.add_argument("--cbg-list", help="Optional CSV selecting CBGs within the named city")
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    prepare_inputs(args.acs_source, args.urban_source, args.city_name,
                   args.output_dir, args.year, args.cbg_list)


if __name__ == "__main__":
    main()
