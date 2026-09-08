"""Test population initialization with temporary input fixtures."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_initialization.extract_cbg_data import (
    ACS_COLUMNS, CBG, GEO_COLUMNS, EDU_COLUMNS, URBAN_FILES, prepare_inputs,
    extract_cbg_data, load_cbg_list)
from agent_initialization.generate_agents_profile import generate_agents
from agent_initialization.citizen_initialization import build_agent_profiles, save_profiles
from agent_initialization.cbg_sample_agent import build_cbg_profiles


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = ["010010201001", "010010201002", "010010201003"]
        self.cbgs = pd.DataFrame({CBG: self.ids})
        self.acs = self.root / "acs.tar.gz"
        self.urban = self.root / "urban.zip"
        self.directory = self.root / "acs"
        self.directory.mkdir()
        with tarfile.open(self.acs, "w:gz") as archive:
            for name, columns in {**ACS_COLUMNS, "cbg_geographic_data.csv": GEO_COLUMNS[1:]}.items():
                frame = self.cbgs.copy()
                for column in columns:
                    frame[column] = [1, 2, 3]
                content = frame.to_csv(index=False).encode()
                member = tarfile.TarInfo("nested/data/" + name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
                (self.directory / name).write_bytes(content)
        base = pd.DataFrame({"CBG Code": self.ids, "City Name": "Test city", "Year": 2019})
        tables = {"lookup": base.drop(columns="Year"),
                  "income": base.assign(**{"Median Household Income": [32100, 65400, 98700]}),
                  "population": base.assign(Population=[100, 200, 300]),
                  "education": base.assign(**{c: [10, 20, 30] for c in EDU_COLUMNS})}
        with zipfile.ZipFile(self.urban, "w") as archive:
            for kind, frame in tables.items():
                archive.writestr("nested/" + URBAN_FILES[kind], frame.to_csv(index=False))

    def test_archive_directory_and_profiles(self):
        census, context = prepare_inputs(self.acs, self.urban, "Test city", self.root / "output")
        direct, geo = extract_cbg_data(self.cbgs, self.directory)
        pd.testing.assert_frame_equal(census, direct)
        self.assertEqual(context.home_cbg_income.tolist(), [32100, 65400, 98700])
        self.assertEqual(context.home_cbg_edu.tolist(), ["Lowest", "Low", "High"])
        agents = generate_agents(census, num_agents=2, seed=42)
        pd.testing.assert_frame_equal(agents, generate_agents(census, num_agents=2, seed=42))
        self.assertFalse(agents.equals(generate_agents(census, num_agents=2, seed=43)))
        profiles = build_agent_profiles(agents, context, 2)
        self.assertEqual(len(profiles), 6)
        self.assertEqual(profiles[0]["CBG"], self.ids[0])
        self.assertEqual(profiles[0]["home_cbg_income"], 32100)
        cbgs = build_cbg_profiles(agents, context, 2)
        for profile in cbgs:
            for field in ("sex_distribution", "age_distribution", "race_distribution"):
                self.assertAlmostEqual(sum(profile[field].values()), 1)
            self.assertEqual(sum(profile["industry_counts"].values()), 2)
        save_profiles(cbgs, self.root / "cbgs.json")
        self.assertEqual(len(json.loads((self.root / "cbgs.json").read_text())), 3)
        self.assertFalse((self.root / "output" / "nested").exists())

    def test_missing_columns_and_context(self):
        census, context = prepare_inputs(self.acs, self.urban, "Test city", self.root / "output")
        with self.assertRaisesRegex(ValueError, "Missing ACS"):
            generate_agents(census.drop(columns="B01001e3"))
        with self.assertRaisesRegex(ValueError, "No matching rows"):
            prepare_inputs(self.acs, self.urban, "Absent city", self.root / "missing")
        agents = generate_agents(census, num_agents=2)
        with self.assertRaisesRegex(ValueError, "no home-CBG context"):
            build_agent_profiles(agents, context.iloc[:1], 2)
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            build_agent_profiles(agents, context, 3)

    def test_cli_modes(self):
        script = Path(__file__).resolve().parents[1] / "initialize_agents.py"
        raw = self.root / "raw"
        subprocess.run([sys.executable, str(script), "--acs-source", str(self.acs),
            "--urban-source", str(self.urban), "--city-name", "Test city", "--city-key", "TEST",
            "--agents-per-cbg", "2", "--output-dir", str(raw)], check=True, capture_output=True)
        prepared = self.root / "prepared"
        subprocess.run([sys.executable, str(script), "--census-table", str(raw / "cbg_extracted_data.csv"),
            "--context-table", str(raw / "cbg_context.csv"), "--city-key", "TEST",
            "--agents-per-cbg", "2", "--output-dir", str(prepared)], check=True, capture_output=True)
        for name in ("agent_profiles_TEST.json", "cbg_profiles_TEST.json"):
            self.assertEqual(json.loads((raw / name).read_text()), json.loads((prepared / name).read_text()))

    def test_cbg_list_leading_zero_and_duplicate(self):
        path = self.root / "cbgs.csv"
        pd.DataFrame({"CBG Code": ["10010201001"]}).to_csv(path, index=False)
        self.assertEqual(load_cbg_list(path)[CBG].tolist(), [self.ids[0]])
        pd.DataFrame({CBG: [self.ids[0], self.ids[0]]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_cbg_list(path)


if __name__ == "__main__":
    unittest.main()
