import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import entropy, gaussian_kde

from evaluation_kl import (RevisedKLMetrics, calculate_kl, fixed_distance_hist,
                           weighted_quantile, radii, RG_BINS)
from TIMA_analyze import DataProcessor, POI_CATEGORIES


class RevisedKLTests(unittest.TestCase):
    def setUp(self):
        self.cbgs = ['010010001001', '010010001002', '010010001003', '010010001004']
        self.coords = {c: (40.0, -74.0) for c in self.cbgs}
        self.pois = {'p1': (40.01, -74.0), 'p2': (40.05, -74.0), 'p3': (40.15, -74.0)}
        self.profiles = {c: {'home_cbg_population': 100 * (i + 1)} for i, c in enumerate(self.cbgs)}
        self.agents = {str(i): {'CBG': c} for i, c in enumerate(self.cbgs)}
        self.devices = {c: 10 for c in self.cbgs}
        self.real, self.sim = [], []
        for i, c in enumerate(self.cbgs):
            for j, poi in enumerate(self.pois):
                base = {'home_cbg': c, 'poi_cbg': self.cbgs[0], 'poi_id': poi,
                        'category': POI_CATEGORIES[j], 'dist_km': [1., 5., 20.][j]}
                self.real.append(dict(base, count=(i + 1) ** (j + 1)))
                self.sim.append(dict(base, count=(i + 2) * (j + 1) + i * j ** 2))
        self.real, self.sim = pd.DataFrame(self.real), pd.DataFrame(self.sim)
        self.ev = RevisedKLMetrics(self.real, self.sim, self.agents, self.profiles,
                                   POI_CATEGORIES, self.coords, self.pois, self.devices)

    def test_direction_and_smoothing(self):
        p, q = np.array([0., 2., 8.]), np.array([1., 4., 5.])
        self.assertAlmostEqual(calculate_kl(p, q), entropy(p + 1e-10, q + 1e-10), places=14)
        self.assertNotAlmostEqual(calculate_kl(p, q), calculate_kl(q, p))

    def test_fixed_support_and_overflow(self):
        values, weights = np.array([1., 2., 10.]), np.array([500., 499., 1.])
        cap = weighted_quantile(values, weights, .995)
        self.assertEqual(cap, 2.)
        hist = fixed_distance_hist(values, weights, np.linspace(0, cap, 51), cap)
        self.assertEqual(len(hist), 51)
        self.assertEqual(hist[-1], 1.)
        self.assertEqual(hist.sum(), 1000.)
        sim = fixed_distance_hist(np.array([100.]), np.array([4.]), np.linspace(0, cap, 51), cap)
        self.assertEqual(sim[-1], 4.)

    def test_trip_filter_and_population_weights(self):
        reference = self.ev.trip_distance()
        extra = self.real.iloc[[0]].copy()
        extra['dist_km'], extra['count'] = 101., 1000000
        self.ev.real = pd.concat([self.real, extra], ignore_index=True)
        self.ev._tables = None
        self.assertAlmostEqual(self.ev.trip_distance(), reference, places=14)
        real, sim = self.ev._table1_inputs()
        rd, sd = real.dist_km.to_numpy(), sim.dist_km.to_numpy()
        rw, sw = real.weight.to_numpy(), sim.weight.to_numpy()
        cap = weighted_quantile(rd, rw, .995)
        edges = np.linspace(0, cap, 51)
        expected = entropy(fixed_distance_hist(sd, sw, edges, cap) + 1e-10,
                           fixed_distance_hist(rd, rw, edges, cap) + 1e-10)
        self.assertAlmostEqual(reference, expected, places=13)
        self.assertEqual(real.weight.iloc[0], self.real['count'].iloc[0] * 10.)
        self.assertEqual(sim.weight.iloc[0], self.sim['count'].iloc[0] * 100.)

    def test_poi_weighted_cbg_kde(self):
        grid = np.linspace(0., 1., 512)
        rt = np.array([self.real[self.real.home_cbg == c]['count'].sum() * (i+1)*10
                       for i, c in enumerate(self.cbgs)])
        st = np.array([self.sim[self.sim.home_cbg == c]['count'].sum() for c in self.cbgs])
        values = []
        for category in POI_CATEGORIES[:3]:
            rp = np.array([self.real[(self.real.home_cbg == c) & (self.real.category == category)]['count'].sum()
                           / self.real[self.real.home_cbg == c]['count'].sum() for c in self.cbgs])
            sp = np.array([self.sim[(self.sim.home_cbg == c) & (self.sim.category == category)]['count'].sum()
                           / self.sim[self.sim.home_cbg == c]['count'].sum() for c in self.cbgs])
            pr = gaussian_kde(rp, weights=rt)(grid)
            ps = gaussian_kde(sp, weights=st * np.arange(1, 5) * 100)(grid)
            values.append(entropy(ps / ps.sum() + 1e-10, pr / pr.sum() + 1e-10))
        self.assertAlmostEqual(self.ev.poi_proportion(), np.mean(values), places=12)

    def test_radius_of_gyration_uses_activity_center(self):
        # A single destination has zero spread even when it is far from home.
        self.assertEqual(len(radii(self.real[self.real.poi_id == 'p3'], self.coords, self.pois)), 0)
        rr, sr = radii(self.real, self.coords, self.pois), radii(self.sim, self.coords, self.pois)
        expected = entropy(np.histogram(sr, RG_BINS, density=True)[0] + 1e-10,
                           np.histogram(rr, RG_BINS, density=True)[0] + 1e-10)
        median, actual = self.ev.radius_of_gyration()
        self.assertAlmostEqual(median, np.median(sr), places=14)
        self.assertAlmostEqual(actual, expected, places=13)

    def test_missing_panel_does_not_substitute_unweighted_metric(self):
        ev = RevisedKLMetrics(self.real, self.sim, self.agents, self.profiles,
                              POI_CATEGORIES, self.coords, self.pois)
        with self.assertWarns(UserWarning):
            self.assertTrue(np.isnan(ev.trip_distance()))

    def test_metadata_aliases_and_home_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({'census_block_group': self.cbgs, 'latitude': [40.] * 4,
                          'longitude': [-74.] * 4}).to_csv(root/'geo.csv', index=False)
            pd.DataFrame({'poi_id': ['p1'], 'latitude': [40.01], 'longitude': [-74.],
                          'naics_code': [np.nan]}).to_csv(root/'poi.csv', index=False)
            pd.DataFrame({'safegraph_place_id': ['p1'], 'poi_cbg': [self.cbgs[0]],
                          'visitor_home_cbgs': [json.dumps({self.cbgs[0]: 3})]}).to_csv(root/'weekly.csv', index=False)
            (root/'sim.jsonl').write_text(json.dumps({'home_cbg': self.cbgs[0], 'agent_id': 0,
                                                     'poi_id': 'Home', 'dist_km': 100})+'\n')
            config = {'paths': {'ranges': str(root/'absent.json'), 'cbg_geo_data': str(root/'geo.csv'),
                                'poi_data_pattern': str(root/'poi.csv'), 'weekly_patterns': str(root/'weekly.csv'),
                                'output_dir': str(root), 'output_filename': 'sim.jsonl'}}
            proc = DataProcessor(config)
            proc.load_metadata()
            real, sim = proc.process_real_data(), proc.process_sim_data()
            self.assertEqual(real.home_cbg.iloc[0], self.cbgs[0])
            self.assertGreater(real.dist_km.iloc[0], 0)
            self.assertEqual(sim.dist_km.iloc[0], 0)
            self.assertEqual(sim.poi_cbg.iloc[0], self.cbgs[0])


if __name__ == '__main__':
    unittest.main()
