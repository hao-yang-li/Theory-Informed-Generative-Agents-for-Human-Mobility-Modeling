import unittest

import numpy as np
import pandas as pd

from TIMA_analyze import MobilityEvaluator, returner_fraction, returner_transition


class ExplorerReturnerTests(unittest.TestCase):
    def setUp(self):
        self.home = '010010001001'
        self.homes = {self.home: (40., -74.)}
        self.pois = {'p1': (40., -74.), 'p2': (40.01, -74.), 'p3': (40.5, -74.)}
        self.real = pd.DataFrame([
            {'home_cbg': self.home, 'poi_cbg': self.home, 'poi_id': poi, 'count': count}
            for poi, count in zip(self.pois, [10, 9, 8])])
        self.sim = self.real.copy()
        self.sim['count'] = [10, 1, 9]

    def evaluator(self, real, sim):
        return MobilityEvaluator(real.copy(), sim.copy(), {}, {}, {}, self.homes, self.pois)

    def test_identical_data_has_zero_mae(self):
        result = self.evaluator(self.real, self.real).analyze_explorer_returner()
        self.assertEqual(result['mae'], 0.)
        np.testing.assert_array_equal(result['empirical_fraction'][:3], [0., 0., 1.])

    def test_changing_empirical_data_changes_comparison(self):
        different = self.evaluator(self.real, self.sim).analyze_explorer_returner()
        identical = self.evaluator(self.sim, self.sim).analyze_explorer_returner()
        self.assertAlmostEqual(different['mae'], 1/50)
        self.assertEqual(identical['mae'], 0.)
        self.assertNotEqual(different['empirical_k_star'], identical['empirical_k_star'])

    def test_cbgs_use_counts_not_agent_identity(self):
        expanded = self.real.loc[self.real.index.repeat(self.real['count'])].copy()
        expanded['count'] = 1
        expanded['agent_id'] = np.arange(len(expanded))
        a = returner_fraction(self.real, self.homes, self.pois, np.arange(1, 51))
        b = returner_fraction(expanded, self.homes, self.pois, np.arange(1, 51))
        np.testing.assert_array_equal(a, b)

    def test_log_interpolation_and_invalid_data(self):
        self.assertAlmostEqual(returner_transition([1, 2, 3], [0., 0., 1.]), np.sqrt(6))
        with self.assertWarns(UserWarning):
            curve = returner_fraction(self.real, {}, self.pois, np.arange(1, 51))
        self.assertTrue(np.isnan(curve).all())
        self.assertTrue(np.isnan(returner_transition(np.arange(1, 51), curve)))


if __name__ == '__main__':
    unittest.main()
