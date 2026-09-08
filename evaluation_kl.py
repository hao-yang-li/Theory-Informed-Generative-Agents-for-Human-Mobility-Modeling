"""KL calculations for TIMA mobility evaluation.

Inputs retain raw visit counts. Population/device expansion applies to the
Table 1 metrics; radius of gyration uses CBG-aggregated raw visit counts.
"""

from collections import Counter
import warnings

import numpy as np
from scipy.stats import entropy, gaussian_kde

EPSILON = 1e-10
MAX_DISTANCE_KM = 100.0
N_REGULAR_DISTANCE_BINS = 50
DISTANCE_CAP_QUANTILE = 0.995
KDE_GRID_POINTS = 512
RG_BINS = np.logspace(-1, 2, 50)


def calculate_kl(simulated, empirical):
    """Natural-log D_KL(simulated || empirical), with epsilon smoothing."""
    p = np.asarray(simulated, dtype=float) + EPSILON
    q = np.asarray(empirical, dtype=float) + EPSILON
    return float(entropy(p / p.sum(), q / q.sum()))


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    if not len(cumulative) or cumulative[-1] <= 0:
        return float("nan")
    index = np.searchsorted(cumulative, quantile * cumulative[-1], side="left")
    return float(values[order[min(index, len(order) - 1)]])


def fixed_distance_hist(distances, weights, edges, cap):
    regular, _ = np.histogram(
        distances[distances <= cap], bins=edges, weights=weights[distances <= cap]
    )
    return np.append(regular.astype(float), float(weights[distances > cap].sum()))


def kde_distribution(data, weights, grid):
    data = np.asarray(data, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(data) & np.isfinite(weights) & (weights > 0)
    data, weights = data[mask], weights[mask]
    if len(data) < 2 or np.allclose(data, data[0]):
        return np.zeros_like(grid)
    try:
        density = gaussian_kde(data, weights=weights)(grid)
    except (np.linalg.LinAlgError, ValueError):
        return np.zeros_like(grid)
    return density / density.sum() if density.sum() > 0 else np.zeros_like(grid)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    value = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(np.clip(value, 0, 1))))


def radii(frame, cbg_coords, poi_coords):
    """Visit-weighted radius about the activity center, one value per home CBG."""
    values = []
    for cbg, group in frame.groupby('home_cbg', sort=False):
        home = cbg_coords.get(str(cbg))
        if home is None:
            continue
        valid = []
        for poi, count in group.groupby('poi_id', sort=False)['count'].sum().items():
            point = poi_coords.get(str(poi))
            if point is not None and haversine(*home, *point) <= 500.0:
                valid.append((point, float(count)))
        total = sum(count for _, count in valid)
        if total <= 0:
            continue
        center_lat = sum(point[0] * count for point, count in valid) / total
        center_lon = sum(point[1] * count for point, count in valid) / total
        rg = np.sqrt(sum(haversine(center_lat, center_lon, *point)**2 * count
                         for point, count in valid) / total)
        if rg > 0.1:
            values.append(rg)
    return np.asarray(values, dtype=float)


class RevisedKLMetrics:
    def __init__(self, real, sim, agent_map, cbg_map, categories,
                 cbg_coords, poi_coords, devices=None, sampling_threshold=0.0):
        self.real, self.sim = real, sim
        self.categories = categories
        self.cbg_coords, self.poi_coords = cbg_coords, poi_coords
        self.devices = devices
        self.valid_cbgs = set()
        self.real_scale, self.sim_scale = {}, {}
        self._tables = None
        self._rg = None
        if devices is not None:
            population = {str(c): float(p.get('home_cbg_population') or 0)
                          for c, p in cbg_map.items()}
            self.valid_cbgs = {c for c, pop in population.items()
                               if pop > 0 and float(devices.get(c, 0) or 0) > 0
                               and float(devices[c]) / pop >= sampling_threshold}
            agents = Counter(str(a['CBG']) for a in agent_map.values())
            self.real_scale = {c: population[c] / float(devices[c]) for c in self.valid_cbgs}
            self.sim_scale = {c: population[c] / float(agents.get(c, 10) or 10)
                              for c in self.valid_cbgs}

    def _table1_inputs(self):
        if self._tables is not None:
            return self._tables
        if self.devices is None:
            warnings.warn("Trip-distance and POI KL require paths.home_panel_summary "
                          "for the evaluation week; these metrics are reported as NaN.")
            return None
        tables = []
        for frame, scale in ((self.real, self.real_scale), (self.sim, self.sim_scale)):
            use = frame.loc[
                frame['home_cbg'].isin(self.valid_cbgs)
                & frame['poi_cbg'].isin(self.valid_cbgs)
                & frame['dist_km'].between(0, MAX_DISTANCE_KM)
            ].copy()
            use['weight'] = use['count'] * use['home_cbg'].map(scale)
            tables.append(use)
        self._tables = tuple(tables)
        return self._tables

    def trip_distance(self):
        tables = self._table1_inputs()
        if tables is None:
            return float('nan')
        real, sim = tables
        rd, rw = real['dist_km'].to_numpy(), real['weight'].to_numpy()
        sd, sw = sim['dist_km'].to_numpy(), sim['weight'].to_numpy()
        cap = weighted_quantile(rd, rw, DISTANCE_CAP_QUANTILE)
        if not np.isfinite(cap) or cap <= 0 or sw.sum() <= 0:
            warnings.warn("Trip-distance KL requires positive empirical support and simulated flow.")
            return float('nan')
        edges = np.linspace(0.0, cap, N_REGULAR_DISTANCE_BINS + 1)
        return calculate_kl(fixed_distance_hist(sd, sw, edges, cap),
                            fixed_distance_hist(rd, rw, edges, cap))

    def poi_proportion(self):
        tables = self._table1_inputs()
        if tables is None:
            return float('nan')
        real, sim = tables
        cbgs = sorted(self.valid_cbgs)
        real_total = real.groupby('home_cbg')['weight'].sum().reindex(cbgs, fill_value=0).to_numpy()
        sim_total = sim.groupby('home_cbg')['count'].sum().reindex(cbgs, fill_value=0).to_numpy()
        real_weights = np.where(real_total > 0, real_total, 1.0)
        sim_weights = sim_total * np.asarray([self.sim_scale[c] for c in cbgs])
        grid = np.linspace(0.0, 1.0, KDE_GRID_POINTS)
        values = []
        for category in self.categories:
            rc = real.loc[real.category == category].groupby('home_cbg')['weight'].sum().reindex(cbgs, fill_value=0).to_numpy()
            sc = sim.loc[sim.category == category].groupby('home_cbg')['count'].sum().reindex(cbgs, fill_value=0).to_numpy()
            rp = np.divide(rc, real_total, out=np.zeros_like(real_total, dtype=float), where=real_total > 0)
            sp = np.divide(sc, sim_total, out=np.zeros_like(sim_total, dtype=float), where=sim_total > 0)
            pr = kde_distribution(rp, real_weights, grid)
            ps = kde_distribution(sp, sim_weights, grid)
            if pr.sum() > 0 and ps.sum() > 0:
                values.append(calculate_kl(ps, pr))
        if not values:
            warnings.warn("POI-proportion KL has no non-degenerate category distributions.")
        return float(np.mean(values)) if values else float('nan')

    def radius_of_gyration(self):
        if self._rg is not None:
            return self._rg
        real_rg = radii(self.real, self.cbg_coords, self.poi_coords)
        sim_rg = radii(self.sim, self.cbg_coords, self.poi_coords)
        if not np.histogram(real_rg, RG_BINS)[0].sum() or not np.histogram(sim_rg, RG_BINS)[0].sum():
            warnings.warn("Radius-of-gyration KL requires empirical and simulated radii within 0.1–100 km.")
            value = float('nan')
        else:
            real_hist = np.histogram(real_rg, bins=RG_BINS, density=True)[0]
            sim_hist = np.histogram(sim_rg, bins=RG_BINS, density=True)[0]
            value = calculate_kl(sim_hist, real_hist)
        self._rg = (float(np.median(sim_rg)) if len(sim_rg) else float('nan'), value)
        return self._rg
