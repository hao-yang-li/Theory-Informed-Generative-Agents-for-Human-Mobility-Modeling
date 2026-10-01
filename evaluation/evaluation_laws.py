import warnings
import numpy as np


def rank_frequency(frame, poi_coords, min_locations=5, max_rank=50):
    rows = []
    accepted = frame.loc[frame.poi_id.isin(poi_coords)]
    for _, group in accepted.groupby('home_cbg', sort=False):
        counts = group.groupby('poi_id', sort=False)['count'].sum().to_numpy(float)
        if len(counts) < min_locations or counts.sum() <= 0:
            continue
        row = np.full(max_rank, np.nan)
        n = min(len(counts), max_rank)
        row[:n] = np.sort(counts)[::-1][:n] / counts.sum()
        rows.append(row)
    if not rows:
        warnings.warn('Rank-frequency evaluation requires a home-CBG profile with at least five POIs.')
        return np.full(max_rank, np.nan)
    values = np.asarray(rows)
    denominator = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0), denominator,
                     out=np.zeros(max_rank), where=denominator > 0)


def rank_frequency_rmse(real, sim, poi_coords):
    empirical = rank_frequency(real, poi_coords)
    simulated = rank_frequency(sim, poi_coords)
    return float(np.sqrt(np.mean((np.log(simulated + 1e-10)
                                  - np.log(empirical + 1e-10)) ** 2)))
