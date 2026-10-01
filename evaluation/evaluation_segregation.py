import warnings
import numpy as np
import pandas as pd


def income_rank_correlation(visits, cbg_map):
    incomes = pd.Series({str(c): float(p['home_cbg_income']) for c, p in cbg_map.items()
                         if p.get('home_cbg_income') is not None
                         and pd.notna(p['home_cbg_income'])
                         and float(p['home_cbg_income']) > 0})
    ranks = incomes.rank(pct=True)
    flows = visits.groupby(['home_cbg', 'poi_cbg']).weight.sum()
    rows = [(ranks[h], ranks[d], w) for (h, d), w in flows.items()
            if h in ranks and d in ranks and w > 0]
    if len(rows) < 2:
        return float('nan')
    x, y, weights = np.asarray(rows).T
    covariance = np.cov(x, y, aweights=weights)
    denominator = np.sqrt(covariance[0, 0] * covariance[1, 1])
    return float(covariance[0, 1] / denominator) if denominator > 0 else float('nan')


def income_dissimilarity(incomes):
    incomes = np.asarray(incomes, dtype=float)
    if incomes.ndim != 1 or not np.isfinite(incomes).all():
        raise ValueError('Income reference must be a finite one-dimensional array.')
    n = len(incomes)
    if n < 2:
        raise ValueError('Income dissimilarity requires at least two CBGs.')
    ranks = pd.Series(incomes).rank(method='average').to_numpy()
    matrix = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        distance = np.abs(ranks - ranks[i])
        ordered = np.sort(distance)
        left = np.searchsorted(ordered, distance, side='left')
        right = np.searchsorted(ordered, distance, side='right')
        row = (left + 0.5 * ((right - left) > 1)) / (n - 1)
        row[distance == 0] = 0
        matrix[i] = row
    return matrix


def experienced_segregation(visits, cbg_map):
    reference = [(str(c), float(p['home_cbg_income'])) for c, p in cbg_map.items()
                 if p.get('home_cbg_income') is not None and pd.notna(p['home_cbg_income'])]
    if len(reference) < 2:
        warnings.warn('Experienced segregation requires at least two income CBGs.')
        return float('nan')
    index = {c: i for i, (c, _) in enumerate(reference)}
    matrix = income_dissimilarity([v for _, v in reference])
    use = visits.loc[visits.home_cbg.isin(index), ['home_cbg', 'poi_id', 'weight']].copy()
    if not np.isfinite(use.weight).all() or (use.weight < 0).any():
        raise ValueError('Visit weights must be finite and nonnegative.')
    use = use.loc[use.weight > 0]
    if use.empty:
        warnings.warn('Experienced segregation has no eligible visits.')
        return float('nan')
    if use.poi_id.isna().any():
        raise ValueError('POI identifiers are required for experienced segregation.')
    flow = use.groupby(['poi_id', 'home_cbg'], sort=False).weight.sum()
    numerator = denominator = 0.
    for _, counts in flow.groupby(level='poi_id', sort=False):
        ix = np.array([index[c] for c in counts.index.get_level_values('home_cbg')])
        weights = counts.to_numpy(dtype=float)
        total = weights.sum()
        numerator += float(weights @ (matrix[np.ix_(ix, ix)] @ (weights / total)))
        denominator += total
    return 1. - numerator / denominator
