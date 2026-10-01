from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path


def load_boundaries(path, cbg_column='GEOID', year=2019):
    import geopandas as gpd
    if Path(path).suffix.lower() == '.csv':
        from shapely import wkt
        frame = pd.read_csv(path, dtype={cbg_column: str})
        if 'Year' in frame:
            frame = frame.loc[pd.to_numeric(frame.Year, errors='raise').eq(year)].copy()
        geo = gpd.GeoDataFrame(frame, geometry=frame.Boundary.map(wkt.loads), crs='EPSG:4326')
    else:
        geo = gpd.read_file(path)
        if 'Year' in geo:
            geo = geo.loc[pd.to_numeric(geo.Year, errors='raise').eq(year)].copy()
    geo = geo.rename(columns={cbg_column: 'cbg'})
    geo['cbg'] = geo.cbg.astype(str).str.zfill(12)
    if geo.cbg.duplicated().any() or geo.geometry.isna().any() or geo.geometry.is_empty.any():
        raise ValueError('Boundaries require unique CBG identifiers and nonempty geometries.')
    return geo


def hotspot_statistics(geo, flows, groups, valid_cbgs):
    from libpysal.weights import Queen
    from esda import G_Local
    area = geo.loc[geo.cbg.isin(valid_cbgs)].sort_values('cbg').reset_index(drop=True)
    if area.cbg.duplicated().any() or set(area.cbg) != set(valid_cbgs):
        raise ValueError('Boundaries must contain exactly one geometry for every eligible CBG.')
    weights = Queen.from_dataframe(area, ids=area.cbg.tolist())
    records, overlaps = [], {}
    for group in ['High', 'Low']:
        hot = {}
        for model in ['Real Data', 'TIMA']:
            counts = Counter()
            for (origin, destination), value in flows[model].items():
                if origin in valid_cbgs and destination in valid_cbgs and groups.get(origin) == group:
                    counts[destination] += value
            values = np.array([counts[c] for c in area.cbg], dtype=float)
            result = G_Local(values, weights, transform='B', star=False, permutations=0)
            z = result.Zs
            p = 2 * norm.sf(np.abs(z))
            mask = np.isfinite(z) & (z > 0) & (p < .05)
            hot[model] = mask
            records.extend(dict(cbg=c, group=group, model=model, visits=float(v),
                gi=float(g), z=float(zz), p_two_sided=float(pp), hotspot=bool(h))
                for c, v, g, zz, pp, h in zip(area.cbg, values, result.Gs, z, p, mask))
        real, sim = hot['Real Data'], hot['TIMA']
        intersection, union = int((real & sim).sum()), int((real | sim).sum())
        overlaps[group] = dict(empirical_hotspots=int(real.sum()), tima_hotspots=int(sim.sum()),
            intersection=intersection, union=union,
            recall=intersection / int(real.sum()) if real.any() else float('nan'),
            precision=intersection / int(sim.sum()) if sim.any() else float('nan'),
            jaccard=intersection / union if union else float('nan'))
    return pd.DataFrame(records), dict(groups=overlaps, n_cbgs=len(area))
