import numpy as np
import pandas as pd
from agent_initialization.demographics import income_level, education_level

VALID_GROUPS = {
    'race': {'White', 'Black', 'Other'},
    'sex': {'Male', 'Female'},
    'age_group': {'Under 18 years', '18 to 60 years', 'Over 60 years'},
    'industry': {
        'Agriculture forestry fishing and hunting and mining', 'Construction',
        'Manufacturing', 'Wholesale trade', 'Retail trade',
        'Transportation and warehousing and utilities', 'Information',
        'Finance and insurance and real estate and rental and leasing',
        'Professional scientific and management and administrative and waste management services',
        'Educational services and health care and social assistance',
        'Arts entertainment and recreation and accommodation and food services',
        'Other services except public administration', 'Public administration'},
}


def attribute(profile, dimension, is_agent=False):
    if dimension == 'income':
        return income_level(profile)
    if dimension == 'education':
        return education_level(profile)
    if is_agent:
        value = profile.get(dimension)
    else:
        field = dict(race='race_distribution', sex='sex_distribution',
                     age_group='age_distribution', industry='industry_counts')[dimension]
        distribution = profile.get(field) or {}
        value = max(distribution, key=distribution.get) if distribution else None
    return value if value in VALID_GROUPS[dimension] else None


def stratified_cpc(real, sim, cbg_profiles, agent_profiles):
    dimensions = ['income', 'education', 'race', 'industry', 'sex', 'age_group']
    flows = [frame.groupby(['home_tract', 'poi_tract']).weight.sum() for frame in (real, sim)]
    means = []
    for dimension in dimensions:
        labels = {c: attribute(p, dimension) for c, p in cbg_profiles.items()}
        agents = [attribute(p, dimension, True) for p in agent_profiles.values()]
        frame = pd.DataFrame({'tract': [str(c)[:11] for c in labels], 'label': list(labels.values())})
        modes = frame.dropna().groupby('tract').label.agg(lambda x: x.mode().iloc[0]).to_dict()
        groups = (['Low', 'Medium', 'High'] if dimension in {'income', 'education'} else
                  ['Male', 'Female'] if dimension == 'sex' else
                  sorted({v for v in [*labels.values(), *agents] if v is not None}))
        scores = []
        for group in groups:
            parts = [f.loc[[modes.get(home) == group for home in f.index.get_level_values(0)]] for f in flows]
            pair = pd.concat(parts, axis=1, keys=['real', 'sim']).fillna(0.)
            totals = pair.sum()
            scores.append(float(np.minimum(pair.real / totals.real, pair.sim / totals.sim).sum())
                          if (totals > 0).all() else 0.)
        if scores:
            means.append(np.mean(scores))
    return float(np.mean(means)) if means else float('nan')
