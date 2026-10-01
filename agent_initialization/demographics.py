import random

import numpy as np
import pandas as pd

LEVELS = ("Low", "Medium", "High")


def _level(profile, field):
    value = profile.get(field)
    if value not in LEVELS:
        raise ValueError(f"{field} must be an explicit Low, Medium, or High label; regenerate profiles.")
    return value


def income_level(profile):
    return _level(profile, "home_cbg_income_level")


def education_level(profile):
    return _level(profile, "home_cbg_edu")


def assign_city_labels(context):
    required = {"census_block_group", "City", "home_cbg_income", "High_Edu_Pop"}
    missing = required - set(context.columns)
    if missing:
        raise ValueError(f"Context is missing columns: {sorted(missing)}")
    result = context.copy()
    if result.empty or result["City"].isna().any():
        raise ValueError("Context requires a nonempty city for every CBG.")
    if result["census_block_group"].astype(str).duplicated().any():
        raise ValueError("Context contains duplicate CBGs.")
    for _, city in result.groupby("City", sort=True):
        income = pd.to_numeric(city["home_cbg_income"], errors="raise")
        education = pd.to_numeric(city["High_Edu_Pop"], errors="raise")
        for name, values in (("home_cbg_income", income), ("High_Edu_Pop", education)):
            if not (np.isfinite(values) & (values >= 0)).all():
                raise ValueError(f"{name} must contain finite nonnegative values.")
        valid_income = income[income > 0]
        if valid_income.empty:
            raise ValueError("Each city needs at least one positive income CBG.")

        def tertiles(values):
            q1, q2 = np.quantile(values, [1 / 3, 2 / 3], method="linear")
            return pd.Series(np.where(values <= q1, "Low",
                             np.where(values <= q2, "Medium", "High")), index=values.index)

        labels = tertiles(valid_income)
        zeros = city.loc[income.eq(0)].sort_values("census_block_group", key=lambda s: s.astype(str))
        weights = [int(labels.eq(level).sum()) for level in LEVELS]
        draws = random.Random(42).choices(LEVELS, weights=weights, k=len(zeros))
        result.loc[labels.index, "home_cbg_income_level"] = labels
        result.loc[zeros.index, "home_cbg_income_level"] = draws
        result.loc[city.index, "home_cbg_edu"] = tertiles(education)
        result.loc[city.index, "income_level_imputed"] = income.eq(0)
    result["income_level_imputed"] = result["income_level_imputed"].astype(bool)
    return result
