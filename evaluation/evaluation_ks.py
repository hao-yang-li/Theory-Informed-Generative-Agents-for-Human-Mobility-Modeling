import warnings
import numpy as np
from scipy.stats import ks_2samp


def visits_per_location_ks(real, simulated):
    real, simulated = np.asarray(real, dtype=float), np.asarray(simulated, dtype=float)
    if real.ndim != 1 or simulated.shape != real.shape or not real.size:
        raise ValueError("Visit vectors must be nonempty and have the same one-dimensional shape.")
    if not (np.isfinite(real).all() and np.isfinite(simulated).all()) or (real < 0).any() or (simulated < 0).any():
        raise ValueError("Visit counts must be finite and nonnegative.")
    if real.sum() <= 0 or simulated.sum() <= 0:
        warnings.warn("Visits-per-location KS requires positive empirical and simulated totals.")
        return float('nan')
    return float(ks_2samp(real / real.sum(), simulated / simulated.sum(), method='asymp').statistic)
