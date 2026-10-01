import math


def population_allocation(populations, target, minimum=1):
    if not populations or minimum < 1 or target < minimum * len(populations):
        raise ValueError("Target total must cover the per-CBG minimum.")
    if any(not math.isfinite(float(p)) or p < 0 for p in populations.values()):
        raise ValueError("CBG populations must be finite and nonnegative.")
    total = sum(populations.values())
    if total <= 0:
        raise ValueError("Total population must be positive.")
    quotas = {c: target * p / total for c, p in populations.items()}
    counts = {c: max(minimum, math.floor(q)) for c, q in quotas.items()}
    difference = target - sum(counts.values())
    if difference > 0:
        order = sorted(counts, key=lambda c: (quotas[c] - math.floor(quotas[c]), c), reverse=True)
        for c in order[:difference]:
            counts[c] += 1
    elif difference < 0:
        order = sorted((c for c in counts if counts[c] > minimum),
                       key=lambda c: (quotas[c] - math.floor(quotas[c]), c))
        cursor = 0
        while difference < 0:
            c = order[cursor % len(order)]
            if counts[c] > minimum:
                counts[c] -= 1
                difference += 1
            cursor += 1
    return counts
