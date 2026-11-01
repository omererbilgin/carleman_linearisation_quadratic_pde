# Algorthmic structure:
#                           Removing a rate means removing every whole TSE that passes through it
#                           A mask is applied at every Taylor step, doing it only at the end would keep transfers from removed coordinates
#                           The original and altered readouts gives the two rankings in the article

#                           Start reading this file from: compute

from fractions import Fraction
import numpy as np
from carleman.analysis.support import supported_vorticity_matrix


def remaining_responses(problem, exclusions, time=1., terms=32):
    matrix, initial, rates = (problem[key] for key in ("matrix", "initial", "rates"))
    vort = problem["vorticity_indices"]
    pairs, weights = problem["palinstrophy_indices"], problem["palinstrophy_weights"]
    # one column per exclusion, an empty tuple gives the unaltered response
    keep = np.column_stack([~np.isin(rates, removed) for removed in exclusions])
    term = initial[:, None] * keep
    omega = term[vort].copy()
    palinstrophy = weights @ term[pairs]
    tail_w = np.zeros_like(omega)
    tail_p = np.zeros_like(palinstrophy)
    for degree in range(1, terms+1):
        term = (matrix @ term) * (time/degree)
        # filter at every step, otherwise a removed eigenspace can feed later transfers
        term *= keep
        field, value = term[vort], weights @ term[pairs]
        omega += field
        palinstrophy += value
        if degree > terms-8:
            tail_w += abs(field)
            tail_p += abs(value)
    # this is a convergence check on the last eight terms, not a rigorous error bound
    tail = max(float(tail_w.max()), float(tail_p.max()))
    return omega.T * problem["phase"], palinstrophy, tail


def compute(*, order=10, radius=30, epsilon=Fraction(2), viscosity=.001,
            time=1., terms=32, batch_size=8, problem=None):
    if terms < 8 or batch_size < 1:
        raise ValueError("at least eight Taylor terms and a positive batch size are required")
    if problem is None:
        problem = supported_vorticity_matrix(order, radius, epsilon, viscosity)
    baseline_w, baseline_p, tail = remaining_responses(problem, [()], time, terms)
    norm_w, norm_p = np.linalg.norm(baseline_w[0]), abs(float(baseline_p[0]))
    if norm_w == 0 or norm_p == 0:
        raise ValueError("relative errors need nonzero original responses")
    groups = {alpha: [] for alpha in (1, 2)}
    candidates = np.unique(problem["rates"])
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start+batch_size]
        omega, palinstrophy, current_tail = remaining_responses(
            problem, [(rate,) for rate in batch], time, terms,
        )
        tail = max(tail, current_tail)
        for rate, field, value in zip(batch, omega, palinstrophy):
            # the common 2*pi from Parseval cancels in this ratio
            groups[1].append({
                "alpha": 1, "unit_viscosity_decay_rate": int(rate),
                "relative_removal_error_percent": float(100*np.linalg.norm(field-baseline_w[0])/norm_w),
            })
            # read P from level two, squaring the altered vorticity is a different removal
            groups[2].append({
                "alpha": 2, "unit_viscosity_decay_rate": int(rate),
                "relative_removal_error_percent": float(100*abs(value-baseline_p[0])/norm_p),
                "remaining_palinstrophy": float(value),
                "signed_palinstrophy_change_percent": float(100*(value-baseline_p[0])/norm_p),
            })
    # the article uses t=1, nu=.001; larger times or viscosities may need more terms
    if not np.isfinite(tail) or tail > 1e-10:
        raise RuntimeError(f"Taylor tail is {tail:.3g}; increase --terms and check convergence")
    rankings = []
    for alpha, rows in groups.items():
        rows.sort(key=lambda row: (-row["relative_removal_error_percent"], row["unit_viscosity_decay_rate"]))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        rankings.append({"alpha": alpha, "ranking": rows})
    return {
        "inputs": {"R": radius, "N": order, "epsilon": str(epsilon), "nu": viscosity, "time": time},
        "rule": "Remove every complete TSE containing each eigenvalue, including beta=alpha; retain all other TSEs unchanged",
        "comparison": "Full unaltered response; vorticity L2 norm and absolute total palinstrophy difference",
        "original_vorticity_L2": float(2*np.pi*norm_w),
        "original_palinstrophy": float(baseline_p[0]),
        "candidate_count": len(candidates),
        "taylor_terms": terms, "maximum_absolute_tail_sum": tail,
        "groups": rankings,
    }
