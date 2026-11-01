# Algorthmic structure:
#                           Initial support gives the Fourier products which can feed the measured degree
#                           Couplings is summed inside each eigenspace before testing a complete TSE, keeping the order of the rates
#                           Residues and integer numerators lets the temporal responses be evaluated without losing the cancellations

#                           Start reading this file from: supported_problem, then count_sequences

from collections import Counter, defaultdict
from fractions import Fraction
import heapq
from math import factorial, lcm, prod
from itertools import combinations_with_replacement

import mpmath as mp
import numpy as np


def supported_problem(order=10, radius=30, epsilon=Fraction(3, 10)):
    if order < 2:
        raise ValueError("order must be at least two")
    if radius < 2:
        raise ValueError("the article's initial condition requires radius >= 2")
    epsilon = Fraction(epsilon)
    # encode (kx,ky) as one index, but keep the full square lattice ordering
    width = 2 * radius + 1

    def mode_index(mode):
        return (mode[0] + radius) * width + mode[1] + radius

    def index_mode(index):
        return index // width - radius, index % width - radius

    # amplitude and power of i, keep these exact
    initial_modes = {
        mode_index(mode): (value, phase)
        for mode, value, phase in (
            ((1, -1), Fraction(1, 4), 0),
            ((-1, 1), Fraction(1, 4), 0),
            ((1, 1), Fraction(-1, 4), 0),
            ((-1, -1), Fraction(-1, 4), 0),
            ((2, 1), -epsilon / 2, 1),
            ((-2, -1), epsilon / 2, 1),
        )
    }

    # initial products with total wavevector zero
    nodes = set()
    for degree in range(2, order + 1):
        for monomial in combinations_with_replacement(sorted(initial_modes), degree):
            wavevectors = [index_mode(index) for index in monomial]
            if (sum(mode[0] for mode in wavevectors) == 0
                    and sum(mode[1] for mode in wavevectors) == 0):
                nodes.add(monomial)

    frontier = list(nodes)
    transfers = {}

    # merge p and q into p+q until all the reachable lower degrees are included
    while frontier:
        monomial = frontier.pop()
        targets = defaultdict(Fraction)
        if len(monomial) > 2:
            for left, right in combinations_with_replacement(sorted(set(monomial)), 2):
                # need two occurrences before using a repeated source pair
                if left == right and monomial.count(left) < 2:
                    continue
                p, q = index_mode(left), index_mode(right)
                replacement_mode = p[0] + q[0], p[1] + q[1]
                if any(abs(component) > radius for component in replacement_mode):
                    continue
                p_squared = p[0]**2 + p[1]**2
                q_squared = q[0]**2 + q[1]**2
                if not p_squared or not q_squared:
                    continue
                # combine both ordered Fourier couplings into one symmetric-pair coefficient
                coupling = ((p[1] * q[0] - p[0] * q[1])
                            * (Fraction(1, p_squared) - Fraction(1, q_squared)))
                if not coupling:
                    continue
                replacement = mode_index(replacement_mode)
                remainder = list(monomial)
                remainder.remove(left)
                remainder.remove(right)
                target = tuple(sorted(remainder + [replacement]))

                # same replacement can appear several times in the target
                targets[target] += coupling * target.count(replacement)
                if target not in nodes:
                    nodes.add(target)
                    frontier.append(target)
        # sum duplicate targets before removing zero couplings
        transfers[monomial] = {target: value for target, value in targets.items() if value}

    # degree first, so the later recurrences can use already computed lower levels
    monomials = sorted(nodes, key=lambda monomial: (len(monomial), monomial))
    decay = [sum(sum(component**2 for component in index_mode(index))
                 for index in monomial) for monomial in monomials]
    initial = []
    for monomial in monomials:
        value, phase = Fraction(1), 0
        for index in monomial:
            amplitude, exponent = initial_modes.get(index, (Fraction(0), 0))
            value *= amplitude
            phase += exponent
        if value and phase % 2:
            raise ArithmeticError("zero-wavevector initial monomial is not real")
        # i**phase is real in this sector, keep its sign separately from the amplitude
        initial.append(value * (-1)**(phase // 2))
    degrees = [len(monomial) for monomial in monomials]
    # palinstrophy readout divided by pi**2, it acts only on degree two
    output = [Fraction(2 * rate) if degree == 2 else Fraction(0)
              for degree, rate in zip(degrees, decay)]
    return {
        "radius": radius, "order": order, "epsilon": epsilon, "monomials": monomials,
        "transfers": transfers, "decay": decay, "initial": initial,
        "output": output, "degrees": degrees,
    }


def count_sequences(problem, max_order=None):
    max_order = problem["order"] if max_order is None else max_order
    if not 2 <= max_order <= problem["order"]:
        raise ValueError("max_order must lie between two and the constructed order")
    rates = problem["decay"]
    initial = problem["initial"]
    positions = {monomial: index for index, monomial in enumerate(problem["monomials"])}
    outgoing = [[] for _ in rates]
    for source, monomial in enumerate(problem["monomials"]):
        for target, coupling in problem["transfers"][monomial].items():
            # follow the row vector backwards from the measured degree toward the initial degree
            outgoing[positions[target]].append((source, coupling))
    counts = Counter()
    max_repeated = defaultdict(int)
    visited = 0

    def visit(vector, sequence, degree):
        nonlocal visited
        visited += 1
        # apply the initial condition only after summing all coordinates in this sequence
        numerator = sum((weight * initial[index] for index, weight in vector.items()
                         if initial[index]), Fraction(0))
        # different coordinate paths can cancel, only the summed nonzero numerator is counted
        if numerator:
            counts[degree] += 1
            max_repeated[degree] = max(max_repeated[degree],
                                       max(Counter(sequence).values()))
        # a zero numerator at this degree can still lead to a nonzero higher-degree response
        if degree == max_order:
            return

        # apply the coupling before splitting by rate, the order matters here
        next_vector = defaultdict(Fraction)
        for target, weight in vector.items():
            for source, coupling in outgoing[target]:
                next_vector[source] += weight * coupling
        # the Fourier diagonal blocks are diagonal, grouping by rate applies their projectors
        groups = defaultdict(dict)
        for index, weight in next_vector.items():
            if weight:
                groups[rates[index]][index] = weight
        for rate, group in groups.items():
            visit(group, sequence + (rate,), degree + 1)

    groups = defaultdict(dict)
    # start at the degree-two palinstrophy row and work upward through source degrees
    for index, weight in enumerate(problem["output"]):
        if weight:
            groups[rates[index]][index] = weight
    for rate, group in groups.items():
        visit(group, (rate,), 2)
    return dict(counts), dict(max_repeated), visited


def divide_by_pole(coefficients, rate):
    result = defaultdict(Fraction)
    for (other, power), value in coefficients.items():
        if not value:
            continue

        # same rate raises the pole order, otherwise use partial fractions
        if rate == other:
            result[rate, power + 1] += value
        else:
            # use exact fractions here, close poles can otherwise cause large cancellation errors
            gap = rate - other
            for index in range(1, power + 1):
                result[other, index] += value * (-1)**(power - index) / gap**(power - index + 1)
            result[rate, 1] += value * (-1)**power / gap**power
    return {key: value for key, value in result.items() if value}


def observable_resolvent_by_degree(problem, viscosity=Fraction(1, 1000)):
    viscosity = Fraction(viscosity)
    positions = {monomial: index for index, monomial in enumerate(problem["monomials"])}
    partials = []
    results = {degree: defaultdict(Fraction) for degree in range(2, problem["order"] + 1)}

    # each column only needs lower-degree columns, which are already in partials
    for monomial, rate, output, initial in zip(
            problem["monomials"], problem["decay"], problem["output"], problem["initial"]):
        # add all lower-degree incoming residues before dividing by this diagonal pole
        rhs = defaultdict(Fraction)
        for target, coupling in problem["transfers"][monomial].items():
            for key, value in partials[positions[target]].items():
                rhs[key] += coupling * value
        coefficients = defaultdict(Fraction, divide_by_pole(rhs, viscosity * rate))
        # direct observable term, separate from transfers coming from lower degrees
        coefficients[viscosity * rate, 1] += output
        coefficients = {key: value for key, value in coefficients.items() if value}
        partials.append(coefficients)
        if initial:
            # contract with the initial value, but keep its source degree for the beta groups
            result = results[len(monomial)]
            for key, value in coefficients.items():
                result[key] += initial * value
    return {degree: {key: value for key, value in result.items() if value}
            for degree, result in results.items()}


def cumulative_residues(by_degree, order):
    total = defaultdict(Fraction)
    result = {}
    # include source degrees up to N, cancel poles after adding them
    for degree in range(2, order + 1):
        for key, value in by_degree[degree].items():
            total[key] += value
        result[degree] = {key: value for key, value in total.items() if value}
    return result


def make_report(problem, counts, repetitions, visited, residues, viscosity):
    order, radius = problem["order"], problem["radius"]
    rows = []
    for truncation in range(2, order + 1):
        included = {degree: counts.get(degree, 0) for degree in range(2, truncation + 1)}
        rows.append({
            "order": truncation,
            "counts_by_source_degree": included,
            "total_sequences": sum(included.values()),
            "maximum_observable_pole_order": max(power for rate, power in residues[truncation]),
            "maximum_single_sequence_repetition": max(repetitions.get(degree, 0)
                                                       for degree in included),
        })
    width = 2 * radius + 1
    return {
        "inputs": {
            "radius": radius, "maximum_order": order, "viscosity": str(viscosity),
            "epsilon": str(problem["epsilon"]), "domain": "[0, 2*pi)^2",
            "initial_vorticity": f"sin(x)*sin(y) + ({problem['epsilon']})*sin(2*x+y)",
            "fourier_cutoff": "integer wavevectors with |k_x|, |k_y| <= radius",
        },
        "method": "exact rational resolvent recurrence on the supported zero-wavevector sector",
        "supported_sector": {
            "dimension": len(problem["monomials"]),
            "nonzero_off_diagonal_entries": sum(map(len, problem["transfers"].values())),
            "coordinates_by_degree": dict(sorted(Counter(problem["degrees"]).items())),
            "maximum_factor_wavevector_component": max(
                abs(component) for monomial in problem["monomials"] for index in monomial
                for component in (index // width - radius, index % width - radius)),
        },
        "visited_sequence_prefixes": visited,
        "counts_by_source_degree": dict(sorted(counts.items())),
        "maximum_repetitions_by_source_degree": dict(sorted(repetitions.items())),
        "rows": rows,
        "exact_resolvent_coefficients": {
            truncation: [{"decay_rate": str(rate), "pole_order": power, "coefficient": str(value)}
                         for (rate, power), value in sorted(coefficients.items())]
            for truncation, coefficients in residues.items()
        },
    }


def responses_by_source_degree(report, time=2.0):
    import mpmath as mp

    previous = {}
    responses = []
    # large signed residues can nearly cancel, evaluate before converting to float
    with mp.workdps(90):
        for degree in range(2, report["inputs"]["maximum_order"] + 1):
            terms = report["exact_resolvent_coefficients"]
            current = {
                (Fraction(term["decay_rate"]), term["pole_order"]): Fraction(term["coefficient"])
                for term in terms.get(str(degree), terms.get(degree, []))
            }
            # subtract P_(N-1) from P_N to get just this source degree
            difference = {
                key: current.get(key, Fraction(0)) - previous.get(key, Fraction(0))
                for key in current.keys() | previous.keys()
            }
            # at zero time only simple-pole terms survive the inverse transform
            if time == 0:
                initial = sum((value for (_, power), value in difference.items() if power == 1),
                              Fraction(0))
                value = mp.mpf(initial.numerator) / initial.denominator
            else:
                value = mp.fsum(
                    (mp.mpf(coefficient.numerator) / coefficient.denominator)
                    * mp.mpf(time)**(power - 1) / factorial(power - 1)
                    * mp.exp(-mp.mpf(rate.numerator) / rate.denominator * time)
                    for (rate, power), coefficient in difference.items() if coefficient
                )
            responses.append((degree, mp.pi**2 * value))
            previous = current
        total = mp.fsum(value for _, value in responses)
        return {
            "radius": report["inputs"]["radius"],
            "order": report["inputs"]["maximum_order"],
            "viscosity": report["inputs"]["viscosity"],
            "epsilon": report["inputs"].get("epsilon", "3/10"),
            "time": time,
            "total_palinstrophy": float(total),
            "percentage_definition": "100 * P_beta(time) / sum_beta P_beta(time), retaining signs",
            "rows": [{"beta": degree, "palinstrophy_response": float(value),
                      "percentage_total": float(100 * value / total)}
                     for degree, value in responses],
        }


def source_history(report, times):
    evaluated = [responses_by_source_degree(report, float(t)) for t in times]
    return {
        "order": report["inputs"]["maximum_order"],
        "radius": report["inputs"]["radius"],
        "epsilon": report["inputs"]["epsilon"],
        "viscosity": report["inputs"]["viscosity"],
        "times": np.asarray(times).tolist(),
        "total_palinstrophy": [item["total_palinstrophy"] for item in evaluated],
        # one time history per beta, in the same row order at every time
        "rows": [
            {"beta": row["beta"], "palinstrophy_response": [
                item["rows"][i]["palinstrophy_response"] for item in evaluated
            ]}
            for i, row in enumerate(evaluated[0]["rows"])
        ],
    }


def numerator_candidates(problem, beta, capacity):
    positions = {monomial: i for i, monomial in enumerate(problem["monomials"])}
    rates = problem["decay"]

    # clear denominators per level so the recursion can use integer numerators
    layer_scales = {degree: 1 for degree in range(2, beta)}
    for source, targets in problem["transfers"].items():
        if len(source) <= beta:
            for target, coupling in targets.items():
                layer_scales[len(target)] = lcm(layer_scales[len(target)], coupling.denominator)
    initial_scale = lcm(*(value.denominator for degree, value in
                         zip(problem["degrees"], problem["initial"]) if degree == beta))
    initial = [int(value * initial_scale) if degree == beta else 0
               for degree, value in zip(problem["degrees"], problem["initial"])]
    outgoing = [[] for _ in rates]
    for source, targets in problem["transfers"].items():
        if len(source) <= beta:
            for target, coupling in targets.items():
                # the recursion starts at the observable, so store the reverse transfer direction
                outgoing[positions[target]].append(
                    (positions[source], int(coupling * layer_scales[len(target)])))
    scale = initial_scale * prod(layer_scales.values())
    candidates = []
    active = visited = signed_sum = absolute_sum = 0
    degree_counts = Counter()

    def visit(vector, sequence, degree):
        nonlocal active, visited, signed_sum, absolute_sum
        visited += 1
        # a complete sequence has reached its requested initial degree
        if degree == beta:
            # contract the whole source eigenspace before deciding if the sequence is active
            numerator = sum(weight * initial[i] for i, weight in vector.items() if initial[i])
            if numerator:
                active += 1
                signed_sum += numerator
                absolute_sum += abs(numerator)
                degree_counts[max(Counter(sequence).values())-1] += 1
                candidate = (abs(numerator), sequence, numerator)
                # keep the largest numerators, the smallest retained one is at the heap root
                if len(candidates) < capacity:
                    heapq.heappush(candidates, candidate)
                elif candidate > candidates[0]:
                    heapq.heapreplace(candidates, candidate)
            return

        # sum coordinate paths first, then split by the next eigenspace
        next_vector = defaultdict(int)
        for target, weight in vector.items():
            for source, coupling in outgoing[target]:
                next_vector[source] += weight * coupling
        groups = defaultdict(dict)
        for index, weight in next_vector.items():
            if weight:
                # same rate means the same next eigenspace, but dont reorder earlier rates
                groups[rates[index]][index] = weight
        for rate, group in groups.items():
            visit(group, sequence + (rate,), degree + 1)

    seeds = defaultdict(dict)
    for index, weight in enumerate(problem["output"]):
        if weight:
            assert weight.denominator == 1
            seeds[rates[index]][index] = weight.numerator
    for rate, seed in seeds.items():
        visit(seed, (rate,), 2)
    return candidates, scale, {
        "active_ordered_sequences": active,
        "visited_prefixes": visited,
        "leading_numerator_sum_exact": str(Fraction(signed_sum, scale)),
        "absolute_numerator_sum_exact": str(Fraction(absolute_sum, scale)),
        "maximum_polynomial_degree_counts": dict(sorted(degree_counts.items())),
    }


def inverse_kernel(rates, viscosity, time):
    # each rate adds (s+nu*r) to the denominator, so the first time power is t**degree
    degree = len(rates)-1
    # shift by the largest rate so the series nodes below are nonnegative
    largest = max(rates)

    # only the denominator can be reordered, the numerator stays as it was
    nodes = [mp.mpf(largest-rate)*viscosity*time for rate in sorted(rates)]
    terms = 80
    largest_node = max(nodes)
    # check the tail for this time/rate range, 80 terms is not enough for every input
    relative_remainder_bound = mp.exp(largest_node)*largest_node**(terms+1)/factorial(terms+1)
    if relative_remainder_bound > mp.mpf("1e-80"):
        raise ValueError("time/rate range needs more inverse-kernel series terms")
    homogeneous = [mp.mpf(1)] + [mp.mpf(0)]*terms

    # k goes upwards, otherwise repeated powers of this node would be missed
    for node in nodes:
        for k in range(1, terms+1):
            homogeneous[k] += node*homogeneous[k-1]
    return mp.exp(-viscosity*largest*time)*mp.fsum(
        value/factorial(degree+k) for k, value in enumerate(homogeneous))*time**degree


def polynomial_degree(rates):
    # highest polynomial power of one complete sequence, not of the final summed response
    return max(Counter(rates).values())-1


def homogeneous_coefficients(nodes, terms):
    nodes = np.asarray(nodes, dtype=np.longdouble)
    if nodes.ndim == 1:
        nodes = nodes[None, :]
    coefficients = np.zeros((len(nodes), terms+1), dtype=np.longdouble)
    coefficients[:, 0] = 1

    # one factor 1/(1-node*z) at a time, k needs to go upwards
    # coefficients[k-1] already includes this node, reusing it gives its higher powers
    for column in range(nodes.shape[1]):
        for k in range(1, terms+1):
            coefficients[:, k] += nodes[:, column]*coefficients[:, k-1]
    return coefficients


def degree_moments(data, viscosity, *, terms=20, chunk_size=16384):
    viscosity = Fraction(viscosity)
    # stored integer numerators all use this one denominator
    scale = int(data['shared_numerator_denominator'])
    records = data['groups']
    highest = max(polynomial_degree(row['factor_rates']) for row in records)
    moments = np.zeros((highest+1, terms+1), dtype=np.longdouble)
    counts = Counter()
    exact_moments = defaultdict(lambda: [0, 0, 0])
    absolute_sums = Counter()
    repeated_eigenvalues = defaultdict(Counter)
    highest_coefficients = defaultdict(Fraction)
    max_rate = max(max(row['factor_rates']) for row in records)
    nu = np.longdouble(str(viscosity.numerator))/np.longdouble(str(viscosity.denominator))

    # batches so the temporary series arrays dont get too large
    for start in range(0, len(records), chunk_size):
        chunk = records[start:start+chunk_size]
        rates = np.asarray([row['factor_rates'] for row in chunk], dtype=np.int64)
        degrees = np.empty(len(chunk), dtype=int)
        weights = np.empty(len(chunk), dtype=np.longdouble)
        for i, row in enumerate(chunk):
            multiplicities = Counter(row['factor_rates'])
            # classify by the most repeated rate, without splitting the sequence into poles
            degree = max(multiplicities.values())-1
            degrees[i] = degree
            numerator = int(row['summed_numerator_integer'])
            weights[i] = (np.longdouble(row['summed_numerator_integer'])
                          / np.longdouble(str(scale)))
            # one denominator group may represent several different ordered sequences
            counts[degree] += row['ordered_sequences']
            absolute_sums[degree] += int(row['absolute_numerator_sum_integer'])
            sum_rates = sum(row['factor_rates'])
            # first moments are 1, sum(r), and ((sum r)**2+sum(r**2))/2
            h2 = (sum_rates**2 + sum(rate*rate for rate in row['factor_rates']))//2
            exact_moments[degree][0] += numerator
            exact_moments[degree][1] += numerator*sum_rates
            exact_moments[degree][2] += numerator*h2
            for rate, multiplicity in multiplicities.items():
                if multiplicity == degree+1:
                    repeated_eigenvalues[degree][rate] += row['ordered_sequences']
                    if degree == highest:
                        # for the highest pole, evaluate the remaining factors at its repeated eigenvalue
                        others = [other for other in row['factor_rates'] if other != rate]
                        highest_coefficients[-viscosity*rate] += (
                            Fraction(numerator, scale)
                            / (viscosity**len(others)*prod(other-rate for other in others)))
        h = homogeneous_coefficients(-nu*rates, terms)
        for degree in range(highest+1):
            selected = degrees == degree
            moments[degree] += (weights[selected, None]*h[selected]).sum(axis=0,
                                                                       dtype=np.longdouble)

    # use exact sums for the first moments, the floating point sums cancel badly
    for degree in range(highest+1):
        for k in range(min(2, terms)+1):
            exact = Fraction(exact_moments[degree][k], scale)*(-viscosity)**k
            moments[degree, k] = (np.longdouble(str(exact.numerator))
                                 / np.longdouble(str(exact.denominator)))
    summary = [{'maximum_polynomial_degree': degree, 'ordered_sequences': counts[degree],
                'leading_numerator_exact': str(Fraction(exact_moments[degree][0], scale)),
                'absolute_sequence_numerator_sum_exact': str(Fraction(absolute_sums[degree], scale)),
                'eigenvalues_at_maximum_degree': {
                    str(-viscosity*rate): count
                    for rate, count in sorted(repeated_eigenvalues[degree].items())}}
               for degree in range(highest+1)]
    summary[-1]['highest_degree_coefficients_from_sequences_exact'] = {
        str(eigenvalue): str(value) for eigenvalue, value in sorted(highest_coefficients.items())}
    return moments, summary, max_rate


def evaluate_moments(moments, beta, times):
    result = np.zeros((len(times), len(moments)))
    with mp.workdps(70):
        # one row per sequence class, one column per term in its kernel series
        for degree, row in enumerate(moments):
            coefficients = [mp.mpf(str(value)) for value in row]
            for i, time in enumerate(times):
                t = mp.mpf(float(time))
                # restore the palinstrophy pi**2 factor and the inverse-Laplace factorials
                result[i, degree] = float(mp.pi**2*mp.fsum(
                    value*t**(beta-2+k)/factorial(beta-2+k)
                    for k, value in enumerate(coefficients)))
    return result


def endpoint_responses(candidates, scale, beta, *, chunk_size=16384, terms=20):
    result = np.empty(len(candidates), dtype=np.longdouble)
    # a beta-to-2 sequence starts at time power beta-2
    denominator = np.array([str(factorial(beta-2+k)) for k in range(terms+1)],
                           dtype=np.longdouble)
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start:start+chunk_size]
        rates = np.array([sequence for _, sequence, _ in chunk], dtype=np.int64)
        weights = np.array([str(numerator) for _, _, numerator in chunk], dtype=np.longdouble)
        # convert the common integer scale only at evaluation time
        weights /= np.longdouble(str(scale))
        h = homogeneous_coefficients(-np.longdouble(".001")*rates, terms)
        result[start:start+len(chunk)] = np.longdouble(str(np.pi**2))*weights*(h/denominator).sum(axis=1)
    return result


def selected_moments(candidates, indices, scale, *, terms=20, chunk_size=16384):
    moments = np.zeros(terms+1, dtype=np.longdouble)
    exact = [0, 0, 0]

    # add series moments instead of storing a history for every sequence
    for start in range(0, len(indices), chunk_size):
        chunk = [candidates[int(index)] for index in indices[start:start+chunk_size]]
        rates = np.array([sequence for _, sequence, _ in chunk], dtype=np.int64)
        weights = np.array([str(abs(numerator)) for _, _, numerator in chunk], dtype=np.longdouble)
        weights /= np.longdouble(str(scale))
        h = homogeneous_coefficients(-np.longdouble(".001")*rates, terms)
        moments += (weights[:, None]*h).sum(axis=0, dtype=np.longdouble)
        for _, sequence, numerator in chunk:
            rate_sum = sum(sequence)
            # keep the first three weighted moments as exact integers until the batch sum is done
            h2 = (rate_sum**2+sum(rate*rate for rate in sequence))//2
            exact[0] += abs(numerator)
            exact[1] += abs(numerator)*rate_sum
            exact[2] += abs(numerator)*h2
    for k in range(3):
        value = Fraction(exact[k], scale)*Fraction(-1, 1000)**k
        moments[k] = np.longdouble(str(value.numerator))/np.longdouble(str(value.denominator))
    return moments


def choose_subset(values, original_endpoint, direction, target_percent, strict_cap):
    possible = []

    # bit i means remove sequence i, beta=4 has nine sequences so 512 subsets
    for mask in range(1 << len(values)):
        indices = tuple(i for i in range(len(values)) if mask & (1 << i))
        # subtracting a negative sequence gives a positive endpoint change
        change = -100*mp.fsum(values[i] for i in indices)/original_endpoint
        if direction*change < 0:
            continue
        if strict_cap and abs(change) > target_percent:
            continue
        possible.append((abs(change-direction*target_percent), len(indices), indices, change))
    # nearest target first, then fewer removed sequences to break a tie
    return min(possible)


def binary_selection(values, sign, target):
    available = np.flatnonzero(np.sign(values) == sign)
    magnitudes = abs(values[available])
    assert magnitudes.sum(dtype=np.longdouble) >= target
    chosen = []
    remaining = np.longdouble(target)
    # take the largest response that still fits in the remaining target
    # each index can be removed once, its amplitude isnt adjustable
    for position in np.argsort(magnitudes)[::-1]:
        if magnitudes[position] <= remaining:
            chosen.append(int(available[position]))
            remaining -= magnitudes[position]
        if remaining < np.longdouble("2e-11"):
            break
    chosen_set = set(chosen)
    unused = np.array([int(i) for i in available if int(i) not in chosen_set], dtype=np.int64)
    # one unused response may give a closer result even if it overshoots
    if len(unused):
        extra = int(unused[np.argmin(abs(abs(values[unused]) - remaining))])
        if abs(abs(values[extra]) - remaining) < remaining:
            chosen.append(extra)
    return np.array(chosen, dtype=np.int64)


def selection_histories(problem,counts,history,percent=1.):
    times=np.asarray(history["times"])
    original=np.asarray(history["total_palinstrophy"])
    target=np.longdouble(str(original[-1]))*np.longdouble(str(percent))/100
    cases=[]
    for beta in (4,6,8,10):
        if beta>problem["order"]:
            continue
        candidates,scale,enumeration=numerator_candidates(problem,beta,counts[beta])
        assert len(candidates)==enumeration["active_ordered_sequences"]==counts[beta]
        # fix the sequence order so equal subset errors use the same tie break
        if beta==4:
            candidates.sort(key=lambda item:tuple(reversed(item[1])))
        values=endpoint_responses(candidates,scale,beta)
        expected=np.array(next(row["palinstrophy_response"] for row in history["rows"] if row["beta"]==beta))
        np.testing.assert_allclose(float(values.sum(dtype=np.longdouble)),expected[-1],atol=2e-12,rtol=0)
        for direction,sign,name in ((1,-1,"increase"),(-1,1,"decrease")):
            if beta==4:
                with mp.workdps(100):
                    precise=[mp.pi**2*mp.mpf(numerator)/scale*inverse_kernel(sequence,mp.mpf(".001"),mp.mpf(1))
                             for _,sequence,numerator in candidates]
                    _,_,chosen,_=choose_subset(precise,mp.mpf(str(original[-1])),direction,mp.mpf(str(percent)),False)
            else:
                chosen=binary_selection(values,sign,target)
            positive=np.array([int(i) for i in chosen if candidates[int(i)][2]>0],dtype=int)
            negative=np.array([int(i) for i in chosen if candidates[int(i)][2]<0],dtype=int)
            # selected_moments stores magnitudes, put the signs back before subtracting the response
            moments=selected_moments(candidates,positive,scale)-selected_moments(candidates,negative,scale)
            removed=evaluate_moments(moments[None,:],beta,times)[:,0]
            np.testing.assert_allclose(removed[-1],float(values[np.asarray(chosen,dtype=int)].sum(dtype=np.longdouble)),atol=2e-12,rtol=0)
            # each beta is removed from the original, changes arent accumulated
            altered=original-removed
            change=float(100*(altered[-1]/original[-1]-1))
            assert direction*change>0 and altered[0]==original[0]
            if beta!=4:
                assert abs(change-direction*percent)<1e-4
            cases.append(dict(beta=beta,name=name,removed_TSEs=len(chosen),endpoint_change_percent=change,
                              altered_palinstrophy=altered.tolist()))
    return cases
