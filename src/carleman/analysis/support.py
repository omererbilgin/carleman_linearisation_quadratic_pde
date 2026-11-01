# Algorthmic structure:
#                           The nonzero initial monomials is used as the starting support
#                           Merging Fourier pairs gives lower degrees that can be reached, these also needs to be kept
#                           The supported matrix keeps all wavevectors, the slow projection only keeps total wavevector zero

#                           Start reading this file from: supported_vorticity_matrix

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np
from scipy.sparse import coo_matrix

from carleman.problems.fourier_vorticity import (
    article_initial_coefficients, fourier_vorticity_operators,
)


def supported_slow_projection(order=8, radius=3):
    operators = fourier_vorticity_operators(radius, 1.)
    initial = article_initial_coefficients(operators)
    modes = operators.modes
    # support of the article initial condition, not of an arbitrary input field
    seeds = np.flatnonzero(initial)
    quadratic = {}
    entries = operators.F2_symmetric.tocoo()
    for row, col, value in zip(entries.row, entries.col, entries.data):
        pair = operators.symmetric_pairs[col]
        quadratic.setdefault(pair, []).append(
            (int(row), Fraction(float(value)).limit_denominator(1000000))
        )

    # only zero total wavevector survives the spatial integral
    nodes = set()
    for degree in range(2, order + 1):
        for monomial in combinations_with_replacement(seeds, degree):
            if not np.any(np.sum(modes[list(monomial)], axis=0)):
                nodes.add(monomial)
    frontier = list(nodes)
    columns = {}

    # follow the new monomials too, they can feed lower degrees
    while frontier:
        monomial = frontier.pop()
        targets = {}
        if len(monomial) > 2:
            for a, b in combinations_with_replacement(sorted(set(monomial)), 2):
                # cant take the same factor twice if it appears only once
                if a == b and monomial.count(a) < 2:
                    continue
                for replacement, value in quadratic.get((a, b), []):
                    remainder = list(monomial)
                    remainder.remove(a)
                    remainder.remove(b)
                    # sorting puts the new product back into the symmetric basis
                    target = tuple(sorted(remainder + [replacement]))
                    # the replacement may occur more than once in the differentiated target
                    coefficient = value * target.count(replacement)
                    targets[target] = targets.get(target, Fraction(0)) + coefficient
                    if target not in nodes:
                        nodes.add(target)
                        frontier.append(target)
        columns[monomial] = targets

    nodes = sorted(nodes, key=lambda m: (len(m), m))
    # use the same sorted order for the transfers, rates and readout row
    positions = {m: j for j, m in enumerate(nodes)}
    decay = [int(np.sum(modes[list(m)]**2)) for m in nodes]
    left = []

    # use the left eigenvector relation at eigenvalue -2
    # sum_target coupling*l_target = (decay_source-2)*l_source
    # lower degrees come first, their weights are already available here
    # the degree-two weights start the recursion, this only constructs the slow readout
    for j, monomial in enumerate(nodes):
        if len(monomial) == 2:
            left.append(Fraction(4) if decay[j] == 2 else Fraction(0))
        else:
            incoming = sum((value * left[positions[target]]
                            for target, value in columns[monomial].items()), Fraction(0))
            left.append(incoming / (decay[j] - 2))

    # apply the initial products to the readout, keeping the source degrees separate
    amplitudes = {degree: Fraction(0) for degree in range(2, order + 1)}
    for monomial, weight in zip(nodes, left):
        value, phase = Fraction(1), 0
        for index in monomial:
            if initial[index].imag:
                value *= Fraction(float(initial[index].imag)).limit_denominator()
                # each imaginary initial coefficient contributes one power of i
                phase += 1
            else:
                value *= Fraction(float(initial[index].real)).limit_denominator()
        if not value:
            continue
        # zero total wavevector should leave a real contribution here
        assert phase % 2 == 0
        amplitudes[len(monomial)] += weight * value * (-1)**(phase // 2)
    return operators, nodes, columns, left, decay, amplitudes


def supported_vorticity_matrix(order=10, radius=30, epsilon=Fraction(2), viscosity=.001):
    if order < 2 or radius < 2:
        raise ValueError("order and radius must be at least two")
    epsilon = Fraction(epsilon)
    width = 2 * radius + 1

    def encode(x, y):
        return (x + radius) * width + y + radius

    @lru_cache(None)
    def mode(index):
        return index // width - radius, index % width - radius

    @lru_cache(None)
    def rate(index):
        x, y = mode(index)
        return x*x + y*y

    @lru_cache(None)
    def merge(first, second):
        px, py = mode(first)
        qx, qy = mode(second)
        p2, q2 = rate(first), rate(second)
        cross = py*qx - px*qy
        # equal lengths and parallel wavevectors give no quadratic transfer
        if not cross or not p2 or not q2 or p2 == q2:
            return None
        x, y = px + qx, py + qy
        if abs(x) > radius or abs(y) > radius:
            return None
        # both ordered Fourier terms belong to this one symmetric pair
        return encode(x, y), Fraction(cross * (q2-p2), p2*q2)

    amplitudes = {
        encode(1, -1): (Fraction(1, 4), 0),
        encode(-1, 1): (Fraction(1, 4), 0),
        encode(1, 1): (Fraction(-1, 4), 0),
        encode(-1, -1): (Fraction(-1, 4), 0),
        encode(2, 1): (-epsilon/2, 1),
        encode(-2, -1): (epsilon/2, 1),
    }
    seeds = sorted(index for index, (value, _) in amplitudes.items() if value)
    # keep all wavevectors here, vorticity needs more than the integrated sector
    nodes = {monomial for degree in range(1, order+1)
             for monomial in combinations_with_replacement(seeds, degree)}
    # a zero initial coordinate can still receive a nonzero product
    # merge p,q into p+q and check the new monomial for further transfers too
    # stop when there are no new coordinates, diffusion only changes the diagonal
    # changing the nonzero initial modes needs this closure again
    frontier = list(nodes)
    transfers = {}
    while frontier:
        monomial = frontier.pop()
        targets = defaultdict(Fraction)
        if len(monomial) > 1:
            for first, second in combinations_with_replacement(sorted(set(monomial)), 2):
                if first == second and monomial.count(first) < 2:
                    continue
                replacement = merge(first, second)
                if replacement is None:
                    continue
                index, coupling = replacement
                remainder = list(monomial)
                remainder.remove(first)
                remainder.remove(second)
                target = tuple(sorted(remainder + [index]))
                # count the replacement in the target product
                # for a*b*c with a*b -> c, the target is c*c, its derivative gives two equal terms
                # counting the removed pair in the source would miss this 2
                targets[target] += coupling * target.count(index)
                if target not in nodes:
                    nodes.add(target)
                    frontier.append(target)
        transfers[monomial] = {target: value for target, value in targets.items() if value}

    monomials = sorted(nodes, key=lambda item: (len(item), item))
    positions = {item: i for i, item in enumerate(monomials)}
    degrees = np.array([len(item) for item in monomials])
    rates = np.array([sum(rate(index) for index in item) for item in monomials])
    rows, columns = list(range(len(monomials))), list(range(len(monomials)))
    values = (-viscosity * rates).tolist()
    initial = []
    pairs = []
    for source, monomial in enumerate(monomials):
        for target, coupling in transfers.pop(monomial).items():
            rows.append(positions[target])
            columns.append(source)
            values.append(float(coupling))
        value, exponent = Fraction(1), 0
        for index in monomial:
            amplitude, phase = amplitudes.get(index, (Fraction(0), 0))
            value *= amplitude
            exponent += phase
            if not value:
                break
        # each sine coefficient brings a power of i into the product
        # store (-1)**(exponent//2) here, an odd exponent still leaves one i
        # for this initial condition the total wavevector parity fixes that remaining phase
        # put it back when reading vorticity, the zero-wavevector pairs are already real
        initial.append(float(value * (-1)**(exponent//2)))
        if len(monomial) == 2:
            p, q = map(mode, monomial)
            if p[0]+q[0] == 0 and p[1]+q[1] == 0:
                pairs.append(source)

    matrix = coo_matrix((values, (rows, columns)), shape=(len(monomials),)*2).tocsr()
    matrix.eliminate_zeros()
    vort = np.flatnonzero(degrees == 1)
    modes = np.array([mode(monomials[index][0]) for index in vort])
    pairs = np.asarray(pairs, dtype=int)
    return {
        "matrix": matrix, "initial": np.asarray(initial), "rates": rates, "degrees": degrees,
        "vorticity_indices": vort, "palinstrophy_indices": pairs,
        # each (k,-k) product occurs once, hence twice the ordered-pair weight
        "palinstrophy_weights": 2*np.pi**2 * rates[pairs],
        "modes": modes, "phase": np.where(np.sum(modes, axis=1) % 2, 1j, 1.),
    }
