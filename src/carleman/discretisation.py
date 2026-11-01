# Algorthmic structure:
#                           The spatial derivatives are put in finite difference matrices on the given grid
#                           Powers of each component is stored in the symmetric basis, repeated orderings arent stored again
#                           The product rule gives the blocks in terms of F_j, the initial products must follow the same ordering

#                           Start reading this file from: carleman_discretisation

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from scipy import sparse

from carleman.linearisation import CarlemanData


@dataclass(frozen=True)
class FiniteDifferenceGrid:
    spatial_domains: tuple[tuple[float, float], ...]
    grid_shape: tuple[int, ...]
    boundary_conditions: tuple[str, ...]
    axes: tuple[Any, ...]
    points: Any
    step_sizes: tuple[float, ...]
    first_derivatives: tuple[Any, ...]
    second_derivatives: tuple[Any, ...]


def _local_difference_matrices(
    grid_points: int,
    step_size: float,
    boundary: str,
):
    first = np.zeros((grid_points, grid_points), dtype=float)
    second = np.zeros((grid_points, grid_points), dtype=float)
    for row in range(grid_points):
        # wrap the neighbour indices only for the periodic grid
        if boundary == "Periodic":
            first[row, (row + 1) % grid_points] += 1 / (2 * step_size)
            first[row, (row - 1) % grid_points] -= 1 / (2 * step_size)
            second[row, row] -= 2 / step_size**2
            second[row, (row + 1) % grid_points] += 1 / step_size**2
            second[row, (row - 1) % grid_points] += 1 / step_size**2
        else:
            # boundary values are zero, so they dont need columns in the matrix
            if row + 1 < grid_points:
                first[row, row + 1] = 1 / (2 * step_size)
                second[row, row + 1] = 1 / step_size**2
            if row - 1 >= 0:
                first[row, row - 1] = -1 / (2 * step_size)
                second[row, row - 1] = 1 / step_size**2
            second[row, row] = -2 / step_size**2
    return sparse.csr_matrix(first), sparse.csr_matrix(second)


def finite_difference_operators(
    grid_shape: int | Iterable[int],
    spatial_domains,
    *,
    boundary: str | Iterable[str] = "HomogeneousDirichlet",
) -> FiniteDifferenceGrid:
    if (
        isinstance(spatial_domains, (tuple, list))
        and len(spatial_domains) == 2
        and all(np.isscalar(value) for value in spatial_domains)
    ):
        domains = (tuple(float(value) for value in spatial_domains),)
    else:
        domains = tuple(
            tuple(float(value) for value in domain)
            for domain in spatial_domains
        )

    if isinstance(grid_shape, int):
        shape = (grid_shape,)
    else:
        shape = tuple(int(size) for size in grid_shape)
    if len(shape) != len(domains):
        raise ValueError("grid_shape and spatial_domains must have the same dimension")

    if isinstance(boundary, str):
        # same boundary condition on all axes unless a list was given
        boundaries = (boundary,) * len(shape)
    else:
        boundaries = tuple(boundary)
    boundaries = tuple(
        "HomogeneousDirichlet" if value == "Dirichlet" else value
        for value in boundaries
    )
    if len(boundaries) != len(shape) or any(
        value not in {"Periodic", "HomogeneousDirichlet"}
        for value in boundaries
    ):
        raise ValueError(
            "boundary must specify Periodic or HomogeneousDirichlet per dimension"
        )

    axes = []
    step_sizes = []
    local_first = []
    local_second = []

    # periodic: dont repeat the endpoint, Dirichlet: only store the interior
    for (left, right), size, condition in zip(domains, shape, boundaries):
        if condition == "Periodic":
            step = (right - left) / size
            axis = np.linspace(left, right - step, size)
        else:
            # size is the interior count here, there are size+1 intervals
            step = (right - left) / (size + 1)
            axis = np.linspace(left + step, right - step, size)
        first, second = _local_difference_matrices(size, step, condition)
        axes.append(axis)
        step_sizes.append(step)
        local_first.append(first)
        local_second.append(second)

    identities = [sparse.eye(size, format="csr") for size in shape]

    def kronecker_slot(local_operator, slot):
        # the last grid coordinate changes fastest when flattened
        # so in 2D use D_x tensor I_y for x, and I_x tensor D_y for y
        # keep the factors in this order, equal matrix sizes can hide a swapped axis
        factors = [
            local_operator if position == slot else identities[position]
            for position in range(len(shape))
        ]
        result = factors[0]
        for factor in factors[1:]:
            result = sparse.kron(result, factor, format="csr")
        return result.tocsr()

    first_derivatives = tuple(
        kronecker_slot(operator, slot)
        for slot, operator in enumerate(local_first)
    )
    second_derivatives = tuple(
        kronecker_slot(operator, slot)
        for slot, operator in enumerate(local_second)
    )
    meshes = np.meshgrid(*axes, indexing="ij")
    # flatten in C order, the last spatial coordinate changes fastest
    points = np.column_stack([mesh.reshape(-1) for mesh in meshes])
    return FiniteDifferenceGrid(
        spatial_domains=domains,
        grid_shape=shape,
        boundary_conditions=boundaries,
        axes=tuple(axes),
        points=points,
        step_sizes=tuple(step_sizes),
        first_derivatives=first_derivatives,
        second_derivatives=second_derivatives,
    )


@lru_cache(maxsize=None)
def symmetric_multi_indices(dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    if dimension < 1 or degree < 0:
        raise ValueError("dimension must be positive and degree nonnegative")
    if dimension == 1:
        return ((degree,),)
    # choose the first exponent, then distribute the remaining degree
    return tuple(
        (first,) + tail
        for first in range(degree, -1, -1)
        for tail in symmetric_multi_indices(dimension - 1, degree - first)
    )


def _set_sparse_entries(shape, entries):
    if not entries:
        return sparse.csr_matrix(shape)
    rows, columns, values = zip(*entries)
    # different derivative terms can hit the same entry, CSR adds them
    return sparse.coo_matrix((values, (rows, columns)), shape=shape).tocsr()


def _symmetric_blocks(data: CarlemanData) -> list[list[Any]]:
    dimension = data.state_dimension
    # alpha=(2,1,0) means x0**2*x1, stored once
    # there are no factorial weights in this basis
    # the 2 from differentiating x0**2 still belongs in the matrix
    basis = {
        level: symmetric_multi_indices(dimension, level)
        for level in range(1, data.order + 1)
    }
    indices = {
        level: {alpha: position for position, alpha in enumerate(basis[level])}
        for level in basis
    }
    blocks: list[list[Any]] = []

    for target in range(1, data.order + 1):
        row_blocks = []
        for source in range(1, data.order + 1):
            entries: list[tuple[int, int, float]] = []
            target_basis = basis[target]
            source_index = indices[source]

            # target is the product being differentiated, source is the product it reads
            # replacing one factor by F0, F1 or F2 gives degree target-1, target or target+1
            # for example d_t(x_i**2) reads 2*x_i*F2_i(x,x) from degree three
            # this puts the quadratic block above the diagonal
            # with F0 the source degree is one lower, degree zero is outside these blocks
            if source == target - 1:
                for row, alpha in enumerate(target_basis):
                    for i, exponent in enumerate(alpha):
                        if exponent:
                            beta = list(alpha)
                            beta[i] -= 1
                            column = source_index.get(tuple(beta))
                            if column is not None and data.forcing[i] != 0:
                                entries.append((row, column, exponent * data.forcing[i]))
            # with F1 replace one factor by one factor
            elif source == target:
                for row, alpha in enumerate(target_basis):
                    for i, exponent in enumerate(alpha):
                        if not exponent:
                            continue
                        for j in range(dimension):
                            # the derivative of x_i**alpha_i brings down alpha_i
                            value = exponent * data.linear[i, j]
                            if value == 0:
                                continue
                            beta = list(alpha)
                            beta[i] -= 1
                            beta[j] += 1
                            column = source_index.get(tuple(beta))
                            if column is not None:
                                entries.append((row, column, value))
            # with F2 replace one factor by a pair
            elif source == target + 1:
                for row, alpha in enumerate(target_basis):
                    for i, exponent in enumerate(alpha):
                        if not exponent:
                            continue
                        for j in range(dimension):
                            for ell in range(dimension):
                                value = exponent * data.quadratic[i, j, ell]
                                if value == 0:
                                    continue
                                beta = list(alpha)
                                beta[i] -= 1
                                beta[j] += 1
                                # if j==ell this correctly adds the same factor twice
                                beta[ell] += 1
                                column = source_index.get(tuple(beta))
                                if column is not None:
                                    entries.append((row, column, value))
            row_blocks.append(
                _set_sparse_entries(
                    (len(target_basis), len(basis[source])), entries
                )
            )
        blocks.append(row_blocks)
    return blocks


def _symmetric_state(base_state, data: CarlemanData):
    values = []

    # keep the same monomial order as the matrix columns
    for level in range(1, data.order + 1):
        for alpha in symmetric_multi_indices(data.state_dimension, level):
            value = 1.0
            for component, exponent in zip(base_state, alpha):
                # zero exponent contributes 1, including when the component itself is zero
                value *= 1.0 if exponent == 0 else component**exponent
            values.append(value)
    return np.asarray(values, dtype=float)


@dataclass
class DiscretisationData:
    blocks: list[list[Any]]
    initial_state: Any

    def assemble_matrix(self):
        return sparse.bmat(self.blocks, format="csr")


def carleman_discretisation(
    data: CarlemanData,
    initial_state,
    *,
    max_explicit_dimension: int = 250_000,
) -> DiscretisationData:
    # check the lifted size before allocating all the blocks
    if data.lifted_dimension_symmetric > max_explicit_dimension:
        raise MemoryError(
            f"lifted dimension {data.lifted_dimension_symmetric:,}; "
            f"limit {max_explicit_dimension:,}, reduce the order or grid size"
        )
    blocks = _symmetric_blocks(data)
    return DiscretisationData(
        blocks=blocks,
        initial_state=_symmetric_state(initial_state, data),
    )
