# Algorthmic structure:
#                           The retained Fourier lattice gives the diffusion and quadratic advection operators F_j
#                           Ordered pairs and symmetric pairs has different column ordering, both gives the same quadratic action
#                           The Carleman blocks are built degree by degree, the vorticity is read from the first level after evolving

#                           Start reading this file from: fourier_vorticity_operators, then fourier_vorticity_carleman

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb
from typing import Iterable, Literal, Mapping

import numpy as np
from scipy import sparse


BasisName = Literal["symmetric", "ordered"]


def _clean_sparse(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    result = matrix.tocsr()
    # different paths can land on the same matrix entry
    result.sum_duplicates()
    # remove entries that cancelled, they shouldnt become support connections
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _symmetric_level_basis(
    dimension: int,
    degree: int,
) -> tuple[tuple[int, ...], ...]:
    # raw products, one sorted factor tuple for each monomial
    return tuple(combinations_with_replacement(range(dimension), degree))


def _sparse_entries_by_row(matrix: sparse.spmatrix):
    matrix = matrix.tocsr()
    entries = []
    for row in range(matrix.shape[0]):
        # these two offsets delimit the row in CSR storage
        start, stop = matrix.indptr[row : row + 2]
        entries.append(
            tuple(
                (int(column), float(value))
                for column, value in zip(
                    matrix.indices[start:stop],
                    matrix.data[start:stop],
                )
                if value != 0
            )
        )
    return tuple(entries)


def _compress_ordered_quadratic(
    ordered: sparse.spmatrix,
    dimension: int,
) -> tuple[sparse.csr_matrix, tuple[tuple[int, int], ...]]:
    pairs = tuple(combinations_with_replacement(range(dimension), 2))
    pair_index = {pair: position for position, pair in enumerate(pairs)}
    coordinates = ordered.tocoo()
    rows = []
    columns = []
    values = []

    # w_p*w_q and w_q*w_p share one symmetric column
    # add their coefficients, averaging would lose half the contribution
    # for p=q there is only one ordered column, dont double it
    for row, ordered_column, value in zip(
        coordinates.row,
        coordinates.col,
        coordinates.data,
    ):
        # the right factor changes fastest in the ordered Kronecker columns
        left, right = divmod(int(ordered_column), dimension)
        columns.append(pair_index[tuple(sorted((left, right)))])
        rows.append(int(row))
        values.append(float(value))
    compressed = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(dimension, len(pairs)),
        dtype=float,
    )
    return _clean_sparse(compressed), pairs


@dataclass(frozen=True)
class FourierVorticityOperators:
    radius: int
    viscosity: float
    modes: np.ndarray
    F0: np.ndarray
    F1: sparse.csr_matrix
    F2: sparse.csr_matrix
    F2_symmetric: sparse.csr_matrix
    symmetric_pairs: tuple[tuple[int, int], ...]

    @property
    def state_dimension(self) -> int:
        return int(self.modes.shape[0])

    @property
    def F2_ordered(self) -> sparse.csr_matrix:
        return self.F2

    @property
    def mode_to_index(self) -> dict[tuple[int, int], int]:
        return {
            (int(mode[0]), int(mode[1])): index
            for index, mode in enumerate(self.modes)
        }

    def mode_index(self, mode: Iterable[int]) -> int:
        key = tuple(int(component) for component in mode)
        return self.mode_to_index[key]

    def ordered_quadratic_tensor(self) -> np.ndarray:
        dimension = self.state_dimension
        # keep the output index first, then the two input factors
        return self.F2.toarray().reshape(dimension, dimension, dimension)

    def symmetric_level_two(self, coefficients) -> np.ndarray:
        coefficients = np.asarray(coefficients)
        # use the same pair order as F2_symmetric
        return np.asarray(
            [
                coefficients[left] * coefficients[right]
                for left, right in self.symmetric_pairs
            ],
            dtype=np.result_type(coefficients.dtype, float),
        )

    def rhs(self, coefficients) -> np.ndarray:
        coefficients = np.asarray(coefficients)
        # evaluate the nonlinear modal ODE, before any Carleman truncation
        ordered_level_two = np.kron(coefficients, coefficients)
        return np.asarray(
            self.F0 + self.F1 @ coefficients + self.F2 @ ordered_level_two
        )

    def coefficient_vector(
        self,
        coefficients: Mapping[tuple[int, int], complex],
    ) -> np.ndarray:
        result = np.zeros(self.state_dimension, dtype=complex)
        indices = self.mode_to_index
        for mode, value in coefficients.items():
            key = tuple(int(component) for component in mode)
            result[indices[key]] = value
        return result


@dataclass(frozen=True)
class FourierVorticityCarleman:
    operators: FourierVorticityOperators
    order: int
    basis: BasisName
    level_dimensions: tuple[int, ...]
    blocks: tuple[tuple[sparse.csr_matrix, ...], ...]
    carleman_matrix: sparse.csr_matrix
    lifted_forcing: np.ndarray

    @property
    def modes(self) -> np.ndarray:
        return self.operators.modes

    def lift(self, coefficients) -> np.ndarray:
        coefficients = np.asarray(coefficients)
        dimension = self.operators.state_dimension
        levels = []
        # ordered keeps every factor ordering, symmetric keeps each product once
        if self.basis == "ordered":
            level = coefficients.copy()
            for degree in range(1, self.order + 1):
                if degree > 1:
                    # append one more physical factor to all the products from the previous level
                    level = np.kron(level, coefficients)
                levels.append(level)
        else:
            dtype = np.result_type(coefficients.dtype, float)
            for degree in range(1, self.order + 1):
                values = []
                for factors in _symmetric_level_basis(dimension, degree):
                    value = np.array(1, dtype=dtype)
                    for factor in factors:
                        # raw product, no square-root or factorial weights
                        value = value * coefficients[factor]
                    values.append(value)
                levels.append(np.asarray(values, dtype=dtype))
        return np.concatenate(levels)


@dataclass(frozen=True)
class FourierVorticityPDEReference:
    times: np.ndarray
    modes: np.ndarray
    coefficients: np.ndarray
    grid_size: int
    dealias_cutoff: int
    function_evaluations: int


def article_initial_coefficients(
    operators: FourierVorticityOperators,
) -> np.ndarray:
    return operators.coefficient_vector(
        {
            (1, -1): 1 / 4,
            (-1, 1): 1 / 4,
            (1, 1): -1 / 4,
            (-1, -1): -1 / 4,
            (2, 1): -3j / 20,
            (-2, -1): 3j / 20,
        }
    )


def embed_fourier_coefficients(
    source_modes,
    coefficients,
    target_modes,
) -> np.ndarray:
    source_modes = np.asarray(source_modes, dtype=int)
    target_modes = np.asarray(target_modes, dtype=int)
    coefficients = np.asarray(coefficients)
    target_indices = {
        (int(mode[0]), int(mode[1])): index
        for index, mode in enumerate(target_modes)
    }
    # modes absent from the smaller lattice start with zero coefficients
    result = np.zeros(
        coefficients.shape[:-1] + (target_modes.shape[0],),
        dtype=np.result_type(coefficients.dtype, complex),
    )
    for source_index, mode in enumerate(source_modes):
        key = (int(mode[0]), int(mode[1]))
        target_index = target_indices[key]
        result[..., target_index] = coefficients[..., source_index]
    return result


def _validated_times(times: Iterable[float]) -> np.ndarray:
    result = np.asarray(tuple(float(time) for time in times), dtype=float)
    if result[0] != 0:
        raise ValueError("times must begin at zero")
    return result


def _evolve_carleman_coefficients(
    system: FourierVorticityCarleman,
    initial_coefficients: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    from scipy.sparse.linalg import expm_multiply

    dimension = system.operators.state_dimension
    if np.any(system.lifted_forcing != 0):
        raise NotImplementedError(
            "affine lifted evolution is not implemented for nonzero F0"
        )
    if times.size == 1:
        return initial_coefficients[None, :].copy()

    lifted_initial = system.lift(initial_coefficients)
    matrix_trace = float(system.carleman_matrix.diagonal().sum())
    intervals = np.diff(times)
    uniform_times = np.allclose(
        intervals,
        intervals[0],
        rtol=1e-12,
        atol=1e-15,
    )
    # each snapshot is exp(t*C)*z0 from the same initial state
    # equal time steps lets expm_multiply reuse work across the interval
    # for irregular times evaluate each t separately, the vector still holds all levels
    if uniform_times:
        lifted_states = expm_multiply(
            system.carleman_matrix,
            lifted_initial,
            start=0.0,
            stop=float(times[-1]),
            num=times.size,
            endpoint=True,
            traceA=matrix_trace,
        )
    else:
        # for irregular times, evaluate each snapshot from the same initial state
        lifted_states = np.stack(
            [
                expm_multiply(
                    time * system.carleman_matrix,
                    lifted_initial,
                    traceA=time * matrix_trace,
                )
                for time in times
            ]
        )
    # only level 1 is needed to recover the physical vorticity
    return np.asarray(lifted_states[:, :dimension])


def solve_vorticity_pde_reference(
    initial_modes,
    initial_coefficients,
    times: Iterable[float],
    *,
    viscosity: float = 1 / 50,
    grid_size: int = 48,
    relative_tolerance: float = 1e-11,
    absolute_tolerance: float = 1e-13,
) -> FourierVorticityPDEReference:
    from scipy.integrate import solve_ivp

    times = _validated_times(times)
    viscosity = float(viscosity)
    grid_size = int(grid_size)

    initial_modes = np.asarray(initial_modes)
    initial_coefficients = np.asarray(initial_coefficients, dtype=complex)
    if not np.issubdtype(initial_modes.dtype, np.integer):
        if not np.all(initial_modes == np.rint(initial_modes)):
            raise ValueError("initial_modes must contain integer wavevectors")
    initial_modes = np.asarray(initial_modes, dtype=int)

    # this PDE reference uses a larger FFT grid, separate from the retained Carleman lattice
    dealias_cutoff = (grid_size - 1) // 3
    if np.max(np.abs(initial_modes), initial=0) > dealias_cutoff:
        raise ValueError(
            "grid_size is too small for the supplied initial Fourier modes"
        )
    wave_numbers = np.fft.fftfreq(grid_size, d=1 / grid_size)
    first_wave_number = wave_numbers[:, None]
    second_wave_number = wave_numbers[None, :]
    squared_wave_number = (
        first_wave_number * first_wave_number
        + second_wave_number * second_wave_number
    )
    inverse_squared_wave_number = np.zeros_like(squared_wave_number)
    nonzero = squared_wave_number != 0
    # zero mode stays zero, the stream function is chosen to have zero mean
    inverse_squared_wave_number[nonzero] = 1 / squared_wave_number[nonzero]
    dealias_mask = (
        (np.abs(first_wave_number) <= dealias_cutoff)
        & (np.abs(second_wave_number) <= dealias_cutoff)
    )

    initial_spectrum = np.zeros((grid_size, grid_size), dtype=complex)
    # negative wavevectors use the wrapped FFT indices
    for mode, value in zip(initial_modes, initial_coefficients):
        initial_spectrum[
            int(mode[0]) % grid_size,
            int(mode[1]) % grid_size,
        ] += value
    initial_spectrum[~dealias_mask] = 0
    # stored values are Fourier-series coefficients, undo NumPy ifft2 normalisation
    transform_scale = grid_size * grid_size

    def pde_right_hand_side(_time, flat_spectrum):
        vorticity_spectrum = flat_spectrum.reshape(grid_size, grid_size)
        # solve -Delta psi=omega mode by mode
        streamfunction_spectrum = (
            inverse_squared_wave_number * vorticity_spectrum
        )
        # velocity is (psi_y,-psi_x), so take derivatives before the inverse FFT
        x_velocity = np.fft.ifft2(
            1j
            * second_wave_number
            * streamfunction_spectrum
            * transform_scale
        )
        y_velocity = np.fft.ifft2(
            -1j
            * first_wave_number
            * streamfunction_spectrum
            * transform_scale
        )
        x_derivative = np.fft.ifft2(
            1j * first_wave_number * vorticity_spectrum * transform_scale
        )
        y_derivative = np.fft.ifft2(
            1j * second_wave_number * vorticity_spectrum * transform_scale
        )

        # multiply in physical space, then remove the modes above the 2/3 cutoff
        advection_spectrum = np.fft.fft2(
            -(x_velocity * x_derivative + y_velocity * y_derivative)
        ) / transform_scale
        advection_spectrum[~dealias_mask] = 0
        derivative = (
            -viscosity * squared_wave_number * vorticity_spectrum
            + advection_spectrum
        )
        derivative[~dealias_mask] = 0
        # advection and diffusion should not change the spatial mean
        derivative[0, 0] = 0
        return derivative.reshape(-1)

    if times.size == 1:
        full_coefficients = initial_spectrum[None, :, :]
        function_evaluations = 0
    else:
        solution = solve_ivp(
            pde_right_hand_side,
            (0.0, float(times[-1])),
            initial_spectrum.reshape(-1),
            method="DOP853",
            t_eval=times,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        if not solution.success:
            raise RuntimeError(
                "continuous vorticity PDE reference integration failed: "
                f"{solution.message}"
            )
        full_coefficients = np.asarray(solution.y.T).reshape(
            times.size,
            grid_size,
            grid_size,
        )
        function_evaluations = int(solution.nfev)

    modes = np.asarray(
        [
            (first, second)
            for first in range(-dealias_cutoff, dealias_cutoff + 1)
            for second in range(-dealias_cutoff, dealias_cutoff + 1)
        ],
        dtype=int,
    )
    # put the FFT output back in lexicographic wavevector order
    coefficients = np.stack(
        [
            full_coefficients[
                :,
                int(mode[0]) % grid_size,
                int(mode[1]) % grid_size,
            ]
            for mode in modes
        ],
        axis=-1,
    )
    return FourierVorticityPDEReference(
        times=times,
        modes=modes,
        coefficients=coefficients,
        grid_size=grid_size,
        dealias_cutoff=dealias_cutoff,
        function_evaluations=function_evaluations,
    )


def fourier_vorticity_operators(
    radius: int = 3,
    viscosity: float = 1 / 50,
) -> FourierVorticityOperators:
    radius = int(radius)
    viscosity = float(viscosity)
    modes = np.asarray(
        [
            (first, second)
            for first in range(-radius, radius + 1)
            for second in range(-radius, radius + 1)
        ],
        dtype=int,
    )
    dimension = int(modes.shape[0])
    mode_to_index = {
        (int(mode[0]), int(mode[1])): index
        for index, mode in enumerate(modes)
    }

    # diffusion is diagonal in Fourier space, with rate -nu*|k|**2
    squared_norms = np.einsum("ij,ij->i", modes, modes)
    F1 = sparse.diags(
        -viscosity * squared_norms,
        offsets=0,
        shape=(dimension, dimension),
        format="csr",
        dtype=float,
    )
    F1.eliminate_zeros()

    rows = []
    columns = []
    values = []

    # a pair p,q only feeds ell=p+q, so ell and p determines q
    # the velocity gives 1/|p|**2 and the derivatives give the cross product
    # store (q,p) at q_index*dimension+p_index to match the Kronecker state
    # discard q outside the lattice, wrapping would put its contribution in a different mode
    for output_index, output_mode in enumerate(modes):
        ell = (int(output_mode[0]), int(output_mode[1]))
        for p_index, p_mode in enumerate(modes):
            p1, p2 = int(p_mode[0]), int(p_mode[1])
            denominator = p1 * p1 + p2 * p2
            # the mean mode has no Biot-Savart velocity
            if denominator == 0:
                continue
            q = (ell[0] - p1, ell[1] - p2)
            # no modular wrapping here, this is a retained-mode projection
            q_index = mode_to_index.get(q)
            if q_index is None:
                continue
            numerator = p2 * q[0] - p1 * q[1]
            if numerator == 0:
                continue
            rows.append(output_index)
            # column order is (q,p), with p changing fastest
            columns.append(q_index * dimension + p_index)
            values.append(numerator / denominator)

    F2 = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(dimension, dimension * dimension),
        dtype=float,
    )
    F2 = _clean_sparse(F2)
    F2_symmetric, pairs = _compress_ordered_quadratic(F2, dimension)
    return FourierVorticityOperators(
        radius=radius,
        viscosity=viscosity,
        modes=modes,
        F0=np.zeros(dimension, dtype=float),
        F1=F1,
        F2=F2,
        F2_symmetric=F2_symmetric,
        symmetric_pairs=pairs,
    )


def _kron_three(left, middle, right) -> sparse.csr_matrix:
    return sparse.kron(
        sparse.kron(left, middle, format="csr"),
        right,
        format="csr",
    )


def _ordered_carleman_blocks(
    operators: FourierVorticityOperators,
    order: int,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[sparse.csr_matrix, ...], ...],
]:
    dimension = operators.state_dimension
    level_dimensions = tuple(dimension**degree for degree in range(1, order + 1))
    identities = {
        power: sparse.eye(dimension**power, format="csr", dtype=float)
        for power in range(order + 1)
    }
    forcing_column = sparse.csr_matrix(operators.F0.reshape(-1, 1))
    block_rows = []
    # differentiate one slot and leave identities on the other factors
    # at degree two F1 gives F1 tensor I + I tensor F1
    # use the same slots for F2, with one extra source factor
    # these products keep their order, collecting equal products happens in the symmetric path
    for target_degree in range(1, order + 1):
        row = []
        for source_degree in range(1, order + 1):
            shape = (
                level_dimensions[target_degree - 1],
                level_dimensions[source_degree - 1],
            )
            block = sparse.csr_matrix(shape, dtype=float)
            # insert F0 at the differentiated slot, identities on the remaining factors
            if source_degree == target_degree - 1:
                for slot in range(target_degree):
                    block += _kron_three(
                        identities[slot],
                        forcing_column,
                        identities[target_degree - slot - 1],
                    )
            # with F1 keep the degree, with F2 read from one degree higher
            elif source_degree == target_degree:
                for slot in range(target_degree):
                    block += _kron_three(
                        identities[slot],
                        operators.F1,
                        identities[target_degree - slot - 1],
                    )
            elif source_degree == target_degree + 1:
                for slot in range(target_degree):
                    block += _kron_three(
                        identities[slot],
                        operators.F2,
                        identities[target_degree - slot - 1],
                    )
            row.append(_clean_sparse(block))
        block_rows.append(tuple(row))
    return level_dimensions, tuple(block_rows)


def _symmetric_carleman_blocks(
    operators: FourierVorticityOperators,
    order: int,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[sparse.csr_matrix, ...], ...],
]:
    dimension = operators.state_dimension
    bases = {
        degree: _symmetric_level_basis(dimension, degree)
        for degree in range(1, order + 1)
    }
    indices = {
        degree: {monomial: index for index, monomial in enumerate(basis)}
        for degree, basis in bases.items()
    }
    level_dimensions = tuple(len(bases[degree]) for degree in range(1, order + 1))
    # only visit nonzero replacements in F1 and F2
    linear_entries = _sparse_entries_by_row(operators.F1)
    quadratic_entries = _sparse_entries_by_row(operators.F2)
    block_rows = []

    for target_degree in range(1, order + 1):
        row_blocks = []
        target_basis = bases[target_degree]
        for source_degree in range(1, order + 1):
            rows = []
            columns = []
            values = []
            if abs(source_degree - target_degree) <= 1:
                source_index = indices[source_degree]
                for row, factors in enumerate(target_basis):
                    multiplicities = {
                        factor: factors.count(factor)
                        for factor in set(factors)
                    }

                    # for (a,a,b), differentiating either a gives the same product
                    # remove a once and multiply by 2, dont remove both copies
                    # the F2 columns are still ordered, sparse assembly adds equal resulting products
                    for differentiated, multiplicity in multiplicities.items():
                        remainder = list(factors)
                        remainder.remove(differentiated)
                        if source_degree == target_degree - 1:
                            value = multiplicity * operators.F0[differentiated]
                            column = source_index.get(tuple(remainder))
                            if value != 0 and column is not None:
                                rows.append(row)
                                columns.append(column)
                                values.append(value)
                        elif source_degree == target_degree:
                            for replacement, coefficient in linear_entries[
                                differentiated
                            ]:
                                monomial = tuple(sorted(remainder + [replacement]))
                                column = source_index.get(monomial)
                                if column is not None:
                                    rows.append(row)
                                    columns.append(column)
                                    values.append(multiplicity * coefficient)
                        elif source_degree == target_degree + 1:
                            for ordered_column, coefficient in quadratic_entries[
                                differentiated
                            ]:
                                # split the ordered F2 column into the two replacement factors
                                left, right = divmod(ordered_column, dimension)
                                monomial = tuple(sorted(remainder + [left, right]))
                                column = source_index.get(monomial)
                                if column is not None:
                                    rows.append(row)
                                    columns.append(column)
                                    values.append(multiplicity * coefficient)
            block = sparse.coo_matrix(
                (values, (rows, columns)),
                shape=(len(target_basis), len(bases[source_degree])),
                dtype=float,
            )
            row_blocks.append(_clean_sparse(block))
        block_rows.append(tuple(row_blocks))
    return level_dimensions, tuple(block_rows)


def fourier_vorticity_carleman(
    radius: int = 3,
    viscosity: float = 1 / 50,
    order: int = 3,
    *,
    basis: BasisName = "symmetric",
    max_explicit_dimension: int = 250_000,
) -> FourierVorticityCarleman:
    order = int(order)
    if basis not in {"symmetric", "ordered"}:
        raise ValueError("basis must be 'symmetric' or 'ordered'")
    operators = fourier_vorticity_operators(radius, viscosity)
    dimension = operators.state_dimension
    if basis == "symmetric":
        anticipated_dimension = sum(
            comb(dimension + degree - 1, degree)
            for degree in range(1, order + 1)
        )
    else:
        anticipated_dimension = sum(
            dimension**degree for degree in range(1, order + 1)
        )
    if anticipated_dimension > max_explicit_dimension:
        raise MemoryError(
            f"Refusing explicit {basis} Carleman dimension "
            f"{anticipated_dimension:,}; limit {max_explicit_dimension:,}. "
            "Increase max_explicit_dimension explicitly or reduce R/order."
        )

    if basis == "symmetric":
        level_dimensions, blocks = _symmetric_carleman_blocks(operators, order)
    else:
        level_dimensions, blocks = _ordered_carleman_blocks(operators, order)
    matrix = _clean_sparse(sparse.bmat(blocks, format="csr"))
    lifted_forcing = np.zeros(sum(level_dimensions), dtype=float)
    # the degree-zero source enters level one here, higher F0 blocks are in matrix
    lifted_forcing[:dimension] = operators.F0
    return FourierVorticityCarleman(
        operators=operators,
        order=order,
        basis=basis,
        level_dimensions=level_dimensions,
        blocks=blocks,
        carleman_matrix=matrix,
        lifted_forcing=lifted_forcing,
    )
