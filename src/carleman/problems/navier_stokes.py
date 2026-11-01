# Biot-Savart law formualtion, embedded mathematics for Carleman operators for vorticity eq.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from carleman.problems.fourier_vorticity import _validated_times


@dataclass(frozen=True)
class PrimitiveNavierStokesReference:
    times: np.ndarray
    modes: np.ndarray
    velocity_coefficients: np.ndarray
    pressure_coefficients: np.ndarray
    vorticity_coefficients: np.ndarray
    grid_size: int
    dealias_cutoff: int
    function_evaluations: int
    maximum_divergence_coefficient: float


def article_initial_velocity(x, y, epsilon: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    # initial velocity comes from psi=sin(x)*sin(y)/2 + epsilon*sin(2*x+y)/5
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    first = np.sin(x) * np.cos(y)
    second = np.cos(2 * x + y)
    x_velocity = 0.5 * first + epsilon / 5 * second
    y_velocity = -0.5 * np.cos(x) * np.sin(y) - 2 * epsilon / 5 * second
    return x_velocity, y_velocity


def velocity_from_vorticity_coefficients(
    modes,
    coefficients,
) -> np.ndarray:
    modes = np.asarray(modes, dtype=int)
    coefficients = np.asarray(coefficients, dtype=complex)

    # divide by |k|**2 for Biot-Savart, leave k=0 out
    squared_wave_number = np.einsum("ij,ij->i", modes, modes)
    nonzero = squared_wave_number != 0
    x_velocity = np.zeros_like(coefficients, dtype=complex)
    y_velocity = np.zeros_like(coefficients, dtype=complex)
    x_velocity[..., nonzero] = (
        1j
        * modes[nonzero, 1]
        / squared_wave_number[nonzero]
        * coefficients[..., nonzero]
    )
    y_velocity[..., nonzero] = (
        -1j
        * modes[nonzero, 0]
        / squared_wave_number[nonzero]
        * coefficients[..., nonzero]
    )
    return np.stack((x_velocity, y_velocity), axis=-1)


def solve_primitive_navier_stokes_reference(
    times: Iterable[float],
    *,
    viscosity: float = 1 / 50,
    grid_size: int = 48,
    relative_tolerance: float = 1e-11,
    absolute_tolerance: float = 1e-13,
    epsilon: float = 0.3,
) -> PrimitiveNavierStokesReference:
    from scipy.integrate import solve_ivp

    times = _validated_times(times)
    viscosity, grid_size = float(viscosity), int(grid_size)
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8 for the article initial modes")
    # periodic grid, dont include 2*pi twice
    coordinates = 2 * np.pi * np.arange(grid_size) / grid_size
    mesh_x, mesh_y = np.meshgrid(coordinates, coordinates, indexing="ij")
    initial_x_velocity, initial_y_velocity = article_initial_velocity(
        mesh_x,
        mesh_y,
        epsilon,
    )

    wave_numbers = np.fft.fftfreq(grid_size, d=1 / grid_size)
    first_wave_number = wave_numbers[:, None]
    second_wave_number = wave_numbers[None, :]
    squared_wave_number = (
        first_wave_number**2 + second_wave_number**2
    )
    inverse_squared_wave_number = np.zeros_like(squared_wave_number)
    nonzero = squared_wave_number != 0
    inverse_squared_wave_number[nonzero] = 1 / squared_wave_number[nonzero]
    dealias_cutoff = (grid_size - 1) // 3
    dealias_mask = (
        (np.abs(first_wave_number) <= dealias_cutoff)
        & (np.abs(second_wave_number) <= dealias_cutoff)
    )
    # divide fft2 by grid_size**2 to get Fourier-series coefficients
    # multiply by the same scale before ifft2, otherwise the advection amplitude changes
    # use the two-thirds mask on both initial data and product spectra
    transform_scale = grid_size * grid_size

    initial_x_spectrum = np.fft.fft2(initial_x_velocity) / transform_scale
    initial_y_spectrum = np.fft.fft2(initial_y_velocity) / transform_scale
    initial_x_spectrum[~dealias_mask] = 0
    initial_y_spectrum[~dealias_mask] = 0

    # initial data is divergence free, this only removes roundoff
    initial_longitudinal = (
        first_wave_number * initial_x_spectrum
        + second_wave_number * initial_y_spectrum
    )
    initial_x_spectrum -= (
        first_wave_number
        * inverse_squared_wave_number
        * initial_longitudinal
    )
    initial_y_spectrum -= (
        second_wave_number
        * inverse_squared_wave_number
        * initial_longitudinal
    )
    initial_x_spectrum[0, 0] = 0
    initial_y_spectrum[0, 0] = 0

    state_size = grid_size * grid_size

    def nonlinear_and_pressure(x_spectrum, y_spectrum):
        x_spectrum = np.asarray(x_spectrum).reshape(grid_size, grid_size)
        y_spectrum = np.asarray(y_spectrum).reshape(grid_size, grid_size)
        # recover both velocity components to multiply the advection terms in physical space
        x_velocity = np.fft.ifft2(x_spectrum * transform_scale)
        y_velocity = np.fft.ifft2(y_spectrum * transform_scale)
        x_derivative_x = np.fft.ifft2(
            1j * first_wave_number * x_spectrum * transform_scale
        )
        x_derivative_y = np.fft.ifft2(
            1j * second_wave_number * x_spectrum * transform_scale
        )
        y_derivative_x = np.fft.ifft2(
            1j * first_wave_number * y_spectrum * transform_scale
        )
        y_derivative_y = np.fft.ifft2(
            1j * second_wave_number * y_spectrum * transform_scale
        )
        nonlinear_x = np.fft.fft2(
            x_velocity * x_derivative_x
            + y_velocity * x_derivative_y
        ) / transform_scale
        nonlinear_y = np.fft.fft2(
            x_velocity * y_derivative_x
            + y_velocity * y_derivative_y
        ) / transform_scale
        # remove the unresolved product modes before solving for pressure
        nonlinear_x[~dealias_mask] = 0
        nonlinear_y[~dealias_mask] = 0

        # pressure cancels the part of advection parallel to k
        # with n_hat for v.grad(v), p_hat=i*(k dot n_hat)/|k|**2
        # then -n_hat-i*k*p_hat has zero divergence
        # leave the mean pressure at zero, its gradient doesnt affect velocity
        pressure = (
            1j
            * (
                first_wave_number * nonlinear_x
                + second_wave_number * nonlinear_y
            )
            * inverse_squared_wave_number
        )
        pressure[0, 0] = 0
        return nonlinear_x, nonlinear_y, pressure

    def right_hand_side(_time, flat_velocity):
        # solve_ivp takes one vector, the two velocity components are stacked in it
        x_spectrum = flat_velocity[:state_size].reshape(
            grid_size,
            grid_size,
        )
        y_spectrum = flat_velocity[state_size:].reshape(
            grid_size,
            grid_size,
        )
        nonlinear_x, nonlinear_y, pressure = nonlinear_and_pressure(
            x_spectrum,
            y_spectrum,
        )
        # advection, pressure gradient and viscosity, with the same Fourier sign convention
        derivative_x = (
            -nonlinear_x
            - 1j * first_wave_number * pressure
            - viscosity * squared_wave_number * x_spectrum
        )
        derivative_y = (
            -nonlinear_y
            - 1j * second_wave_number * pressure
            - viscosity * squared_wave_number * y_spectrum
        )
        derivative_x[~dealias_mask] = 0
        derivative_y[~dealias_mask] = 0
        derivative_x[0, 0] = 0
        derivative_y[0, 0] = 0
        return np.concatenate((derivative_x.ravel(), derivative_y.ravel()))

    initial_state = np.concatenate(
        (initial_x_spectrum.ravel(), initial_y_spectrum.ravel())
    )
    if times.size == 1:
        full_velocity = initial_state[None, :]
        function_evaluations = 0
    else:
        solution = solve_ivp(
            right_hand_side,
            (0.0, float(times[-1])),
            initial_state,
            method="DOP853",
            t_eval=times,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        if not solution.success:
            raise RuntimeError(
                "primitive Navier--Stokes reference integration failed: "
                f"{solution.message}"
            )
        full_velocity = np.asarray(solution.y.T)
        function_evaluations = int(solution.nfev)

    full_velocity = full_velocity.reshape(times.size, 2, grid_size, grid_size)
    full_x_spectrum = full_velocity[:, 0]
    full_y_spectrum = full_velocity[:, 1]
    # recover pressure for every saved velocity state, including t=0
    full_pressure = np.stack(
        [
            nonlinear_and_pressure(x_spectrum, y_spectrum)[2]
            for x_spectrum, y_spectrum in zip(
                full_x_spectrum,
                full_y_spectrum,
            )
        ]
    )
    # curl of the directly evolved velocity, this isnt the Carleman vorticity
    full_vorticity = (
        1j * first_wave_number[None, ...] * full_y_spectrum
        - 1j * second_wave_number[None, ...] * full_x_spectrum
    )
    full_divergence = (
        1j * first_wave_number[None, ...] * full_x_spectrum
        + 1j * second_wave_number[None, ...] * full_y_spectrum
    )

    modes = np.asarray(
        [
            (first, second)
            for first in range(-dealias_cutoff, dealias_cutoff + 1)
            for second in range(-dealias_cutoff, dealias_cutoff + 1)
        ],
        dtype=int,
    )
    # extract the retained FFT modes in the same order used by the comparison
    indices = tuple(
        (int(mode[0]) % grid_size, int(mode[1]) % grid_size)
        for mode in modes
    )
    x_coefficients = np.stack(
        [full_x_spectrum[:, first, second] for first, second in indices],
        axis=-1,
    )
    y_coefficients = np.stack(
        [full_y_spectrum[:, first, second] for first, second in indices],
        axis=-1,
    )
    pressure_coefficients = np.stack(
        [full_pressure[:, first, second] for first, second in indices],
        axis=-1,
    )
    vorticity_coefficients = np.stack(
        [full_vorticity[:, first, second] for first, second in indices],
        axis=-1,
    )

    return PrimitiveNavierStokesReference(
        times=times,
        modes=modes,
        velocity_coefficients=np.stack(
            (x_coefficients, y_coefficients),
            axis=-1,
        ),
        pressure_coefficients=pressure_coefficients,
        vorticity_coefficients=vorticity_coefficients,
        grid_size=grid_size,
        dealias_cutoff=dealias_cutoff,
        function_evaluations=function_evaluations,
        maximum_divergence_coefficient=float(np.max(np.abs(full_divergence))),
    )
