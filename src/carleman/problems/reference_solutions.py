# Cole hopf transformation for burgers equation:
#                           exact analytical solution for sinx and some other initial conditions 
#                           Start reading this file from: burgers_exact, then the numerical reference function needed

from __future__ import annotations

from typing import Callable


def burgers_exact(
    *,
    viscosity: float,
    a: float,
) -> Callable:
    import numpy as np

    # this closed form uses b=nu in the rational sine initial profile
    def solution(x, t):
        x_array = np.asarray(x)
        return (
            2
            * np.pi
            * viscosity
            * np.exp(-np.pi**2 * viscosity * t)
            * np.sin(np.pi * x_array)
            / (a + np.exp(-np.pi**2 * viscosity * t) * np.cos(np.pi * x_array))
        )

    return solution


def burgers_sine_exact(
    *,
    viscosity: float,
    fourier_terms: int = 128,
) -> Callable:
    import numpy as np

    from scipy.special import ive

    viscosity = float(viscosity)
    fourier_terms = int(fourier_terms)
    mode_numbers = np.arange(fourier_terms + 1, dtype=float)
    concentration = 1.0 / (2.0 * viscosity)
    # ive removes the common exp(concentration) from the Bessel coefficients
    # it cancels in H_x/H anyway, keeping it would overflow for small viscosity
    bessel_coefficients = ive(mode_numbers, concentration)

    def solution(x, t):
        time_value = float(t)
        x_array = np.asarray(x, dtype=float)
        if time_value == 0.0:
            result = np.sin(x_array)
            return float(result) if result.ndim == 0 else result

        flat_x = x_array.reshape(-1)
        decayed = bessel_coefficients * np.exp(
            -viscosity * mode_numbers**2 * time_value
        )
        phases = np.outer(mode_numbers[1:], flat_x)
        # the constant mode enters H only, differentiating it gives zero in H_x
        denominator = decayed[0] + 2.0 * np.sum(
            decayed[1:, None] * np.cos(phases),
            axis=0,
        )
        numerator = 4.0 * viscosity * np.sum(
            (mode_numbers[1:] * decayed[1:])[:, None] * np.sin(phases),
            axis=0,
        )
        result = (numerator / denominator).reshape(x_array.shape)
        return float(result) if result.ndim == 0 else result

    return solution
