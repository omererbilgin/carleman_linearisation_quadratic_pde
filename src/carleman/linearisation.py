# Algorthmic structure:
#                           F_0, F_1 and F_2 are supplied explicitly, identifying them from the PDE isnt done here
#                           The arrays are kept together with the retained order, they still describe the original quadratic system
#                           The block dimensions follows from the number of components and the monomial degrees

#                           Start reading this file from: from_operators

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CarlemanData:
    order: int
    linear: Any
    # quadratic is indexed as [output, left factor, right factor]
    quadratic: Any
    forcing: Any
    state_dimension: int | None = None

    def __post_init__(self) -> None:
        self.linear = np.asarray(self.linear, dtype=float)
        self.state_dimension = self.linear.shape[0]
        self.quadratic = np.asarray(self.quadratic, dtype=float)
        # keep one F0 value per physical component
        self.forcing = np.asarray(self.forcing, dtype=float).reshape(self.state_dimension)

    @property
    def lifted_dimension_symmetric(self) -> int:
        from math import comb
        # sum the symmetric level sizes, excluding the constant degree-zero product
        return comb(self.state_dimension + self.order, self.order) - 1


def from_operators(
    order: int,
    linear: Any,
    quadratic: Any,
    forcing: Any = 0.0,
) -> CarlemanData:
    linear_array = np.asarray(linear, dtype=float)
    dimension = linear_array.shape[0]
    if np.isscalar(forcing):
        # a scalar forcing value is applied to each component
        forcing = np.full(dimension, float(forcing))
    return CarlemanData(order, linear_array, quadratic, forcing)


def carleman_linearisation(
    equation: Any,
    order: int,
    F0: Any,
    F1: Any,
    F2: Any,
) -> CarlemanData:
    from carleman.equation_formatting import equation_data

    # only describe the equation here, the operator arrays come from the caller
    metadata = equation_data(equation)
    if metadata.equation_type != "Quadratic":
        raise ValueError(f"Unsupported equation type: {metadata.equation_type}")
    return from_operators(order, F1, F2, F0)
