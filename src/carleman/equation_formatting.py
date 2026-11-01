# Algorthmic structure:
#                           The time derivative on the left gives the dependent field and its spatial variables
#                           The powers of the field is counted term by term on the right, then an equation type is assigned
#                           This is only metadata, derivative factors arent counted correctly yet for terms like u*u_x

#                           Start reading this file from: equation_data

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EquationData:
    equation: Any
    field: Any
    spatial_variables: tuple[Any, ...]
    time_variable: Any
    rhs: Any
    field_polynomial_degrees: tuple[int, ...]
    equation_type: str


def _sympy_data(equation: Any) -> EquationData:
    import sympy as sp

    lhs = equation.lhs
    derivatives = list(lhs.atoms(sp.Derivative))
    if len(derivatives) != 1:
        raise ValueError("one time derivative expected on the left")
    derivative = derivatives[0]
    if derivative.derivative_count != 1:
        raise ValueError("expected a first derivative")

    field_application = derivative.expr
    field = field_application.func
    variables = tuple(field_application.args)
    # the differentiated argument is time, the other field arguments are space
    time_variable = derivative.variables[0]
    spatial_variables = tuple(variable for variable in variables if variable != time_variable)

    rhs = equation.rhs
    degrees = []
    # this only counts plain u factors and powers of u
    # sympy keeps u_x as a derivative object, so u*u_x gets counted as degree one here
    # this isnt enough for identifying the PDE operators, supply F1 and F2 explicitly
    for term in sp.Add.make_args(sp.expand(rhs)):
        degree = 0
        for factor, power in term.as_powers_dict().items():
            if getattr(factor, "func", None) == field:
                degree += int(power) if getattr(power, "is_integer", False) else 1
        degrees.append(degree)
    degree_tuple = tuple(sorted(set(degrees or [0])))
    highest = max(degree_tuple)
    equation_type = "Linear" if highest <= 1 else "Quadratic" if highest == 2 else "Nth Degree"
    return EquationData(
        equation,
        field,
        spatial_variables,
        time_variable,
        rhs,
        degree_tuple,
        equation_type,
    )


def equation_data(equation: Any) -> EquationData:
    if isinstance(equation, EquationData):
        return equation
    return _sympy_data(equation)
