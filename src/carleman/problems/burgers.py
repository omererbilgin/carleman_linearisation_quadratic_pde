# Burgers specific symbolic mathematics:
#                           For the discrete path, F_1 and F_2 are written using the first and second derivative matrices
#                           For the continuous path, heat moments of the log potential is evolved before dividing the amplitude series
#                           These paths has their own representation, the finite difference grid isnt used by the continuous one

#                           Start reading this file from: burgers_operators, then continuous_burgers_carleman for the continuous path

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from carleman.discretisation import finite_difference_operators


@dataclass(frozen=True)
class ContinuousBurgersCarleman:
    initial_log_potential: Any
    spatial_variable: Any
    time_variable: Any
    viscosity: Any
    spatial_domain: tuple[float, float]
    heat_moment_functions: tuple[Any, ...]
    source_expressions: tuple[Any, ...]
    order_expressions: tuple[Any, ...]

    @property
    def carleman_order(self):
        return len(self.source_expressions)

    @property
    def physical_expression(self):
        return self.order_expressions[-1]

    @cached_property
    def initial_expression(self):
        import sympy as sp
        return -2 * self.viscosity * sp.diff(
            self.initial_log_potential, self.spatial_variable)

    @cached_property
    def _compiled(self):
        import sympy as sp
        means = sp.symbols(f"h1:{self.carleman_order + 1}")
        derivatives = sp.symbols(f"d1:{self.carleman_order + 1}")
        replace = dict(zip(self.heat_moment_functions, means))
        # treat each heat moment and its x derivative as independent inputs to lambdify
        replace.update({sp.diff(h, self.spatial_variable): d
                        for h, d in zip(self.heat_moment_functions, derivatives)})
        expressions = [v.xreplace(replace) for v in self.source_expressions]
        return (
            sp.lambdify(self.spatial_variable, self.initial_log_potential, "numpy"),
            sp.lambdify(self.spatial_variable, self.initial_expression, "numpy"),
            sp.lambdify((*means, *derivatives), expressions, "numpy", cse=True),
        )

    def heat_moments(self, positions, time, *, quadrature_order=512):
        import numpy as np
        from scipy.special import factorial, roots_legendre

        viscosity = float(self.viscosity)
        if not np.isfinite(viscosity) or viscosity <= 0:
            raise ValueError("viscosity must be a positive number for evaluation")
        if not np.isfinite(time) or time <= 0:
            raise ValueError("heat_moments requires a positive time")
        x = np.asarray(positions, dtype=float)
        left, right = self.spatial_domain
        length = right - left
        nodes, weights = roots_legendre(quadrature_order)
        # map Gauss-Legendre nodes and weights from [-1,1] to the physical interval
        nodes = left + (nodes + 1) * length / 2
        weights = weights * length / 2
        degrees = np.arange(1, self.carleman_order + 1)
        g = np.broadcast_to(self._compiled[0](nodes), nodes.shape)
        # these are the initial coefficients of exp(z*g), one column per degree
        initial_moments = g[:, None] ** degrees / factorial(degrees)
        values = np.empty((x.size, self.carleman_order))
        derivatives = np.empty_like(values)

        # the heat potential needs zero normal derivative at both ends
        # then u=-2*nu*d_x log(potential) also vanishes there
        # the reflected Gaussians cancel those boundary derivatives
        # evolve g**j/j! with the kernel and its x derivative with gradient
        # quadrature_order controls the integral accuracy, not the Carleman order
        # beyond this distance the Gaussian factor is smaller than exp(-45)
        cutoff = 2 * np.sqrt(viscosity * time * 45)
        images = int(np.ceil(1 + cutoff / (2 * length)))
        normalisation = np.sqrt(4 * np.pi * viscosity * time)
        # batch the evaluation points so the image-kernel arrays stay manageable
        for start in range(0, x.size, 128):
            xx = x.ravel()[start:start + 128, None]
            direct = xx - nodes[None, :]
            # the reflected part is what gives zero normal derivative at the boundary
            reflected = xx + nodes[None, :] - 2 * left
            kernel = np.zeros_like(direct)
            gradient = np.zeros_like(direct)
            for image in range(-images, images + 1):
                for distance in (direct + 2 * length * image,
                                 reflected + 2 * length * image):
                    gaussian = np.exp(-distance**2 / (4 * viscosity * time)) / normalisation
                    kernel += gaussian
                    gradient -= distance * gaussian / (2 * viscosity * time)
            values[start:start + len(xx)] = (kernel * weights) @ initial_moments
            derivatives[start:start + len(xx)] = (gradient * weights) @ initial_moments
        shape = (self.carleman_order, *x.shape)
        # move the degree index to the first axis, followed by the original position shape
        return values.T.reshape(shape), derivatives.T.reshape(shape)

    def evaluate_sources(self, positions, time, *, quadrature_order=512):
        import numpy as np
        x = np.asarray(positions, dtype=float)
        if time == 0:
            result = np.zeros((self.carleman_order, *x.shape))
            # at t=0 only the first initial-amplitude degree survives
            result[0] = np.broadcast_to(self._compiled[1](x), x.shape)
            return result
        h, hx = self.heat_moments(x, time, quadrature_order=quadrature_order)
        terms = self._compiled[2](*h, *hx)
        return np.stack([np.broadcast_to(v, x.shape) for v in terms])

    def evaluate_orders(self, positions, time, *, quadrature_order=512):
        import numpy as np
        # order N is the sum of source contributions 1 through N
        return np.cumsum(self.evaluate_sources(
            positions, time, quadrature_order=quadrature_order), axis=0)


def continuous_burgers_carleman(
    initial_log_potential, spatial_variable, time_variable, *, viscosity,
    carleman_order, spatial_domain=(-1.0, 1.0),
):
    import sympy as sp

    carleman_order = int(carleman_order)
    x, t = map(sp.sympify, (spatial_variable, time_variable))
    domain = tuple(map(float, spatial_domain))
    nu = sp.sympify(viscosity)
    g = sp.sympify(initial_log_potential)
    # use H_j for heat[g**j/j!], keep the heat evolution symbolic
    moments = tuple(sp.Function(f"H_{j}")(x, t)
                    for j in range(1, carleman_order + 1))
    terms, expressions = [], []

    # write H=1+z*H_1+z**2*H_2+... and expand -2*nu*H_x/H in z
    # its coefficient u_n is -2*nu*d_x(H_n)-sum_k H_k*u_(n-k)
    # so the earlier u terms needs to be available before computing the next one
    # these are amplitude degrees, the heat evolution inside H_n is kept in time
    for j, h in enumerate(moments):
        value = -2 * nu * sp.diff(h, x)
        value -= sum(moments[k - 1] * terms[j - k] for k in range(1, j + 1))
        terms.append(sp.expand(value))
        expressions.append(sp.expand(sum(terms)))
    return ContinuousBurgersCarleman(
        g, x, t, nu, domain, moments, tuple(terms), tuple(expressions))


def burgers_operators(
    grid_points: int,
    domain: tuple[float, float],
    viscosity: float,
    *,
    boundary: str = "HomogeneousDirichlet",
):
    finite_differences = finite_difference_operators(
        grid_points,
        domain,
        boundary=boundary,
    )
    first_operator = finite_differences.first_derivatives[0]
    second_operator = finite_differences.second_derivatives[0]
    first = (
        first_operator.toarray()
        if hasattr(first_operator, "toarray")
        else np.asarray(first_operator)
    )
    second = (
        second_operator.toarray()
        if hasattr(second_operator, "toarray")
        else np.asarray(second_operator)
    )
    grid = finite_differences.axes[0]
    # u_t=nu*D2*u - u*(D1*u), with multiplication point by point
    linear = viscosity * second
    quadratic = np.zeros((grid_points, grid_points, grid_points))
    for output in range(grid_points):
        # the undifferentiated u belongs to the output grid point
        quadratic[output, output, :] = -first[output, :]
    return linear, quadratic, np.zeros(grid_points), grid


def comparison_data(profile="sine", grid_points=64, maximum_order=3):
    import sympy as sp
    from scipy.sparse.linalg import expm_multiply
    from carleman.analytic import continuous_ordered_horner_taylor
    from carleman.discretisation import carleman_discretisation
    from carleman.linearisation import from_operators
    from carleman.problems.reference_solutions import burgers_exact, burgers_sine_exact

    sine = profile == "sine"
    domain = (-np.pi, np.pi) if sine else (-1., 1.)
    times = np.array([.5, 1., 2.] if sine else [.1, .5, 1., 2.])
    x, t = sp.symbols("x t", real=True)
    nu = sp.Rational(1, 10)
    initial = sp.sin(x) if sine else 2*sp.pi*nu*sp.sin(sp.pi*x)/(sp.Rational(21,20)+sp.cos(sp.pi*x))
    evaluate_initial = sp.lambdify(x, initial, "numpy")
    F1, F2, F0, grid = burgers_operators(grid_points, domain, float(nu))
    exact = (burgers_sine_exact(viscosity=float(nu)) if sine else
             burgers_exact(viscosity=float(nu), a=1.05))
    orders = (maximum_order,) if sine else tuple(range(1, maximum_order+1))
    curves = {}
    for order in orders:
        lifted = carleman_discretisation(from_operators(order,F1,F2,F0), evaluate_initial(grid))
        matrix = lifted.assemble_matrix()
        curves[order] = np.array([expm_multiply(time*matrix,lifted.initial_state)[:grid_points] for time in times])
    # only N=3 is used by the continuous sine figure, K is also three in the article
    expression = None
    if sine:
        model = continuous_ordered_horner_taylor(
            initial,(x,),t,carleman_order=maximum_order,taylor_order=3,
            linear_operator=lambda f,c: nu*sp.diff(f,c[0],2),
            quadratic_operator=lambda f,left,right: -sp.diff(f,right[0]))
        expression = model.physical_expression
    return dict(profile=profile,domain=domain,times=times,grid=grid,orders=orders,
                curves=curves,reference=exact,expression=expression,F0=F0,F1=F1,F2=F2,matrix=matrix)
