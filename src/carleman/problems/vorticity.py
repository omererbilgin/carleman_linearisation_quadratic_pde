# Vorticity specific symbolic matheamtics with the initial considiton sinxsiny \epsilon sin2x+y:
#                           The Green operator acts on exact exponential terms, without a projection radius
#                           Independent coordinate groups is kept until the quadratic action has finished
#                           The same ordered Horner solver as Burgers supplies the continuous expression

#                           Start reading this file from: vorticity_continuous

import sympy as sp
from carleman.analytic import continuous_ordered_horner_taylor


def vorticity_continuous(order=3, taylor_order=3, viscosity=sp.Rational(1, 1000), epsilon=2):
    x, y, t = sp.symbols("x y t", real=True)
    nu = sp.sympify(viscosity)
    initial = sp.sin(x)*sp.sin(y) + sp.sympify(epsilon)*sp.sin(2*x+y)

    def green_velocity(expression, left, right, component):
        terms = []
        # periodic Green operator on exact exponentials, no Fourier cutoff or mesh
        for term in sp.Add.make_args(sp.expand(expression.rewrite(sp.exp))):
            powers = term.as_powers_dict()
            phase = powers.get(sp.E, sp.S.Zero) + sp.Add(*(
                base.args[0]*power for base, power in powers.items() if base.func == sp.exp))
            wave = tuple(sp.expand(sp.diff(phase, coordinate)/sp.I) for coordinate in right)
            squared = sp.expand(sum(value**2 for value in wave))
            # the constant mode has no velocity from the periodic Green operator
            if squared == 0:
                continue
            multiplier = sp.I*wave[1]/squared if component == 0 else -sp.I*wave[0]/squared
            # integrate the right coordinate, then evaluate the velocity at the left one
            terms.append(multiplier * term.xreplace(dict(zip(right, left))))
        return sp.expand(sp.Add(*terms))

    def linear(expression, coordinates):
        return nu * sum(sp.diff(expression, coordinate, 2) for coordinate in coordinates)

    def quadratic(expression, left, right):
        # apply -v(right).grad_left omega(left), then the Green integral
        return -sum(green_velocity(sp.diff(expression, left[j]), left, right, j)
                    for j in range(2))

    result = continuous_ordered_horner_taylor(
        initial, (x, y), t, carleman_order=order, taylor_order=taylor_order,
        linear_operator=linear, quadratic_operator=quadratic,
    )
    # exponentials here are exact spatial expressions, not numerical Fourier coefficients
    expression = result.physical_expression
    evaluate = sp.lambdify((x, y, t), expression, "numpy", cse=True, docstring_limit=0)
    return {"model": result, "expression": expression, "evaluate": evaluate}
