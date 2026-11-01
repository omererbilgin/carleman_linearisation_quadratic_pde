# Algorthmic structure:
#                           Tuples help the nature of the kronecker product to be implemented
#                           Non commutative multiplication of the block matrices follow it
#                           The block matrices may contain also matrices, in terms of F_j, the necessary iterations must be done during the symbolic multiplications

#                           Start reading this file from: continuous_ordered_horner_taylor

# The code presented here follows Vaszary's formulations in: https://arxiv.org/abs/2412.00014

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ContinuousCarlemanResult:
    physical_expression: Any
    # level_expressions still use separate coordinates for each factor
    level_expressions: tuple[Any, ...]
    coordinate_copies: tuple[tuple[tuple[Any, ...], ...], ...]
    spatial_variables: tuple[Any, ...]
    time_variable: Any
    carleman_order: int
    taylor_order: int


def continuous_ordered_horner_taylor(
    initial_expression: Any,
    spatial_variables: Sequence[Any],
    time_variable: Any,
    *,
    carleman_order: int,
    taylor_order: int,
    linear_operator,
    quadratic_operator,
    simplify_result: bool = False,
) -> ContinuousCarlemanResult:
    import sympy as sp

    variables = tuple(sp.sympify(variable) for variable in spatial_variables)
    time_symbol = sp.sympify(time_variable)
    # this path takes a scalar field and supplied operators, F0 isnt included
    initial = sp.sympify(initial_expression)

    # each factor needs its own coordinates while taking derivatives
    # for two factors these are ((x1,), (x2,)) or ((x1,y1), (x2,y2))
    # coordinates[0] is the first spatial direction of the group passed to the operator
    # keep u0(x1)*u0(x2) apart here, joining x1 and x2 too early differentiates both factors
    coordinate_copies = tuple(
        tuple(
            tuple(
                sp.Symbol(f"{variable.name}{copy_index + 1}")
                for variable in variables
            )
            for copy_index in range(level)
        )
        for level in range(1, carleman_order + 1)
    )

    # initial level i is the product of i copies of the initial field
    initial_blocks = tuple(
        sp.expand(
            sp.prod(
                initial.xreplace(dict(zip(variables, coordinates)))
                for coordinates in level_coordinates
            )
        )
        for level_coordinates in coordinate_copies
    )

    def carleman_action(blocks):
        result = []
        for level in range(1, carleman_order + 1):
            # level one is stored at index zero
            target_coordinates = coordinate_copies[level - 1]
            # apply F1 once at each factor, as in the product rule
            # for two Burgers factors this is nu*(d_x1**2 + d_x2**2)
            # the other coordinate groups stay fixed during each call
            value = sum(
                (
                    sp.sympify(
                        linear_operator(blocks[level - 1], coordinates)
                    )
                    for coordinates in target_coordinates
                ),
                sp.S.Zero,
            )

            # at the last level there is no higher block, this is the N truncation
            if level < carleman_order:
                source = blocks[level]
                source_coordinates = coordinate_copies[level]

                for position in range(level):
                    term = sp.sympify(
                        quadratic_operator(
                            source,
                            source_coordinates[position],
                            source_coordinates[position + 1],
                        )
                    )
                    # the source has one extra factor, F2 joins position and position+1
                    # for Burgers, first take -d_x2 of u(x1)*u(x2), then set x2=x1
                    # both groups now use one target group, the remaining groups shift back one place
                    # do these substitutions together, otherwise a moved coordinate may get renamed again
                    collapse = {}
                    for source_position, source_group in enumerate(
                        source_coordinates
                    ):
                        if source_position <= position:
                            target_position = source_position
                        else:
                            target_position = source_position - 1
                        collapse.update(
                            zip(
                                source_group,
                                target_coordinates[target_position],
                            )
                        )
                    value += term.xreplace(collapse)
            result.append(sp.expand(value))
        return tuple(result)

    value = initial_blocks

    # start from the last divisor and work backwards
    # with K=2 this gives z0 + t*C*(z0 + t*C*z0/2)
    # add the original initial products at each step, dont replace them with value
    # here N limits the monomial levels, K limits the time powers
    for divisor in range(taylor_order, 0, -1):
        action = carleman_action(value)
        value = tuple(
            sp.expand(initial_block + time_symbol * block / divisor)
            for initial_block, block in zip(initial_blocks, action)
        )

    # level 1 is the physical field, rename its dummy coordinates back to x,y,...
    physical_substitution = dict(zip(coordinate_copies[0][0], variables))
    physical_expression = value[0].xreplace(physical_substitution)
    if simplify_result:
        physical_expression = sp.simplify(physical_expression)

    return ContinuousCarlemanResult(
        physical_expression=physical_expression,
        level_expressions=value,
        coordinate_copies=coordinate_copies,
        spatial_variables=variables,
        time_variable=time_symbol,
        carleman_order=carleman_order,
        taylor_order=taylor_order,
    )
