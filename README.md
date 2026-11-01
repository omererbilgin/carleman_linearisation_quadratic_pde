# Carleman linearisation

This framework includes the necessary algorithms for constructing the Carleman matrix, and then solving it the PDE.

# Transfer Sequence of Eigenspaces

this framework includes (under src/carleman/analysis) making a connection of block matrices of the Carleman matrix in the Laplace domain. The eigenspaces are then introduced for these block matrices interacting. By algebraic multiplicity, and Laplace inverse, each TSE gives the info of associated eigenvalues of the eigenspaces of the block matrices interacting, and the contribution to the solution/response it results in.

# Examples

burgers' eq and vorticity eq. examples are given in burgers.py and vorticity.py compactly, for the embedded mathematics (boundary, initial conditions, carleman operators), check out src/problems
