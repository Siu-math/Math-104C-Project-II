from .elliptic import solve_laplace_gauss_seidel
from .parabolic import (
    solve_backward_difference_gauss_seidel,
    solve_crank_nicolson_gauss_seidel,
    solve_forward_difference,
)


ELLIPTIC_METHODS = [("Gauss-Seidel", solve_laplace_gauss_seidel)]
HEAT_METHODS = [
    ("Forward Difference", solve_forward_difference),
    ("Backward Difference", solve_backward_difference_gauss_seidel),
    ("Crank-Nicolson", solve_crank_nicolson_gauss_seidel),
]
