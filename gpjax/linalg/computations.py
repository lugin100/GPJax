import jax
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular as scipy_solve


def solve_triangular(L, b, **kwargs):
    r"""Compute the linear system Lx = b.
        Args:
            L: Linear Operator
            b: Right hand side, may be a matrix interpreted as batch of vectors.
            kwargs: keyword arguments to be passed to the solver
        Returns:
            x: Solution of Lx = b
    """
    if isinstance(L, jax.Array):
        return scipy_solve(L, b, **kwargs)
    return scipy_solve(L.as_matrix(), b, **kwargs)

def solve_block_triangular(L11, L21, L22, b1, b2):
    r"""Compute $(L L^T)^{-1} b$ where 
    $L$ is assumed to be a lower-triangular block matrix
    $L = [[L11, 0], [L21, L22]]$ and b is a vector or matrix $[[b1, b2]]$.
    """
    # Block forward substitution
    y1 = solve_triangular(L11, b1, lower=True)
    y2 = solve_triangular(L22, b2 - L21 @ y1, lower=True)
    # Block backward substitution
    x2 = solve_triangular(L22.mT, y2, lower=False)
    x1 = solve_triangular(L11.transpose(), y1 - L21.T @ x2, lower=False)
    return x1, x2
