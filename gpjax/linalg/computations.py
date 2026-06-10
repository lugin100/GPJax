import jax
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular as scipy_solve
import lineax as lx

def solve_triangular(L, b, lower=True):
    r"""Compute the linear system Lx = b.
        Args:
            L: Linear Operator
            b: Right hand side, may be a matrix interpreted as batch of vectors.
            lower:  If True, L is assumed to be lower triangular (default). 
                    If False, L is assumed to be upper triangular
        Returns:
            x: Solution of Lx = b
    """
    if isinstance(L, jax.Array):
        return scipy_solve(L, b, lower=lower, trans="N")

    #solve = lambda b: scipy_solve(L.as_matrix(), b, **kwargs)
    #solver = lx.Normal(lx.GMRES(rtol=1e-9, atol=1e-9))
    #solve = lambda b: lx.linear_solve(L, b, solver).value
    solve = lambda b: mv_triangular_solve(L, b, lower)
    if b.ndim == 1:
        return solve(b)
    else: # b.ndim == 2
        return jax.vmap(solve, in_axes=1, out_axes=1)(b)

def mv_triangular_solve(L: lx.AbstractLinearOperator, b, lower):
    r"""Compute the linear system Lx = b using forward or backward substitution.

    Args:
        L: Triangular linear operator.
        b: Right-hand side vector.
        lower: Whether L is lower or upper triangular.
    Returns:
        x: Solution of Lx = b.
    """
    b = jnp.asarray(b)
    n = b.shape[0]

    def row(i):
        e = jax.nn.one_hot(i, n, dtype=b.dtype)
        return L.transpose().mv(e)

    def body(m, x):
            rhs = b[m]
            row_m = row(m)
            rhs = rhs - jnp.dot(row_m, x)
            diag = row_m[m]
            return x.at[m].set(rhs / diag)
    x0 = jnp.zeros_like(b)

    if lower:
        return jax.lax.fori_loop(0, n, body, x0)
    else:
        def reverse_body(k, x):
            m = n - 1 - k
            return body(m, x)

        return jax.lax.fori_loop(0, n, reverse_body, x0)


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
    x1 = solve_triangular(L11.transpose(), y1 - L21.mT @ x2, lower=False)
    return x1, x2
