import jax
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular as scipy_solve
import lineax as lx

def generate_solver(L):
    r"""Given a Kronecker linear operator, 
    returns a callable mapping a vector (or matrix) b to the solution of Lx=b.
        Args:
            L: Linear Operator
    """
    if isinstance(L, lx.KroneckerLinearOperator):
        solver = lx.Kronecker()
    else:
        solver = lx.AutoLinearSolver(well_posed=True)
    state = solver.init(L, options={})
    solve = lambda b: lx.linear_solve(L, b, solver, state=state).value
    state_T, _ = solver.transpose(state, {})
    solve_T = lambda b: lx.linear_solve(L, b, solver, state=state_T).value

    def solve_with_L(B, transpose=False):
        if isinstance(B, lx.KroneckerLinearOperator):
            return solver.compute_for_Kronecker(state, B, {})[0]
        if isinstance(B, lx.AbstractLinearOperator):
            B = B.as_matrix()
        solve_fn = solve_T if transpose else solve
        if B.ndim == 1:
            return solve_fn(B)
        else: # b.ndim == 2
            return jax.vmap(solve_fn, in_axes=1, out_axes=1)(B)

    return solve_with_L


def solve_block_triangular(solve_L11, L21, solve_L22, b1, b2):
    r"""Compute $L^{-1} b$ where 
    $L$ is assumed to be a lower-triangular block matrix
    $L = [[L11, 0], [L21, L22]]$ and b is a vector or matrix $[[b1, b2]]$.
    """
    x1 = solve_L11(b1)
    x2 = solve_L22(b2 - L21 @ x1)
    return x1, x2
