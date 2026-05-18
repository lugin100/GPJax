from jax import numpy as jnp
from jax.scipy.linalg import qr, solve_triangular

def compute_Kronecker_Cholesky(A, B):
    r"""Compute the Cholesky decomposition of a Kronecker product:
    $$ A \otimes B = LL^T$$

    Args:
        A: Num[Array, '...']
        B: Num[Array, '...']
    Returns:
        $L$
    """
    Lambda_A, U_A = jnp.linalg.eigh(A)
    Lambda_B, U_B = jnp.linalg.eigh(B)
    _, R_A = qr(jnp.diag(jnp.sqrt(Lambda_A)) @ U_A.mT)
    _, R_B = qr(jnp.diag(jnp.sqrt(Lambda_B)) @ U_B.mT)
    L = jnp.kron(R_A, R_B).mT
    return L


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
    x1 = solve_triangular(L11.mT, y1 - L21.T @ x2, lower=False)
    return jnp.concatenate([x1, x2], axis=0)
