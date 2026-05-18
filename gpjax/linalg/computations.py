from jax import numpy as jnp
from jax.scipy.linalg import qr

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