"""Linear algebra module for GPJax."""

from gpjax.linalg.custom_operators import BlockDiag, Kronecker
from gpjax.linalg.utils import add_jitter, cholesky_factor, logdet
from gpjax.linalg.computations import solve_block_triangular

__all__ = [
    "BlockDiag",
    "Kronecker",
    "add_jitter",
    "cholesky_factor",
    "logdet",
    "solve_block_triangular"
]
