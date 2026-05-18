"""Linear algebra module for GPJax."""

from gpjax.linalg.custom_operators import BlockDiag, Kronecker
from gpjax.linalg.utils import add_jitter, cholesky_factor, logdet
from gpjax.linalg.computations import compute_Kronecker_Cholesky

__all__ = [
    "BlockDiag",
    "Kronecker",
    "add_jitter",
    "cholesky_factor",
    "logdet",
    "compute_Kronecker_Cholesky"
]
