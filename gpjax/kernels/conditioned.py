import jax.numpy as jnp
from gpjax.kernels.base import AbstractKernel
from typing import Callable
from jaxtyping import Float
from gpjax.typing import Array, ScalarFloat
from jax.scipy.linalg import cho_solve

class ConditionedKernel(AbstractKernel):
    r"""A kernel that is conditioned on data.

    Args:
        k: Base kernel that is to be conditioned on.
        L: lower cholesky factor of training Gram matrix.
        kX_train: function mapping kernel input x to k(x, X_train).
    """
    k: AbstractKernel
    L: Array
    kX_train: Callable


    def __init__(self, k: AbstractKernel, L: Array, kX_train: Callable):
        self.k = k
        self.L = L
        self.kX_train = kX_train
        super().__init__()


    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        k_xX = self.kX_train(x)
        k_Xy = self.kX_train(y)
        representer_weight = cho_solve((self.L, True), k_Xy)
        prior = self.k(x,y)
        return prior - jnp.dot(k_xX, representer_weight)
