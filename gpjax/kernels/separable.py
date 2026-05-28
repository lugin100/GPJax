from gpjax.kernels import AbstractKernel
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Num,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)
from typing import Tuple

class SeparableKernel(AbstractKernel):
    r"""A kernel that is the multiplication of two kernels on different domains.

    Args:
        k_A: Base kernel on domain A
        k_B: Base kernel on domain B
    """
    k_A: AbstractKernel
    k_B: AbstractKernel


    def __init__(self,
        k_A: AbstractKernel,
        k_B: AbstractKernel
    ):
        self.k_A = k_A
        self.k_B = k_B
        super().__init__()


    def __call__(
        self,
        x: Tuple[Float[Array, " D"]],
        y: Tuple[Float[Array, " D"]],
    ) -> ScalarFloat:
        k_A = self.k_A(x[0], y[0])
        k_B = self.k_B(x[1], y[1])
        return k_A * k_B


    def cross_covariance(
            self, x: Tuple[Num[Array, "N D"]], y: Tuple[Num[Array, "M D"]]
        ) -> Array:
        k_A = self.k_A.cross_covariance(x[0], y[0])
        k_B = self.k_B.cross_covariance(x[1], y[1])
        return jnp.kron(k_A, k_B)


    def gram(self, x: Tuple[Num[Array, "N D"]]) -> Array:
        r"""Compute the gram matrix of the kernel.

        Args:
            x: the input matrix of shape `(N, D)`.

        Returns:
            The gram matrix of the kernel of shape `(N, N)`.
        """
        k_A = self.k_A.gram(x[0])
        k_B = self.k_B.gram(x[1])
        return jnp.kron(k_A, k_B)