import jax
import jax.numpy as jnp
import gpjax
from gpjax.kernels.base import AbstractKernel, _compute_base_init, _val
from gpjax.typing import ScalarFloat
from jaxtyping import Float, Array
from paramax import AbstractUnwrappable
from gpjax.parameters import NonNegativeReal
import beartype.typing as tp
import equinox as eqx
from typing import Callable
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)

class Gibbs(AbstractKernel):
    lengthscale_fn: Callable
    variance: AbstractUnwrappable

    def __init__(
        self,
        lengthscale_fn: Callable,
        active_dims: tp.Union[list[int], slice, None] = None,
        variance: tp.Union[ScalarFloat, AbstractUnwrappable] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        active_dims, n_dims, compute_engine = _compute_base_init(
            active_dims, n_dims, compute_engine
        )
        self.lengthscale_fn = lengthscale_fn
        if isinstance(variance, AbstractUnwrappable):
            self.variance = variance
        else:
            self.variance = NonNegativeReal(variance)

        self.active_dims = active_dims
        self.n_dims = n_dims
        self.compute_engine = compute_engine


    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"]
    ) -> ScalarFloat:
        lx = self.lengthscale_fn(x)
        ly = self.lengthscale_fn(y)

        dist_sq = jnp.sum((x - y) ** 2, axis=-1)

        prefactor = jnp.sqrt(2.0 * jnp.dot(lx,ly) / jnp.dot(lx**2,ly**2))

        exponential = jnp.exp(-dist_sq / jnp.dot(lx**2,ly**2))

        return _val(self.variance) * prefactor * exponential
