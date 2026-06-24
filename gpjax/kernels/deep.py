import equinox as eqx
from gpjax.kernels.computations import DenseKernelComputation
from jaxtyping import (
    Array,
    Float,
)
from gpjax.kernels.base import AbstractKernel


class DeepKernelFunction(AbstractKernel):
    base_kernel: AbstractKernel
    network: eqx.Module

    def __init__(
        self,
        base_kernel: AbstractKernel = None,
        network: eqx.Module = None,
    ):
        self.base_kernel = base_kernel
        self.network = network
        super().__init__(compute_engine=DenseKernelComputation())

    def __call__(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, "1"]:
        xt = self.network(x)
        yt = self.network(y)
        return self.base_kernel(xt, yt)