# Copyright 2022 The thomaspinder Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import jax
import jax.numpy as jnp
from jaxtyping import Float
import numpyro.distributions as npd

from gpjax.kernels.base import _val
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.stationary.utils import (
    build_student_t_distribution,
    euclidean_distance,
)
from gpjax.typing import Array

class Matern32(StationaryKernel):
    r"""The Matérn kernel with smoothness parameter fixed at 1.5.

    Computes the covariance for pairs of inputs $(x, y)$ with
    lengthscale parameter $\ell$ and variance $\sigma^2$.

    $$
    k(x, y) = \sigma^2 \Bigg(1+ \frac{\sqrt{3}\lvert x-y \rvert}{\ell} \ \Bigg)\exp\Bigg(-\frac{\sqrt{3}\lvert x-y\rvert}{\ell} \Bigg)
    $$
    """

    name: str = "Matérn32"

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> Float[Array, ""]:
        out = matern32_kernel(
            self.slice_input(x),
            self.slice_input(y),
            _val(self.lengthscale),
            _val(self.variance))
        return out

    @property
    def spectral_density(self) -> npd.StudentT:
        return build_student_t_distribution(nu=3)


def matern32_kernel(x, y, lengthscale, variance):
    x = x / lengthscale
    y = y / lengthscale
    s = jnp.sum((x - y) ** 2)
    
    # Standard matern kernel definition
    tau_safe = jnp.sqrt(s + 1e-36)
    k_large = (
        variance
        * (1.0 + jnp.sqrt(3.0) * tau_safe)
        * jnp.exp(-jnp.sqrt(3.0) * tau_safe)
    )
    # Taylor around tau=0
    k_small = variance * (1.0 - (3.0/2.0)*s)

    # Use Taylor approximation when s is small
    # This guarantees autodifferentiability up to order 2
    return jnp.where(s < 1e-12, k_small, k_large).squeeze()

