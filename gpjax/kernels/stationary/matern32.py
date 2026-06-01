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
    k(x, y) = \sigma^2 \exp \Bigg(1+ \frac{\sqrt{3}\lvert x-y \rvert}{\ell} \ \Bigg)\exp\Bigg(-\frac{\sqrt{3}\lvert x-y\rvert}{\ell} \Bigg)
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


@jax.custom_jvp
def matern32_kernel(x, y, lengthscale, variance):
    x = x / lengthscale
    y = y / lengthscale
    tau = euclidean_distance(x, y)
    K = (
        variance
        * (1.0 + jnp.sqrt(3.0) * tau)
        * jnp.exp(-jnp.sqrt(3.0) * tau)
    )
    return K.squeeze()


@matern32_kernel.defjvp
def matern32_kernel_jvp(primals, tangents):
    x, y, lengthscale, variance = primals
    x_dot, y_dot, l_dot, v_dot = tangents

    diff = (x - y)/ lengthscale
    # Add eps to avoid r = 0, which would make 2nd derivative undefined
    eps = 1e-12
    r = jnp.sqrt(jnp.sum(diff**2, axis=-1) + eps)
    tau = r

    exp_term = jnp.exp(-jnp.sqrt(3.0) * tau)

    primal_out = variance * (1.0 + jnp.sqrt(3.0) * tau) * exp_term

    dk_dx = -3.0 * variance * exp_term * diff / lengthscale
    dk_dy = -dk_dx
    dk_dl = 3.0 * variance * exp_term * diff**2 / lengthscale
    # diff might have D>1 even if lengthsclae is scalar (isotropic)
    if lengthscale.ndim == 0:
        dk_dl = jnp.sum(dk_dl)
    dk_dv = (1.0 + jnp.sqrt(3.0) * tau) * exp_term
    print(dk_dl.shape)
    print(l_dot.shape)
    print(jnp.dot(dk_dl, l_dot).shape)
    tangent_out = (
        jnp.dot(dk_dx, x_dot)
        + jnp.dot(dk_dy, y_dot)
        + jnp.dot(dk_dl, l_dot)
        + dk_dv * v_dot
    )
    return primal_out, tangent_out
