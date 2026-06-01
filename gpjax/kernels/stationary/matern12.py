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
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class Matern12(StationaryKernel):
    r"""The Matérn kernel with smoothness parameter fixed at 0.5.

    Computes the covariance on a pair of inputs $(x, y)$ with
    lengthscale parameter $\ell$ and variance $\sigma^2$.

    $$
    k(x, y) = \sigma^2\exp\Bigg(-\frac{\lvert x-y \rvert}{2\ell}\Bigg)
    $$
    """

    name: str = "Matérn12"

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> Float[Array, ""]:
        out = matern12_kernel(
            self.slice_input(x),
            self.slice_input(y),
            _val(self.lengthscale),
            _val(self.variance))
        return out


    @property
    def spectral_density(self) -> npd.StudentT:
        return build_student_t_distribution(nu=3)


@jax.custom_jvp
def matern12_kernel(x, y, lengthscale, variance):
    x = x / lengthscale
    y = y / lengthscale
    tau = euclidean_distance(x, y)
    K = variance * jnp.exp(-tau)
    return K.squeeze()


@matern12_kernel.defjvp
def matern12_kernel_jvp(primals, tangents):
    x, y, lengthscale, variance = primals
    x_dot, y_dot, l_dot, v_dot = tangents

    x = x / lengthscale
    y = y / lengthscale
    # Clip with eps to avoid tau = 0, which would make 2nd derivative unstable
    eps = 1e-12
    tau = jnp.maximum(euclidean_distance(x, y), eps)

    exp_term = jnp.exp(-tau)

    primal_out = variance * exp_term

    dk_dx = -variance * exp_term * (x - y) / (lengthscale**2 * tau)
    dk_dy = -dk_dx
    dk_dv = exp_term
    dk_dl = variance * exp_term * tau / lengthscale
    # For n_dim>1, lengthscale can still be scalar (isotropic)
    # In this case, sum up derivatives along dimension
    if lengthscale.ndim == 0:
        dk_dl = jnp.sum(dk_dl)

    tangent_out = (
        jnp.dot(dk_dx, x_dot)
        + jnp.dot(dk_dy, y_dot)
        + jnp.dot(dk_dl, l_dot)
        + dk_dv * v_dot     # variance is always scalar
    )
    return primal_out, tangent_out


    @property
    def spectral_density(self) -> npd.StudentT:
        return build_student_t_distribution(nu=1)
