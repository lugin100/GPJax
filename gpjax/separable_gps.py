import beartype.typing as tp
from typing import Literal
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import (
    solve_triangular,
    cholesky
)
from jaxtyping import (
    Float,
    Num,
)
from gpjax.typing import Array
from gpjax.dataset import SeparableDataset
import lineax as lx
from gpjax.distributions import GaussianDistribution
from gpjax.likelihoods import AbstractLikelihood, Gaussian
from gpjax.gps import AbstractPrior, AbstractPosterior
from gpjax.linalg import add_jitter, solve_block_triangular
from gpjax.linalg.custom_operators import Kronecker

L = tp.TypeVar("L", bound=AbstractLikelihood)
G = tp.TypeVar("G", bound=Gaussian)
P = tp.TypeVar("P", bound=AbstractPrior)
PO = tp.TypeVar("PO", bound=AbstractPosterior)


class SeparablePrior(eqx.Module):
    r"""Gaussian process prior over two spaces $\mathbb{A}$ and $\mathbb{B}$, where mean and kernel are given as:
    $$m((a,b)) = m_A(a)\cdot m_B(b)$$
    $$k((a,b), (a^\prime,b^\prime)) = k_A(a, a^\prime)\cdot k_B(b, b^\prime)$$
    """

    prior_A: P
    prior_B: P

    def __init__(
        self,
        prior_A: P,
        prior_B: P
    ):
        r"""Construct a Gaussian process prior from two priors defined on separable domains.

        Args:
            prior_A: Prior defined on domain A.
            prior_B: Prior defined on domain B.
        """
        self.prior_A = prior_A
        self.prior_B = prior_B

    def __call__(
        self,
        test_inputs_A: Num[Array, "N D"],
        test_inputs_B: Num[Array, "M E"],
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Evaluate the Gaussian process at the given points.

        The output of this function is a ``GaussianDistribution`` from which
        the latent function's mean and covariance can be evaluated and the
        distribution can be sampled.

        Args:
            test_inputs_A: Input locations where the GP should be evaluated.
            return_covariance_type: Literal denoting whether to return the full covariance
                of the joint predictive distribution at the test_inputs (dense)
                or just the the standard-deviation of the predictive distribution at
                the test_inputs.

        Returns:
            GaussianDistribution: A multivariate normal random variable representation
                of the Gaussian process.
        """
        return self.predict(
            test_inputs_A,
            test_inputs_B,
            return_covariance_type=return_covariance_type,
        )

    def full_mean(self, A, B):
        mean_A = self.prior_A.mean_function(A)
        mean_B = self.prior_B.mean_function(B)
        return jnp.kron(mean_A, mean_B).squeeze()

    def full_gram(self, A, B):
        gram_A = self.prior_A.kernel.gram(A)
        gram_B = self.prior_B.kernel.gram(B)
        return Kronecker(gram_A, gram_B)


    def predict(
        self,
        test_inputs_A: Num[Array, "N D"],
        test_inputs_B: Num[Array, "M E"],
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Compute the predictive prior distribution for a given set of
        parameters. The output of this function is a ``GaussianDistribution``
        over each combination of inputs for A and B.

        Args:
            test_inputs_A (Float[Array, "N D"]): The inputs at which to evaluate prior A.
            test_inputs_B (Float[Array, "M E"]): The inputs at which to evaluate prior B.

            return_covariance_type: Literal denoting whether to return the full covariance
                of the joint predictive distribution at the test_inputs (dense)
                or just the standard-deviation of the predictive distribution at
                the test inputs.

        Returns:
            GaussianDistribution: A multivariate normal random variable representation
                of the Gaussian process.
        """
        NM = test_inputs_A.shape[0] * test_inputs_B.shape[0]
        jitterOperator = self.jitter * lx.IdentityLinearOperator(NM)

        def _return_full_covariance(t_A, t_B):
            Kaa = self.prior_A.kernel.gram(t_A)
            Kbb = self.prior_B.kernel.gram(t_B)
            return Kronecker(Kaa, Kbb) + jitterOperator

        def _return_diagonal_covariance(t_A, t_B):
        	Kaa = self.prior_A.kernel.diagonal(t_A)
        	Kbb = self.prior_B.kernel.diagonal(t_B)
        	Kxx = jnp.kron(Kaa, Kbb)
        	return lx.DiagonalLinearOperator(Kxx) + jitterOperator

        mean_at_test = self.full_mean(test_inputs_A, test_inputs_B)

        cov = jax.lax.cond(
            return_covariance_type == "dense",
            _return_full_covariance,
            _return_diagonal_covariance,
            test_inputs_A,
            test_inputs_B
        )

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean_at_test.squeeze()), scale=cov
        )

    def __mul__(self, other: L):
        r"""Combine the prior with a likelihood to form a posterior distribution.

        The product of a prior and likelihood is proportional to the posterior
        distribution. By computing the product of a GP prior and a likelihood
        object, a posterior GP object will be returned. Mathematically, this can
        be described by:
        ```math
        p(f(\cdot) \mid y) \propto p(y \mid f(\cdot))p(f(\cdot)),
        ```
        where $p(y | f(\cdot))$ is the likelihood and $p(f(\cdot))$ is the prior.

        Args:
            other (Likelihood): The likelihood distribution of the observed dataset.

        Returns
            Posterior: The relevant GP posterior for the given prior and
                likelihood.
        """
        is_gaussian_likelihood = isinstance(other, Gaussian)
        if is_gaussian_likelihood:
        	return SeparablePosterior(prior=self, likelihood=other)
       	raise NotImplementedError(
       		"SeparablePrior only supports Gaussian likelihoods."
       		)

    def __rmul__(self, other):
        r"""Combine the prior with a likelihood to form a posterior distribution.

        Reimplement the multiplication operator to allow for order-invariant
        product of a likelihood and a prior i.e., likelihood * prior.

        Args:
            other (Likelihood): The likelihood distribution of the observed
                dataset.

        Returns
            Posterior: The relevant GP posterior for the given prior and
                likelihood.
        """
        return self.__mul__(other)


class SeparablePosterior():
    r"""Posterior to a separable GP prior with Gaussian likelihood."""

    def __init__(
        self,
        prior: SeparablePrior,
        likelihood: G,

    ):
        r"""Construct a Gaussian process posterior.

        Args:
            prior (SeparablePrior): The prior distribution.
            likelihood (Gaussian): The likelihood distribution.
        """
        self.prior = prior
        self.likelihood = likelihood
        self.mean_function_A = prior.prior_A.mean_function
        self.mean_function_B = prior.prior_B.mean_function
        self.kernel_A = prior.prior_A.kernel
        self.kernel_B = prior.prior_B.kernel
        self.conditioned_on_data = False
        self.conditioned_on_functional = False



    def condition_on_data(self, train_data: SeparableDataset, jitter=1e-6):
        r"""Condition the posterior on data.

        Args:
            train_data (SeparableDataset): Data to condition on.
        """
        if self.conditioned_on_data:
            raise ValueError("Can only condition on data once")
        self.A = train_data.A
        self.B = train_data.B
        self.Kaa = add_jitter(self.kernel_A.gram(self.A).as_matrix(), jitter)
        self.Kbb = add_jitter(self.kernel_B.gram(self.B).as_matrix(), jitter)
        L_A = cholesky(self.Kaa, lower=True)
        L_B = cholesky(self.Kbb, lower=True)
        self.L_11 = jnp.kron(L_A, L_B)
        self.residual_data = self.compute_data_residual(train_data)
        self.conditioned_on_data = True


    def compute_data_residual(self, train_data):
        r"""Diffrence between training targets and prior mean evaluated at training points."""
        prior_pred = jnp.kron(self.mean_function_A(train_data.A), self.mean_function_B(train_data.B))
        return train_data.y - prior_pred


    def condition_on_functional(self, functional, y):
        r"""Condition the posterior on a linear functional: $L[u] = y$.

        Args:
            functional (Callable): A callable mapping 
                a PDE solution function $u: \mathbb{R}^d \mapsto \mathbb{R}$ to a vector in $\mathbb{R}^l$.
            y: A vector in $\mathbb{R}^l 
        """
        if not self.conditioned_on_functional:
            self.y_functional = y
            self.functional = functional
            self.conditioned_on_functional = True
        else:
            self.y_functional = jnp.concatenate((self.y_functional, y))
            old_functional = self.functional
            def new_functional(x):
                return jnp.concatenate((old_functional(x), functional(x)))
            self.functional = new_functional

    def predict(
        self,
        test_inputs_A: Num[Array, "N D"],
        test_inputs_B: Num[Array, "M E"],
        jitter = 1e-6,
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Infer the posterior distribution at given inputs,
            taking all previous conditionings into account.

        Args:
            test_inputs_A: Where to infer on domain A.
            test_inputs_B: Where to infer on domain B.
            jitter (float): A small constant added to the diagonal of the
                covariance matrix to ensure numerical stability.

        Returns:
            Gaussian distribution over values at test_inputs.
        """
        if not self.conditioned_on_data:
            raise ValueError("Can not predict on posterior that has not been conditioned on data. Use prior.predict() instead.")
        #noise = self.likelihood.noise_vector(train_data.n)

        # Compute K_test_train
        Kata = self.kernel_A.cross_covariance(test_inputs_A, self.A)
        Kbtb = self.kernel_B.cross_covariance(test_inputs_B, self.B)
        K_test_train = jnp.kron(Kata, Kbtb)

        # Compute prior mean
        mean_A_test = self.mean_function_A(test_inputs_A)
        mean_B_test = self.mean_function_B(test_inputs_B)
        prior_mean = jnp.kron(mean_A_test, mean_B_test).squeeze()

        # Compute prior covariance
        Katat = add_jitter(self.kernel_A.gram(test_inputs_A), jitter)
        Kbtbt = add_jitter(self.kernel_B.gram(test_inputs_B), jitter)
        prior_cov = Kronecker(Katat, Kbtbt)

        if not self.conditioned_on_functional:
            res = solve_triangular(self.L_11, self.residual_data, lower=True)
            res = solve_triangular(self.L_11, res, lower=True, trans="T")
            res = K_test_train @ res

            X = solve_triangular(self.L_11, K_test_train.mT, lower=True)
            X = solve_triangular(self.L_11, X, lower=True, trans="T")
            X = K_test_train @ X

        else:
            kL = lambda b: self.functional(lambda b_prime: self.kernel_B(b, b_prime))
            kLB = jax.vmap(kL)(self.B)
            kLBt = jax.vmap(kL)(test_inputs_B)
            kLZ = jnp.kron(Kata.mT, kLB)

            LkL = jax.vmap(lambda i: self.functional(lambda x: kL(x)[i]))(jnp.arange(self.y_functional.shape[0]))
            LkLZ = jnp.kron(Katat.as_matrix(), LkL)
            
            L_21 = solve_triangular(self.L_11, kLZ, lower=True).mT
            S = LkLZ - L_21 @ L_21.mT
            S = add_jitter(S, jitter)
            L_22 = cholesky(S, lower=True)

            new_y = jnp.kron(jnp.ones((test_inputs_A.shape[0],1)), self.y_functional)
            mLZ = jnp.kron(self.mean_function_A(test_inputs_A), self.functional(lambda x: self.mean_function_B(jnp.atleast_2d(x)).squeeze())[:,None])
            residual_functional = new_y - mLZ
            K_test_functional = jnp.kron(Katat.as_matrix(), kLBt)

            blocks = solve_block_triangular(self.L_11, L_21, L_22, self.residual_data, residual_functional)
            res = K_test_train @ blocks[0] + K_test_functional @ blocks[1]

            blocks = solve_block_triangular(self.L_11, L_21, L_22, K_test_train.mT, K_test_functional.mT)
            X = K_test_train @ blocks[0] + K_test_functional @ blocks[1]

        mean = prior_mean[:,None] + res

        X = lx.MatrixLinearOperator(X)
        cov = prior_cov - X

        return GaussianDistribution(loc=jnp.atleast_1d(mean.squeeze()), scale=cov)


    def __call__(
        self,
        test_inputs_A: Num[Array, "N D"],
        test_inputs_B: Num[Array, "M E"],
        jitter = 1e-6,
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Infer the posterior distribution at given inputs,
            taking all previous conditionings into account.

        Args:
            test_inputs_A: Where to infer on domain A.
            test_inputs_B: Where to infer on domain B.
            jitter (float): A small constant added to the diagonal of the
                covariance matrix to ensure numerical stability.

        Returns:
            Gaussian distribution over values at test_inputs.
        """
        return self.predict(
        test_inputs_A,
        test_inputs_B,
        jitter=jitter,
        return_covariance_type=return_covariance_type,
    )
