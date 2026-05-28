import beartype.typing as tp
from typing import Literal
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, cho_solve
from jaxtyping import (
    Float,
    Num,
)
import lineax as lx
from gpjax.typing import Array
from gpjax.dataset import SeparableDataset
from gpjax.mean_functions import ConditionedMean
from gpjax.kernels import ConditionedKernel
from gpjax.distributions import GaussianDistribution
from gpjax.likelihoods import AbstractLikelihood, Gaussian
from gpjax.gps import AbstractPrior, AbstractPosterior
from gpjax.linalg import add_jitter, compute_Kronecker_Cholesky, solve_block_triangular
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
        self.y = train_data.y
        self.Kaa = add_jitter(self.kernel_A.gram(self.A).as_matrix(), jitter)
        #self.L_A = cholesky(self.Kaa)
        #self.L_B = cholesky(self.Kbb)
        self.conditioned_on_data = True


    def condition_on_functional(self, functional, y):
        r"""Condition the posterior on a linear functional: $L[u] = y$.

        Args:
            functional (Callable): A callable mapping 
                a PDE solution function $u: \mathbb{R}^d \mapsto \mathbb{R}$ to a vector in $\mathbb{R}^l$.
            y: A vector in $\mathbb{R}^l 
        """
        old_mean = self.mean_function_B
        old_kernel = self.kernel_B
        kL = lambda b: functional(lambda b_prime: old_kernel(b, b_prime))
        LkL = jax.vmap(lambda i: functional(lambda x: kL(x)[i]))(jnp.arange(y.shape[0]))
        print("Min Eig LkL: ", jnp.linalg.eigh(LkL)[0].min())

        #LkL = functional(lambda b: functional(lambda b_prime: old_kernel(b, b_prime)))
        L_L = cholesky(LkL, lower=True)

        # GPJax defines mean functions as NxD -> Nx1, whereas functional expects D -> 1
        single_mean = lambda x: old_mean(jnp.atleast_2d(x)).squeeze()
        Lm = functional(single_mean)
        residual = y - Lm[:,None]
        self.mean_function_B = ConditionedMean(old_mean, L_L, residual, jax.vmap(kL))
        self.kernel_B = ConditionedKernel(old_kernel, L_L, kL)

    def compute_data_residual(self):
        r"""Difference between training targets and mean evaluated at training points."""
        pred = jnp.kron(self.mean_function_A(self.A), self.mean_function_B(self.B))
        return self.y - pred


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
        prior_mean = jnp.kron(self.mean_function_A(test_inputs_A), self.mean_function_B(test_inputs_B))
        print(jnp.any(jnp.isnan(prior_mean)))
        self.Kbb = add_jitter(self.kernel_B.gram(self.B).as_matrix(), jitter)
        L = compute_Kronecker_Cholesky(self.Kaa, self.Kbb)

        # Kernel computations
        Kata = self.kernel_A.cross_covariance(test_inputs_A, self.A)
        Kbtb = self.kernel_B.cross_covariance(test_inputs_B, self.B)
        K_test_train = jnp.kron(Kata, Kbtb)

        Katat = add_jitter(self.kernel_A.gram(test_inputs_A), jitter)
        Kbtbt = add_jitter(self.kernel_B.gram(test_inputs_B), jitter)
        prior_cov = Kronecker(Katat, Kbtbt)

        residual = self.compute_data_residual()
        residual = cho_solve((L, True), residual)
        residual = K_test_train @ residual
        X = cho_solve((L, True), K_test_train.mT)
        X = K_test_train @ X

        mean = prior_mean + residual
        print("Mean has Nan: ", jnp.any(jnp.isnan(mean)))

        cov = prior_cov - lx.MatrixLinearOperator(X)
        print("Min covariance value: ", cov.as_matrix().min())

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
