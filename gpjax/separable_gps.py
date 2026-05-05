import beartype.typing as tp
from typing import Literal
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.numpy.linalg import qr
from jax.scipy.linalg import solve_triangular
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


class SeparablePosterior(eqx.Module, tp.Generic[P, L]):

    prior: SeparablePrior
    likelihood: tp.Any

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

    def condition_on_data(self, train_data: SeparableDataset):
        A, B, y = train_data.A, train_data.B, train_data.y
        Kaa = self.prior.prior_A.kernel.gram(A)
        Kbb = self.prior.prior_B.kernel.gram(B)
        L = compute_Gram_Cholesky(Kaa, Kbb)
        return ConditionedSeparablePosterior(
            self,
            train_data,
            Kaa,
            Kbb,
            L)


def compute_Gram_Cholesky(Kaa, Kbb):
    Lambda_A, U_A = jnp.linalg.eigh(Kaa.as_matrix())
    Lambda_B, U_B = jnp.linalg.eigh(Kbb.as_matrix())
    _, R_A = qr(jnp.diag(jnp.sqrt(Lambda_A)) @ U_A.mT)
    _, R_B = qr(jnp.diag(jnp.sqrt(Lambda_B)) @ U_B.mT)
    L = jnp.kron(R_A, R_B).mT
    return L


class ConditionedSeparablePosterior():

    prior: SeparablePrior
    likelihood: tp.Any
    train_data: SeparableDataset
    Kaa: lx.AbstractLinearOperator
    Kbb: lx.AbstractLinearOperator
    L: Num[Array, '...']
    jitter = 1e-6

    def __init__(
        self,
        posterior: SeparablePosterior,
        train_data: SeparableDataset,
        Kaa: lx.AbstractLinearOperator,
        Kbb: lx.AbstractLinearOperator,
        L: Num[Array, '...']):
        
        self.prior = posterior.prior
        self.likelihood = posterior.likelihood
        self.train_data = train_data
        self.Kaa = Kaa
        self.Kbb = Kbb
        self.L = L


    def __call__(
        self,
        test_inputs_A: Num[Array, "N D"],
        test_inputs_B: Num[Array, "M E"],
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        return self.predict(
        test_inputs_A,
        test_inputs_B,
        return_covariance_type=return_covariance_type,
    )

    def predict(
        self,
        test_inputs_A: Num[Array, "N D"],
        test_inputs_B: Num[Array, "M E"],
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        mean_function_A = self.prior.prior_A.mean_function
        mean_function_B = self.prior.prior_B.mean_function
        A, B, y = self.train_data.A, self.train_data.B, self.train_data.y
        #noise = self.likelihood.noise_vector(train_data.n)

        Kata = self.prior.prior_A.kernel.cross_covariance(test_inputs_A, A)
        Kbtb = self.prior.prior_B.kernel.cross_covariance(test_inputs_B, B)
        Kata = lx.MatrixLinearOperator(Kata)
        Kbtb = lx.MatrixLinearOperator(Kbtb)
        Kaat = Kata.transpose()
        Kbbt = Kbtb.transpose()
        Katat = self.prior.prior_A.kernel.gram(test_inputs_A)
        Kbtbt = self.prior.prior_B.kernel.gram(test_inputs_B)

        prior_mean = jnp.kron(mean_function_A(test_inputs_A), mean_function_B(test_inputs_B)).squeeze()
        res = y - jnp.kron(mean_function_A(A), mean_function_B(B))
        res = solve_triangular(self.L, res, lower=True)
        res = solve_triangular(self.L, res, lower=True, trans="T")
        res = Kronecker(Kata, Kbtb).mv(res)

        mean = prior_mean + res

        prior_cov = Kronecker(Katat, Kbtbt)

        X = Kronecker(Kaat, Kbbt).as_matrix()
        X = solve_triangular(self.L, X, lower=True)
        X = solve_triangular(self.L, X, lower=True, trans="T")
        # Compute Kron(Kata, Kbtb) @ X by vmapping over vec trick
        X = jax.vmap(Kronecker(Kata, Kbtb).mv, in_axes=1, out_axes=1)(X)
        X = lx.MatrixLinearOperator(X)
        jitterOperator = self.jitter * lx.IdentityLinearOperator(X.in_structure())
        cov = prior_cov - X + jitterOperator

        return GaussianDistribution(loc=jnp.atleast_1d(mean.squeeze()), scale=cov)
