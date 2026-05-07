import beartype.typing as tp
from typing import Literal
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import (
    qr,
    solve_triangular,
#    cholesky
)
from jax.lax.linalg import cholesky
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
    r"""Posterior to a separable GP prior with Gaussian likelihood."""

    prior: SeparablePrior
    likelihood: tp.Any
    mean_function_A: tp.Any
    mean_function_B: tp.Any
    kernel_A: tp.Any
    kernel_B: tp.Any

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


    def condition_on_data(self, train_data: SeparableDataset):
        r"""Condition the posterior on data.

        Args:
            train_data (SeparableDataset): Data to condition on.

        Returns:
            ConditionedSeparablePosterior
        """
        A, B, y = train_data.A, train_data.B, train_data.y
        Kaa = self.kernel_A.gram(A)
        Kbb = self.kernel_B.gram(B)
        L = _compute_Kronecker_Cholesky(Kaa.as_matrix(), Kbb.as_matrix())
        
        prior_eval = jnp.kron(self.mean_function_A(A), self.mean_function_B(B))
        return ConditionedSeparablePosterior(
            self,
            train_data,
            Kaa,
            Kbb,
            L,
            prior_eval)


def _compute_Kronecker_Cholesky(A, B):
    r"""Compute the Cholesky decomposition of a Kronecker product:
    $$ A \otimes B = LL^T$$

    Args:
        A: Num[Array, '...']
        B: Num[Array, '...']
    Returns:
        $L$
    """
    Lambda_A, U_A = jnp.linalg.eigh(A)
    Lambda_B, U_B = jnp.linalg.eigh(B)
    _, R_A = qr(jnp.diag(jnp.sqrt(Lambda_A)) @ U_A.mT)
    _, R_B = qr(jnp.diag(jnp.sqrt(Lambda_B)) @ U_B.mT)
    L = jnp.kron(R_A, R_B).mT
    return L


class ConditionedSeparablePosterior():
    r"""A separable posterior conditioned on data and optionally linear functionals."""
    prior: SeparablePrior
    likelihood: tp.Any
    train_data: SeparableDataset
    Kaa: lx.AbstractLinearOperator
    Kbb: lx.AbstractLinearOperator
    L: Num[Array, '...']

    def __init__(
        self,
        posterior: SeparablePosterior,
        train_data: SeparableDataset,
        Kaa: lx.AbstractLinearOperator,
        Kbb: lx.AbstractLinearOperator,
        L: Num[Array, '...'],
        prior_eval: Num[Array, '...']
        ):
        
        self.prior = posterior.prior
        self.likelihood = posterior.likelihood
        self.A = train_data.A
        self.B = train_data.B
        self.y = train_data.y
        self.Kaa = Kaa
        self.Kbb = Kbb
        self.L = L
        self.prior_eval = prior_eval
        self.mean_function_A = posterior.mean_function_A
        self.mean_function_B = posterior.mean_function_B
        self.kernel_A = posterior.kernel_A
        self.kernel_B = posterior.kernel_B

    def __call__(
        self,
        test_inputs_A: Num[Array, "N D"],
        test_inputs_B: Num[Array, "M E"],
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Infer the posterior distribution at given inputs.

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
        return_covariance_type=return_covariance_type,
    )

    def predict(
        self,
        test_inputs_A: Num[Array, "N D"],
        test_inputs_B: Num[Array, "M E"],
        jitter = 1e-6,
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Infer the posterior distribution at given inputs.

        Args:
            test_inputs_A: Where to infer on domain A.
            test_inputs_B: Where to infer on domain B.
            jitter (float): A small constant added to the diagonal of the
                covariance matrix to ensure numerical stability.

        Returns:
            Gaussian distribution over values at test_inputs.
        """
        #noise = self.likelihood.noise_vector(train_data.n)

        Kata = self.kernel_A.cross_covariance(test_inputs_A, self.A)
        Kbtb = self.kernel_B.cross_covariance(test_inputs_B, self.B)
        KbtL = self.functional(lambda b: self.kernel_B.cross_covariance(test_inputs_B, b))
        K_test_train = jnp.kron(Kata, Kbtb)
        K_test_functional = jnp.kron(Kata, KbtL)
        K_test_conditions = jnp.concatenate((K_test_train, K_test_functional), axis=1)

        #Kata = lx.MatrixLinearOperator(Kata)
        #Kbtb = lx.MatrixLinearOperator(Kbtb)
        #Kaat = Kata.transpose()
        #Kbbt = Kbtb.transpose()
        Katat = self.kernel_A.gram(test_inputs_A)
        Kbtbt = self.kernel_B.gram(test_inputs_B)

        prior_mean = jnp.kron(self.mean_function_A(test_inputs_A), self.mean_function_B(test_inputs_B)).squeeze()
        
        res = self.y - self.prior_eval
        res = solve_triangular(self.L, res, lower=True)
        res = solve_triangular(self.L, res, lower=True, trans="T")
        #res = Kronecker(Kata, Kbtb).mv(res)
        res = K_test_conditions @ res

        mean = prior_mean[:,None] + res

        prior_cov = Kronecker(Katat, Kbtbt)

        #X = Kronecker(Kaat, Kbbt).as_matrix()
        X = K_test_conditions.mT
        self.L = self.L + jitter * jnp.eye(self.L.shape[0])
        X = solve_triangular(self.L, X, lower=True)
        X = solve_triangular(self.L, X, lower=True, trans="T")
        # Compute Kron(Kata, Kbtb) @ X by vmapping over vec trick
        #X = jax.vmap(Kronecker(Kata, Kbtb).mv, in_axes=1, out_axes=1)(X)
        X = K_test_conditions @ X
        X = lx.MatrixLinearOperator(X)
        jitterOperator = jitter * lx.IdentityLinearOperator(X.in_structure())
        cov = prior_cov - X + jitterOperator
        return GaussianDistribution(loc=jnp.atleast_1d(mean.squeeze()), scale=cov)


    def condition_on_functional(self, functional, y, jitter=1e-3):
        LkB = functional(lambda b: self.kernel_B.cross_covariance(b, self.B))
        
        #kLB = LkB.mT
        LkZ = jnp.kron(self.Kaa.as_matrix(), LkB)
        kLZ = LkZ.mT
        LkL = functional(lambda b: functional(lambda b_prime: self.kernel_B.cross_covariance(b, b_prime)))
        LkL = jnp.kron(self.Kaa.as_matrix(), LkL)
        L_11 = self.L
        L_21 = solve_triangular(self.L, kLZ).mT
        L_12 = jnp.zeros_like(L_21.mT)
        S = LkL - L_21 @ L_21.mT + jitter * jnp.eye(LkL.shape[0])
        eigvals = jnp.linalg.eigvalsh(S)
        print(eigvals.min())
        L_22 = cholesky(S, symmetrize_input=True)
        L_new = jnp.block([[L_11, L_12], [L_21, L_22]])
        self.L = L_new
        print(jnp.any(jnp.isnan(S)))
        print(jnp.any(jnp.isnan(L_22)))
        self.y = jnp.concatenate((self.y, jnp.tile(y, (self.Kaa.as_matrix().shape[0],1))), axis=0)
        
        functional_eval = jnp.kron(self.mean_function_A(self.A), functional(self.mean_function_B))
        self.prior_eval = jnp.concatenate((self.prior_eval, functional_eval), axis=0)
        self.functional = functional
        return self