import beartype.typing as tp
from typing import Literal
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky

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
from gpjax.linalg import add_jitter, solve_block_triangular, generate_solver

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
        return jnp.kron(mean_A, mean_B)

    def full_gram(self, A, B):
        gram_A = self.prior_A.kernel.gram(A)
        gram_B = self.prior_B.kernel.gram(B)
        return lx.KroneckerLinearOperator(gram_A, gram_B)


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
            return lx.KroneckerLinearOperator(Kaa, Kbb) + jitterOperator

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


class SeparablePosterior(AbstractPosterior):
    r"""Posterior to a separable GP prior with Gaussian likelihood."""

    def condition_on_data(self, train_data: SeparableDataset, jitter=1e-6):
        r"""Condition the posterior on data.

        Args:
            train_data (SeparableDataset): Data to condition on.
        """
        A = train_data.A
        B = train_data.B
        Kaa = add_jitter(self.prior.prior_A.kernel.gram(A).as_matrix(), jitter)
        Kbb = add_jitter(self.prior.prior_B.kernel.gram(B).as_matrix(), jitter)
        L_A = lx.MatrixLinearOperator(cholesky(Kaa, lower=True))
        L_B = lx.MatrixLinearOperator(cholesky(Kbb, lower=True))
        L_A = lx.TaggedLinearOperator(L_A, lx.lower_triangular_tag)
        L_B = lx.TaggedLinearOperator(L_B, lx.lower_triangular_tag)
        L_11 = lx.KroneckerLinearOperator(L_A, L_B)
        solve_with_L11 = generate_solver(L_11)
        residual = self.compute_data_residual(train_data)
        return ConditionedSeparablePosterior(self, A, B, residual, solve_with_L11)

    def compute_data_residual(self, train_data):
        r"""Difference between training targets and prior mean evaluated at training points."""
        prior_pred = jnp.kron(
            self.prior.prior_A.mean_function(train_data.A),
            self.prior.prior_B.mean_function(train_data.B)
            )
        return (train_data.y - prior_pred).reshape(-1, 1)

    def predict(self):
        raise ValueError("Condition on data before predicting")


class ConditionedSeparablePosterior():

    def __init__(self, separablePosterior, A, B, residual, solve_with_L11):
        self.likelihood = separablePosterior.likelihood
        self.mean_function_A = separablePosterior.prior.prior_A.mean_function
        self.mean_function_B = separablePosterior.prior.prior_B.mean_function
        self.kernel_A = separablePosterior.prior.prior_A.kernel
        self.kernel_B = separablePosterior.prior.prior_B.kernel
        self.A = A
        self.B = B
        self.residual_data = residual
        self.solve_with_L11 = solve_with_L11
        self.conditioned_on_functional = False

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

    def compute_functional_matrices(self, use_cached_functionals):
        if use_cached_functionals:
            if hasattr(self, "LkL") and hasattr(self, "kLB") and hasattr(self, "kL"):
                return self. kL, self.kLB, self.LkL
        # else recompute
        kL = lambda b: self.functional(lambda b_prime: self.kernel_B(b, b_prime))
        kLB = jax.vmap(kL)(self.B)
        LkL = jax.vmap(lambda i: self.functional(lambda x: kL(x)[i]))(jnp.arange(self.y_functional.shape[0]))
        return kL, lx.MatrixLinearOperator(kLB), lx.MatrixLinearOperator(LkL)

    def predict(
        self,
        test_inputs_A: Num[Array, "N D"],
        test_inputs_B: Num[Array, "M E"],
        jitter = 1e-6,
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
        use_cached_functionals: bool = False,
    ) -> GaussianDistribution:
        r"""Infer the posterior distribution at given inputs,
            taking all previous conditionings into account.

        Args:
            test_inputs_A: Where to infer on domain A.
            test_inputs_B: Where to infer on domain B.
            jitter (float): A small constant added to the diagonal of the
                covariance matrix to ensure numerical stability.
            use_cached_functionals: Indicate that functionals have not changed since last inference call,
                                    meaning cached computation can be used
        Returns:
            Gaussian distribution over values at test_inputs.
        """
        P = self.likelihood.num_outputs
        # TODO: What about noise?
        #noise = self.likelihood.noise_vector(train_data.n)


        # Used for LkLZ and prior_cov
        self.Katat = add_jitter(self.kernel_A.gram(test_inputs_A), jitter)

        # Compute K_test_train
        Kata = lx.MatrixLinearOperator(self.kernel_A.cross_covariance(test_inputs_A, self.A))
        Kbtb = lx.MatrixLinearOperator(self.kernel_B.cross_covariance(test_inputs_B, self.B))
        K_test_train = lx.KroneckerLinearOperator(Kata, Kbtb)

        # Parse 'return_covariance_type' input
        mapping = {"dense": True, "diagonal": False}
        if return_covariance_type.lower() not in mapping:
            raise ValueError(f"'return_covariance_type' must be 'dense' or 'diagonal', got '{return_covariance_type}'")
        dense = mapping[return_covariance_type.lower()]

        if not self.conditioned_on_functional:
            L_inv_res = self.solve_with_L11(self.residual_data)
            L_inv_K_train_test = self.solve_with_L11(K_test_train.transpose())

            mean_update = L_inv_K_train_test.transpose() @ L_inv_res

            if dense:
                cov_update = L_inv_K_train_test.transpose() @ L_inv_K_train_test
            else:
                cov_update = L_inv_K_train_test.squared_sum()
        else:

            self.kL, self.kLB, self.LkL = self.compute_functional_matrices(use_cached_functionals)
            kLBt = lx.MatrixLinearOperator(jax.vmap(self.kL)(test_inputs_B))
            kLZ = lx.KroneckerLinearOperator(Kata.transpose(), self.kLB)

            LkLZ = lx.KroneckerLinearOperator(self.Katat, self.LkL)

            L_21 = self.solve_with_L11(kLZ).transpose()
            S = LkLZ - L_21 @ L_21.transpose()
            S = add_jitter(S, jitter).as_matrix()
            L_22 = cholesky(S, lower=True)
            L_22 = lx.MatrixLinearOperator(L_22, lx.lower_triangular_tag)
            solve_with_L22 = generate_solver(L_22)
            new_y = jnp.kron(jnp.ones((test_inputs_A.shape[0],1)), self.y_functional)
            mLZ = jnp.kron(self.mean_function_A(test_inputs_A), self.functional(lambda x: self.mean_function_B(jnp.atleast_2d(x)).squeeze())[:,None])
            residual_functional = new_y - mLZ
            K_test_functional = lx.KroneckerLinearOperator(self.Katat, kLBt)

            L_inv_res = solve_block_triangular(self.solve_with_L11, L_21, solve_with_L22, self.residual_data, residual_functional)
            L_inv_K_train_test = solve_block_triangular(self.solve_with_L11, L_21, solve_with_L22, K_test_train.transpose(), K_test_functional.transpose())

            mean_update = L_inv_K_train_test[0].transpose() @ L_inv_res[0] + L_inv_K_train_test[1].mT @ L_inv_res[1]

            if dense:
                cov_update = L_inv_K_train_test[0].transpose() @ L_inv_K_train_test[0]
                cov_update = cov_update + lx.MatrixLinearOperator(L_inv_K_train_test[1].mT @ L_inv_K_train_test[1])
            else:
                cov_update = L_inv_K_train_test[0].squared_sum()
                cov_update = cov_update + lx.DiagonalLinearOperator(jnp.sum(L_inv_K_train_test[1]**2, axis=0))

        prior_mean = self.prior_mean(test_inputs_A, test_inputs_B)
        prior_mean = jnp.tile(prior_mean, (P, 1)) if P > 1 else prior_mean
        mean = prior_mean + mean_update
        cov = self.prior_cov(test_inputs_B, dense) - cov_update

        return GaussianDistribution(loc=jnp.atleast_1d(mean.squeeze()), scale=cov)

    def prior_mean(self, A_test, B_test):
        mean_A = self.mean_function_A(A_test)
        mean_B = self.mean_function_B(B_test)
        prior_mean = jnp.kron(mean_A, mean_B)
        return prior_mean

    def prior_cov(self, B_test, dense):
        if dense:
            Katat = self.Katat
            Kbtbt = self.kernel_B.gram(B_test)
        else:
            Katat = lx.DiagonalLinearOperator(self.Katat.as_matrix().diagonal())
            Kbtbt = self.kernel_B.diagonal(B_test)
        prior_cov = lx.KroneckerLinearOperator(Katat, Kbtbt)
        return prior_cov

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
