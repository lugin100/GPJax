from typing import Literal
import beartype.typing as tp
from jaxtyping import (
    Float,
    Num,
)
from gpjax.typing import Array

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky
import equinox as eqx
import lineax as lx

from gpjax.dataset import SeparableDataset
from gpjax.distributions import GaussianDistribution
from gpjax.likelihoods import AbstractLikelihood, Gaussian
from gpjax.gps import AbstractPrior, AbstractPosterior
from gpjax.linalg import add_jitter, solve_block_triangular, generate_solver

L = tp.TypeVar("L", bound=AbstractLikelihood)
P = tp.TypeVar("P", bound=AbstractPrior)


class SeparablePrior(eqx.Module):
    r"""Gaussian process prior over two spaces $\mathbb{P}$ and $\mathbb{X}$,
    where mean and kernel are given as:
    $$m((p,x)) = m_P(p)\cdot m_X(x)$$
    $$k((p,x), (p^\prime,x^\prime)) = k_P(p, p^\prime) \cdot k_B(b, b^\prime)$$
    """

    prior_P: P
    prior_X: P

    def __init__(
        self,
        prior_P: P,
        prior_X: P
    ):
        r"""Construct a Gaussian process prior from two priors defined on separable domains.

        Args:
            prior_P: Prior defined on domain P.
            prior_X: Prior defined on domain X.
        """
        self.prior_P = prior_P
        self.prior_X = prior_X

    def __call__(
        self,
        P_test: Num[Array, "N D"],
        X_test: Num[Array, "M E"],
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Evaluate the Gaussian process at the given points.

        The output of this function is a ``GaussianDistribution`` from which
        the latent function's mean and covariance can be evaluated and the
        distribution can be sampled.

        Args:
            P_test (Num[Array, "N D"]): The inputs at which to evaluate prior P.
            X_test (Num[Array, "M E"]): The inputs at which to evaluate prior X.            
            return_covariance_type: Literal denoting whether to return the full
                covariance of the joint predictive distribution at the
                test inputs (dense) or just the the standard-deviation of the
                predictive distribution at the test inputs.

        Returns:
            GaussianDistribution: A multivariate normal random variable
                representation of the Gaussian process.
        """
        return self.predict(
            P_test,
            X_test,
            return_covariance_type=return_covariance_type,
        )

    def full_mean(self, P, X):
        mean_P = self.prior_P.mean_function(P)
        mean_X = self.prior_X.mean_function(X)
        return jnp.kron(mean_P, mean_X)

    def full_gram(self, P, X):
        gram_P = self.prior_P.kernel.gram(P)
        gram_X = self.prior_X.kernel.gram(X)
        return lx.KroneckerLinearOperator(gram_P, gram_X)

    def predict(
        self,
        P_test: Num[Array, "N D"],
        X_test: Num[Array, "M E"],
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
    ) -> GaussianDistribution:
        r"""Compute the predictive prior distribution for a given set of
        parameters. The output of this function is a ``GaussianDistribution``
        over each combination of inputs for P and X.

        Args:
            P_test (Num[Array, "N D"]): The inputs at which to evaluate prior P.
            X_test (Num[Array, "M E"]): The inputs at which to evaluate prior X.

            return_covariance_type: Literal denoting whether to return the full
                covariance of the joint predictive distribution at the
                test inputs (dense) or just the the standard-deviation of the
                predictive distribution at the test inputs.

        Returns:
            GaussianDistribution: A multivariate normal random variable
                representation of the Gaussian process.
        """
        def _return_full_covariance(t_P, t_X):
            Kpp = self.prior_P.kernel.gram(t_P)
            Kxx = self.prior_X.kernel.gram(t_X)
            return add_jitter(lx.KroneckerLinearOperator(Kpp, Kxx), jitter)

        def _return_diagonal_covariance(t_P, t_X):
            Kpp = self.prior_P.kernel.diagonal(t_P)
            Kxx = self.prior_X.kernel.diagonal(t_X)
            Kzz = jnp.kron(Kpp, Kxx)
            return add_jitter(lx.DiagonalLinearOperator(Kzz), jitter)

        mean_at_test = self.full_mean(P_test, X_test)

        cov = jax.lax.cond(
            return_covariance_type == "dense",
            _return_full_covariance,
            _return_diagonal_covariance,
            P_test,
            X_test
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
        P = train_data.A
        X = train_data.B
        Kpp = add_jitter(self.prior.prior_P.kernel.gram(P).as_matrix(), jitter)
        Kxx = add_jitter(self.prior.prior_X.kernel.gram(X).as_matrix(), jitter)
        L_P = lx.MatrixLinearOperator(cholesky(Kpp, lower=True))
        L_X = lx.MatrixLinearOperator(cholesky(Kxx, lower=True))
        L_P = lx.TaggedLinearOperator(L_P, lx.lower_triangular_tag)
        L_X = lx.TaggedLinearOperator(L_X, lx.lower_triangular_tag)
        L_11 = lx.KroneckerLinearOperator(L_P, L_X)
        solve_with_L11 = generate_solver(L_11)
        residual = self.compute_data_residual(train_data)
        return ConditionedSeparablePosterior(self, P, X, residual, solve_with_L11)

    def compute_data_residual(self, train_data):
        r"""Difference between training targets and
        prior mean evaluated at training points."""
        prior_pred = jnp.kron(
            self.prior.prior_P.mean_function(train_data.A),
            self.prior.prior_X.mean_function(train_data.B)
            )
        return (train_data.y - prior_pred).reshape(-1, 1)

    def predict(self):
        raise ValueError("Condition on data before predicting")


class ConditionedSeparablePosterior():
    r"""Posterior to a separable GP prior that is conditoned on data.

    Does not subclass AbstractPosterior in order to not be an eqx.Module,
    which simplifies implementation."""

    def __init__(self, separablePosterior, P, X, residual, solve_with_L11):
        self.likelihood = separablePosterior.likelihood
        self.mean_function_P = separablePosterior.prior.prior_P.mean_function
        self.mean_function_X = separablePosterior.prior.prior_X.mean_function
        self.kernel_P = separablePosterior.prior.prior_P.kernel
        self.kernel_X = separablePosterior.prior.prior_X.kernel
        self.P = P
        self.X = X
        self.residual_data = residual
        self.solve_with_L11 = solve_with_L11
        self.conditioned_on_functional = False

    def condition_on_functional(self, functional, y, uncertainty=0., vectorize=False):
        r"""Condition the posterior on a linear functional: $L[u] = y$.

        Args:
            functional (Callable): A callable mapping a PDE solution function
            $u: \mathbb{R}^d \mapsto \mathbb{R}$ to a vector in $\mathbb{R}^l$.
            If 'vectorize'==True, a callable mapping a batch of PDE solutions 
            $u: \mathbb{R}^d \mapsto \mathbb{R}^m$ to a batch of vectors ($\mathbb{R}^{l \times m}$)
            y: A vector in $\mathbb{R}^l
            uncertainty: Assumed variance of the residual between functional output and y.
                Can be zero, in which case the functional is assumed to be exact (default).
            vectorize: Whether to batch-compute the functional Gram matrix for a speed-up.
                If True, assumes 'functional' to operate on a multivariate function, see above.

            Note:
                Vectorization is only applied if all conditionings are called with 'vectorize'=True.
        """
        if not self.conditioned_on_functional:
            self.y_functional = y
            self.functional = functional
            self.conditioned_on_functional = True
            self.Tau = jnp.ones(len(y)) * uncertainty
            self.vectorize = vectorize
        else:
            self.y_functional = jnp.concatenate((self.y_functional, y))
            old_functional = self.functional

            def new_functional(u):
                return jnp.concatenate((old_functional(u), functional(u)))
            self.functional = new_functional
            new_Tau = jnp.ones(len(y)) * uncertainty
            self.Tau = jnp.concatenate((self.Tau, new_Tau))
            self.vectorize = self.vectorize & vectorize

    def compute_functional_matrices(self, use_cached_functionals):
        if use_cached_functionals:
            if hasattr(self, "LkL") and hasattr(self, "kLX") and hasattr(self, "kL"):
                return self.kL, self.kLX, self.LkL
        # else recompute
        kL = lambda x: self.functional(lambda x_prime: self.kernel_X(x, x_prime))
        kLX = jax.vmap(kL)(self.X)

        if self.vectorize:
            LkL = self.functional(kL)
        else:
            LkL = jax.vmap(lambda i: self.functional(
                lambda u: kL(u)[i]))(jnp.arange(self.y_functional.shape[0])
                )
        LkL = add_jitter(LkL, 1e-6)
        assert jnp.allclose(LkL - LkL.mT, 0), "The functional Gram matrix is not symmetric. Make sure that the functionals are linear."
        assert jnp.linalg.eigvalsh(LkL).min() > 0, "The functional Gram matrix is not pos. definite. Make sure the kernel is psd and numerically stable under the functionals operations."
        return kL, lx.MatrixLinearOperator(kLX), lx.MatrixLinearOperator(LkL)


    def predict(
        self,
        P_test: Num[Array, "N D"],
        X_test: Num[Array, "M E"],
        jitter=1e-6,
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
        use_cached_functionals: bool = False,
    ) -> GaussianDistribution:
        r"""Infer the posterior distribution at given inputs,
            taking all previous conditionings into account.

        Args:
            P_test: Where to infer on domain P.
            X_test: Where to infer on domain X.
            jitter (float): A small constant added to the diagonal of the
                covariance matrix to ensure numerical stability.
            use_cached_functionals: Indicate that functionals have not changed
                since last inference call, so use cached computations.
        Returns:
            Gaussian distribution over values at test inputs.
        """
        
        # Compute Kptpt for LkLZ and prior_cov
        self.Kptpt = add_jitter(self.kernel_P.gram(P_test), jitter)

        # Compute K_test_train
        Kptp = lx.MatrixLinearOperator(self.kernel_P.cross_covariance(P_test, self.P))
        Kxtx = lx.MatrixLinearOperator(self.kernel_X.cross_covariance(X_test, self.X))
        K_test_train = lx.KroneckerLinearOperator(Kptp, Kxtx)

        # Parse 'return_covariance_type' input
        mapping = {"dense": True, "diagonal": False}
        if return_covariance_type.lower() not in mapping:
            msg = f'''return_covariance_type must be 'dense' or 'diagonal',
            got {return_covariance_type}'''
            raise ValueError(msg)
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

            self.kL, self.kLX, self.LkL = self.compute_functional_matrices(use_cached_functionals)
            kLXt = lx.MatrixLinearOperator(jax.vmap(self.kL)(X_test))
            kLZ = lx.KroneckerLinearOperator(Kptp.transpose(), self.kLX)
            LkLZ = lx.KroneckerLinearOperator(self.Kptpt, self.LkL)
            variance = lx.KroneckerLinearOperator(
                lx.DiagonalLinearOperator(jnp.ones(len(P_test))),
                lx.DiagonalLinearOperator(self.Tau))
            L_21 = self.solve_with_L11(kLZ).transpose()
            S = LkLZ + variance - L_21 @ L_21.transpose()
            S = add_jitter(S, jitter).as_matrix()
            L_22 = cholesky(S, lower=True)
            L_22 = lx.MatrixLinearOperator(L_22, lx.lower_triangular_tag)
            solve_with_L22 = generate_solver(L_22)
            new_y = jnp.kron(jnp.ones((len(P_test),1)), self.y_functional)
            mLZ = jnp.kron(self.mean_function_P(P_test), self.functional(lambda x: self.mean_function_X(jnp.atleast_2d(x)).squeeze())[:,None])
            residual_functional = new_y - mLZ
            K_test_functional = lx.KroneckerLinearOperator(self.Kptpt, kLXt)

            L_inv_res = solve_block_triangular(self.solve_with_L11, L_21, solve_with_L22, self.residual_data, residual_functional)
            L_inv_K_train_test = solve_block_triangular(self.solve_with_L11, L_21, solve_with_L22, K_test_train.transpose(), K_test_functional.transpose())

            mean_update = L_inv_K_train_test[0].transpose() @ L_inv_res[0] + L_inv_K_train_test[1].mT @ L_inv_res[1]

            if dense:
                cov_update = L_inv_K_train_test[0].transpose() @ L_inv_K_train_test[0]
                cov_update = cov_update + lx.MatrixLinearOperator(L_inv_K_train_test[1].mT @ L_inv_K_train_test[1])
            else:
                cov_update = L_inv_K_train_test[0].squared_sum()
                cov_update = cov_update + lx.DiagonalLinearOperator(jnp.sum(L_inv_K_train_test[1]**2, axis=0))

        prior_mean = self.prior_mean(P_test, X_test)
        mean = prior_mean + mean_update
        cov = self.prior_cov(X_test, dense) - cov_update

        return GaussianDistribution(loc=jnp.atleast_1d(mean.squeeze()), scale=cov)

    def prior_mean(self, P_test, X_test):
        mean_P = self.mean_function_P(P_test)
        mean_X = self.mean_function_X(X_test)
        prior_mean = jnp.kron(mean_P, mean_X)
        return prior_mean

    def prior_cov(self, X_test, dense):
        if dense:
            Kptpt = self.Kptpt
            Kxtxt = self.kernel_X.gram(X_test)
        else:
            Kptpt = lx.DiagonalLinearOperator(self.Kptpt.as_matrix().diagonal())
            Kxtxt = self.kernel_X.diagonal(X_test)
        prior_cov = lx.KroneckerLinearOperator(Kptpt, Kxtxt)
        return prior_cov

    def __call__(
        self,
        p_test: Num[Array, "D"],
        jitter=1e-6,
        *,
        return_covariance_type: Literal["dense", "diagonal"] = "dense",
        use_cached_functionals: bool = False,
    ) -> GaussianDistribution:
        r"""Return a function that maps test inputs from X to the posterior distribution at given inputs (p_test, X_test),
            taking all previous conditionings into account.

        Args:
            p_test: Single inference point on domain P.
            jitter (float): A small constant added to the diagonal of the
                covariance matrix to ensure numerical stability.

        Returns:
            Callable mapping X_test to a Gaussian distribution over values at test inputs.
        """
        return lambda X_test: self.predict(
            jnp.atleast_2d(p_test),
            X_test,
            jitter=jitter,
            return_covariance_type=return_covariance_type,
            use_cached_functionals=use_cached_functionals,
        )

def debug_print(mat, name):
    mat = mat.as_matrix() if isinstance(mat, lx.AbstractLinearOperator) else mat
    print(f"{name}.shape: {mat.shape}")
    print(f"{name} min eig: {jnp.linalg.eigvalsh(mat).min()}")
    print(f"{name} cond: {jnp.linalg.cond(mat)}")
    print(f"{name} symmetric: {(mat - mat.T).max()  < 1e-8}") 