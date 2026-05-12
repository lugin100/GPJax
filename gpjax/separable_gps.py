import beartype.typing as tp
from typing import Literal
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import (
    qr,
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


class SeparablePosterior(tp.Generic[P, L]):
    r"""Posterior to a separable GP prior with Gaussian likelihood."""

    prior: SeparablePrior
    likelihood: tp.Any
    mean_function_A: tp.Any
    mean_function_B: tp.Any
    kernel_A: tp.Any
    kernel_B: tp.Any
    L: tp.Any
    residual_list: tp.Any
    K_test_condition_list: tp.Any
    computations: tp.Any
    A: tp.Any
    B: tp.Any

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

        self.L_11 = None
        self.L_12 = None
        self.L_22 = None
        self.residual_list = []
        self.computations = []


    def condition_on_data(self, train_data: SeparableDataset, jitter=1e-6):
        r"""Condition the posterior on data.

        Args:
            train_data (SeparableDataset): Data to condition on.
        """
        if self.L_11 is not None:
            raise ValueError("Can only condition on data once")

        self.A = train_data.A
        self.B = train_data.B

        def condition_using_test_points(Kata, Kbtb, Katat, A_test, B_test):

            # Compute L
            Kaa = add_jitter(self.kernel_A.gram(self.A).as_matrix(), jitter)
            Kbb = add_jitter(self.kernel_B.gram(self.B).as_matrix(), jitter)
            self.L_11 = _compute_Kronecker_Cholesky(Kaa, Kbb)

            # Append residuals
            residual = train_data.y - jnp.kron(self.mean_function_A(self.A), self.mean_function_B(self.B))
            self.residual_list.append(residual)

            # Append K_test_conditions
            self.K_test_train = jnp.kron(Kata, Kbtb)
            

        self.computations.append(condition_using_test_points)


    def condition_on_functional(self, functional, y, jitter=1e-6):
        if len(self.computations) == 0:
            raise ValueError("Must call condition_on_functional after condition_on_data")

        def condition_using_test_points(Kata, Kbtb, Katat, A_test, B_test):
            LkB = functional(lambda b: self.kernel_B.cross_covariance(b, self.B))        
            LkLB = functional(lambda b: functional(lambda b_prime: self.kernel_B.cross_covariance(b, b_prime)))
            LkZ = jnp.kron(Kata, LkB)
            kLZ = LkZ.mT
            LkLZ = jnp.kron(Katat, LkLB)

            # Update L
            self.L_21 = stable_solve_triangular(self.L_11, kLZ).mT
            S = LkLZ - self.L_21 @ self.L_21.mT
            S = add_jitter(S, jitter)
            eigvals = jnp.linalg.eigvalsh(S)
            print("Min Eigenval of S: ", eigvals.min())
            self.L_22 = cholesky(S)

            # Append residuals
            new_y = jnp.tile(y, (A_test.shape[0],1))
            mLZ = jnp.kron(self.mean_function_A(A_test), functional(self.mean_function_B))
            self.residual_list.append(new_y - mLZ)

            # Append K_test_conditions
            kLBt = functional(lambda b_prime: self.kernel_B.cross_covariance(B_test, b_prime))
            self.K_test_condition = jnp.kron(Katat, kLBt)
            

        self.computations.append(condition_using_test_points)


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
        if len(self.computations) == 0:
            raise ValueError("Cannot predict using an unconditioned posterior")

        #noise = self.likelihood.noise_vector(train_data.n)

        # Kernel computations
        Kata = self.kernel_A.cross_covariance(test_inputs_A, self.A)
        Kbtb = self.kernel_B.cross_covariance(test_inputs_B, self.B)
        Katat = self.kernel_A.gram(test_inputs_A)
        Kbtbt = self.kernel_B.gram(test_inputs_B)

        # Evaluate lazy conditioning now
        [computation(Kata, Kbtb, Katat.as_matrix(), test_inputs_A, test_inputs_B) for computation in self.computations]
        L_12 = jnp.zeros_like(self.L_21.mT)
        L = jnp.block([[self.L_11, L_12], [self.L_21, self.L_22]])
        L = add_jitter(L, jitter)
        print("cond(L): ", jnp.linalg.cond(L))
        K_test_conditions = jnp.concatenate((self.K_test_train,self.K_test_condition), axis=1)

        # Posterior mean
        prior_mean = jnp.kron(self.mean_function_A(test_inputs_A), self.mean_function_B(test_inputs_B)).squeeze()

        r1, r2 = self.residual_list
        res = solve_block_triangular(self.L_11, self.L_21, self.L_22, r1, r2)
        res = K_test_conditions @ res
        mean = prior_mean[:,None] + res
        print("Mean has Nan: ", jnp.any(jnp.isnan(mean)))
        
        # Posterior covariance
        prior_cov = Kronecker(Katat, Kbtbt)

        K1 = self.K_test_train.mT
        K2 = self.K_test_condition.mT
        X = solve_block_triangular(self.L_11, self.L_21, self.L_22, K1, K2)
        X = K_test_conditions @ X

        X = lx.MatrixLinearOperator(X)
        cov = prior_cov - X
        print("Min covariance value: ", cov.as_matrix().min())

        return GaussianDistribution(loc=jnp.atleast_1d(mean.squeeze()), scale=cov)

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


def solve_block_triangular(L11, L21, L22, b1, b2):
    r"""Compute $(L L^T)^{-1} b$ where 
    $L$ is assumed to be a lower-triangular block matrix
    $L = [[L11, 0], [L21, L22]]$ and b is a vector or matrix $[b1, b2]$.
    """
    # Block forward substitution
    y1 = solve_triangular(L11, b1, lower=True)
    y2 = solve_triangular(L22, b2 - L21 @ y1, lower=True)
    # Block backward substitution
    x2 = solve_triangular(L22.mT, y2, lower=False)
    x1 = solve_triangular(L11.mT, y1 - L21.T @ x2, lower=False)
    return jnp.concatenate([x1, x2], axis=0)


def stable_solve_triangular(M, B, **kwargs):
    Q, R = qr(M)
    return solve_triangular(R, Q.T @ B, **kwargs)


def add_jitter(op, jitter):
    if isinstance(op, Array):
        return op + jitter * jnp.eye(op.shape[0])
    if isinstance(op, lx.AbstractLinearOperator):
        return op + jitter * lx.IdentityLinearOperator(op.in_structure())
    else:
        raise ValueError
