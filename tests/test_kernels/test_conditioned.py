import jax
import jax.numpy as jnp
from gpjax.kernels import ConditionedKernel
from gpjax.kernels.stationary import Matern32
from jax.scipy.linalg import cholesky


key = jax.random.PRNGKey(0)

n_train = 5
n_test = 40
dim = 2

key1, key2 = jax.random.split(key)

X_test = jax.random.normal(key1, (n_train, dim))
X_train = jax.random.normal(key2, (n_train, dim))

# Base kernel
k = Matern32(
    lengthscale=jnp.array(1.5),
    variance=jnp.array(2.0),
)

def test_conditioned_kernel_symmetric():

    kL = lambda x: jax.vmap(lambda y: k(x,y))(X_train)
    LkL = k.gram(X_train).as_matrix()
    L = cholesky(LkL, lower=True)
    
    sut = ConditionedKernel(k, L, kL)
    G = sut.gram(X_test).as_matrix()

    # check symmetry
    assert jnp.max(jnp.abs(G - G.T)) < 1e-12


def test_conditioned_kernel_psd():
    
    kL = lambda x: jax.vmap(lambda y: k(x,y))(X_train)
    LkL = k.gram(X_train).as_matrix()
    L = cholesky(LkL, lower=True)
    print(L)
    sut = ConditionedKernel(k, L, kL)
    G = sut.gram(X_test).as_matrix()

    # check psd-ness
    eigvals = jnp.linalg.eigvalsh(G)
    assert eigvals.min() > 0

def test_conditioned_kernel_functional_symmetric():
    
    functional = lambda u: jax.vmap(u)(X_train)

    kL = lambda x: functional(lambda y: k(x, y))
    LkL = functional(lambda x: functional(lambda y: k(x, y)))
    L = cholesky(LkL, lower=True)
    
    sut = ConditionedKernel(k, L, kL)
    G = sut.gram(X_test).as_matrix()

    # check symmetry
    assert jnp.max(jnp.abs(G - G.T)) < 1e-12


def test_conditioned_kernel_functional_psd():
    
    functional = lambda u: jax.vmap(u)(X_train)

    kL = lambda x: functional(lambda y: k(x, y))
    LkL = functional(lambda x: functional(lambda y: k(x, y)))
    L = cholesky(LkL, lower=True)
    
    sut = ConditionedKernel(k, L, kL)
    G = sut.gram(X_test).as_matrix()

    # check psd-ness
    eigvals = jnp.linalg.eigvalsh(G)
    assert eigvals.min() > 0
