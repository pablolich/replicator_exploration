import jax

import jax.numpy as jnp
import jax.scipy as jspy
import jax.random as random
import numpy as np
import equinox as eqx

from einops import rearrange, einsum

import sys
import os

# Comment out if not on Apple Silicon
os.environ["JAX_PLATFORM_NAME"] = "cpu"
jax.config.update("jax_platform_name", "cpu")


# @eqx.filter_jit
def sampleA(n, k, key):
    T = random.normal(key, shape=(2 * k) * [n])
    A = T - jnp.transpose(T, jnp.concatenate([jnp.arange(k, 2 * k), jnp.arange(k)]))

    normalization = jnp.ogrid[2 * k * [range(1, n + 1)]]

    for normalizer in normalization:
        A = A / normalizer

    return A


@jax.jit
def computeF(A, evalpoints):
    F = A
    n, k = A.shape[0], len(A.shape) // 2
    V = jnp.vander(evalpoints, n, increasing=True)

    for _ in range(2 * k):
        F = jnp.tensordot(F, V, axes=[0, 1])

    return rearrange(
        F,
        " ".join("i" + str(i) for i in range(k))
        + " "
        + " ".join("j" + str(i) for i in range(k))
        + " -> "
        + "("
        + " ".join("i" + str(i) for i in range(k))
        + ") "
        + "("
        + " ".join("j" + str(i) for i in range(k))
        + ")",
    )


@eqx.filter_jit
def computeT(n):
    def leg_coeffs(coeffs):
        new_coeffs = np.polynomial.legendre.leg2poly(coeffs)
        return np.concatenate([new_coeffs, np.zeros(n - len(new_coeffs))])

    return jnp.array(list(map(leg_coeffs, np.eye(n)))).transpose()


@jax.jit
def computeB(A):
    n, k = A.shape[0], len(A.shape) // 2
    T = computeT(n)
    inv_T = jspy.linalg.solve_triangular(T, jnp.eye(n, n), lower=False)

    B = A

    for _ in range(2 * k):
        B = jnp.tensordot(B, inv_T, axes=[0, 1])

    return B


@eqx.filter_jit
def computeQ(B_mat, n, k, r):
    vals, vecs = jnp.linalg.eig(B_mat)

    imag_vals = jnp.imag(vals)

    indsort = jnp.argsort(imag_vals)

    vals = vals[indsort]
    imag_vals = imag_vals[indsort]
    vecs = vecs[:, indsort]

    vals = jnp.concatenate([vals[: r // 2], vals[: n**k - r // 2 - 1 : -1]])
    vecs = jnp.concatenate([vecs[:, : r // 2], vecs[:, : n**k - r // 2 - 1 : -1]], 1)

    # vals = vals[::-1]
    # vecs = vecs[:, ::-1]

    vals = rearrange(vals, "(two n) -> n two", two=2)
    vals = vals[:, ::-1]
    vals = rearrange(vals, "n two -> (n two)")
    vecs = rearrange(vecs, "i (two n) -> i n two", two=2)
    vecs = vecs[:, :, ::-1]
    vecs = rearrange(vecs, "i n two -> i (n two)")

    M = 1 / jnp.sqrt(2) * jnp.array([[1, -1j], [1, 1j]])

    V = vecs @ jnp.kron(jnp.eye(r // 2), M)

    Q = V * jnp.sqrt(jnp.abs(vals))

    return jnp.real(Q), r
