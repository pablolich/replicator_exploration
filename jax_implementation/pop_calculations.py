import jax

import jax.numpy as jnp
import jax.scipy as jspy
import jax.random as random
import numpy as np
import equinox as eqx

from einops import rearrange, einsum

import sys
import os

from matrix_generators import computeT

# Comment out if not on Apple Silicon
os.environ["JAX_PLATFORM_NAME"] = "cpu"
jax.config.update("jax_platform_name", "cpu")

@eqx.filter_jit
def evaluateBVec(Q, evalpoints, n, r, k):
    eval_poly = jnp.vander(evalpoints, n, increasing=True)

    M = computeT(n).transpose() @ eval_poly.transpose()

    Q_mat_string = "(" + "".join("k" + str(l) + " " for l in range(k))[:-1] + ") j"
    Q_tensor_string = "".join("k" + str(l) + " " for l in range(k))[:-1] + " j"

    labels_dict = {"k" + str(l): n for l in range(k)}

    Q = rearrange(Q, Q_mat_string + " -> " + Q_tensor_string, **labels_dict)

    M_strings = "".join("k" + str(l) + " i" + str(l) + ", " for l in range(k))[:-2]

    out_tensor_string = "".join("i" + str(l) + " " for l in range(k))[:-1] + " j"
    out_mat_string = "(" + "".join("i" + str(l) + " " for l in range(k))[:-1] + ") j"

    full_string = Q_tensor_string + ", " + M_strings + " -> " + out_tensor_string

    b_vec = einsum(
        Q,
        *k * [M],
        full_string,
    )

    b_vec = rearrange(b_vec, out_tensor_string + " -> " + out_mat_string)

    return b_vec
