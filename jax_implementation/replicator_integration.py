import jax

import jax.numpy as jnp
import jax.random as random

import diffrax
import sys
import os
from einops import rearrange, einsum
import time
import matplotlib.pyplot as plt

# Comment out if not on Apple Silicon
os.environ["JAX_PLATFORM_NAME"] = "cpu"
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from matrix_generators import sampleA, computeF, computeB, computeQ
from pop_calculations import evaluateBVec

if __name__ == "__main__":
    key = random.key(0)
    n = 4
    k = 3
    r = n**k // 2 * 2
    dt0 = 0.1
    threshold = 1e-8
    N = 30  # integration resolution
    tspan = (0, 10)  # integration time span
    num_times = 30
    times = jnp.linspace(tspan[0], tspan[1], num=num_times).tolist()
    a, b = (-1, 1)  # trait domain
    evalpoints = jnp.linspace(a, b, num=N)
    # resolution window size
    deltax = (b - a) / (N - 1)

    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=times)
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)

    wvec = deltax * jnp.ones([N])
    wvec = wvec.at[0].set(deltax / 2).at[-1].set(deltax / 2)

    wvec_string = ", ".join("i" + str(i) for i in range(k))
    second_wvec_string = " ".join("i" + str(i) for i in range(k))

    wvec = einsum(*k * [wvec], wvec_string + " -> " + second_wvec_string)
    wvec = rearrange(wvec, second_wvec_string + " -> (" + second_wvec_string + ")")

    A = sampleA(n, k, key)

    F = computeF(A, evalpoints)

    B = computeB(A)

    B_mat = jnp.reshape(B, [n**k, n**k])

    Q, r = computeQ(B_mat, n, k, r)

    b_vec = evaluateBVec(Q, evalpoints, n, r, k)

    initial_parameters = jnp.array(r * [1])
    p0 = jnp.exp(b_vec @ initial_parameters)
    mass = jnp.dot(p0, wvec)

    normalizing_row = jnp.linalg.lstsq(Q, jnp.eye(n**k)[:, 0])[0]
    initial_parameters -= normalizing_row * jnp.log(mass)

    W = jnp.kron(jnp.eye(r // 2), jnp.array([[0, 1], [-1, 0]]))

    gradP = jax.jit(jax.grad(lambda theta: jnp.dot(jnp.exp(b_vec @ theta), wvec)))

    @jax.jit
    def latent_equation(t, theta, *args):
        return W @ gradP(theta)

    @jax.jit
    def discrete_equation(t, p, *args):
        return p * (F @ (p * wvec))

    latent_term = diffrax.ODETerm(latent_equation)
    discrete_term = diffrax.ODETerm(discrete_equation)

    t = time.time()
    latent_sol = diffrax.diffeqsolve(
        latent_term,
        solver=solver,
        t0=tspan[0],
        t1=tspan[1],
        dt0=dt0,
        y0=initial_parameters,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    print(time.time() - t)

    latent_sol_populations = jax.vmap(lambda theta: b_vec @ theta)(
        latent_sol.ys
    )

    t = time.time()
    discrete_sol = diffrax.diffeqsolve(
        discrete_term,
        solver=solver,
        t0=tspan[0],
        t1=tspan[1],
        dt0=dt0,
        y0=jnp.exp(b_vec @ initial_parameters),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    print(time.time() - t)
