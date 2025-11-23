"""
Backward Euler solver for complex eikonal equation.
"""

from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array


@jax.jit
def evolve(u0: Complex[Array, "M"],
           v0: Complex[Array, "M"],
           zn: Float[Array, "N"],
           fn: Complex[Array, "N M"]):
    """
    Solves the complex ODE $dw/dz + w^2 + f(z) = 0$ using the Backward Euler
    method on predefined points, zn, and source values at these points, fn. This
    version is JIT-accelerated.

    **Parameters**:
    u0, v0: array of complex
        Initial values of u and v fields at z=0.
    zn: array of float
        Sequence of points in Z to evolve the fields to.
    fn: array of complex
        Sequence of sources in Z to evolve the fields to. The elements must be
        broadcastable with v.

    **Returns**:
    tuple of arrays
        Solutions at all the input points.
    """
    def scan_fun(val, slice):
        u, v, z, f = val
        z_, f_ = slice

        dz = z_ - z
        #f__ = 0.5 * (f + f_)  # Worse
        f__ = f_              # Better

        v_ = (-1 + jnp.sqrt(1 - 4*dz * (dz*f__ - v))) / (2*dz)
        #u_ = u + dz * 0.5*(v_ + v)  # Same
        u_ = u + dz * v_            # Same

        return (u_, v_, z_, f_), (u_, v_)

    init_val = (u0, v0, zn[0], fn[0])
    _, (u, v) = jax.lax.scan(scan_fun, init_val, (zn[1:], fn[1:]))

    return jnp.concatenate([jnp.array([u0]), u]), jnp.concatenate([jnp.array([v0]), v])


@partial(jax.jit, static_argnames=('dz_func', 'f_func'))
def evolve_to(u0: Complex[Array, "M"],
              v0: Complex[Array, "M"],
              z0, z_max,
              dz_func,
              f_func):
    """
    Solves the complex ODE $dw/dz + w^2 + f(z) = 0$ using the Backward Euler
    method. This version is JIT-accelerated.

    **Parameters**:
    u0, v0: array of complex
        Initial values of u and v fields at z=0.
    z0, z_max: float
        The initial and maximum values of z to solve for.
    dz_func: function (z, u, v) -> dz
        A function that takes the current z, u, and v and returns the step size dz.
    f_func: function (z, u) -> f
        A function ther returns the source at a given position.

    **Returns**:
    tuple of arrays
        Solutions at the point `z_max`.
    """

    def cond_fun(val):
        _, _, z, _ = val
        return z < z_max

    def body_fun(val):
        u, v, z, f = val
        dz = dz_func(z, u, v)
        z_ = jnp.minimum(z + dz, z_max)
        f_ = f_func(z_, u)

        u__, v__ = evolve(u, v, jnp.array([z, z_]), jnp.array([f, f_]))

        return u__[-1], v__[-1], z_, f_

    init_val = (u0, v0, z0, f_func(z0, u0))
    u, v, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)

    return u, v
