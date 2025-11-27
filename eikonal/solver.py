"Backward Euler solver for complex eikonal equation."

from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import Float, Complex, Array


def core_be(u, v, dz, k, k_, q):
    """
    Solves the complex ODE $dw/dz + w^2 + k^2 + q = 0$ using the Backward Euler
    scheme on predefined points, zn.
    """
    f_  = k_**2 + q
    v_ = (-1 + jnp.sqrt(1 - 4*dz * (dz*f_ - v))) / (2*dz)
    u_ = u + dz * v_
    return u_, v_


def core_cn(u, v, dz, k, k_, q):
    """
    Solves the complex ODE $dw/dz + w^2 + k^2 + q = 0$ using the Crank-Nicolson
    method on predefined points, zn.
    (v_ - v)/dz + [(v + v_)/2]^2 + f = 0
    """
    f = 0.5*(k**2 + k_**2) + q
    v_ = ( - (2 + dz*v) + jnp.sqrt( (2 + dz*v)**2 + 4*dz*v - dz**2*(v**2 + 4*f) ) ) / dz
    u_ = u + dz * 0.5*(v_ + v)
    return u_, v_


def core_cn_alt(u, v, dz, k, k_, q):
    """
    Solves the complex ODE $dw/dz + w^2 + k^2 + q = 0$ using the Crank-Nicolson
    method on predefined points, zn.
    (v_ - v)/dz + (v^2 + v_^2)/2 + f = 0
    """
    f = 0.5*(k**2 + k_**2) + q
    v_ = ( -1 + jnp.sqrt( 1 + 2*dz*v - dz**2*(v**2 + 2*f) ) ) / dz
    u_ = u + dz * 0.5*(v_ + v)
    return u_, v_


def core_svea(u, v, dz, k, k_, q):
    """
    Solves the complex ODE $dw/dz + w^2 + k^2 + q = 0$ using the SVEA method.
    """
    dphi   = jnp.sqrt((k**2).real)
    dalpha = - 0.25/dz * jnp.log((k_**2).real / (k**2).real) - 0.5*(k**2).imag

    u_ = u + (dalpha + 1j*dphi) * dz
    v_ = jnp.empty_like(u_)
    return u_, v_


@partial(jax.jit, static_argnames=('dz_func', 'k_func', 'perp_func', 'core_func'))
def propagate(
              u0: Complex[Array, "M"],
              v0: Complex[Array, "M"],
              saveat: Float[Array, "N"],
              dz_func,
              k_func,
              perp_func,
              core_func):
    """
    Solves the complex ODE $dw/dz + w^2 = rhs(z, u)$ using a given step function.

    **Parameters**:
    u0, v0: array of complex
        Initial values of u and v fields at z=0.
    saveat: array of float
        Points in Z to save the fields at.
    dz_func: function
        A function that takes the current z, u, and v and returns the step size dz.
    k_func: function
        A function that returns a complex wavenumber, $k (n + i\\chi)$.
    perp_func: function
        A function that returns perpendicular (transverse) sources.
    core_func: function
        A function that performs a single step.

    **Returns**:
    tuple of arrays
        Solutions at the points `saveat`.
    """
    def scan_fun(val, z_):
        u, v, z = val
        k = k_func(z)
        q = perp_func(z, u)

        def cond_fun(val):
            _, _, z_prev, _, _ = val
            return z_prev < z_

        def body_fun(val):
            u_prev, v_prev, z_prev, k_prev, q_prev = val
            dz = dz_func(z_prev, u_prev, v_prev)
            z_next = jnp.minimum(z_prev + dz, z_)
            dz = z_next - z_prev

            k_next = k_func(z_next)

            u_next, v_next = core_func(u_prev, v_prev, dz, k_prev, k_next, q_prev)

            q_next = perp_func(z_next, u_next)

            return u_next, v_next, z_next, k_next, q_next

        init_while = (u, v, z, k, q)
        u, v, z, k, q = jax.lax.while_loop(cond_fun, body_fun, init_while)
        return (u, v, z), (u, v)

    init_scan = (u0, v0, saveat[0])
    _, (u, v) = jax.lax.scan(scan_fun, init_scan, saveat[1:])

    return jnp.concatenate([jnp.array([u0]), u]), jnp.concatenate([jnp.array([v0]), v])
