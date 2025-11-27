"""
In this test, we run three solvers: one have been implemented using [Diffrax](https://github.com/patrick-kidger/diffrax), and two are of the Eikonal library. Two of them solve the wave propagation problem at high spatial resolution, and the third does it at a lower resolution.
"""

import time

import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt
from diffrax import PIDController, Dopri5, Tsit5, Kvaerno3

import matplotlib.pyplot as plt

from _context import eikonal
from eikonal.components import *
from eikonal import solver
import util


# A placeholder for data
class Data:
    def __init__(self, name=None):
        self.name = name


def S(x, smo):
    """Smoothed step function"""
    return 0.5 + 0.5*jax.scipy.special.erf(x/smo)


def refr_index(z, args):
    """Complex refraction index distribution"""
    z_max, n0, n, smo = args
    return n0 + (n - n0) * S(z - z_max/3, smo) * S(2*z_max/3 - z, smo) + 0j
    #return n0


def solve(u0, v0, x, z_max, dz, k, n0, n, smo, core_func, dz_factor=1.0, sigma=0.0):
    """
    Eikonal solver wrapper.
    """
    saveat = jnp.arange(0., z_max, dz)
    dx = x[1] - x[0]

    """
    def k_func(z):
        # Base part of refractive index
        cn = refr_index(z, (z_max, n0, n, smo))

        # Define absorbing boundary layer parameters
        boundary_width_ratio = 0.15 # 15% of the domain on each side
        chi_max = 100 / k # Absorption strength

        domain_width = x[-1] - x[0]
        boundary_width = domain_width * boundary_width_ratio

        # Distance from the start of the absorbing layer (inside the domain)
        dist_from_edge = jnp.maximum(0, jnp.abs(x) - (domain_width / 2 - boundary_width))

        # Quadratic ramp for absorption
        chi = chi_max * (dist_from_edge / boundary_width)**2

        # Combine both parts for the full complex refractive index
        return k * (cn + 1j*chi)
    """

    def k_func(z):
        # Base part of refractive index
        cn = refr_index(z, (z_max, n0, n, smo))
        # Distance from the start of the absorbing layer (inside the domain)
        dom = x[-1] - x[0]
        dist = jnp.abs(x) - 0.5*dom
        return k * (cn + jnp.where(dist < 0.1*dom, 1j*n0/2, 0j))
        #return k * cn

    def perp_func(z, u):
        q_half = jnp.diff(u) / dx

        q_right = jnp.zeros_like(u)
        q_left  = jnp.zeros_like(u)

        q_right = q_right.at[:-1].set(q_half)
        q_left  = q_left .at[1:] .set(q_half)

        q_left  = q_left .at[0] .set(sigma * q_half[0])
        q_right = q_right.at[-1].set(sigma * q_half[-1])

        perp = (q_right - q_left) / dx + (0.5*(q_right + q_left))**2
        #perp = (q_right - q_left) / dx + 0.5*(q_right**2 + q_left**2)
        return perp

    def dz_func(z, u, v):
        return dz * dz_factor

    u, v = solver.propagate(u0, v0, saveat, dz_func, k_func, perp_func, core_func)

    return saveat, u, v


if __name__ == "__main__":
    wv = 0.632  # [um]
    k = 2*jnp.pi/wv
    # Background refraction index
    n0 = 1.
    # Refractive layer
    n = 1.5
    # Boundary condition parameter
    sigma = 1.0  # Free flow
    #sigma = 0.0  # Adiabatic bounds
    #sigma = -1.0  # Reflecting bounds
    # Spatial grids
    D = 10*wv
    dx = 0.05  # [um]
    x = jnp.arange(-D/2, D/2, dx)
    #z_max = 100 * wv
    #dz = 0.05  # [um]
    z_max = 15 * wv
    dz = z_max / 1000
    z = jnp.arange(0., z_max, dz)
    print("x.shape:", x.shape)
    print("z.shape:", z.shape)

    # Profile smoothing
    smo = 4*dz
    #smo = 4*wv

    #
    # Initial state of the field

    # Making a lens
    f = 100 * wv
    #u0 = 0 - 1j * k * x**2 / (2*f)
    D_lens = 0.9*D/2
    #u0 = 0 - 1j * k / (2*f) * jnp.where(jnp.abs(x) < D_lens, x**2, D_lens**2)
    u0 = 0 - 1j * k / (2*f) * jnp.minimum(x**2, D_lens**2)
    v0 = jnp.full_like(x, 0 + 1j*k*n0, dtype=complex)

    #
    # Run

    print("\n* Eikonal Crank-Nicolson full solver")
    # Warming-up
    solve(0*u0, 0*v0, x, z_max, dz, k, n0, n, smo, solver.core_cn, sigma=sigma)
    # Run
    t0 = time.time()
    data = Data("Eikonal full")
    data.z, data.w, data.dw = solve(u0, v0, x, z_max, dz, k, n0, n, smo, solver.core_cn, dz_factor=1e-3, sigma=sigma)
    data.u = jnp.exp(data.w)
    print("data.z.shape:", data.z.shape)
    print("data.u.shape:", data.u.shape)
    print("data.u.dtype:", data.u.dtype)
    print("time:", time.time() - t0)

    #import sys
    #sys.exit(0)

    #
    # Plot

    _, ax = util.fig_init((4, 1), dims=(10, 4))

    #X, Z = jnp.meshgrid(x, dfx.z)

    X, Z = jnp.meshgrid(x, data.z)

    ax_ = ax[0]
    ax_.set_title("Amplitude")
    im = ax_.pcolormesh(Z/wv, X/wv, jnp.abs(data.u)**2, cmap='inferno')
    plt.colorbar(im, ax=ax_)

    ax_ = ax[1]
    ax_.set_title("Phase")
    ax_.pcolormesh(Z/wv, X/wv, jnp.angle(data.u), cmap='twilight')

    N_max = 40
    ax_ = ax[2]
    ax_.plot(data.z[:N_max]/wv, data.u[:,0][:N_max].real)
    ax_.plot(data.z[:N_max]/wv, data.u[:,-1][:N_max].real)

    ax_ = ax[3]
    ax_.plot(data.z[:N_max]/wv, data.u[:,0][:N_max].imag)
    ax_.plot(data.z[:N_max]/wv, data.u[:,-1][:N_max].imag)

    for ax_ in ax.ravel():
        ax_.set_xlabel("z/λ")
        #ax_.set_ylabel("x/λ")

    plt.show()
    plt.close()
