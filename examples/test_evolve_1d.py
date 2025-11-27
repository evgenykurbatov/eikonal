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


def solve_diffrax(u0, v0, z_max, dz, k, n0, n, smo):
    """
    The Diffrax ODE solver. Here, the eikonal equation is splitted for real and
    imaginary parts.
    """

    def rhs(z, w, _):
        α_1, α_2, φ_1, φ_2 = w

        cn = refr_index(z, (z_max, n0, n, smo))
        kn, kχ = k*cn.real, k*cn.imag

        dα_1 = α_2
        dα_2 = - α_2**2 + φ_2**2 - kn**2 + kχ**2
        dφ_1 = φ_2
        dφ_2 = - 2 * (α_2*φ_2 + kn*kχ)

        return dα_1, dα_2, dφ_1, dφ_2

    z = jnp.arange(0., z_max, dz)
    w0 = (u0.real, v0.real, u0.imag, v0.imag)

    term = ODETerm(rhs)
    #solver = diffrax.Dopri8()
    #solver = diffrax.Tsit5()
    solver = diffrax.Kvaerno5()
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-6)
    #stepsize_controller = PIDController(rtol=1e-6, atol=1e-7)
    #stepsize_controller = PIDController(rtol=1e-5, atol=1e-7,
    #                                    pcoeff=0.08, icoeff=0.03, dcoeff=0,
    #                                    safety=0.85, factormax=1.8, factormin=0.3,)
    saveat = SaveAt(ts=z)

    def do_solve():
        sol = diffeqsolve(term, solver, z[0], z[-1], 1e-3*dz, w0,
                          max_steps=1_000_000,
                          saveat=saveat, stepsize_controller=stepsize_controller)
        return sol

    sol = jax.jit(do_solve)()
    u = jnp.array(sol.ys[0]) + 1j*jnp.array(sol.ys[2])
    v = jnp.array(sol.ys[1]) + 1j*jnp.array(sol.ys[3])

    return z, u, v


def solve_eikonal(u0, v0, z_max, dz, k, n0, n, smo, core_func, dz_factor=1.0):
    """
    Eikonal solver wrapper.
    """
    saveat = jnp.arange(0., z_max, dz)

    def k_func(z):
        cn = refr_index(z, (z_max, n0, n, smo))
        return k * cn

    def perp_func(z, u):
        return jnp.zeros_like(u)

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
    # Spatial grids
    x = jnp.asarray(0.)
    z_max = 100*wv
    dz = 0.05  # [um]
    #dz = wv/100
    z = jnp.arange(0., z_max, dz)
    # Profile smoothing
    smo = 4*dz
    #smo = 4*wv

    # Initial state of the field
    u0 = 0 + 0j
    v0 = 0 + 1j*k*n0

    #
    # Run

    print("\n* Diffrax solver")
    # Run
    t0 = time.time()
    dfx = Data("Diffrax")
    dfx.z, dfx.u, dfx.v = solve_diffrax(u0, v0, z_max, dz, k, n0, n, smo)
    print("dfx.z.shape", dfx.z.shape)
    print("dfx.u.shape", dfx.u.shape)
    print("dfx.u.dtype", dfx.u.dtype)
    print("time:", time.time() - t0)

    print("\n* Eikonal full solver")
    # Warming-up
    solve_eikonal(0*u0, 0*v0, z_max, dz, 0*k, n0, n, smo, solver.core_cn)
    # Run
    t0 = time.time()
    eik0 = Data("Eikonal full")
    eik0.z, eik0.u, eik0.v = solve_eikonal(u0, v0, z_max, dz, k, n0, n, smo, solver.core_cn, dz_factor=0.1)
    print("eik0.z.shape", eik0.z.shape)
    print("eik0.u.shape", eik0.u.shape)
    print("eik0.u.dtype", eik0.u.dtype)
    print("time:", time.time() - t0)

    print("\n* Eikonal SVEA solver")
    # Warming-up
    solve_eikonal(0*u0, 0*v0, z_max, dz, 0*k, n0, n, smo, solver.core_svea)
    # Run
    t0 = time.time()
    svea = Data("Eikonal SVEA")
    svea.z, svea.u, svea.v = solve_eikonal(u0, v0, z_max, dz, k, n0, n, smo, solver.core_svea)
    print("svea.z.shape", svea.z.shape)
    print("svea.u.shape", svea.u.shape)
    print("svea.u.dtype", svea.u.dtype)
    print("time:", time.time() - t0)

    #
    # Plot

    _, ax = util.fig_init((4, 1), dims=(10, 4))

    ax_ = ax[0]
    ax_.set_title("u.real")
    ax_.axhline(u0.real, ls=':', c='gray')
    ax_.axhline(u0.real - 0.5*jnp.log(n/n0), ls=':', c='gray')
    ax_.plot(dfx.z/wv, dfx.u.real, 'k', label=dfx.name)
    ax_.plot(eik0.z/wv, eik0.u.real, '-', label=eik0.name)
    ax_.plot(svea.z/wv, svea.u.real, '-o', label=svea.name)
    ax_.legend(frameon=False)

    ax_ = ax[1]
    ax_.set_title("u.imag")
    ax_.plot(dfx.z/wv, dfx.u.imag, 'k', label=dfx.name)
    ax_.plot(eik0.z/wv, eik0.u.imag, '-', label=eik0.name)
    ax_.plot(svea.z/wv, svea.u.imag, '-o', label=svea.name)
    ax_.legend(frameon=False)

    ax_ = ax[2]
    ax_.set_title("v.real")
    ax_.plot(dfx.z/wv, dfx.v.real, 'k', label=dfx.name)
    ax_.plot(eik0.z/wv, eik0.v.real, '-', label=eik0.name)
    ax_.legend(frameon=False)

    ax_ = ax[3]
    ax_.set_title("v.imag")
    ax_.axhline(k*n0, ls=':', c='gray')
    ax_.axhline(k*n, ls=':', c='gray')
    ax_.plot(dfx.z/wv, dfx.v.imag, 'k', label=dfx.name)
    ax_.plot(eik0.z/wv, eik0.v.imag, '-', label=eik0.name)
    ax_.legend(frameon=False)



    for ax_ in ax.ravel():
        ax_.set_xlabel("z/λ")

    plt.show()
    plt.close()