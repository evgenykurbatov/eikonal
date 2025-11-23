
from tqdm import tqdm

import matplotlib.pyplot as plt

from _context import eikonal
from eikonal.components import *
from eikonal.solver import *

import util


def S(x):
    """Smoothed step function"""
    return 0.5 + 0.5*jax.scipy.special.erf(x/smo)



if __name__ == "__main__":
    wv = 0.600
    k = 2*jnp.pi/wv
    # Background refraction index
    n0 = 1.
    # Spatial grids
    L = 100 * wv
    x = jnp.asarray(0.)
    z = jnp.linspace(0., L, 500)
    # Profile smoothing
    smo = wv / 2

    # Initial state of the field
    u = [1 + 0j]
    v = [0 + 1j*k*n0]

    def dz_func(z, u, v):
        # Constant spatial step in z
        #return smo
        return wv/(2*jnp.pi*n0)

    def f_func(z, u):
        # Refractive layer
        n1 = 1.5
        #n = n0 + (n1 - n0) * S(z - L/2)
        n = n0 + (n1 - n0) * S(z - L/3) * S(2*L/3 - z)
        return (k*(n + 0j))**2

    #
    # Solve

    # Warming-up
    evolve_to(0*u[0], 0*v[0], 0., smo, dz_func, f_func)

    # Evolve
    for l in tqdm(range(1, len(z))):
        u_, v_ = evolve_to(u[l-1], v[l-1], z[l-1], z[l], dz_func, f_func)
        u.append(u_)
        v.append(v_)
    u = jnp.array(u)
    v = jnp.array(v)
    print("u.shape:", u.shape)

    w = jnp.exp(u)

    #import sys
    #sys.exit(0)

    #
    # Plot

    _, ax = util.fig_init((4, 1), dims=(8, 4))

    ax_ = ax[0]
    ax_.plot(z, u.real, '-o')
    ax_.set_xlabel("z [um]")

    ax_ = ax[1]
    ax_.plot(z, u.imag, '-o')
    ax_.set_xlabel("z [um]")

    ax_ = ax[2]
    ax_.plot(z, v.real, '-o')
    ax_.set_xlabel("z [um]")

    ax_ = ax[3]
    ax_.plot(z, v.imag, '-o')
    ax_.set_xlabel("z [um]")

    plt.show()
    plt.close()
