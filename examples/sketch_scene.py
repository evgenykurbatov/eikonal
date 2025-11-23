
import matplotlib.pyplot as plt
import util

from _context import eikonal
from eikonal.components import *
from eikonal.solver import *



if __name__ == "__main__":

    #
    # Set up

    wv = 0.600
    k = 2*jnp.pi/wv
    n = 1.5
    smo = wv/10

    r0 = jnp.array([0., 1.])
    D = 10e3
    H = D/5
    lens = Lens(r0, D=D, c1=1/D, c2=0., H=H, refr_index=n+0j)
    f, FFD, BFD = lens.focals()
    print("lens f, FFD, BFD:", f, FFD, BFD)

    bucket = Components([lens], smo_default=smo)

    #Delta = 10*wv
    Delta = wv/10
    x = jnp.arange(-0.6*D, 0.6*D, 50*wv)
    z = jnp.arange(-H, 1.5*f, Delta)
    X, Z = jnp.meshgrid(x, z)
    R = jnp.stack((X, Z))
    print("X.shape:", X.shape)
    print("R.shape:", R.shape)

    # Initial state of the field
    #u0 = jnp.full_like(x, 1.+0j, dtype=complex)
    #v0 = jnp.full_like(x, 1j*k*n, dtype=complex)
    u0 = jnp.where(jnp.abs(x) < D/2, 1.+0j, 0j)
    v0 = jnp.where(jnp.abs(x) < D/2, 1j*k*n, 0j)

    # Source of refraction
    k2eps = (k * bucket.dispersion(R).T)**2

    #
    # Solve

    # Warming-up
    jax.jit(evolve)(0*u0, 0*v0, z, 0*refr_index2)
    # Evolve
    u, v = jax.jit(evolve)(u0, v0, z, refr_index2)

    w = jnp.exp(u)

    #
    # Plot

    _, ax = util.fig_init((3, 1))

    ax_ = ax[0]
    edge = jnp.zeros_like(X.T, dtype=bool)
    for comp in bucket.bucket:
        edge |= comp.edge(R)
    im = ax_.pcolormesh(Z, X, edge)
    plt.colorbar(im, ax=ax_)
    ax_.set_xlabel("z [um]")
    ax_.set_ylabel("x [um]")

    ax_ = ax[1]
    #im = ax_.imshow(jnp.abs(w.T)**2, extent=(z[0], z[-1], x[-1], x[0]))
    im = ax_.pcolormesh(Z, X, jnp.abs(w.T)**2)
    plt.colorbar(im, ax=ax_)
    ax_.set_xlabel("z [um]")
    ax_.set_ylabel("x [um]")

    ax_ = ax[2]
    #ax_.plot(z, u[:,200].real, label="Re")
    ax_.plot(z, u[:,200].imag, label="Im")
    ax_.legend(frameon=True)
    ax_.set_xlabel("z [um]")

    plt.show()
    plt.close()
