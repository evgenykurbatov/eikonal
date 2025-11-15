
import matplotlib.pyplot as plt
import util

from _context import eikonal
from eikonal.components import *



if __name__ == "__main__":
    wv = 0.600
    smo = wv/10
    r0 = jnp.array([0., 1.])
    D = 10e3
    H = D/5

    bucket = Components([Slit(r0, D, H, eps=0.14+3.98j),
                         Lens(r0, D=D, c1=1/D, c2=0., H=H, eps=1.5+0j)])

    x = jnp.arange(-D, D, 50*wv)
    z = jnp.arange(-H, 2*H, 50*wv)
    #x = jnp.arange(-D/2-10*smo, -D/2+10*smo, smo/10)
    #z = jnp.arange(r0[1]+H-10*smo, r0[1]+H+10*smo, smo/10)
    X, Z = jnp.meshgrid(x, z)
    R = jnp.stack((X, Z)).T
    print("X.shape:", X.shape)
    print("R.shape:", R.shape)


    _, ax = util.fig_init((1, 2))

    ax_ = ax[0]
    disp = bucket.dispersion(R)
    disp_im = util.complex_to_rgb(disp)
    ax_.imshow(disp_im, extent=(z[0], z[-1], x[-1], x[0]))
    ax_.set_xlabel("z [um]")
    ax_.set_ylabel("x [um]")

    ax_ = ax[1]
    edge = jnp.zeros_like(X.T, dtype=bool)
    for comp in bucket.bucket:
        edge |= comp.edge(R)
    im = ax_.imshow(edge, extent=(z[0], z[-1], x[-1], x[0]))
    plt.colorbar(im, ax=ax_)
    ax_.set_xlabel("z [um]")
    ax_.set_ylabel("x [um]")

    plt.show()
    plt.close()
