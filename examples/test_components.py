
import matplotlib.pyplot as plt
import util

from _context import eikonal
from eikonal.components import *



if __name__ == "__main__":
    wv = 0.600
    smo = wv/10
    r0 = jnp.array([0., 0.])
    D = 10e3
    H = D/5

    bucket = Components([Slit(r0, D, H, refr_index=0.14+3.98j),
                         Lens(r0, D=D, c1=1/D, c2=0., H=H, refr_index=1.5+0j)],
                        smo_default=smo)
    print("bucket.smo_default:", bucket.smo_default)

    x = jnp.arange(-D, D, 50*wv)
    z = jnp.arange(-H, 2*H, 50*wv)
    #x = jnp.arange(-D/2-10*smo, -D/2+10*smo, smo/10)
    #z = jnp.arange(r0[1]+H-10*smo, r0[1]+H+10*smo, smo/10)
    print("x.shape:", x.shape)
    print("z.shape:", z.shape)
    # [len(z), len(x)]
    X, Z = jnp.meshgrid(x, z)
    # [2, len(z), len(x)]
    R = jnp.stack((X, Z))
    print("X.shape:", X.shape)
    print("R.shape:", R.shape)


    _, ax = util.fig_init((1, 2))

    ax_ = ax[0]
    refr = bucket.refr_index(R)
    refr_im = util.complex_to_rgb(refr)
    ax_.pcolormesh(Z, X, refr_im)
    ax_.set_xlabel("z [um]")
    ax_.set_ylabel("x [um]")

    ax_ = ax[1]
    edge = jnp.zeros_like(X, dtype=bool)
    for comp in bucket.bucket:
        edge |= comp.edge(R)
    im = ax_.pcolormesh(Z, X, edge)
    plt.colorbar(im, ax=ax_)
    ax_.set_xlabel("z [um]")
    ax_.set_ylabel("x [um]")

    plt.show()
    plt.close()
