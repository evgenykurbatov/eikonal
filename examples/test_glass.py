
import matplotlib.pyplot as plt
import util

from _context import eikonal
from eikonal.glass import *



if __name__ == "__main__":
    plt.figure()

    for glass in [SF5, F2, N_BK7]:
        wv = jnp.linspace(glass.wv_min, glass.wv_max, 100)
        n = jax.vmap( lambda wv_: glass.refraction_index(wv_) )(wv)
        plt.plot(wv, n, label=glass.name)
    plt.legend(frameon=False)
    plt.xlabel("wv [um]")
    plt.ylabel("Refraction index")

    plt.show()
