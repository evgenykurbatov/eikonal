"""
Glasses.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


# Some reference wavelengths
wv_ref_F = 0.48613  # [um], H blue line
wv_ref_e = 0.54607  # [um], Hg green line
wv_ref_d = 0.58756  # [um], He yellow line
wv_ref_C = 0.65627  # [um], H red line


class Glass(eqx.Module):
    name: str = eqx.field(static=True)
    wv_min: float
    wv_max: float
    B: Float[Array, ""]
    C: Float[Array, ""]

    def refraction_index(self, wv: float) -> float:
        """
        Sellmeier interpolating formula for refraction index.
        """
        return jnp.sqrt( 1. + (self.B * wv**2 / (wv**2 - self.C)).sum() )


N_BK7 = Glass(
    name='N-BK7',  # or 'Crown'
    wv_min=0.365,  # [um]
    wv_max=2.5,    # [um]
    B=jnp.array([1.03961212, 0.231792344, 1.01046945]),
    C=jnp.array([0.00600069867, 0.0200179144, 103.560653]),
)


F2 = Glass(
    name='F2',     # or 'Flint'
    wv_min=0.365,  # [um]
    wv_max=2.5,    # [um]
    B=jnp.array([1.34533359, 0.209073176, 0.937357162]),
    C=jnp.array([0.00997743871, 0.0470450767, 111.886764]),
)


SF5 = Glass(
    name='SF5',    # or 'Dense Flint'
    wv_min=0.365,  # [um]
    wv_max=2.5,    # [um]
    B=jnp.array([1.52481889, 0.187085527, 1.42729015]),
    C=jnp.array([0.011254756, 0.0588995392, 129.141675]),
)
