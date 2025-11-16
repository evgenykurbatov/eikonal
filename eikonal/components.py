"""
Optical components.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Complex, Bool, Array



class Components:
    """A placeholder for optical components.
    """
    bucket = []
    refr_index0: complex
    smo_default = jnp.asarray(0.06)  # 0.600/10 [um]


    def __init__(self,
                 bucket: list | None = None,
                 refr_index0: complex | None = 1+0j,
                 smo_default: float | None = None):
        """
        **Parameters**:
        bucket: list | None
            A list of optical components, i.e. instances of the classes `Slit`
            or `Lens`.
        refr_index0: complex | None
            Complex refractive index of the background. Default is `1+0j`.
        smo_default: float | None
            Smoothing scale [um]. Default is 0.06 [um].
        """
        if bucket is not None:
            self.bucket = bucket
        self.refr_index0 = refr_index0
        if smo_default is not None:
            self.smo_default = jnp.array(smo_default)


    def S(self, x: float | Array):
        """Smoothed step function.

        **Returns**:
            0  as  x -> -inf
            1  as  x -> +inf
        """
        return 0.5 + 0.5*jax.scipy.special.erf(x/self.smo)


    def edge(self, r: Float[Array, "2 ..."]) -> Bool[Array, "..."]:
        """Edge detector for mask which represents a component.

        **Parameters**:
        r: array of float
            Array of coordinates of points to check for the edge. The zeroth
            (outermost) index chooses X or Z coordinate.

        **Returns**:
        array of bool
        """
        mask_grad = jnp.array(jnp.gradient( self.mask(r) ))
        # Norm: max(abs(x))
        return jnp.linalg.norm(mask_grad, axis=0, ord=jnp.inf) > 0.25


    def refr_index(self, r: Float[Array, "2 ..."]) -> Complex[Array, "..."]:
        """Calculates complex refractive index at given points.

        **Parameters**:
        r: array of float
            Array of coordinates of points. The zeroth (outermost) index chooses
            X or Z coordinate.

        **Returns**:
        array of complex
        """
        # Fill in the array over the domain dimensions
        res = jnp.full(r.shape[1:], self.refr_index0)
        for elem in self.bucket:
            res += (elem.refr_index - self.refr_index0) * elem.mask(r)
        return res



class Slit(Components, eqx.Module):
    """A slit.
    """
    r0: Float[Array, "2"]
    D:  float
    H:  float
    refr_index: complex
    smo: float


    def __init__(self,
                 r0:  Float[Array, "2"],
                 D:   float,
                 H:   float,
                 refr_index: complex,
                 smo: float | None = None):
        """
        **Parameters**:
        r0: array of float
            Coordinates of the reference point. It is the centre in along X and
            the leftmost point in Z.
        D: float
            Diameter of the slit along X.
        H: float
            Height of the slit along Z.
        refr_index: complex
            Complex refractive index.
        smo: float | None
            Smoothing scale. The `Components.smo_default` is used by default.
        """
        self.r0 = jnp.asarray(r0)
        self.D, self.H, self.refr_index = D, H, refr_index
        if smo is not None:
            self.smo = jnp.asarray(smo)
        else:
            self.smo = self.smo_default


    def mask(self, r: Float[Array, "2 ..."]) -> Float[Array, "..."]:
        """A mask representing the optical element.

        **Parameters**:
        r: array of float
            Array of coordinates of points. The zeroth (outermost) index chooses
            X or Z coordinate.

        **Returns**:
        array of float
        """
        r = jnp.asarray(r)
        x, z = r[0], r[-1]
        X1 = self.r0[0] - 0.5*self.D
        X2 = self.r0[0] + 0.5*self.D
        Z1 = self.r0[1]
        Z2 = self.r0[1] + self.H
        S = self.S
        return (S(X1 - x) + S(x - X2)) * S(z - Z1) * S(Z2 - z)



class Lens(Components, eqx.Module):
    """A lens.
    """
    r0: Float[Array, "2"]
    D:  float
    c1: float
    c2: float
    H:  float
    refr_index: complex
    smo: float


    def __init__(self,
                 r0: Float[Array, "2"],
                 D:  float,
                 c1: float, c2: float,
                 H:  float,
                 refr_index: complex,
                 smo: float | None = None):
        """
        **Parameters**:
        r0: array of float
            Coordinates of the reference point. It is the centre in along X and
            the leftmost point in Z.
        D: float
            Diameter of the slit along X.
        c1, c2: float
            Inverse radii of the front and back surfaces.
        H: float
            Height of the slit along Z.
        refr_index: complex
            Complex refractive index.
        smo: float | None
            Smoothing scale. The `Components.smo_default` is used by default.
        """
        self.r0 = jnp.asarray(r0)
        self.D, self.c1, self.c2, self.H, self.refr_index = D, c1, c2, H, refr_index
        if smo is not None:
            self.smo = jnp.asarray(smo)
        else:
            self.smo = self.smo_default


    def mask(self, r: Float[Array, "2 ..."]) -> Float[Array, "..."]:
        """A mask representing the optical element.

        **Parameters**:
        r: array of float
            Array of coordinates of points. The zeroth (outermost) index chooses
            X or Z coordinate.

        **Returns**:
        array of float
        """
        r = jnp.asarray(r)

        S = self.S
        x, z = r[0], r[-1]
        x_ = x - self.r0[0]
        X1 = self.r0[0] - 0.5*self.D
        X2 = self.r0[0] + 0.5*self.D
        # Front (left) edge
        c1x = self.c1 * x_
        Z1 = lambda x_: self.r0[1] + c1x*x_ / (1. + jnp.sqrt(1. - c1x**2))
        S_Z1 = jnp.where(jnp.abs(c1x) <= 1, S(z - Z1(x_)), 0)
        # Back (right) edge
        c2x = self.c2 * x_
        Z2 = lambda x_: self.r0[1] + self.H + c2x*x_ / (1. + jnp.sqrt(1. - c2x**2))
        S_Z2 = jnp.where(jnp.abs(c2x) <= 1, S(Z2(x_) - z), 0)

        return S(x - X1) * S(X2 - x) * S_Z1 * S_Z2


    def focals(self):
        """Calculates distances related to the lens focals.

        **Returns**:
        tuple of floats
           Focal length, Front focal Distance (FFD), and Back Focal Distance (BFD).
        """
        n = self.refr_index.real
        # Effective focal length
        f = 1. / ( (n-1.) * ( self.c1 - self.c2 + (1-1/n)*self.H*self.c1*self.c2 ) )
        # Front focal distance
        FFD = f * ( 1. + (1-1/n)*self.H*self.c2 )
        # Back focal distance
        BFD = f * ( 1. - (1-1/n)*self.H*self.c1 )
        return f, FFD, BFD
