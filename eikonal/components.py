"""
Optical components.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Complex, Bool, Array



class Components:
    bucket = []
    eps0: complex
    smo_default = jnp.asarray(0.06)  # 0.600/10 [um]


    def __init__(self,
                 bucket: list | None = None,
                 eps0: complex | None = 1+0j,
                 smo_default: float | None = None):
        if bucket is not None:
            self.bucket = bucket
        self.eps0 = eps0
        if smo_default is not None:
            self.smo_default = jnp.array(smo_default)


    def S(self, x: float | Array):
        return 0.5 + 0.5*jax.scipy.special.erf(x/self.smo)


    def edge(self, r: Float[Array, "... 2"]) -> Bool[Array, "..."]:
        mask_grad = jnp.array(jnp.gradient( self.mask(r) ))
        # Norm: max(abs(x))
        return jnp.linalg.norm(mask_grad, axis=0, ord=jnp.inf) > 0.25


    def dispersion(self, r: Float[Array, "... 2"]) -> Complex[Array, "..."]:
        res = jnp.full(r.shape[:-1], self.eps0)
        for comp in self.bucket:
            res += (comp.eps - self.eps0) * comp.mask(r)
        return res



class Slit(Components, eqx.Module):
    r0:  Float[Array, "2"]
    D:   float
    H:   float
    eps: complex
    smo: float


    def __init__(self,
                 r0:  Float[Array, "2"],
                 D:   float,
                 H:   float,
                 eps: complex,
                 smo: float | None = None):
        """
        """
        self.r0 = jnp.asarray(r0)
        self.D, self.H, self.eps = D, H, eps
        if smo is not None:
            self.smo = jnp.asarray(smo)
        else:
            self.smo = self.smo_default


    def mask(self, r: Float[Array, "... 2"]) -> Float[Array, "..."]:
        r = jnp.asarray(r)
        x, z = r[...,0], r[...,-1]
        X1 = self.r0[0] - 0.5*self.D
        X2 = self.r0[0] + 0.5*self.D
        Z1 = self.r0[1]
        Z2 = self.r0[1] + self.H
        S = self.S
        return (S(X1 - x) + S(x - X2)) * S(z - Z1) * S(Z2 - z)



class Lens(Components, eqx.Module):
    r0:  Float[Array, "2"]
    D:   float
    c1:  float
    c2:  float
    H:   float
    eps: complex
    smo: float


    def __init__(self,
                 r0:  Float[Array, "2"],
                 D:   float,
                 c1:  float, c2: float,
                 H:   float,
                 eps: complex,
                 smo: float | None = None):
        """
        """
        self.r0 = jnp.asarray(r0)
        self.D, self.c1, self.c2, self.H, self.eps = D, c1, c2, H, eps
        if smo is not None:
            self.smo = jnp.asarray(smo)
        else:
            self.smo = self.smo_default


    def mask(self, r: Float[Array, "... 2"]) -> Float[Array, "..."]:
        r = jnp.asarray(r)

        S = self.S
        x, z = r[...,0], r[...,-1]
        x_ = x - self.r0[0]
        X1 = self.r0[0] - 0.5*self.D
        X2 = self.r0[0] + 0.5*self.D
        # Front
        c1x = self.c1 * x_
        Z1 = lambda x_: self.r0[1] + c1x*x_ / (1. + jnp.sqrt(1. - c1x**2))
        S_Z1 = jnp.where(jnp.abs(c1x) <= 1, S(z - Z1(x_)), 0)
        # Back
        c2x = self.c2 * x_
        Z2 = lambda x_: self.r0[1] + self.H + c2x*x_ / (1. + jnp.sqrt(1. - c2x**2))
        S_Z2 = jnp.where(jnp.abs(c2x) <= 1, S(Z2(x_) - z), 0)

        return S(x - X1) * S(X2 - x) * S_Z1 * S_Z2


    def focals(self):
        n = self.eps.real
        # Effective focal length
        f = 1. / ( (n-1.) * ( self.c1 - self.c2 + (1-1/n)*self.H*self.c1*self.c2 ) )
        # Front focal distance
        FFD = f * ( 1. + (1-1/n)*self.H*self.c2 )
        # Back focal distance
        BFD = f * ( 1. - (1-1/n)*self.H*self.c1 )
        return f, FFD, BFD
