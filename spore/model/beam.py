"""
Various models of the primary beam.
"""
import copy
from cached_property import cached_property
import numpy as np
from astropy import constants
from astropy import units as un
from _framework import Component as Cmpt
from spore.common_tools import ensure_unit

class Beam(Cmpt):
    """
    A general description of a beam.

    Parameters
    ----------
    nu : float
        Frequency of observation. Should be in MHz.
    """
    _defaults = {}

    def __init__(self,nu0,f0=1,**params):
        self.f0 = np.atleast_1d(f0)
        self.nu0 = ensure_unit(nu0, un.MHz)

        super(Beam,self).__init__(**params)


    def beam(self,l,m):
        """
        Should return the sensitivity of the beam at any l,m.
        """
        pass



class CircularBeam(Beam):

    def _correct_arrays(self,l,m=None,dim=2):
        if len(l.shape) == dim:
            l = np.repeat(l, len(self.sigma)).reshape((l.shape + self.sigma.shape)).T
            if m is not None:
                m = np.repeat(m, len(self.sigma)).reshape((m.shape + self.sigma.shape)).T
        return l,m

    def _r2(self,l,m=None,dim=2):

        l,m = self._correct_arrays(l,m,dim)

        if m is None:
            return l**2
        else:
            return l**2 + m**2

    def beam(self,l,m=None):
        pass


class CircularGaussian(CircularBeam):
    _defaults = {"epsilon":0.42,
                "D":4.}

    def __init__(self,*args,**kwargs):
        super(CircularGaussian,self).__init__(*args,**kwargs)
        self.params['D'] = ensure_unit(self.params['D'], un.m)

    @cached_property
    def sigma(self):
        return (self.params['epsilon'] * constants.c / (self.f0 * self.nu0 * self.params['D'])).to(un.dimensionless_unscaled) * un.rad

    def beam(self,l,m=None):
        "The beam attenuation at l,m"
        r2 = self._r2(l,m)
        r2 = ensure_unit(r2,un.rad**2)
        return np.exp(-r2.T/(2*self.sigma**2)).T

    def beam_1D(self,l,m=None):
        "The beam attenuation at l,m"
        r2 = self._r2(l,m,True)
        r2 = ensure_unit(r2,un.rad**2)
        return np.exp(-r2.T/(2*self.sigma**2)).T

    def fourier_beam(self,u,v=None):
        "The fourier transform of beam attenuation at u,v"

        u2 = self._r2(u,v)
        u2 = ensure_unit(u2,1./un.rad**2)
        return (2*np.pi*self.sigma**2 * np.exp(-2*np.pi**2 * self.sigma**2 * u2.T)).T

    def fourier_beam_1D(self,u,v=None):
        "The fourier transform of beam attenuation at u,v"
        u2 = self._r2(u,v,True)
        u2 = ensure_unit(u2,1/un.rad**2)
        return (2*np.pi*self.sigma**2 * np.exp(-2*np.pi**2 * self.sigma**2 * u2.T)).T