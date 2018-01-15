"""
Classes constructing covariance matrices (and means) for visibilities given
unmodelled point-sources.
"""
import numpy as np
from beam import CircularGaussian
from cached_property import cached_property
from hankel import SymmetricFourierTransform
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from powerbox.powerbox import _magnitude_grid
from scipy.special import gamma
from scipy.integrate import simps
from scipy.linalg import dft
import astropy.units as un
import astropy.constants as cnst

from scipy.special import hyp1f1
from itertools import product
from spore.measure import unit_conversions as uc
from spore.common_tools import ensure_unit
from powerbox.dft import fftfreq


class PointSourceCircularGaussian(object):
    def __init__(self, source_counts, beam_model, u=np.linspace(8, 500, 50),
                 clustering_params={"func": lambda u: np.zeros_like(u)}):
        self.u = ensure_unit(u,1/un.rad)
        self.beam_model = beam_model
        self.source_counts = source_counts
        self.f0 = self.source_counts.f0

        if not np.allclose(np.diff(self.f0), self.f0[1]-self.f0[0]):
            raise ValueError("Frequencies must be specified in regular intervals in real space")

        self.clustering_params = clustering_params

        if not isinstance(self.beam_model, CircularGaussian):
            raise ValueError("This class requires the beam model to be a circular gaussian.")

    @cached_property
    def ugrid(self):
        return _magnitude_grid(self.u, 2)

    @cached_property
    def point_source_power_spec(self):
        """
        The point-source power spectrum of density fluctuations. Should be either `None` or
        a callable function of a single variable, u.
        """
        return self.clustering_params['func']

    @cached_property
    def mean_visibility(self):
        return self.source_counts.total_flux_density*self.beam_model.fourier_beam(_magnitude_grid(self.u, 2)).T

    @cached_property
    def mean_visibility_circular(self):
        return self.source_counts.total_flux_density*self.beam_model.fourier_beam_1D(self.u).T

    @cached_property
    def poisson_covariance(self):
        Nnu = len(self.f0)

        s2 = self.beam_model.sigma(self.f0) ** 2
        SIG = un.steradian**2 * np.outer(s2, s2)/np.add.outer(s2, s2) #the outer doesn't preserve units, but add.outer does. (!?)

        fnu = np.add.outer(self.f0, -self.f0)

        covar = (2*np.pi*self.source_counts.total_squared_flux_density*SIG*np.exp(
            -2*np.pi ** 2*np.outer(fnu ** 2*SIG, self.u ** 2).reshape((Nnu, Nnu, len(self.u)))).T)

        return covar

    @property
    def eta(self):
        return -fftfreq(len(self.f0), d=(self.f0[1] - self.f0[0]) * self.beam_model.nu0.value)[1:len(self.f0) / 2][::-1] / self.beam_model.nu0.unit

    @property
    def redshifts(self):
        return 1420.0 * uc.un.MHz / (self.f0 * self.beam_model.nu0).to(un.MHz) - 1

    @property
    def kpar(self):
        return self.eta.to(uc.hub/uc.un.Mpc, equivalencies=uc.cosmo_21cm_los_equiv(np.min(self.redshifts)))

    @property
    def kperp(self):
        return self.u.to(uc.hub / uc.un.Mpc, equivalencies=uc.cosmo_21cm_angle_equiv(self.redshifts.min()))

    def _beam_term(self, r):
        """Returns the inverse fourier transform of the fourier transformed beam squared"""
        r = ensure_unit(r,un.rad)
        s = self.beam_model.sigma[0]
        return np.pi*s ** 2*np.exp(-r ** 2/(4*s ** 2))

    def _beam_squared(self, r):
        r = ensure_unit(r,un.rad)
        s = self.beam_model.sigma[0]
        return np.exp(-r ** 2/s ** 2)

    def _realspace_power(self, nu1, nu2):
        r = np.logspace(-3, 3, 1000)

        ht1 = SymmetricFourierTransform(ndim=2, N=1000, h=0.003, a=0, b=2*np.pi)
        gres = ht1.transform(
            lambda u: np.sqrt(self.point_source_power_spec(2*np.pi*u))*np.sqrt(self.point_source_power_spec(2*np.pi*u)),
            k=r, inverse=True, ret_err=False)
        # *nu1*nu2
        xir = spline(np.log(r), np.log(gres))
        return lambda r: np.exp(xir(np.log(r)))

    @cached_property
    def clustered_only_covariance(self):
        L = self.u.max()*2.4
        N = min(500, L/(self.u[self.u > 0].min()/1.2))

        udash = np.arange(-L/2, L/2, L/N)
        du = udash[1] - udash[0]
        UD, VD = np.meshgrid(udash, udash)
        u_dash_grid = _magnitude_grid(udash, 2)

        U = self.u
        out = np.zeros((len(U), len(self.f0), len(self.f0)))

        def mappable(i, j):
            nu1 = self.f0[i]
            nu2 = self.f0[j]
            power = self.point_source_power_spec(2*np.pi*u_dash_grid*nu1)

            u_take_ud = np.sqrt(np.add.outer(U, -UD) ** 2 + VD ** 2)
            u_take_udmod = np.sqrt(np.add.outer(U, -(nu1/nu2)*UD) ** 2 + VD ** 2)

            f1 = 2*np.pi*self.beam_model.sigma[0] ** 2*np.exp(
                -2*np.pi ** 2*self.beam_model.sigma[0] ** 2*u_take_ud ** 2)
            f2 = 2*np.pi*self.beam_model.sigma[0] ** 2*np.exp(
                -2*np.pi ** 2*self.beam_model.sigma[0] ** 2*u_take_udmod ** 2)

            beambit = f1*f2/(nu1*nu2)/(nu1/nu2)
            integrand = power*beambit

            out[:, i, j] = simps(simps(integrand, dx=du), dx=du)
            out[:, i, j] *= self.source_counts.total_flux_density[i]*self.source_counts.total_flux_density[j]
            # out[:,j,i] = out[:,i,j]

            return None

        indices = list(product(range(len(self.f0)), range(len(self.f0))))  # np.tril_indices(len(self.nu))
        for i, j in indices:
            mappable(i, j)

        return out

    @property
    def dnu(self):
        return (self.f0[1] - self.f0[0]) * self.beam_model.nu0

    def effective_volume(self, Aeff):
        """
        The approximate effective volume of the model in sr.MHz
                
        Parameters
        ----------
        Aeff : float
            The effective collecting area of the telescope, in m^2.

        Returns
        -------
        vol :
            Effective volume of the model (given the beamwidth and frequency coverage), in sr.MHz.
        """
        Aeff = ensure_unit(Aeff, un.m**2)
        hz_range = self.beam_model.nu0*(self.f0.max() - self.f0.min())
        numax = self.beam_model.nu0 * self.f0.max()

        return un.steradian*(cnst.c ** 2/numax ** 2).to(un.m ** 2)/Aeff * hz_range

    @cached_property
    def total_covariance(self):
        return self.poisson_covariance + self.clustered_only_covariance

    def _get_fourier_vis(self,cov,taper=None,natural_units=False, Aeff=20., diagonal=True):
        cov_fourier = convert_cov_fg_to_cov_fourier(cov, self.dnu, taper=taper)

        if diagonal:
            cov_fourier = np.diagonal(cov_fourier.T).T

        if natural_units:
            return cov_fourier
        else:
            return cov_fourier.to(un.milliKelvin ** 2 * un.Mpc ** 6 / uc.hub ** 6,
                                  equivalencies=uc.radio_to_cosmo_equiv(self.f0.max() * self.beam_model.nu0, Aeff))

    def fourier_vis_covariance_poisson(self,taper=None,natural_units=False, Aeff=20.,diagonal=True):
        return self._get_fourier_vis(self.poisson_covariance, taper,natural_units,Aeff,diagonal)

    def fourier_vis_covariance_clustering(self, taper=None, natural_units=False, Aeff=20.,diagonal=True):
        return self._get_fourier_vis(self.clustered_only_covariance, taper, natural_units, Aeff,diagonal)

    def fourier_vis_covariance_total(self, taper=None, natural_units=False, Aeff=20.,diagonal=True):
        return self._get_fourier_vis(self.total_covariance, taper, natural_units, Aeff,diagonal)

    def power_poisson(self,taper=None,natural_units=False, Aeff=20.,diagonal=True):
        fvis = self.fourier_vis_covariance_poisson(taper=taper, natural_units=natural_units,
                                                   Aeff=Aeff, diagonal=diagonal)
        if natural_units:
            return fvis/ self.effective_volume(Aeff)
        else:
            numax = self.f0.max() * self.beam_model.nu0
            vol = self.effective_volume(Aeff).to(un.Mpc**3/uc.hub**3, equivalencies=uc.radio_to_cosmo_equiv(numax,Aeff))

            return fvis/vol

    def power_clustering(self, taper=None, natural_units=False, Aeff=20., diagonal=True):
        fvis = self.fourier_vis_covariance_clustering(taper=taper, natural_units=natural_units,
                                                   Aeff=Aeff, diagonal=diagonal)
        if natural_units:
            return fvis / self.effective_volume(Aeff)
        else:
            numax = self.f0.max() * self.beam_model.nu0
            vol = self.effective_volume(Aeff).to(un.Mpc ** 3 / uc.hub ** 3,
                                                 equivalencies=uc.radio_to_cosmo_equiv(numax, Aeff))

            return fvis / vol

    def power_total(self, taper=None, natural_units=False, Aeff=20., diagonal=True):
        fvis = self.fourier_vis_covariance_total(taper=taper, natural_units=natural_units,
                                                   Aeff=Aeff, diagonal=diagonal)
        if natural_units:
            return fvis / self.effective_volume(Aeff)
        else:
            numax = self.f0.max() * self.beam_model.nu0
            vol = self.effective_volume(Aeff).to(un.Mpc ** 3 / uc.hub ** 3,
                                                 equivalencies=uc.radio_to_cosmo_equiv(numax, Aeff))

            return fvis / vol

    def _get_power_cov(self, cov, taper=None, natural_units=False, Aeff=20.,diagonal=True):
        cov_fourier = convert_cov_fg_to_cov_ps(cov, self.dnu, taper=taper)

        numax = self.f0.max() * self.beam_model.nu0

        if diagonal:
            cov_fourier = np.diagonal(cov_fourier.T).T

        if not natural_units:
            cov_fourier = cov_fourier.to(un.milliKelvin**4 * un.Mpc**12/ uc.hub**12,
                                    equivalencies=uc.radio_to_cosmo_equiv(numax, Aeff))

            vol = self.effective_volume(Aeff).to(un.Mpc**3/uc.hub**3, equivalencies=uc.radio_to_cosmo_equiv(numax,Aeff))
            return cov_fourier/vol**2
        else:
            return cov_fourier/self.effective_volume(Aeff)**2

    def power_cov_poisson(self, taper=None, natural_units=False, Aeff=20.,diagonal=True):
        return self._get_power_cov(self.poisson_covariance, taper, natural_units, Aeff,diagonal)

    def power_cov_clustering(self, taper=None, natural_units=False, Aeff=20.,diagonal=True):
        return self._get_power_cov(self.clustered_only_covariance, taper, natural_units, Aeff,diagonal)

    def power_cov_total(self, taper=None, natural_units=False, Aeff=20.,diagonal=True):
        return self._get_power_cov(self.total_covariance, taper, natural_units, Aeff,diagonal)


class CircularGaussianPowerLaw(PointSourceCircularGaussian):

    def __init__(self,*args,**kwargs):
        super(CircularGaussianPowerLaw,self).__init__(*args,**kwargs)

        self.clustering_params['u0'] = ensure_unit(self.clustering_params['u0'],1/un.rad)

    @property
    def point_source_power_spec(self):
        return lambda u: (u/self.clustering_params['u0']) ** -self.clustering_params['kappa']

    @cached_property
    def clustered_only_covariance(self):
        u0 = self.clustering_params['u0']
        kappa = self.clustering_params['kappa']
        gm = self.source_counts.spectral_index

        # Scalar quantities
        s = self.beam_model.sigma(self.f0)[0]
        y = (2*np.pi*s ** 2).to(un.sr)
        mu1 = self.source_counts.total_flux_density[0]
        front = y*mu1**2

        # front *= un.sr  # hack for now
        # front = front.to(un.Jy ** 2)  # will fail if above is wrong.
        # #        front = np.pi*y ** 2*(2*np.pi/u0) ** -kappa*self.source_counts.total_flux_density[0] ** 2

        # nu,nu quantities
        P = np.outer(self.f0, 1. / self.f0)  # shape (nu,nu)
        Q = (1 + P) ** 2/(1 + P ** 2)
        allnu = Q**(-kappa/2.) * (1 + P ** 2) ** (kappa/2. -1)/np.outer(self.f0 ** (kappa + gm), self.f0 ** (gm + 2))

        ps_term = (self.u/u0)**-kappa * un.sr
        uQ = np.outer(self.u**2,(Q-2)).reshape(self.u.shape+Q.shape) / un.sr
        lastfac = np.exp(np.pi*y*uQ)

        return front*allnu*(ps_term*lastfac.T).T


def exp_hyp1f1(a, b, c, d, x):
    "Return exp(-ax) * gamma(b)*1F1(b,c,d*x), with saner solutions at high x"
    dx = np.outer(d, x).reshape(d.shape + x.shape)
    return np.where(a*x < 500, gamma(b)*np.exp(-a*x)*hyp1f1(b, c, dx), np.exp(dx - a*x)*(dx) ** (b - c))


def _cov_fourier(cov,dx, taper=None):
    """
    Determine the covariance of a Fourier space vector from the known covariance of a real-space vector.
    
    Parameters
    ----------
    cov : array
        Either square 2-dimensional covariance matrix in real space, or 3D, with first axis a dependent dimension.
        
    dx : float
        The physical interval between bins in the covariance matrix.

    Returns
    -------
    cov_fourier : Covariance of Fourier-space vector (same shape as cov)

    """
    N = cov.shape[-1]

    if taper is not None:
        cov *= np.outer(taper(N),taper(N))

    F = dft(N, "sqrtn")

    out = np.real(np.conjugate(F.T).dot(cov).dot(F).transpose(1, 0, 2)*dx ** 2)

    # Select only positive eta entries
    return out.T[1:N/2,1:N/2].T


def _cov_power(cov,dx, taper=None):
    """
    Determine the covariance of the power spectrum, from the known covariance of a real-space vector (assuming
    a Gaussian distribution)

    Parameters
    ----------
    cov : array
        Either square 2-dimensional covariance matrix in real space, or 3D, with first axis a dependent dimension.

    dx : float
        The physical interval between bins in the covariance matrix.

    Returns
    -------
    cov_power : Covariance of power spectrum (same dimensions as cov, but nu dimensions are halved in size)
    """
    if len(cov.shape) ==2:
        cov = np.atleast_3d(cov).T

    N = cov.shape[-1]
    F = dft(N, "sqrtn")

    if taper is not None:
        cov *= np.outer(taper(N), taper(N))

    out = np.zeros(cov.shape) * cov.unit**2
    for i,c in enumerate(cov):
        out[i,:,:] = np.real(np.conjugate(F.T).dot(c).dot(c).dot(F))

    out *= dx**4

    # Select only positive eta entries
    out = out.T[1:N/2,1:N/2].T

    return np.squeeze(out)

def convert_cov_fg_to_cov_ps(cov, dnu, taper=None):
    """
    Convert foreground covariance to covariance of the power spectrum (with extra Hz^4 units)

    Parameters
    ----------
    cov
    nu
    u
    Aeff
    window
    verbose

    Returns
    -------
    """
    return _cov_power(cov, dnu, taper=taper)

def convert_cov_fg_to_cov_fourier(cov, dnu, taper=None):
    """
    Convert foreground covariance to covariance of the power spectrum (with extra Hz^4 units)

    Parameters
    ----------
    cov
    nu
    u
    Aeff
    window
    verbose

    Returns
    -------

    """
    return _cov_fourier(cov, dnu, taper=taper)
