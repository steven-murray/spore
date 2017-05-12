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
        self.nu = self.source_counts.nu

        if not np.allclose(np.diff(self.nu), self.nu[1]-self.nu[0]):
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
        Nnu = len(self.nu)

        s2 = self.beam_model.sigma ** 2
        SIG = un.steradian * np.outer(s2, s2)/np.add.outer(s2, s2)

        fnu = np.add.outer(self.nu, -self.nu)

        covar = (2*np.pi*self.source_counts.total_squared_flux_density*SIG*np.exp(
            -2*np.pi ** 2*np.outer(fnu ** 2*SIG, self.u ** 2).reshape((Nnu, Nnu, len(self.u)))).T)

        return covar

    @property
    def eta(self):
        return -fftfreq(len(self.nu), d=(self.nu[1] - self.nu[0])*self.beam_model.nu0)[1:len(self.nu)/2][::-1]

    @property
    def redshifts(self):
        return 1420.0 * uc.un.MHz / (self.nu*self.beam_model.nu0).to(un.MHz) - 1

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
        out = np.zeros((len(U), len(self.nu), len(self.nu)))

        def mappable(i, j):
            nu1 = self.nu[i]
            nu2 = self.nu[j]
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

        indices = list(product(range(len(self.nu)), range(len(self.nu))))  # np.tril_indices(len(self.nu))
        for i, j in indices:  # zip(indices[0],indices[1]):
            mappable(i, j)
            # ProcessPool().map(mappable,indices[0],indices[1])

            # for i,nu1 in enumerate(self.nu):
            #     power = self.point_source_power_spec(2*np.pi*u_dash_grid*nu1)
            #
            #     for j,nu2 in enumerate(self.nu):
            #         if j>i:
            #             continue
            #
            #         u_take_ud = np.sqrt(np.add.outer(U, -UD) ** 2 + VD ** 2)
            #         u_take_udmod = np.sqrt(np.add.outer(U, -(nu1/nu2)*UD) ** 2 + VD ** 2)
            #
            #         f1 = 2*np.pi*self.beam_model.sigma[i] ** 2*np.exp(
            #             -2*np.pi ** 2*self.beam_model.sigma[i] ** 2*u_take_ud ** 2)
            #         f2 = 2*np.pi*self.beam_model.sigma[j] ** 2*np.exp(
            #             -2*np.pi ** 2*self.beam_model.sigma[j] ** 2*u_take_udmod ** 2)
            #
            #         beambit = f1*f2#/(nu1*nu2)**2
            #         integrand = power*beambit
            #
            #         out[:,i,j] = simps(simps(integrand, dx=du), dx=du)
            #         out[:,i,j] *= self.source_counts.total_flux_density[i]*self.source_counts.total_flux_density[j]
            #         out[:,j,i] = out[:,i,j]
        return out

    @property
    def dnu(self):
        return (self.nu[1] - self.nu[0])*self.beam_model.nu0

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
        hz_range = self.beam_model.nu0*(self.nu.max() - self.nu.min())
        numax = self.beam_model.nu0 * self.nu.max()

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
            return cov_fourier.to(un.milliKelvin**2 * un.Mpc**6/ uc.hub**6,
                                  equivalencies=uc.radio_to_cosmo_equiv(self.nu.max()*self.beam_model.nu0,Aeff))

    def fourier_vis_covariance_poisson(self,taper=None,natural_units=False, Aeff=20.,diagonal=True):
        return self._get_fourier_vis(self.poisson_covariance, taper,natural_units,Aeff,diagonal)

    def fourier_vis_covariance_clustering(self, taper=None, natural_units=False, Aeff=20.,diagonal=True):
        return self._get_fourier_vis(self.clustered_only_covariance, taper, natural_units, Aeff,diagonal)

    def fourier_vis_covariance_total(self, taper=None, natural_units=False, Aeff=20.,diagonal=True):
        return self._get_fourier_vis(self.total_covariance, taper, natural_units, Aeff,diagonal)


    def _get_power_cov(self, cov, taper=None, natural_units=False, Aeff=20.,diagonal=True):
        cov_fourier = convert_cov_fg_to_cov_ps(cov, self.dnu, taper=taper)

        numax = self.nu.max()*self.beam_model.nu0

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


    # def _get_power_spectrum_2d(self, cov, Aeff,window=None):
    #     return get_ps_from_cov(cov, self.nu*self.beam_model.nu0, self.u, Aeff,window=window)
    #
    # def power_spec_2d_poisson(self, Aeff,window=None):
    #     return self._get_power_spectrum_2d(self.poisson_covariance, Aeff,window=window)
    #
    # def power_spec_2d_clustering(self, Aeff,window=None):
    #     return self._get_power_spectrum_2d(self.clustered_only_covariance, Aeff,window=window)
    #
    # def power_spec_2d(self, Aeff,window=None):
    #     return self._get_power_spectrum_2d(self.total_covariance, Aeff,window=window)


class CircularGaussianPowerLaw(PointSourceCircularGaussian):

    def __init__(self,*args,**kwargs):
        super(CircularGaussianPowerLaw,self).__init__(*args,**kwargs)

        self.clustering_params['u0'] = ensure_unit(self.clustering_params['u0'],1/un.rad)

    @property
    def point_source_power_spec(self):
        return lambda u: (u/self.clustering_params['u0']) ** -self.clustering_params['kappa']

    # def _realspace_power(self, nu1, nu2):
    #     u0 = self.clustering_params['u0']
    #     K = self.clustering_params['kappa']
    #
    #     A = u0 ** K*gamma(1 - K/2)/(2*np.pi*2 ** (K - 1)*gamma(K/2))  # (nu1*nu2)**(-K/2) *
    #     alpha = 2 - K
    #
    #     return lambda r: A*r ** -alpha

    #    @cached_property
    #    def clustered_only_covariance(self):
    #        u0 = self.clustering_params['u0']
    #        kappa = self.clustering_params['kappa']
    #
    #        T = self.source_counts.total_flux_density[0] ** 2/np.outer(self.nu,self.nu) ** (1 + self.source_counts.spectral_index)
    #        #T = np.atleast_3d(T) # shape (1,nu,nu)
    #
    #       s = self.beam_model.sigma[0]
    #        y = 2*np.pi*s ** 2
    #        P = np.outer(self.nu, 1./self.nu)  # shape (nu,nu)
    #
    #        T /= P
    #
    #        nupart = np.outer(np.exp(-2*np.pi*y*self.u ** 2),(2*np.pi*self.nu) ** -kappa)  # shape (u,nu)
    #
    #        front = 2*np.pi*y ** 2* u0 ** kappa * np.einsum("ij,kj->kji",T,nupart) # shape (u,nu,nu) -- non-symmetric
    #
    #        a, b, c = 1-kappa/2, np.pi*y*(1 + P ** 2), np.transpose(2*np.pi*y*np.outer(self.u,(1 + P)).reshape(self.u.shape+P.shape),(0,2,1))
    #        b = np.atleast_3d(b).T  # shape (1,nu,nu)
    #
    #        out =  front*0.5*np.transpose((b.T**(-a)*gamma(a)*hyp1f1(a,1,c.T**2/(4*b.T))),(2,0,1))
    #        # function goes nan instead of just being zero at high u
    #        out[np.isnan(out)] = 0.0
    #        return out

#     @cached_property
#     def clustered_only_covariance(self):
#         u0 = self.clustering_params['u0']
#         kappa = self.clustering_params['kappa']
#         gm = self.source_counts.spectral_index
#
#         # Scalar quantities
#         s = self.beam_model.sigma[0]
#         y = 2*np.pi*s ** 2
#         a = 1 - kappa/2
#         # front should probably be (2*np.pi*np.sqrt(np.pi*y)/u0)**-kappa *y * self.source_counts.total_flux_density[0] ** 2
#         front = (1/(u0*np.sqrt(y)))**-kappa * y * self.source_counts.total_flux_density[0] ** 2
#         front *= un.sr # hack for now
#         front = front.to(un.Jy**2) #will fail if above is wrong.
# #        front = np.pi*y ** 2*(2*np.pi/u0) ** -kappa*self.source_counts.total_flux_density[0] ** 2
#
#         # nu,nu quantities
#         P = np.outer(self.nu, 1./self.nu)  # shape (nu,nu)
#         Q = (1 + P) ** 2/(1 + P ** 2)
#         allnu = (1 + P ** 2) ** (-a)/np.outer(self.nu ** (kappa + gm), self.nu ** (gm + 2))
#
#         lastfac = exp_hyp1f1(2*np.pi*y, a, 1, np.pi*y*Q, self.u ** 2)
#
#         return front*allnu*lastfac.T

    @cached_property
    def clustered_only_covariance(self):
        u0 = self.clustering_params['u0']
        kappa = self.clustering_params['kappa']
        gm = self.source_counts.spectral_index

        # Scalar quantities
        s = self.beam_model.sigma[0]
        y = (2*np.pi*s ** 2).to(un.sr)
        mu1 = self.source_counts.total_flux_density[0]
        front = y*mu1**2

        # front *= un.sr  # hack for now
        # front = front.to(un.Jy ** 2)  # will fail if above is wrong.
        # #        front = np.pi*y ** 2*(2*np.pi/u0) ** -kappa*self.source_counts.total_flux_density[0] ** 2

        # nu,nu quantities
        P = np.outer(self.nu, 1./self.nu)  # shape (nu,nu)
        Q = (1 + P) ** 2/(1 + P ** 2)
        allnu = Q**(-kappa/2.) * (1 + P ** 2) ** (kappa/2. -1)/np.outer(self.nu ** (kappa + gm), self.nu ** (gm + 2))

        ps_term = (self.u/u0)**-kappa * un.sr
        uQ = np.outer(self.u**2,(Q-2)).reshape(self.u.shape+Q.shape) / un.sr
        print uQ.unit, y.unit
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


#    kpar = eta.to(uc.hub/uc.un.Mpc, equivalencies=uc.cosmo_21cm_los_equiv(z))
#    kperp = u.to(uc.hub/uc.un.Mpc, equivalencies=uc.cosmo_21cm_angle_equiv(z))
#    out_ps = uc.jyhz_to_mKMpc(ps, nu, Aeff, verbose)
#    return out_ps, kpar, kperp

#
# def convert_cov_fg_to_cov_ps(cov, nu, u,Aeff,taper=None):
#     """
#     Convert foreground covariance to covariance of the power spectrum (with extra Hz^4 units)
#
#     Parameters
#     ----------
#     cov
#     nu
#     u
#     Aeff
#     window
#     verbose
#
#     Returns
#     -------
#
#     """
#     if not hasattr(cov, "unit"):
#         cov = cov * uc.un.Jy**2
#
#     if not hasattr(u, "unit"):
#         u = u / uc.un.radian
#
#     if not hasattr(nu, "unit"):
#         nu = nu * uc.un.MHz
#
#     ps = _convert_cov_to_2d_ps(cov,window=window)*uc.un.Jy**2 * nu.unit**2
#
#     N = len(nu)
#     z = (1420.0 * uc.un.MHz / nu.to(uc.un.MHz).min()) - 1
#     eta = fftfreq(N, d=nu[1] - nu[0])[1:N / 2]
#
#     kpar = eta.to(uc.hub / uc.un.Mpc, equivalencies=uc.cosmo_21cm_los_equiv(z))
#     kperp = u.to(uc.hub / uc.un.Mpc, equivalencies=uc.cosmo_21cm_angle_equiv(z))
#     out_ps = uc.jyhz_to_mKMpc(ps, nu, Aeff, verbose)
#     return out_ps, kpar, kperp