import numpy as np
import spore.common.unit_conversions as uc

from powerbox import LogNormalPowerBox, PowerBox
from scipy.interpolate import griddata
from _framework import Component as Cmpt
from spore.common.unit_conversions import ensure_unit
from astropy.units import rad

from healpy import sphtfunc, pixelfunc

try:
    from pygsm import GlobalSkyModel2016, GlobalSkyModel
    HAVE_PYGSM = True
except ImportError:
    HAVE_PYGSM = False

class SpatialDistribution(Cmpt):
    """
    Base class for generating on-sky spatial distributions.

    This base class merely provides a gridding of the sky, in l,m (sine-projection) co-ordinates. To produce the sky
    brightness on this grid, a specific sub-class must be instantiated.

    Parameters
    ----------
    f0 : float or array of floats, optional
        A vector of frequencies, as ratios to a reference frequency, nu0.

    sky_size : float or array, optional
        Defines the size of the sky (in l,m units) at the reference frequency. If scalar, will be divided by f0 to
        get sky sizes at varying frequencies. Otherwise, left as-is.

    ncells : int, optional
        The number of cells per side to divide the sky up into.

    seed : float, optional
        A seed to ensure the same results are returned.
    """
    _defaults = {}

    def __init__(self, f0=1, sky_size=1., ncells=100, seed=None, *args, **kwargs):
        super(SpatialDistribution, self).__init__(*args, **kwargs)
        self.f0 = np.atleast_1d(f0)

        if np.isscalar(sky_size):
            self.sky_size = ensure_unit(sky_size/self.f0, rad)
        else:
            self.sky_size = ensure_unit(sky_size, rad)

        self.ncells = ncells
        self.seed = seed

    @property
    def resolution(self):
        return self.sky_size/self.ncells

    @property
    def lgrid(self):
        return [np.arange(-s.value/2,s.value/2,r.value)[:self.ncells] * uc.un.rad for s,r in zip(self.sky_size,self.resolution)]

    @property
    def cell_area(self):
        return self.resolution ** 2


class PureClustering(SpatialDistribution):
    """
    Defines a base class for all distributions that have an isotropic power spectrum with no (Poisson) shot noise.

    See :class:`SpatialDistribution` for relevant parameters. In addition, ``power_spectrum`` is a function that returns
    the (volume-normalised) power spectrum of the brightness as a function of fourier scale u.
    """

    _defaults = dict(power_spectrum = lambda u : np.zeros_like(u))

    def sky(self, sbar):
        """
        A gridded sky-brightness distribution, at multiple frequencies

        Parameters
        ----------
        sbar : array
            The mean brightness of the sky, at each f0.

        Returns
        -------
        sky : (N,M,M)-array
            With N frequency bins, and M equi-distant bins in (l,m) (corresponding to `lgrid`), this gives a sky
            brightness (in Jy) in each. Note that lgrid is frequency dependent, so that the FT at each frequency
            corresponds to the same baseline length.
        """
        raise NotImplementedError("Please use a specific subclass!")


class PoissonClustering(SpatialDistribution):
    """
    Defines a base class for all distributions that have an isotropic power spectrum with (Poisson) shot noise.

    See :class:`SpatialDistribution` for relevant parameters. In addition, ``power_spectrum`` is a function that returns
    the (volume-normalised) power spectrum of the brightness as a function of fourier scale u.
    """
    _defaults = dict(power_spectrum = lambda u : np.zeros_like(u))

    def source_positions(self, nbar):
        """
        (N,2) array of source positions in l,m
        """
        pass

    def sky(self, pos, fluxes, spec_indices):
        """
        A gridded sky-brightness distribution, at multiple frequencies

        Parameters
        ----------
        pos : (n,2)-array
            An array of source positions, in (l,m) co-ordinates.

        fluxes : 1D-array
            A vector of flux density values as defined at `f0=1`, with same length as `pos`

        spec_indices : 1D-array
            A vector of spectral index values as defined at `f0=1`, with same length as `pos`.

        Returns
        -------
        sky : (N,M,M)-array
            With N frequency bins, and M equi-distant bins in (l,m) (corresponding to `lgrid`), this gives a sky
            brightness (in Jy) in each. Note that lgrid is frequency dependent, so that the FT at each frequency
            corresponds to the same baseline length.
        """
        raise NotImplementedError("Please use a specific subclass!")


class PureClustering_FlatSky(PureClustering):
    """
    Defines an arbitrary isotropic sky distribution without Poisson scatter, in which l,m co-ordinates are treated as
    the co-ordinates of isotropy.
    """
    _defaults = PureClustering._defaults
    _defaults.update({"use_lognormal":True})

    @property
    def _powerbox(self):
        if self.params['use_lognormal']:
            return LogNormalPowerBox(N=self.ncells, pk=self.params['power_spectrum'],
                                     dim=2, boxlength=np.max(self.sky_size).value,a=0,b=2*np.pi)
        else:
            return PowerBox(N=self.ncells, pk=self.params['power_spectrum'],
                            dim=2, boxlength=np.max(self.sky_size).value,a=0,b=2*np.pi)

    def sky(self, sbar):

        sbar = np.atleast_1d(sbar)

        assert len(sbar) == len(self.f0), "sbar must be a vector of length len(f0)"

        if self.seed is not None:
            np.random.seed(self.seed)

        density = self._powerbox.delta_x() + 1
        coords_x, coords_y = np.meshgrid(self.lgrid[0], self.lgrid[0])

        Sbins = np.zeros((len(self.f0), self.ncells, self.ncells))
        if np.std(density)>0:
            for i, f0 in enumerate(self.f0):
                if i == 0:
                    Sbins[i] = sbar[i] * density
                else:
                    x, y = np.meshgrid(self.lgrid[i], self.lgrid[i])
                    Sbins[i] = sbar[i] * griddata((coords_x.flatten(), coords_y.flatten()), density.flatten(), (x, y),
                                                  method="cubic")
        else:
            for i in range(len(self.f0)):
                Sbins[i] = sbar[i]

        return Sbins


class PoissonClustering_FlatSky(PoissonClustering, PureClustering_FlatSky):
    """
    Defines an arbitrary isotropic sky distribution with Poisson scatter, in which l,m co-ordinates are treated as
    the co-ordinates of isotropy.
    """
    _defaults = PureClustering._defaults
    _defaults.update({"use_lognormal":True})

    def source_positions(self, nbar):
        if self.seed is not None:
            np.random.seed(self.seed)

        pos = self._powerbox.create_discrete_sample(nbar)
        return pos.T

    def sky(self, pos, fluxes, spec_indices):
        sbins = [0] * len(self.sky_size)
        for i, (sz, f0) in enumerate(zip(self.sky_size.value, self.f0)):
            mask = np.logical_and(np.logical_and(pos[0] > -sz / 2, pos[0] < sz / 2),
                                  np.logical_and(pos[1] > -sz / 2, pos[1] < sz / 2))
            sbins[i] = np.histogram2d(pos[0,mask], pos[1,mask], bins=np.concatenate((self.lgrid[i], [sz / 2])),
                                      weights=fluxes[mask] * f0 ** (-spec_indices[mask]))[0] / self.cell_area[i]

        return np.array(sbins)


class PureClustering_Spherical(PureClustering):
    _defaults = PureClustering._defaults
    _defaults.update({"nside":64,
                      "lmax":1000})

    @property
    def _healpix_deltax(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        l = np.arange(1,self.params['lmax'])
        cls = self.params['power_spectrum'](l)
        return sphtfunc.synfast(cls, self.params['nside'])

    def healpix_brightness(self, sbar):
        sbar = np.atleast_1d(sbar)
        dx = self._healpix_deltax
        return [(1+dx)*s for s in sbar]

    def sky(self, sbar):
        healpix = self.healpix_brightness(sbar)

        lm_map = np.zeros((len(self.f0), self.ncells * self.ncells)) * np.nan
        for i,hp in enumerate(healpix):
            # Convert lgrid to co-lat and longitude in radians.
            L, M = np.meshgrid(self.lgrid[i].value, self.lgrid[i].value)
            lm = np.sqrt(L**2+M**2).flatten()
            mask = lm < 1

            theta = np.arcsin(lm[mask])
            phimod = np.arccos(L.flatten()[mask] / lm[mask])
            phi = np.where(M.flatten()[mask] < 0, phimod, -phimod)
            phi[np.isnan(phi)] = 0.0


            # Generate map from interpolation
            lm_map[i][mask] = pixelfunc.get_interp_val(hp, theta, phi)
            #lm_map[i] = lm_map[i].reshape((self.ncells, self.ncells))

            #print i, lm.max(), L.max(), M.max(), lm_map[i][500]

        lm_map = lm_map.reshape((len(self.f0), self.ncells, self.ncells))
        return lm_map


def randsphere(n, theta_range = (0,np.pi), phi_range = (0,2*np.pi)):
    """
    Generate random angular theta, phi points on the sphere


    Parameters
    ----------
    n: integer
        The number of randoms to generate

    Returns
    -------
    theta,phi: tuple of arrays
    """
    phi = np.random.random(n)
    phi = phi*(phi_range[1]-phi_range[0]) + phi_range[0]

    cos_theta_min=np.cos(theta_range[0])
    cos_theta_max=np.cos(theta_range[1])

    v = np.random.random(n)
    v *= (cos_theta_max-cos_theta_min)
    v += cos_theta_min

    theta = np.arccos(v)

    return theta, phi


class PoissonClustering_Spherical(PoissonClustering_FlatSky):
    _defaults = PoissonClustering_FlatSky._defaults
    _defaults.update({"nside":64,
                      "lmax":1000})
    @property
    def _powerbox(self):
        raise AttributeError("'PoissonClustering_Spherical' object has no attribute '_powerbox'")

    @property
    def _healpix_deltax(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        l = np.arange(1,self.params['lmax'])
        cls = self.params['power_spectrum'](l)
        return sphtfunc.synfast(cls, self.params['nside'])

    def source_positions_ang(self, nbar):
        dx = self._healpix_deltax+1

        # Limit the range on the sky that we populate for efficiency
        maxtheta = self.get_maxtheta()
        angular_size = 2*np.pi*(1 - np.cos(maxtheta))

        ntot = np.random.poisson(nbar * angular_size)
        pos = np.zeros((2,ntot))

        ndone = 0
        nleft = ntot

        pixweights = dx / np.max(dx)
        overweight = len(pixweights)/np.sum(pixweights)

        while nleft > 0:
            theta, phi = randsphere(int(nleft*overweight), theta_range=(0,maxtheta))
            pix = pixelfunc.ang2pix(self.params['nside'], theta, phi)
            weights = pixweights[pix]

            rnd = np.random.random(len(theta))
            w, = np.where(rnd < weights)
            if w.size > 0:
                if w.size >  nleft:
                    theta = theta[w][:nleft]
                    phi = phi[w][:nleft]
                else:
                    theta = theta[w]
                    phi = phi[w]

                pos[:,ndone:ndone+min(w.size,nleft)] = np.array([theta,phi])

            ndone += w.size
            nleft -= w.size

        return pos

    def get_maxtheta(self):
        maxl = min(np.sqrt(2) * np.max(self.sky_size.value/2), 1)
        return np.arcsin(maxl)

    def source_positions(self, nbar):
        theta, phi = self.source_positions_ang(nbar)
        l = np.sin(theta) * np.cos(phi)
        m = np.sin(theta) * np.sin(phi)

        return np.array([l,m])

    def sky(self, pos, fluxes, spec_indices):
        return PoissonClustering_FlatSky.sky(self,pos, fluxes, spec_indices)

#
# def rotate_map(hmap, rot_theta, rot_phi):
#     nside = hp.npix2nside(len(hmap))
#
#     # Get theta, phi for non-rotated map
#     t, p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))  # theta, phi
#
#     # Define a rotator
#     r = hp.Rotator(deg=False, rot=[rot_phi, rot_theta])
#
#     # Get theta, phi under rotated co-ordinates
#     trot, prot = r(t, p)
#
#     # Inerpolate map onto these co-ordinates
#     rot_map = hp.get_interp_val(hmap, trot, prot)
#
#     return rot_map

if HAVE_PYGSM:
    class GSM(PureClustering_Spherical):
        _defaults = dict(use_2008=False,
                         theta0=0,
                         phi0=0,
                         nu0 = 150.,
                         low_res=False)

        @property
        def _healpix_deltax(self):
            raise AttributeError("'GSM' object has no attribute '_healpix_deltax'")

        def healpix_brightness(self, sbar):
            if self.params['use_2008']:
                gsm = GlobalSkyModel(unit="MJysr", theta_rot = self.params['theta0'], phi_rot = self.params['phi0'],
                                     resolution="lo" if self.params['low_res'] else "hi")
            else:
                gsm = GlobalSkyModel2016(unit="MJysr", theta_rot = self.params['theta0'], phi_rot = self.params['phi0'],
                                         resolution="lo" if self.params['low_res'] else "hi")  #This only works for the steven-murray/PyGSM fork.

            gsm.generate(self.params['nu0'] * self.f0)

            data = np.atleast_2d(gsm.generated_map_data) * 1e6
            # rot_map = np.zeros_like(data)
            #
            # for i in range(len(self.f0)):
            #     rot_map[i] = 1e6 * rotate_map(data[i], self.params['theta0'], self.params['phi0']) #1e6 because units are MJy

            return data