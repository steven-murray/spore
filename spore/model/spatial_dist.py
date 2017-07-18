import numpy as np
from cached_property import cached_property
import spore.measure.unit_conversions as uc

from powerbox import LogNormalPowerBox, PowerBox
from scipy.interpolate import griddata
from _framework import Component as Cmpt

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
        self.f0 = np.array(f0)

        if np.isscalar(sky_size):
            self.sky_size = sky_size/self.f0
        else:
            self.sky_size = sky_size

        self.ncells = ncells
        self.seed = seed

    @cached_property
    def resolution(self):
        return self.sky_size/self.ncells

    @cached_property
    def lgrid(self):
        return [np.arange(-s.value/2,s.value/2,r.value)[:self.ncells] * uc.un.rad for s,r in zip(self.sky_size,self.resolution)]

    @cached_property
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

    @cached_property
    def _powerbox(self):
        if self.params['use_lognormal']:
            return LogNormalPowerBox(N=self.ncells, pk=self.params['power_spectrum'],
                                     dim=2, boxlength=np.max(self.sky_size).value,a=0,b=2*np.pi)
        else:
            return PowerBox(N=self.ncells, pk=self.params['power_spectrum'],
                            dim=2, boxlength=np.max(self.sky_size).value,a=self.a,b=2*np.pi)

    def sky(self, sbar):
        if self.seed is not None:
            np.random.seed(self.seed)

        density = self._powerbox.delta_x() + 1
        coords_x, coords_y = np.meshgrid(self.lgrid[0], self.lgrid[0])

        Sbins = np.zeros((len(self.f0), self.ncells, self.ncells))
        for i, f0 in enumerate(self.f0):
            if i == 0:
                Sbins[i] = sbar[i] * density
            else:
                x, y = np.meshgrid(self.lgrid[i], self.lgrid[i])
                Sbins[i] = sbar[i] * griddata((coords_x.flatten(), coords_y.flatten()), density.flatten(), (x, y),
                                              method="cubic")

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

        pos = self._powerbox.create_discrete_sample(nbar.value)
        return pos.T

    def sky(self, pos, fluxes, spec_indices):
        sbins = [0] * len(self.sky_size)
        for i, (sz, f0) in enumerate(zip(self.sky_size.value, self.f0)):
            mask = np.logical_and(np.logical_and(pos[0] > -sz / 2, pos[0] < sz / 2),
                                  np.logical_and(pos[1] > -sz / 2, pos[1] < sz / 2))
            sbins[i] = np.histogram2d(pos[0,mask], pos[1,mask], bins=np.concatenate((self.lgrid[i], [sz / 2])),
                                      weights=fluxes[mask] * f0 ** (-spec_indices[mask]))[0] / self.cell_area[i]

        return np.array(sbins)
