"""
A module for producing point-source foreground mocks.
"""
import numpy as np
from cached_property import cached_property

from spore.model.spectral_index import  UniversalDist
from powerbox.powerbox import _magnitude_grid
from powerbox.dft import fft, fftfreq

from spore.fortran_routines.direct_ft import direct_ft

class ForegroundComponentMock(object):
    def __init__(self, beam_model, spec_index_model, spatial_dist):
        self.beam_model = beam_model
        self.spec_index_model = spec_index_model
        self.spatial_dist = spatial_dist

        self.f0 = self.spatial_dist.f0

        if np.max(self.spatial_dist.sky_size) < 6*np.max(self.beam_model.sigma(self.f0)):
            print("WARNING: The sky size does not allow for the beam to attenuate enough")

    @cached_property
    def sky(self):
        raise NotImplementedError("This is not implemented in the base class")

    def beam_attenuation(self):
        """
        Return the beam attenuation/response at each gridpoint on the sky (i.e. on :attr:`.spatial_dist.lgrid`)
        """
        L = np.zeros((len(self.f0), self.spatial_dist.ncells, self.spatial_dist.ncells))
        M = np.zeros((len(self.f0), self.spatial_dist.ncells, self.spatial_dist.ncells))

        for i in range(len(self.f0)):
            L[i], M[i] = np.meshgrid(self.spatial_dist.lgrid[i], self.spatial_dist.lgrid[i])
        return self.beam_model.beam(L, M, self.f0)

    def visible_sky(self):
        """
        The beam-attenuated brightness of the sky. A 3D array, with first index corresponding to `f0`, and last two
        corresponding to sky position (l,m).
        """
        return self.beam_attenuation() * self.sky

    @cached_property
    def visibility(self):
        """
        The visibilities of :meth:`visible_sky` over a uniform (u,v)-grid,  defined as :attr:`ugrid_raw0`.
        """
        # Note that base fft in numpy is equivalent to \int f(x) e^{-2pi i kx} dx
        return ((fft(self.visible_sky(), axes=(-2,-1),a=0,b=2*np.pi)[0]).T*self.spatial_dist.cell_area).T

    @cached_property
    def ugrid_raw0(self):
        """
        The 1D grid of u (fourier dual of l) at nu0. Used as the grid on a side of a 2D grid of the sky.
        """
        return fftfreq(self.spatial_dist.ncells, d=self.spatial_dist.resolution[0],b=2*np.pi)

    @cached_property
    def _ugrid0(self):
        """
        The magnitude of u at each point of a 2D grid of the sky, in the fourier-dual of l.
        """
        return _magnitude_grid(self.ugrid_raw0,2)

    @cached_property
    def _weights_edges_centres(self):
        weights, edges = np.histogram(self._ugrid0.flatten(), bins=self.spatial_dist.ncells)
        binav = np.histogram(self._ugrid0.flatten(), bins=edges, weights=self._ugrid0.flatten())[0]/weights
        return weights, edges, binav

    @cached_property
    def ugrid(self):
        "The 1D grid of u (centres) after angular averaging"
        return self._weights_edges_centres[2]

    @cached_property
    def _ugrid_edges(self):
        "The 1D grid edges of u after angular averaging"
        return self._weights_edges_centres[1]

    @cached_property
    def _grid_weights(self):
        "The number of 2D grid points contributing to each 1D bin of u."
        return self._weights_edges_centres[0]

    def _circavg(self,X):
        "Return the circular average of X in :attr:`ugrid`"
        return np.histogram(self._ugrid0.flatten(), bins=self._ugrid_edges,
                            weights=X.flatten())[0]/self._grid_weights

    @cached_property
    def visibility_1d(self):
        """
        The complex visibility, after circular averaging. Corresponds to :attr:`ugrid`
        """
        vis = np.zeros((len(self.f0), len(self.ugrid)), dtype="complex128")
        for i in range(len(self.f0)):
            vis_rl = self._circavg(np.real(self.visibility[i]))
            vis_im = self._circavg(np.imag(self.visibility[i]))
            vis[i] = vis_rl+1j*vis_im
        return vis

    @cached_property
    def visibility_squared_circular(self):
        """
        The circular average of the modulus squared of the visibility. Corresponds to :attr:`ugrid`.
        """
        n = len(self.f0)
        vis2 = np.zeros((n,n, len(self.ugrid)))

        for i in range(n):
            for j in range(n - i):
                v2 = self.visibility[i]*np.conj(self.visibility[j+i])

                vis2[i,j+i] = self._circavg(v2)
                vis2[j+i,i] = self._circavg(v2)
        return vis2


class PointSourceForegrounds(ForegroundComponentMock):
    """
    A full point-source foreground model.

    This class combines several sub-models to fully specify a sky distribution across flux, frequency and angle,
    and it also provides necessary methods to calculate visibilities and any other derived quantities.

    Parameters
    ----------
    source_counts : :class:`spore.model.source_counts.SourceCounts` instance
        An instance of `SourceCounts` (via a subclass) which contains a model for the source count distribution.

    beam_model : :class:`spore.model.beam_model.Beam` instance
        An instance of `Beam` (via a subclass) which contains a model for the instrument primary beam.

    spec_index_model : :class:`spore.model.spectral_index.SpecIndex` instance
        An instance of `SpecIndex` (via a subclass) which contains a model for the spectral index distribution

    spatial_dist : :class:`spore.model.spatial_dist.SpatialDistribution` instance
        An instance of `SpatialDistribution` (via a subclass) which contains a model for the spatial distribution of flux.
    """

    def __init__(self, source_counts, *args, **kwargs):

        super(PointSourceForegrounds,self).__init__(*args, **kwargs)
        self.source_counts = source_counts

        if not np.all(self.source_counts.f0 == self.f0):
            raise ValueError("The frequencies of observation must be the same for spatial_model and source_counts")


    @cached_property
    def sky(self):
        """
        Returns
        -------
        sbins : (nf0,ncells,ncells)-array
            A gridded realisation of the total flux density in each cell at each frequency. Units Jy/Hz/rad^2
        """
        if not hasattr(self.spatial_dist, "source_positions"):
            return self.spatial_dist.sky(self.source_counts.total_flux_density)

        else:
            pos = self.spatial_dist.source_positions(self.source_counts.total_number_density.value)
            n  = len(pos[0])
            return self.spatial_dist.sky(pos,
                                         self.source_counts.sample_source_counts(n),
                                         self.spec_index_model.sample(n))


class PointSourceForegroundsDirect(PointSourceForegrounds):
    def __init__(self, u0, v0, *args, **kwargs):
        """

        Parameters
        ----------
        u0, v0 : array-like
            The u,v co-ordinates of the baselines at nu0.
        """
        self.u0, self.v0 = u0, v0
        super(PointSourceForegroundsDirect, self).__init__(*args, **kwargs)

        # Ensure the spatial distribution allows for point-sources
        if not hasattr(self.spatial_dist, "source_positions"):
            raise TypeError("The spatial distribution for a direct model must allow for point sources")

    @cached_property
    def _source_positions(self):
        return self.spatial_dist.source_positions(self.source_counts.total_number_density.value)

    @cached_property
    def beam_attenuation(self):
        """
        Return the beam attenuation/response at each source position (shape [N_nu, N_u])
        """
        raise NotImplementedError("beam_attenuation not implemented for the Direct model.")

    @cached_property
    def visible_sky(self):
        raise NotImplementedError("not implemented for the Direct model.")

    @cached_property
    def source_flux(self):
        return self.source_counts.sample_source_counts(len(self._source_positions[0]))

    @cached_property
    def source_spec_slope(self):
        return self.spec_index_model.sample(len(self._source_positions[0]))

    def source_visible_flux(self,f0):
        return self.beam_model.beam(self._source_positions[0], self._source_positions[1], f0) * self.source_flux * f0 ** (
                -self.source_spec_slope)

    @cached_property
    def visibility(self):
        """
        The visibility as a function of frequency.
        """

        visibility = np.zeros((len(self.f0), len(self.u0)),
                              dtype='complex128')

        for i,f0 in enumerate(self.f0):
            visible_flux = self.source_visible_flux(f0)
            visibility[i] = direct_ft(f0,self.u0, self.v0, self._source_positions, visible_flux)

        return visibility


class GalacticForegrounds(ForegroundComponentMock):
    def __init__(self, mean_sky_brightness, *args, **kwargs):
        self.mean_sky_brightness = mean_sky_brightness
        super(GalacticForegrounds, self).__init__(*args, **kwargs)

        assert np.issubclass_(self.spec_index_model, UniversalDist), "For Galactic Foregrounds, spectral index distribution must be Universal."

    @cached_property
    def sky(self):
        return self.spatial_dist.sky(self.mean_sky_brightness * self.f0 ** -self.spec_index_model.sample(1))



class ThermalForegrounds(ForegroundComponentMock):
    def __init__(self, Aeff, Tsys, deltaT, *args, **kwargs):
        self.Aeff = Aeff
        self.Tsys = Tsys
        self.deltaT = deltaT

        super(ThermalForegrounds, self).__init__(*args, **kwargs)

        assert np.all(np.diff(np.diff(self.f0))==0), "f0 should be regular"

    @cached_property
    def sky(self):
        dnu = self.beam_model.nu0 * (self.f0[1] - self.f0[0])
        sigma =  10**26 * 2*un.kB * self.Tsys / self.Aeff/ np.sqrt(dnu * self.deltaT)
