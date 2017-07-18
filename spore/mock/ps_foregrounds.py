"""
A module for producing point-source foreground mocks.
"""
import numpy as np
from cached_property import cached_property
from spore.model.source_counts import PowerLawSourceCounts
from spore.model.beam import CircularGaussian


from spore.model.spatial_dist import PureClustering, PoissonClustering

from powerbox.powerbox import _magnitude_grid
from powerbox.dft import fft, fftfreq

class PointSourceForegrounds(object):
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

    spatial_model : :class:`spore.model.spatial_dist.SpatialDistribution` instance
        An instance of `SpatialDistribution` (via a subclass) which contains a model for the spatial distribution of flux.
    """

    def __init__(self, source_counts, beam_model, spec_index_model, spatial_dist):

        self.source_counts = source_counts
        self.beam_model = beam_model
        self.spec_index_model = spec_index_model
        self.spatial_dist = spatial_dist

        self.f0 = self.source_counts.f0

        if not np.all(self.beam_model.f0 == self.f0):
            raise ValueError("The frequencies of observation must be the same for source_counts and beam_model")

        if not np.all(self.spatial_dist.f0 == self.f0):
            raise ValueError("The frequencies of observation must be the same for spatial_model and source_counts")

        if np.max(self.spatial_dist.sky_size) < 6*np.max(self.beam_model.sigma):
            print("WARNING: The sky size does not allow for the beam to attenuate enough")

    @cached_property
    def sky(self):
        """
        Returns
        -------
        sbins : (ncells,ncells)-array
            A gridded realisation of the total flux density in each cell, at `nu[0]`. Units Jy/Hz/rad^2
        """
        if not hasattr(self.spatial_dist, "source_positions"):
            return self.spatial_dist.sky(self.source_counts.total_flux_density)

        else:
            pos = self.spatial_dist.source_positions(self.source_counts.total_number_density)
            n  = len(pos[0])
            return self.spatial_dist.sky(pos,
                                         self.source_counts.sample_source_counts(n),
                                         self.spec_index_model.sample(n))

    def beam_attenuation(self):

        L = np.zeros((len(self.f0), self.spatial_dist.ncells, self.spatial_dist.ncells))
        M = np.zeros((len(self.f0), self.spatial_dist.ncells, self.spatial_dist.ncells))

        for i in range(len(self.f0)):
            L[i], M[i] = np.meshgrid(self.spatial_dist.lgrid[i], self.spatial_dist.lgrid[i])
        return self.beam_model.beam(L, M)

    def visible_sky(self):
        """
        The beam-attenuated brightness of the sky. A 3D array, with first index corresponding to nu, and last two
        corresponding to sky position (l,m).
        """
        return self.beam_attenuation() * self.sky

    @cached_property
    def visibility(self):
        # Note that base fft in numpy is equivalent to \int f(x) e^{-2pi i kx} dx
        return ((fft(self.visible_sky(), axes=(-2,-1),a=0,b=2*np.pi)[0]).T*self.spatial_dist.cell_area).T

    @cached_property
    def ugrid_raw0(self):
        return fftfreq(self.spatial_dist.ncells, d=self.spatial_dist.resolution[0],b=2*np.pi)

    @cached_property
    def _ugrid0(self):
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
        "Return the circular average of X in u"
        return np.histogram(self._ugrid0.flatten(), bins=self._ugrid_edges,
                            weights=X.flatten())[0]/self._grid_weights

    @cached_property
    def visibility_1d(self):
        vis = np.zeros((len(self.f0), len(self.ugrid)), dtype="complex128")
        for i in range(len(self.f0)):
            vis_rl = self._circavg(np.real(self.visibility[i]))
            vis_im = self._circavg(np.imag(self.visibility[i]))
            vis[i] = vis_rl+1j*vis_im
        return vis

    @cached_property
    def visibility_squared_circular(self):
        n = len(self.f0)
        vis2 = np.zeros((n,n, len(self.ugrid)))

        for i in range(n):
            for j in range(n - i):
                v2 = self.visibility[i]*np.conj(self.visibility[j+i])

                vis2[i,j+i] = self._circavg(v2)
                vis2[j+i,i] = self._circavg(v2)
        return vis2

#
# class ClusteredForegrounds(PoissonProcessForegrounds):
#     a = 0
#     b = 2*np.pi
#
#     def __init__(self,point_source_power_spec, use_lognormal=True, *args,**kwargs):
#         self.point_source_power_spec = point_source_power_spec
#         self.use_lognormal = use_lognormal
#
#         super(ClusteredForegrounds,self).__init__(*args,**kwargs)
#
#     @cached_property
#     def powerbox(self):
#         if self.use_lognormal:
#             return LogNormalPowerBox(N=self.ncells, pk=self.point_source_power_spec,
#                                      dim=2, boxlength=np.max(self.sky_size).value,a=self.a,b=self.b)
#         else:
#             return PowerBox(N=self.ncells, pk=self.point_source_power_spec,
#                             dim=2, boxlength=np.max(self.sky_size).value,a=self.a,b=self.b)
#
#     def _get_pointed_sky(self):
#         if self.seed is not None:
#             np.random.seed(self.seed)
#
#         pos = self.powerbox.create_discrete_sample(self.source_counts.total_number_density.value)
#         S = self.source_counts.sample_source_counts(len(pos), ret_nu_array=False)
#         gamma = self.spec_index_model.sample(len(pos))
#         return S, pos.T, gamma
#
#
# class ClusteredForegroundsOnly(PoissonProcessForegrounds):
#     a = 0
#     b = 2*np.pi
#
#     def __init__(self, point_source_power_spec, use_lognormal=True, *args, **kwargs):
#         self.point_source_power_spec = point_source_power_spec
#         self.use_lognormal = use_lognormal
#         super(ClusteredForegroundsOnly, self).__init__(*args, **kwargs)
#
#     @cached_property
#     def powerbox(self):
#         if self.use_lognormal:
#             return LogNormalPowerBox(N=self.ncells, pk=self.point_source_power_spec,
#                                      dim=2, boxlength=np.max(self.sky_size.value),a=self.a,b=self.b)
#         else:
#             return PowerBox(N=self.ncells, pk=self.point_source_power_spec,
#                             dim=2, boxlength=np.max(self.sky_size).value,a=self.a,b=self.b)
#     @cached_property
#     def sky(self):
#         if self.seed is not None:
#             np.random.seed(self.seed)
#
#         density = self.powerbox.delta_x() + 1
#         coords_x, coords_y = np.meshgrid(self.lgrid[0],self.lgrid[0])
#
#         Sbins = np.zeros((len(self.nu),self.ncells,self.ncells))
#         for i, nu in enumerate(self.nu):
#             if i==0:
#                 Sbins[i] = self.source_counts.total_flux_density[i]*density
#             else:
#                 x,y = np.meshgrid(self.lgrid[i],self.lgrid[i])
#                 Sbins[i] = griddata((coords_x.flatten(), coords_y.flatten()), density.flatten(), (x,y),method="cubic")*self.source_counts.total_flux_density[i]
#
#         return Sbins

