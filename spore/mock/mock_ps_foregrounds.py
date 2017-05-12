"""
A module for producing point-source foreground mocks.
"""
import numpy as np
from cached_property import cached_property
from spore.model.source_counts import PowerLawSourceCounts
from spore.model.beam import CircularGaussian

import spore.measure.unit_conversions as uc

from powerbox import LogNormalPowerBox, PowerBox
from powerbox.powerbox import _magnitude_grid
from powerbox.dft import fft,ifft,fftfreq

from scipy.interpolate import griddata

class PoissonProcessForegrounds(object):
    """
    A point-source foreground mock realisation which has no spatial correlations.

    Parameters
    ----------
    source_counts : SourceCounts instance
        An instance of a :class:`spore.model.source_counts.SourceCounts` class.

    beam_model :

    sky_size :

    ncells :

    seed :



    Assumptions
    -----------
    SED's are the same for every object, and are a power law in region of interest.
    """

    def __init__(self, source_counts, beam_model, sky_size=4,
                 ncells=100,seed=None):

        self.source_counts = source_counts
        self.beam_model = beam_model

        self.nu = self.source_counts.nu

        if not np.all(self.beam_model.nu == self.source_counts.nu):
            raise ValueError("The frequencies of observation must be the same for source_counts and beam_model")

        if sky_size is None:
            self.sky_size = 8 * self.beam_model.sigma
        else:
            self.sky_size = 2 * sky_size*self.beam_model.sigma

        self.ncells = ncells

        if np.max(self.sky_size) < 6*np.max(self.beam_model.sigma):
            print("WARNING: The sky size does not allow for the beam to attenuate enough")

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

    def _get_pointed_sky(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        # Get all sources within the largest window
        nexpected = self.source_counts.total_number_density * np.max(self.sky_size) ** 2
        ntot = np.random.poisson(nexpected)
        S = self.source_counts.sample_source_counts(ntot, ret_nu_array=False)

        sz = np.max(self.sky_size/2)
        pos = np.random.uniform(-sz,sz,size=(2,ntot))

        return S, pos

    @cached_property
    def sky(self):
        """
        Returns
        -------
        sbins : (ncells,ncells)-array
            A gridded realisation of the total flux density in each cell, at `nu[0]`. Units Jy/Hz/rad^2
        """
        # Note that while actually sampling the flux densities of each source is
        # horribly inefficient, it seems necessary. Otherwise the distribution of sbins is discrete.
        S, pos = self._get_pointed_sky()

        sbins = [0]*len(self.sky_size)
        for i, sz in enumerate(self.sky_size.value):
            mask = np.logical_and(np.logical_and(pos[0]>-sz/2,pos[0]<sz/2),np.logical_and(pos[1]>-sz/2,pos[1]<sz/2))
            sbins[i] = np.histogram2d(pos[0,mask],pos[1,mask],bins=np.concatenate((self.lgrid[i],[sz/2])),weights=S[mask])[0] /self.cell_area[i]

        return ((self.nu/self.nu[0])**-self.source_counts.spectral_index * np.array(sbins).T).T


    def beam_attenuation(self):

        L = np.zeros((len(self.nu),self.ncells,self.ncells))
        M = np.zeros((len(self.nu), self.ncells, self.ncells))

        for i in range(len(self.nu)):
            L[i], M[i] = np.meshgrid(self.lgrid[i], self.lgrid[i])
        return self.beam_model.beam(L,M)

    def visible_sky(self):
        """
        The beam-attenuated brightness of the sky. A 3D array, with first index corresponding to nu, and last two
        corresponding to sky position (l,m).
        """
        return self.beam_attenuation() * self.sky

    @cached_property
    def visibility(self):
        # Note that base fft in numpy is equivalent to \int f(x) e^{-2pi i kx} dx
        return ((fft(self.visible_sky(), axes=(-2,-1),a=0,b=2*np.pi)[0]).T*self.cell_area).T

    @cached_property
    def ugrid_raw0(self):
        return fftfreq(self.ncells, d=self.resolution[0],b=2*np.pi)

    @cached_property
    def _ugrid0(self):
        return _magnitude_grid(self.ugrid_raw0,2)

    @cached_property
    def _weights_edges_centres(self):
        weights, edges = np.histogram(self._ugrid0.flatten(), bins=self.ncells)
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
        vis = np.zeros((len(self.nu), len(self.ugrid)),dtype="complex128")
        for i in range(len(self.nu)):
            vis_rl = self._circavg(np.real(self.visibility[i]))
            vis_im = self._circavg(np.imag(self.visibility[i]))
            vis[i] = vis_rl+1j*vis_im
        return vis

    # def visibility_squared(self):
    #     """
    #     A 3D (nu,nu,ncells)-array, with each cross-correlation of nu containing the squared visibility as a function
    #     of u [defined as :meth:`ugrid_vis`].
    #     """
    #     return np.einsum("ijk,ljk->iljk",self.visibility,np.conj(self.visibility))# self.visibility * np.conj(self.visibility)

    @cached_property
    def visibility_squared_circular(self):
        n = len(self.nu)
        vis2 = np.zeros((n,n, len(self.ugrid)))

        for i in range(n):
            for j in range(n - i):
                v2 = self.visibility[i]*np.conj(self.visibility[j+i])

                vis2[i,j+i] = self._circavg(v2)
                vis2[j+i,i] = self._circavg(v2)
        return vis2

    # def _idx(self,i,j, n=None):
    #     "Return the index of a vector corresponding to symmetric array. By default, n is len(nu)"
    #     if n is None:
    #         n = len(self.nu)
    #     return n*i + j if j<i else n*j+i

class ClusteredForegrounds(PoissonProcessForegrounds):
    a = 0
    b = 2*np.pi

    def __init__(self,point_source_power_spec, use_lognormal=True, *args,**kwargs):
        self.point_source_power_spec = point_source_power_spec
        self.use_lognormal = use_lognormal

        super(ClusteredForegrounds,self).__init__(*args,**kwargs)

    @cached_property
    def powerbox(self):
        if self.use_lognormal:
            return LogNormalPowerBox(N=self.ncells, pk=self.point_source_power_spec,
                                     dim=2, boxlength=np.max(self.sky_size).value,a=self.a,b=self.b)
        else:
            return PowerBox(N=self.ncells, pk=self.point_source_power_spec,
                            dim=2, boxlength=np.max(self.sky_size).value,a=self.a,b=self.b)

    def _get_pointed_sky(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        pos = self.powerbox.create_discrete_sample(self.source_counts.total_number_density.value)
        S = self.source_counts.sample_source_counts(len(pos), ret_nu_array=False)
        return S, pos.T


class ClusteredForegroundsOnly(PoissonProcessForegrounds):
    a = 0
    b = 2*np.pi

    def __init__(self, point_source_power_spec, use_lognormal=True, *args, **kwargs):
        self.point_source_power_spec = point_source_power_spec
        self.use_lognormal = use_lognormal
        super(ClusteredForegroundsOnly, self).__init__(*args, **kwargs)

    @cached_property
    def powerbox(self):
        if self.use_lognormal:
            return LogNormalPowerBox(N=self.ncells, pk=self.point_source_power_spec,
                                     dim=2, boxlength=np.max(self.sky_size.value),a=self.a,b=self.b)
        else:
            return PowerBox(N=self.ncells, pk=self.point_source_power_spec,
                            dim=2, boxlength=np.max(self.sky_size).value,a=self.a,b=self.b)
    @cached_property
    def sky(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        density = self.powerbox.delta_x() + 1
        coords_x, coords_y = np.meshgrid(self.lgrid[0],self.lgrid[0])

        Sbins = np.zeros((len(self.nu),self.ncells,self.ncells))
        for i, nu in enumerate(self.nu):
            if i==0:
                Sbins[i] = self.source_counts.total_flux_density[i]*density
            else:
                x,y = np.meshgrid(self.lgrid[i],self.lgrid[i])
                Sbins[i] = griddata((coords_x.flatten(), coords_y.flatten()), density.flatten(), (x,y),method="cubic")*self.source_counts.total_flux_density[i]

        return Sbins


def visibility_covariance(foreground_model=PoissonProcessForegrounds, niter=30,seed=None,
                          ret_realisations=False,a=1,b=1,*args,**kwargs):
    if seed:
        np.random.seed(seed)

    cov = 0
    meanvis = 0
    realisations = []
    for i in range(niter):
        mdl = foreground_model(*args,**kwargs)
        mdl.a = a
        mdl.b = b
        if ret_realisations:
            realisations.append(mdl)

        meanvis += mdl.visibility_1d/niter
        cov += mdl.visibility_squared_circular/niter

    cov -= np.real(np.einsum('ij,kj->ikj',meanvis,np.conj(meanvis)))


    if cov.shape[0] == 1:
        cov = cov[0,0]

    if ret_realisations:
        return mdl.ugrid, cov, realisations
    else:
        return mdl.ugrid, cov
