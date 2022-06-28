"""
This module contains functions for converting a simulation output (eg. 21cmFAST) to a standard visibility.
"""
import numpy as np
from tocm_tools import readbox
from astropy.cosmology import Planck15
from spore.model.beam import CircularGaussian
from powerbox import dft
from scipy.interpolate import RectBivariateSpline
from astropy.constants import k_B
from scipy.interpolate import InterpolatedUnivariateSpline as spline

def pos_to_baselines(pos):
    # ignore up.
    x, y = pos[:, 0], pos[:, 1]
    ind = np.tril_indices(len(x), k=-1)
    Xsep = np.add.outer(x, -x)[ind]
    Ysep = np.add.outer(y, -y)[ind]
    return np.array([Xsep.flatten(), Ysep.flatten()]).T


def baselines_to_u0(baselines, nu0, umax=None, ret_baselines=False):
    u0 =  baselines * nu0 * 1e6/3e8

    if umax:
        n = len(u0)
        indx = np.where(np.logical_and(np.abs(u0[:, 0]) < umax, np.abs(u0[:, 1]) < umax))[0]
        u0 = u0[indx]
        baselines = baselines[indx]

        if len(u0)< n:
            print("%s baselines were removed due to being longer than the maximum of %s."%(n-len(u0), umax))

    if ret_baselines:
        return u0, baselines
    else:
        return u0

def u0_to_u(u0, f0):
    return u0*f0


def get_sim_21cmfast(box):
    """
    This function should act as a kind of template for any other simulation types that might be imported, but at this
    stage it is the only one implemented.

    Parameters
    ----------
    fname : str
        The filename of the simulation to be imported

    Returns
    -------

    """
    if isinstance(box, str):
        sim = readbox(box)
    else:
        sim = box

    fname = sim.param_dict['filename']

    # Get the zstart and end values
    bits = fname.split("_")
    for b in bits:
        if b.startswith("zstart"):
            zstart = float(b[6:])
        if b.startswith("zend"):
            zend = float(b[4:])

    return sim.box_data, sim.param_dict['dim'], zstart, zend, sim.param_dict['BoxSize']


def get_cut_box(box,numin=150., numax=161.15):
    box, N, zstart, zend, L = get_sim_21cmfast(box)

    # GENERATE FREQUENCIES AND CUT THE BOX
    box = box[:, :, ::-1]  # reverse last axis
    d = np.linspace(Planck15.comoving_distance(zstart), Planck15.comoving_distance(zend), N)

    _z = np.linspace(zstart, zend, N)
    dspline = spline(Planck15.comoving_distance(_z), _z)
    z = dspline(d[::-1])
    #    z = np.linspace(zend, zstart, N)
    nu = 1420./(z + 1)

    mask = np.logical_and(nu > numin, nu < numax)

    box = box[:, :, mask]
    z = z[mask]
    d = d[mask].value
    nu = nu[mask]

    return box, N, L * Planck15.h, d*Planck15.h, nu, z


def sim_to_vis(box, antenna_pos, numin=150., numax=180., cosmo=Planck15, beam=CircularGaussian):
    """

    Parameters
    ----------
    box : :class:`tocm_tools.Box` instance or str
        Either a 21cmFAST simulation box, or a string specifying a filename to one.

    antenna_pos : array
        2D array of (x,y) positions of antennae in meters (shape (Nantennate,2)).

    numin : float
        Minimum frequency (in MHz) to include in the "observation"

    numax : float
        Maximum frequency (in MHz) to include in the "observation"

    cosmo : :class:`astropy.cosmology.FLRW` instance
        The cosmology to use for all calculations

    beam : :class:`spore.model.beam.Beam` instance
        The telescope beam model to use.

    Returns
    -------
    uvsample : array
        2D complex array in which the first dimension has length of Nbaselines, and the second has Nnu. This is the
        Fourier Transform of the sky at the baselines, has units of Jy.

    baselines : array
        The baseline vectors of the observation (shape (Nbaseline,2)). Units m.

    nu : array
        1D array of frequencies of observation. Units MHz.
    """

    # READ THE BOX
    box, Nbox, L, d, nu, z = get_cut_box(box,numin,numax)

    lam = 3e8/(nu*1e6)  # in m

    # Convert to specific intensity (Jy/sr)
    #Jy/(sr?) mK->K     to Jy     (J/K) /m^2
    box    *= 1e-3     *  1e26   * 2* k_B/lam**2

    # INITIALISE A BEAM MODEL
    beam = beam(nu.min(), np.linspace(1, nu.max()/nu.min(), len(nu)))

    # Box width at different redshifts
    width = L/cosmo.angular_diameter_distance(z).value

    # Minimum umax available in simulation
    maxl = 2*np.sin(width.max()/2 - width.max()/Nbox/2)
    maxu = Nbox/2/maxl

    # GENERATE BASELINE VECTORS
    baselines = pos_to_baselines(antenna_pos)
    u0, baselines = baselines_to_u0(baselines, nu[0], maxu, ret_baselines=True)

    uvsample = np.zeros((len(baselines), len(nu)), dtype="complex128")


    # ATTENUATE BOX BY THE BEAM
    for i in range(len(nu)):
        dl = width[i]/Nbox

        l = np.sin(np.linspace(-width[i]/2 + dl/2, width[i]/2 - dl/2, Nbox))
        L, M = np.meshgrid(l, l)

        slice = box[:, :, i] * np.exp(-(L ** 2 + M ** 2)/(2*beam.sigma[i] ** 2))

        # Interpolate onto regular l,m grid
        spl_lm = RectBivariateSpline(l,l,slice)
        l = np.linspace(l.min(),l.max(),len(l))
        slice = spl_lm(l,l,grid=True)

        FT, freq = dft.fft(slice, L=l.max()-l.min(), a=0, b=2*np.pi)
        uvsample[:,i] = interpolate_visibility_onto_baselines(FT, freq[0], nu[i]/nu[0], u0)

    return uvsample, u0, nu, {'slice':slice,'box':box,"FT":FT, 'freq':freq, "l":l}


def interpolate_visibility_onto_baselines(visibility, ugrid, f0, u0):

    spl_rl = RectBivariateSpline(ugrid, ugrid, np.real(visibility))
    spl_im = RectBivariateSpline(ugrid, ugrid, np.imag(visibility))
    uv = u0_to_u(u0, f0)

    return spl_rl(uv[:, 0], uv[:, 1], grid=False) + 1j * spl_im(uv[:, 0], uv[:, 1], grid=False)