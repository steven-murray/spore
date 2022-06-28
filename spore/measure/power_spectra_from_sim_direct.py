from powerbox import get_power
from spore.mock.tocm_tools import readbox
import numpy as np
from spore.mock.convert_sim_to_vis import get_cut_box
from spore.model.beam import CircularGaussian
from astropy.cosmology import Planck15
import psutil

def get_power_single_z(fname, get_delta=True, maxN=None, bins=50, forcerun=False):
    """
    Calculate the spherically-averaged power spectrum for a single-redshift box.
    
    Parameters
    ----------
    fname : str
        Filename of 21cmFAST simulation box. Should *not* be a lighttravel box.

    get_delta : bool, optional
        If True, return the dimensionless power spectrum, Delta^2, otherwise, return the volume-normalised power.
        
    maxN : int, optional
        The maximum number of cells per side to use in the estimation. Use to fit within memory. 
        
    bins : int, optional
        Number of k-bins to use (linearly spaced)
        
    forcerun : bool, optional
        Forces the function to run even if it thinks it will use too much memory.
        
    Returns
    -------
    pk : array
        The power spectrum as a function of k. Has units mK^2 if `get_delta` is True, otherwise has units
        mK^2 Mpc^3. 
        
    kbins : array
        The k values corresponding to pk. Has units 1/Mpc.

    """
    if fname.endswith("lighttravel"):
        raise ValueError("This function is intended for obtaining power spectra from boxes at a single redshift")

    box = readbox(fname)
    N = box.dim
    L = box.param_dict['BoxSize']
    box = box.box_data

    if maxN is None or maxN>N:
        maxN = N

    if maxN**3 * 8 * 3 > psutil.virtual_memory().available:
        msg = """
Required memory (%s GB) would be more than that available (%s GB), aborting. If you really want to run it, use forcerun.
"""%(maxN**3 * 8 * 3/1.0e9,  psutil.virtual_memory().available/1.0e9)
        if not forcerun:
            raise ValueError(msg)
        else:
            print "Warning: required memory (%s GB) may be more than that available (%s GB)."%(maxN**3 * 8 * 3/1.0e9,  psutil.virtual_memory().available/1.0e9)

    # Restrict the box for memory
    box = box[:maxN, :maxN, :maxN]
    L *= float(maxN)/N

    pk, kbins = get_power(box, L, bins=bins)

    if get_delta:
        pk *= kbins**3/(2*np.pi**2)
    return pk, kbins


def get_power_lightcone(fname, numin=150., numax=161.15, get_delta=True, bins=50, res_ndim=None, taper=None):
    """
    Calculate the spherically-averaged power spectrum in a segment of a lightcone. 

    Parameters
    ----------
    fname : str
        Filename of 21cmFAST simulation box. Should be a lighttravel box.

    numin, numax : float
        Min/Max frequencies of the "observation", in MHz. Used to cut the box before PS estimation.
        
    get_delta : bool, optional
        If True, return the dimensionless power spectrum, Delta^2, otherwise, return the volume-normalised power.

    bins : int, optional
        Number of k-bins to use (linearly spaced)
        
    res_ndim : int, optional
        Only perform angular averaging over first `res_ndim` dimensions. By default, uses all dimensions. 
        
    Returns
    -------
    pk : array
        The power spectrum as a function of k. Has units mK^2 if `get_delta` is True, otherwise has units
        mK^2 Mpc^3. 

    kbins : array
        The k values corresponding to pk. Has units 1/Mpc.

    """
    if res_ndim is not None and res_ndim != 3 and get_delta:
        raise NotImplementedError("Currently can't get Delta^2 for not fully-averaged data")

    box, N, L, d, nu, z = get_cut_box(fname, numin, numax)
    if taper is not None:
        box = box * taper(len(nu))

    res = get_power(box, [L, L, np.abs(d[-1] - d[0])], bins=bins, res_ndim=res_ndim)

    if get_delta:
        return res[0]*res[1]**3/(2*np.pi**2), res[1]

    else:
        P = res[0]
        kav = res[1]
        k_other = res[2][0]

        P = P[np.logical_not(np.isnan(kav))]

        P = P[:,k_other>0]

        kav = kav[np.logical_not(np.isnan(kav))]
        k_other = k_other[k_other>0]

        return P, kav, k_other, {"N":N,"L":L, "d":d, "nu":nu,"z":z}


def get_power_lightcone_beam(fname, numin=150., numax=161.15, get_delta=True, bins=50, beam_model = CircularGaussian):
    """
    Calculate the spherically-averaged power spectrum in a segment of a lightcone, after attenuation by a 
    given beam.

    Parameters
    ----------
    fname : str
        Filename of 21cmFAST simulation box. Should *not* be a lighttravel box.

    numin, numax : float
        Min/Max frequencies of the "observation", in MHz. Used to cut the box before PS estimation.

    get_delta : bool, optional
        If True, return the dimensionless power spectrum, Delta^2, otherwise, return the volume-normalised power.

    bins : int, optional
        Number of k-bins to use (linearly spaced)

    beam_model : `spore.model.beam.CircularBeam` subclass
        Defines the beam model
        
    Returns
    -------
    pk : array
        The power spectrum as a function of k. Has units mK^2 if `get_delta` is True, otherwise has units
        mK^2 Mpc^3. 

    kbins : array
        The k values corresponding to pk. Has units 1/Mpc.

    """
    box, N, L, d, nu, z = get_cut_box(fname, numin, numax)

    # INITIALISE A BEAM MODEL
    beam = beam_model(nu.min(), np.linspace(1, nu.max()/nu.min(), len(nu)))

    # ATTENUATE BOX BY THE BEAM
    width = L/Planck15.angular_diameter_distance(z).value
    vol = np.ones_like(box)

    for i in range(len(nu)):
        dl = width[i]/N

        l = np.sin(np.linspace(-width[i]/2 + dl/2, width[i]/2 - dl/2, N))
        X, M = np.meshgrid(l, l)

        vol[:, :, i] = np.exp(-(X ** 2 + M ** 2)/(2*beam.sigma[i] ** 2))
        box[:, :, i] = box[:, :, i]*vol[:, :, i]

    volfrac = np.sum(vol)/np.product(vol.shape)

    pk, kbins = get_power(box, [L, L, np.abs(d[-1] - d[0])], bins=bins)

    if get_delta:
        pk *= kbins**3/(2*np.pi**2)
        
    return pk, kbins, volfrac