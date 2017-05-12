"""
This module contains functions which take an input visibility grid and convert it to a power spectrum
"""
import numpy as np
from powerbox import dft
dft.THREADS = 1
from powerbox.tools import angular_average
import unit_conversions as uc
from spore.fortran_routines.resample import lay_ps_map

def power_spec_from_visibility(visibility,baselines, nu, beam_radius = 10., umax=300,n_u=1200, Aeff=20.,
                               taper=None, umin= 0):
    """
    Given an irregularly-sampled visibility at multiple frequencies, determine the 2D power spectrum.

    Parameters
    ----------
    visibility : 2D-array
        An array with the first dimension corresponding to u,v position, and second dimension corresponding
        to frequency. NOTE: Frequency should be ascending.

    baselines : 2D-array
        Array of shape (nbaselines, 2) giving the baseline displacement vectors (x,y) in meters.

    nu : 1D-array
        The frequencies of the observation, in MHz. Should be ascending.

    beam_radius : float
        Determines the extent of the beam kernel used when resampling the u,v points (it is multiplied by the
        Gaussian width to determine total radius). Lower values will decrease computation time.

    Returns
    -------
    ps_2d : 2D-array
        The 2D power spectrum, as a function of kperp, kpar, in mK^2 h^-3 Mpc^3

    kperp : 1D-array
        The kperp values corresponding to ps_2d, in h/Mpc

    kpar : 1D-array
        The kpar values corresponding to ps_2d, in h/Mpc
    """
    # Create a master grid in u,v.
    master_u = np.linspace(-umax, umax, n_u)
    UMASTER, VMASTER = np.meshgrid(master_u, master_u)

    # Resample the visiblities onto the master grid
    ps_map_rl = lay_ps_map(nu, np.real(visibility), baselines, UMASTER, VMASTER, beam_radius)
    ps_map_im = lay_ps_map(nu, np.imag(visibility), baselines, UMASTER, VMASTER, beam_radius)
    ps_map_rl[np.isnan(ps_map_rl)] = 0.
    ps_map_im[np.isnan(ps_map_im)] = 0.

    # Do the FT in the nu plane.
    ps_tot = np.zeros((len(ps_map_rl), len(ps_map_rl), len(nu)),dtype="complex128")
#    def do_fft(i,j):
#        ps_tot[i, j], eta = dft.fft(ps_map_rl[i, j] + 1j*ps_map_im[i, j], L=nu[-1] - nu[0], a=0, b=2*np.pi)
#        return eta

    #eta = ProcessPool(cpu_count()).map(do_fft,range(len(ps_map_rl)),range(len(ps_map_rl)))[0]

    if taper is not None:
        taper = taper(len(nu))
    else:
        taper = 1

    for i in range(len(ps_map_rl)):
       for j in range(len(ps_map_rl)):
           ps_tot[i, j], eta = dft.fft(taper*(ps_map_rl[i, j] + 1j*ps_map_im[i, j]),
                                       L=nu[-1] - nu[0], a=0, b=2*np.pi)

    # Generate PS from FT
    ps = np.abs(ps_tot)**2

    # Perform cylindrical averaging.
    coords = np.sqrt(UMASTER ** 2 + VMASTER ** 2)
    ps_2d = np.zeros((100, len(nu)))
#    def do_angavg(i):
#        ps_2d[:, i], ubins = angular_average(ps[:, :, i], coords, 100)

#    ubins = ProcessPool(cpu_count()).map(do_angavg(range(len(nu))))[0]
    for i in range(len(nu)):
        ps_2d[:, i], ubins = angular_average(ps[:, :, i], coords, 100)

    eta = eta[0]/uc.un.Hz/1e6
    ubins = ubins/uc.un.rad

    z = (1420.0/nu.min()) - 1
    kpar = eta.to(uc.hub / uc.un.Mpc, equivalencies=uc.cosmo_21cm_los_equiv(z))
    kperp = ubins.to(uc.hub / uc.un.Mpc, equivalencies=uc.cosmo_21cm_angle_equiv(z))

#    ps_2d = uc.srMHz_mpc3(ps_2d, nu*uc.un.MHz,Aeff)
    ps_2d = ps_2d * uc.un.Jy**2 * uc.un.MHz**2
    ps_2d = uc.jyhz_to_mKMpc(ps_2d, nu*uc.un.MHz, Aeff, verbose=False)

    # Restrict to the positive kpar
    ps_2d = ps_2d[:, kpar > 0]
    kpar = kpar[kpar > 0]

    # Restrict to positive PS (some may be zero or NaN).
    kperp = kperp[ps_2d[:, 0] > 0]
    ubins = ubins[ps_2d[:, 0] > 0]
    ps_2d = ps_2d[ps_2d[:, 0] > 0]

    # Restrict to valid u range
    kperp = kperp[ubins.value>umin]
    ps_2d = ps_2d[ubins.value>umin]
    ubins = ubins[ubins.value>umin]

    return ps_2d, kperp, kpar, ubins