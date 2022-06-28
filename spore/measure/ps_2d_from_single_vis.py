"""
This module contains functions which take an input visibility grid and convert it to a power spectrum
"""
import numpy as np
from powerbox import dft
dft.THREADS = 1
from powerbox.tools import angular_average_nd
from spore.common import unit_conversions as uc
from spore.fortran_routines.resample import lay_ps_map


def grid_visibilities(visibility,baselines, nu, beam_radius = 10., umax=300,n_u=1200):
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
    vis_rl = lay_ps_map(nu, np.real(visibility), baselines, UMASTER, VMASTER, beam_radius)
    vis_im = lay_ps_map(nu, np.imag(visibility), baselines, UMASTER, VMASTER, beam_radius)
    vis_rl[np.isnan(vis_rl)] = 0.
    vis_im[np.isnan(vis_im)] = 0.

    return vis_rl + 1j * vis_im


def vis_to_3d_ps(vis, dnu, taper=None):
    """
    Apply a taper and perform a FT over the frequency dimension of a grid of visibilities to get the 3D power spectrum.

    Parameters
    ----------
    vis : 3D array
        Array with first axis corresponding to nu and final two axes corresponding to u,v.

    dnu : float
        The (regular) interval between frequency bins.

    taper : callable, optional
        A taper/filter function to apply over the frequency axis.

    Returns
    -------
    ps_3d : 3D array
        3D Power Spectrum, with first axis corresponding to frequency.

    eta : 1D array
        The Fourier-dual of input frequencies.
    """
    if taper is not None:
        taper = taper(len(vis))
    else:
        taper = 1

    # Do the DFT to eta-space
    vistot, eta = dft.fft((taper * vis.value.T).T,
                          L=dnu.value, a=0, b=2 * np.pi,
                          axes=(0,))

    ps_3d = np.abs(vistot)**2  # Form the power spectrum

    # ps_3d = np.zeros(vis.shape)
    # for i in range(vis.shape[1]):
    #     for j in range(vis.shape[1]):
    #         vistot, eta = dft.fft(taper * vis[:,i, j].value,
    #                               L=dnu.value, a=0, b=2 * np.pi)
    #
    #         ps_3d[:,i, j] = np.abs(vistot)**2

    return ps_3d, eta[0]  # eta is a list of arrays, so take first (and only) entry


def ps_3d_to_ps_2d(ps_3d, u, nu, bins=100):
    """
    Take a 3D power spectrum and return a cylindrically-averaged 2D power spectrum.

    Parameters
    ----------
    ps_3d : 3D array
        The power spectrum in 3D, with first axis corresponding to frequency.

    u : 1D array
        The grid-coordinates along a side of the `ps_3d` array. Assumes that the (u,v) grid is square.

    nu : 1D array
        The frequencies corresponding to the first dimension of `ps_3d`.

    bins : int
        Number of (regular linear) bins to form the average into.

    Returns
    -------
    ps_2d : 2D array
        The circularly-averaged PS, with first axis corresponding to frequency.

    ubins : 1D array
        Length `bins` array giving the average central-bin co-ordinate for u after averaging.
    """
    # Perform cylindrical averaging.
    ps_2d, ubins, _ = angular_average_nd(field=ps_3d.T, coords=[u,u,nu], bins=bins, n=2)
    return ps_2d.T, ubins


def correct_raw_2d_ps(ps_2d, kperp, kpar, ubins, umin=0):
    "Make simple cuts on a raw power spectrum to make it suitable for viewing."
    # Restrict to the positive kpar
    ps_2d = ps_2d[kpar > 0, :]
    kpar = kpar[kpar > 0]

    # Restrict to positive PS (some may be zero or NaN).
    # kperp = kperp[ps_2d[0] > 0]
    # ubins = ubins[ps_2d[0] > 0]
    # ps_2d = ps_2d[:, ps_2d[0] > 0]

    # Restrict to valid u range
    kperp = kperp[ubins.value > umin]
    ps_2d = ps_2d[:, ubins.value > umin]
    ubins = ubins[ubins.value > umin]

    return ps_2d, ubins, kpar, kperp


def power_spec_from_visibility(visibility,baselines, nu, beam_radius = 10., umax=300,n_u=1200, Aeff=20., n_ubins=100,
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
    Aeff = uc.ensure_unit(Aeff, uc.un.m**2)
    vis_grid = grid_visibilities(visibility, baselines, nu, beam_radius, umax,n_u)

    # Do the FT in the nu plane.
    ps_3d, eta = vis_to_3d_ps(vis_grid, nu[-1] - nu[0], taper)

    # Perform cylindrical averaging.
    # coords = np.sqrt(UMASTER ** 2 + VMASTER ** 2)
    # ps_2d = np.zeros((100, len(nu)))
    # for i in range(len(nu)):
    #     ps_2d[:, i], ubins = angular_average(ps_3d[:, :, i], coords, 100)

    # Create a master grid in u,v.
    master_u = np.linspace(-umax, umax, n_u)
    ps_2d, ubins = ps_3d_to_ps_2d(ps_3d, master_u, nu, bins=n_ubins)

    eta = eta/uc.un.Hz/1e6
    ubins = ubins/uc.un.rad

    z = (1420.0/nu.min()) - 1
    kpar = eta.to(uc.hub / uc.un.Mpc, equivalencies=uc.cosmo_21cm_los_equiv(z))
    kperp = ubins.to(uc.hub / uc.un.Mpc, equivalencies=uc.cosmo_21cm_angle_equiv(z))


#    ps_2d = uc.srMHz_mpc3(ps_2d, nu*uc.un.MHz,Aeff)
    ps_2d = ps_2d * uc.un.Jy**2 * uc.un.MHz**2
    ps_2d = uc.jyhz_to_mKMpc_per_h(ps_2d, nu*uc.un.MHz, Aeff, verbose=False)

    ps_2d, kperp, kpar, ubins = correct_raw_2d_ps(ps_2d, kperp, kpar, ubins, umin=umin)

    return ps_2d, kperp, kpar, ubins