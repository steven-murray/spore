from astropy import units as un
import numpy as np
from astropy.cosmology import Planck15
import astropy.constants as cnst
from spore.common import ensure_unit


hub = un.def_unit('h',un.dimensionless_unscaled*Planck15.h)
f21 = 1.42e9 #(Hz)

def dm(z):
    return Planck15.comoving_distance(z).value

def eta_to_kpar_fac(z):
    return (2*np.pi*Planck15.H0 * f21 * Planck15.efunc(z)/(cnst.c*(1+z)**2)).to(1/un.Mpc).value

def cosmo_21cm_angle_equiv(z):
    return [(un.rad,un.Mpc,lambda x: dm(z) * x/(2*np.pi),lambda x: 2*np.pi*x/dm(z)),
            (un.rad**-1,un.Mpc**-1,lambda x: 2*np.pi*x/dm(z),lambda x: x*dm(z)/(2*np.pi)),
            (un.steradian,un.Mpc**2,lambda x: dm(z)**2 * x,lambda x: x/dm(z)**2),
            (un.steradian**2,un.Mpc**4,lambda x: dm(z)**4 * x,lambda x: x/dm(z)**4)
            ]

def cosmo_21cm_los_equiv(z):
    return [(un.Hz**-1,un.Mpc**-1,lambda x: x*eta_to_kpar_fac(z),lambda x: x/eta_to_kpar_fac(z)),
            (un.Hz,un.Mpc,lambda x: 2*np.pi * x/eta_to_kpar_fac(z),lambda x: x/(2*np.pi*eta_to_kpar_fac(z)))
           ]

def brightness_temp(Aeff):
    ksr = (Aeff.value/(1e26 * 2 * cnst.k_B.value))
    return [(un.Jy,un.K * un.steradian, lambda x: ksr * x, lambda x: x/ksr),
            (un.Jy**2, un.K**2*un.steradian**2, lambda x: ksr**2 * x, lambda x: x/ksr**2)
            ]

def radio_to_cosmo_equiv(nu,Aeff):
    f21 = 1420 * un.MHz
    z = f21 / nu - 1
    Aeff = ensure_unit(Aeff, un.m**2)

    hz_mpc = un.Hz.to(un.Mpc/hub, equivalencies=cosmo_21cm_los_equiv(z))
    sr_mpc = un.steradian.to(un.Mpc ** 2/hub ** 2, equivalencies=cosmo_21cm_angle_equiv(z))
    jy_Ksr = un.Jy.to(un.K *un.steradian, equivalencies=brightness_temp(Aeff))

    return [(un.steradian*un.Hz, un.Mpc**3/ hub**3, lambda x : x*hz_mpc*sr_mpc, lambda x : x/(hz_mpc*sr_mpc)),
            (un.Jy*un.Hz, un.K * un.Mpc**3/ hub**3, lambda x : x*jy_Ksr*sr_mpc*hz_mpc, lambda x : x/(jy_Ksr*sr_mpc*hz_mpc)),
            (un.Jy**2 * un.Hz**2, un.K**2 * un.Mpc ** 6/hub ** 6, lambda x: x*(jy_Ksr*sr_mpc*hz_mpc)**2,
             lambda x: x/(jy_Ksr*sr_mpc*hz_mpc)**2),
            (un.Jy ** 4*un.Hz ** 4, un.K**4 * un.Mpc ** 12/hub ** 12, lambda x: x*(jy_Ksr*sr_mpc*hz_mpc) ** 4,
             lambda x: x/(jy_Ksr*sr_mpc*hz_mpc) ** 4)
            ]


def jyhz_to_mKMpc_per_h(power, nu, Aeff, verbose=False):
    if not hasattr(Aeff, "unit"):
        Aeff *= un.m ** 2

    # First ensure power is in correct units
    power = power.to(un.Jy ** 2 * un.Hz ** 2)

    f21 = 1420 * un.MHz
    zmin = f21 / nu.max() - 1

    # Convert the MHz into Mpc/h
    hz_mpc = un.Hz.to(un.Mpc, equivalencies=cosmo_21cm_los_equiv(zmin))
    out = power * hz_mpc ** 2
    out /= un.Hz ** 2
    out *= un.Mpc ** 2
    if verbose:
        print("Hz to Mpc: ", hz_mpc)

    # get into K.sr (!?!)
    jy2_K2 = (un.Jy ** 2).to(un.K ** 2 * un.steradian ** 2, equivalencies=brightness_temp(Aeff))
    out *= jy2_K2
    out /= un.Jy ** 2
    out *= un.K ** 2 * un.steradian ** 2
    if verbose:
        print("Jy^2 to K^2 sr^2: ", jy2_K2)

    # Convert sr^2 to Mpc^4/h^4
    sr2_mpc4 = (un.steradian ** 2).to(un.Mpc ** 4, equivalencies=cosmo_21cm_angle_equiv(zmin))
    out *= sr2_mpc4
    out /= un.steradian ** 2
    out *= un.Mpc ** 4
    if verbose:
        print("sr^2 to mpc^4: ", sr2_mpc4)

    # Norm by volume
    BW = un.steradian * (cnst.c ** 2 / nu.min() ** 2).to(un.m ** 2) / Aeff
    if verbose:
        print(BW)
    vol = (BW * (nu.max() - nu.min())).to(un.Mpc**3/hub**3, equivalencies=radio_to_cosmo_equiv(nu.min(),Aeff))

    # vol = (Planck15.comoving_volume(zmax) - Planck15.comoving_volume(zmin)) * BW.value#(2*np.pi*beam_sig**2)
    if verbose:
        print("Volume: ", vol)
    out /= vol
    out = out.to(un.milliKelvin ** 2 * un.Mpc ** 3 / hub ** 3)
    return out