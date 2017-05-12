import glob
import os

import numpy as np
from astropy.cosmology import Planck15

from spore.model import beam


def get_physical_dimensions(nu_min, nu_max, nu_num, sigma_max=0.4, nsigma=2, cosmo=Planck15):
    """
    Parameters
    ----------
    nu_min, nu_max : float
        Lowest/highest frequency in band, in MHz. Note that these (I think) should represent
        the extents of the *centres* of individual channels.

    nu_num : int
        Number of frequency channels.

    sigma_max : float
        The maximum width of the telescope beam in the band (sets the necessary angular extent
        of the simulation).

    nsigma : float
        The number of beam widths to simulate. Default 2 (one on each side of the max).

    Returns
    -------
    zmin : float
        The minimum redshift of the observation

    zmax : float
        The maximum redshift of the observation

    dz : float
        The minimum interval between redshift bins

    width : float
        The maximum necessary width to cover the observation (in Mpc/h)

    depth : float
        The maximum necessary depth to cover the observation (in Mpc/h)
    """

    ###### BASIC VALUES
    nu = np.linspace(nu_min, nu_max, nu_num + 1)
    nu = (nu[1:] + nu[:-1])/2  # actual values of nu at the centre of the bins.

    angular_extent = nsigma*sigma_max

    ###### Box Dimensions
    z = 1420./nu - 1
    d = cosmo.comoving_distance(z)
    deltad = d.max() - d.min()
    width = cosmo.angular_diameter_distance(z).max()*angular_extent
    dd = d[:-1] - d[1:]
    depth = deltad + 0.5*dd[0] + 0.5*dd[-1]
    return z.min(), z.max(), (z[:-1] - z[1:]).min(), width.max().value, depth.value


def set_param(fname, param, val, direc, hastype=True):
    """
    Set a parameter in the parameter files for 21cmFAST

    Parameters
    ----------
    fname : str
        The filename that includes the parameter to change (only the filename, not directory)

    param : str
        The name of the parameter

    val :
        The value of the parameter to write

    direc : str
        The directory in which the filename resides.

    hastype : bool, default True
        Whether the declaration has the type information in it. All parameters in the .H files
        include this, but definitions within .c files may not.
    """
    with open(os.path.join(direc, fname), 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        lst = line.split(" ")
        if lst[0] == "#define" and lst[1] == param.upper():
            if hastype:
                lst[3] = str(val)
            else:
                lst[2] = "(" + str(val) + ")"
            lines[i] = " ".join(lst)
            if not lines[i].endswith("\n"):
                lines[i] += '\n'

    with open(os.path.join(direc, fname), 'w') as f:
        f.writelines(lines)


def fix_program_calls(direc):
    """
    Change program calls in 21cm FAST drivers to include preceding "./".


    Parameters
    ----------
    direc : str
        String pointing to the Programs/ directory of 21cm FAST. Best if this is absolute.
    """
    pwd = os.getcwd()
    os.chdir(direc)
    programs = glob.glob("*.c")
    programs = [x[:-2] for x in programs]
    drivers = [x + ".c" for x in programs if x.startswith("drive_")]

    for d in drivers:
        with open(d, 'r') as f:
            text = f.read()
        for p in programs:
            text = text.replace('"%s'%p, '"./%s'%p)
        with open(d, 'w') as f:
            f.write(text)
    os.chdir(pwd)


def run_boxes(progdir, nu_min, nu_max, nu_num, beamtype=beam.CircularGaussian(), nsigma=2,
              dx=1.0, seed=None, tocm_init_opts={}, tocm_anal_opts={}, tocm_cosmo_opts={},
              tocm_heat_opts={}):
    """
    Run a series of 21cmFAST boxes at varying redshifts to cover the frequency range given.

    This function takes an input min/max frequency, and the number of channels desired,
    calculates the minimum number of *redshift* steps to achieve the desired frequency resolution,
    then runs 21cmFAST (without spin-temperature at this point) for the redshifts it requires.

    The size of the box is determined by the maximum of either the required depth (determined
    by the comoving distance between min and max redshift) or the required width (determined
    by covering a certain fraction of the given beam size)
    Parameters
    ----------
    progdir : str
        Path to the 21cmFAST directory (top-level).

    nu_min : float
        Minimum frequency to cover (in MHz)

    nu_max : float
        Maximum frequency to cover (in MHz)

    nu_num : int
        Number of frequencies to cover

    beamtype : :class:`spore.beam.CircularBeam` instance, default `GaussianBeam`.
        A beam-shape for the telescope.

    nsigma : float, default 2
        The number of beam-widths to capture in the box.

    dx : float, default 1.0
        The approximate resolution, in Mpc, of the 21cmFAST HII box. Note, the hi-res density box
        will be 4 times this resolution.

    seed : int, default None
        A seed to use for the 21cmFAST run.

    tocm_init_opts : dict
        A dictionary of {str:val} key-pairs noting the parameter name and its value to use for the run.
        Applied to INIT_PARAMS.H. Note, some of these are set by default by the other parameters passed here.

    tocm_anal_opts : dict
        A dictionary of {str:val} key-pairs noting the parameter name and its value to use for the run.
        Applied to ANAL_PARAMS.H. Note, some of these are set by default by the other parameters passed here.

    tocm_cosmo_opts : dict
        A dictionary of {str:val} key-pairs noting the parameter name and its value to use for the run.
        Applied to COSMO.H. Note, some of these are set by default by the other parameters passed here.

    tocm_heat_opts : dict
        A dictionary of {str:val} key-pairs noting the parameter name and its value to use for the run.
        Applied to HEAT_PARAMS.H. Note, some of these are set by default by the other parameters passed here.

    Returns
    -------
    filenames : str
        Filenames of the boxes produced.
    """
    # Go to the 21cmFAST directory
    pwd = os.getcwd()
    os.chdir(progdir)

    # Fix driver program calls
    fix_program_calls("Programs/")

    # Generate maximum sigma across bandwidth
    sigma_max = beamtype.sigma(nu_min)

    # Generate physical dimensions of boxes
    zmin, zmax, dz, width, depth = get_physical_dimensions(nu_min, nu_max, nu_num, sigma_max, nsigma)
    boxsize = np.ceil(max(width, depth))
    N = 2 ** (int(np.ceil(np.log(boxsize/dx)/np.log(2))))

    # Check if boxes like this already exist. If so, just exit. Too hard to deal with right now.
    if len(glob.glob("Boxes/*_%s_%sMpc"%(N, boxsize))) > 0:
        raise Exception("There are already boxes with this size and resolution, please move/delete them first.")

    # Set the physical dimensions in the parameter files.
    set_param("INIT_PARAMS.H", "BOX_LEN", boxsize, "Parameter_files")
    set_param("INIT_PARAMS.H", "DIM", N*4, "Parameter_files")
    set_param("INIT_PARAMS.H", "HII_DIM", N, "Parameter_files")
    if seed:
        set_param("INIT_PARAMS.H", "SEED", seed, "Parameter_files")


    print "About to run boxes with these parameters:"
    print "  BOX_LEN: ", boxsize
    print "  DIM    : ", N
    print "  HII DIM: ", N/4
    print "  ZSTART : ", zmax
    print "  ZEND   : ", zmin
    print "  ZSTEP  : ", -dz

    # Set all other parameters passed by the user.
    # NOTE: these can override the automatically-determined parameters!
    for k, v in tocm_anal_opts.items():
        set_param("ANAL_PARAMS.H", k, v, "Parameter_files")
    for k, v in tocm_heat_opts.items():
        set_param("HEAT_PARAMS.H", k, v, "Parameter_files")
    for k, v in tocm_init_opts.items():
        set_param("INIT_PARAMS.H", k, v, "Parameter_files")
    for k, v in tocm_cosmo_opts.items():
        set_param("COSMOLOGY.H", k, v, "Parameter_files")

    # Set the .c file parameters (basically the z params)
    set_param("drive_zscroll_noTs.c", "ZSTART", zmax, "Programs", False)
    set_param("drive_zscroll_noTs.c", "ZEND", zmin, "Programs", False)
    set_param("drive_zscroll_noTs.c", "ZSTEP", -dz, "Programs", False)

    ## Get list of boxes before running
    preboxes = glob.glob("Boxes/delta_T*")
    preboxes = [os.path.basename(p) for p in preboxes]

    # Run 21cmFAST
    os.chdir("Programs")
    os.system("make")
    os.system("./drive_zscroll_noTs")

    ## Get list of boxes after running, and determine new ones in the list.
    postboxes = glob.glob("../Boxes/delta_T*")
    newboxes = [os.path.abspath(p) for p in postboxes if os.path.basename(p) not in preboxes]

    # Go back to initial working dir.
    os.chdir(pwd)

    return newboxes
