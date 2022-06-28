import numpy as np
import os
import re
import sys
import string


class Box(object):
    def __init__(self):
        self.name = 'my name'
        self.box_data = []
        self.dim = []
        self.param_dict = {}
        self.init = False
        self.z = -1
        return

    def info(self):
        # report info on box
        print 'initialised=', self.init
        print 'dim=', self.dim
        return

    def setBox(self, box_data, param_dict):
        # initialise Box from box_data and param_dict
        self.box_data = box_data
        self.param_dict = param_dict
        self.dim = param_dict['dim']
        self.init = True
        self.z = param_dict['z']
        return

    def getBoxStats(self):
        if self.init:
            print 'box_data exists'
            self.mean = self.box_data.mean()
            self.var = self.box_data.var()
        else:
            print 'box_data has not been initialised'
            self.mean = 0.0
            self.var = 0.0

        print 'mean=', self.mean, ' var=', self.var
        return

    def boxToOverdensity(self):
        """Renormalise the box to scaled variable delta=(x-xmean)/xmean"""
        if self.init:
            self.getBoxStats()
            self.box_data = (self.box_data - self.mean)/self.mean

        else:
            print 'box_data has not been initialised'

        self.getBoxStats()
        return

    def boxFromThreshold(self, crit_delta):
        # produce new box with all zeros
        newbox = np.zeros(self.box_data.shape)
        newbox[self.box_data <= crit_delta] = 1
        return newbox

    def thresholdFromVolume(self, vol_frac):
        """given a volume fraction find the threshold delta_crit that
        divides box values below that from box values above that. """

        # work in terms of number of pixels that need to be below threshold
        count_frac = int(vol_frac*self.box_data.size)
        # use ordered list of all box elements
        indx = self.box_data.copy()  # this is memory inefficient
        indx = indx.reshape((indx.size))
        indx.sort()
        # find threshold value
        crit_delta = indx[count_frac]

        return crit_delta


def readbox(filename, verbose=False):
    # read a 21cmfast output file and return a Box object with data

    # parse filename to (1) check its a 21cmFast box (2) get box parameters
    # (3) identify what sort of box it is
    param_dict = parse_filename(filename)
    if verbose:
        print(param_dict)

    # open box and read in data
    dim = param_dict['HIIdim']
    box_data = open_box(filename, dim)

    # tidy data to ensure its in optimal form i.e. trim padding
    box_data = trim_box(box_data)
    # push data into Box class
    box = Box()
    box.setBox(box_data, param_dict)

    return box


def parse_filename(filepath):
    """ 21cmFast uses filename to contain information.  Extract this and
        return a dictionary with all the available information


        This is functional, but could be made better and more robust
    """

    param_dict = {}

    # would be useful to have a bit here to find the file base
    (base, filename) = os.path.split(filepath)
    # print base
    # print filename

    if not filename:
        # handle case where filename doesn't exist
        param_dict['type'] = False
        param_dict['z'] = False
        return param_dict

    # store origin of box and full filename
    param_dict['filename'] = filename
    param_dict['basedir'] = base

    match = re.search('([a-zA-Z]+_)+', filename)
    if match:
        prefixstr = match.group()
        param_dict['type'] = validateBoxType(prefixstr)

    # now fill in key details

    match = re.search('_nf([0-9.]+)', filename)
    if match:
        nf = match.group(1)
        param_dict['nf'] = float(nf)

    match = re.search('_eff([0-9.]+)', filename)
    if match:
        eff = match.group(1)
        param_dict['eff'] = float(eff)

    match = re.search('_HIIfilter([0-9]+)', filename)
    if match:
        HIIfilter = match.group(1)
        param_dict['HIIfilter'] = int(HIIfilter)

    match = re.search('_Mmin([0-9.e+-]+)', filename)
    if match:
        Mmin = match.group(1)
        param_dict['Mmin'] = float(Mmin)

    match = re.search('_RHIImax([0-9.]+)', filename)
    if match:
        RHIImax = match.group(1)
        param_dict['RHIIfilter'] = float(RHIImax)

    # redshift
    match = re.search('_z([0-9.]+)', filename)
    if match:
        z = match.group(1)
        param_dict['z'] = float(z)
    else:
        z = 0.0  # assign z=0.0 to initialisation boxes, which are unlabelled
        param_dict['z'] = float(z)

    # basic box information
    match = re.search('_([0-9]+)_([0-9]+)Mpc', filename)
    if match:
        HIIdim = match.group(1)
        BoxSize = match.group(2)
        param_dict['HIIdim'] = int(HIIdim)
        param_dict['dim'] = int(HIIdim)
        param_dict['BoxSize'] = int(BoxSize)
    else:
        print 'Warning: no information on box size'
        print filename

    # for k in param_dict.keys():
    #  print k,'\t', param_dict[k]

    return param_dict


def validateBoxType(prefixstr):
    """look at the prefix string from filename and identify what kind of
    box it was

    prefix is losely defined here as all characters and underscores up to the
    first non-character/+_ character. Typically a number.  Should do something
    better than this

    prefixes for the initialisation boxes are assigned types in the same way
    as the normal boxes

    High resolution density boxes are labelled differently as init_x and init_k
    """

    typestr = prefixstr

    # search for match of string tag in the zeroth position of the prefixstr

    if (string.find(prefixstr, 'updated_smoothed_deltax') == 0):
        typestr = 'density'
    elif (string.find(prefixstr, 'xH_nohalos') == 0):
        typestr = 'xh'
    elif (string.find(prefixstr, 'delta_T') == 0):
        typestr = 'deltaT'
    elif (string.find(prefixstr, 'updated_vx') == 0):
        typestr = 'vx'
    elif (string.find(prefixstr, 'updated_vy') == 0):
        typestr = 'vy'
    elif (string.find(prefixstr, 'updated_vz') == 0):
        typestr = 'vz'
    elif (string.find(prefixstr, 'smoothed_deltax') == 0):
        typestr = 'density'
    elif (string.find(prefixstr, 'vxoverddot') == 0):
        typestr = 'vx'
    elif (string.find(prefixstr, 'vyoverddot') == 0):
        typestr = 'vy'
    elif (string.find(prefixstr, 'vzoverddot') == 0):
        typestr = 'vz'
    elif (string.find(prefixstr, 'deltax') == 0):
        typestr = 'init_x'
    elif (string.find(prefixstr, 'deltak') == 0):
        typestr = 'init_k'
    else:
        print prefixstr, ' not found in and so unvalidated'

    return typestr


def trim_box(box):
    """ Take a box, work out if it has fft padding, and if so remove it"""
    (dim, dim, dim3) = box.shape
    new_box = box
    if dim3 > dim:
        print 'trimming FFT padding from box'
        new_box = new_box[:, :, 0:dim]

    return new_box


def open_box(filename, dim):
    """Given dimensions of cubic box (dim,dim,dim) storing float data
    read in that data and return a numpy array of the correct dimensions

    Note- fftw padding convention:   P[x,y,z]= z+2*(D/2+1)*(y+D*x)
        - fftw no-padding convention P[x,y,z]=z+D*(y+D*x)
        reshaping must match these conventions
    """

    size = dim*dim*dim
    dtype = 'f'
    fd = open(filename, 'rb')
    read_data = np.fromfile(fd, dtype)
    fd.close()

    if not size == len(read_data):
        print 'Error: Read box does not match expected size=', dim, '...'
        size = len(read_data)
        root3 = int(pow(size, 1.0/3.0))
        if root3*root3*root3 == size:
            print 'but read box is cubic with size=', root3, ',so winging it...'
            dim = root3
            shape = (dim, dim, dim)

        elif root3*root3*2*(root3/2 + 1) == size:
            # even dim boxes get padded with two cells
            print 'but consistent with fftw box with FFT padding'
            dim = root3
            shape = (dim, dim, 2*(dim/2 + 1))
            print 'no. elements=', len(read_data), ' and shape=', shape
        else:
            print 'and box is not cubic.  Aborting.'
            print 'Box was:%d'%len(read_data)
            print 'Expected length and dim were: %i'%root3
            sys.exit(1)
    else:
        shape = (dim, dim, dim)

    # print filename
    # print numpy.sum(read_data)/len(read_data)
    data_box = read_data.reshape(shape)

    data_box = trim_box(data_box)

    return data_box


def save_box(filename, box_data):
    """ Save a numpy box in a binary format"""

    try:
        fd = open(filename, 'wb')
        box_data.tofile(fd)
        fd.close()
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    except:
        print "Unknown error:", sys.exc_info()[0]
        raise

    return


def find_run_boxes(filename):
    """
    given filename for one box in run retrieve all files for that run
    and return them in order by neutral fraction
    """
    param_dict = parse_filename(filename)
    (base, fname) = os.path.split(filename)
    dim = param_dict['HIIdim']
    size = param_dict['BoxSize']
    match = re.search('(^[a-zA-Z0-9]+)', fname)
    tag = match.group()

    # retreive all files that match tag and box size
    searchstr = '_' + str(dim) + '_' + str(size) + 'Mpc'
    filenames = os.listdir(base)

    box_files = []
    for filename in filenames:
        if filename.find(tag) == 0 and filename.find(searchstr) >= 0:
            box_files.append(os.path.join(base, filename))

    return sorted(box_files)
