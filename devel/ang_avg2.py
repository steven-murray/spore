import numpy as np
from pathos.multiprocessing import ProcessPool

#from multiprocessing import Array

def angular_average2(field,coords,bins):
    if not np.iterable(bins):
        bins = np.linspace(coords.min(),coords.max()+0.1,bins+1)
    
    indx = np.digitize(coords,bins)
    weights = np.zeros(indx.max())
    binav = np.zeros(indx.max())
    res = np.zeros(indx.max())
    
    print indx.max()
    def do_i(i):
        mask = indx==i
        weights = np.sum(mask)
        binav[i-1] = np.sum(coords[mask])/weights
        res[i-1] = np.sum(field[mask])/weights
        #return binav,res
    
    pool = ProcessPool(8) 
    pool.map(do_i,range(1,indx.max()+1))
    
    return res,binav