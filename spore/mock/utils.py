import numpy as np
from foregrounds import PointSourceForegrounds

def visibility_covariance(foreground_model = PointSourceForegrounds, niter=30,seed=None,
                          ret_realisations=False, *args,**kwargs):
    if seed:
        np.random.seed(seed)

    cov = 0
    meanvis = 0
    realisations = []
    for i in range(niter):
        mdl = foreground_model(*args, **kwargs)

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
