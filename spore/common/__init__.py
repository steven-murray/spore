"""
Common tools used across the board.
"""
import numpy as np

def ensure_unit(val,unit):
    if hasattr(val,"unit"):
        return val.to(unit)
    else:
        return val*unit


def nfold_outer(a,n=1):
    return np.squeeze(np.product(np.meshgrid(1,*([a]*n)),axis=0))