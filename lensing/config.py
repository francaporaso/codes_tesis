import numpy as np

from dataclasses import dataclass
from lensing.io import *

@dataclass
class Config:
    
    RIN : float|None = None
    ROUT : float|None = None

    # TODO :: maybe this can be in another dataclass
    # to make a for loop over these for different samples
    # or they could be a list to iterate over...
    ZMIN : float|None = None
    ZMAX : float|None = None
    RVMIN : float|None = None
    RVMAX : float|None = None
    DELTAMIN : float|None = None
    DELTAMAX : float|None = None

    N : int|None = None
    NJK : int|None = None
    NCORES : int|None = None
    NSIDE : int|None = 64
    
    SHAPENOISE : bool|None = False
    USE08 : bool|None = False
    
    BINNING : str|None = None
    RDSHFT_COLNAME : str = 'z_cgal_v'
    LENSNAME : str|None = None
    SOURCENAME : str|None = None
    SAVEFILENAME : str|None = None


CONFIG : Config = None

S = None # Table of galaxy data
PIX_TO_IDX = {}

binspace = None

def init_worker(configargs, sourceargs):
    global CONFIG
    global S, PIX_TO_IDX
    global binspace

    CONFIG = Config(**configargs)

    if CONFIG.BINNING == 'lin':
        binspace = np.linspace
    elif CONFIG.BINNING == 'log':
        binspace = np.logspace
    else:
        raise ValueError('BINNING must be "lin" or "log".')
    
    S = sourcecat_load(**sourceargs)
    
    ## making a dict of healpix idx for fast query
    upix, split_idx = np.unique(S['pix'], return_index=True)
    split_idx = np.append(split_idx, len(S))
    for i, pix in enumerate(upix):
        PIX_TO_IDX[int(pix)] = np.arange(split_idx[i], split_idx[i+1])