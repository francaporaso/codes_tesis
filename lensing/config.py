import numpy as np

from dataclasses import dataclass
from lensing.io import *

@dataclass
class GlobalConfig:
    RIN : float|None = None
    ROUT : float|None = None
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
    OUTPUTNAME : str|None = None

    def print(self):
        print(' Lens cat '+f'{": ":.>10}{self.LENSNAME}')
        print(' Source cat '+f'{": ":.>10}{self.SOURCENAME}')
        print(' Output file '+f'{": ":.>10}{self.OUTPUTNAME}')
        print(' NCORES '+f'{": ":.>12}{self.NCORES}\n')
        print(' RMIN '+f'{": ":.>14}{self.RIN:.2f}')
        print(' RMAX '+f'{": ":.>14}{self.ROUT:.2f}')
        print(' N '+f'{": ":.>17}{self.N:<2d}')
        print(' NJK '+f'{": ":.>16}{self.NJK:<2d}')
        print(' Binning '+f'{": ":.>11}{self.BINNING}')
        print(' Shape Noise '+f'{": ":.>7}{self.SHAPENOISE}\n')

@dataclass
class VoidProfileConfig:
    ZMIN : float|None = None
    ZMAX : float|None = None
    RVMIN : float|None = None
    RVMAX : float|None = None
    DELTAMIN : float|None = None
    DELTAMAX : float|None = None

CONFIG : GlobalConfig = None

S = None # Table of galaxy data
PIX_TO_IDX = {}

binspace = None

def init_worker(configargs, sourceargs):
    global CONFIG
    global S, PIX_TO_IDX
    global binspace

    CONFIG = GlobalConfig(**configargs)

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