import numpy as np
from dataclasses import dataclass
from astropy.io import fits
from astropy.table import Table

@dataclass
class DataProfile:
    redshift : np.float64
    R : np.ndarray
    Sigma : np.ndarray
    DSigma_t : np.ndarray
    DSigma_x : np.ndarray
    covS : np.ndarray
    covDSt : np.ndarray
    covDSx : np.ndarray

def read_dataprofile_fits(**kwargs):
    with fits.open(**kwargs) as f:
        hd = f[0].header
        dt = f[1].data
        data = DataProfile(
            R = np.linspace(hd['RIN'],hd['ROUT'],hd['N']),
            redshift = hd['Z_MEAN'],
            Sigma = dt['Sigma'],
            DSigma_t = dt['DSigma_t'],
            DSigma_x = dt['DSigma_x'],
            covS = f[2].data,
            covDSt = f[3].data,
            covDSx = f[4].data,
        )
    return data

def save_chains_h5():
    pass