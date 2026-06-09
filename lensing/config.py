import numpy as np
import toml

# ==== Input globals
# read from config file
class Config:

    def __init__(self, configfile:str='lensing/config.toml'):

        config = toml.load(configfile)
        cat = config['catalog']
        run = config['run']
        profile = config['profile']

        self.lensname = cat['lenses']['name']
        self.sourcename = cat['sources']['name']
        self.randsname = cat['randoms']['name']
        self.redshift = cat['sources']['redshift_col']

        self.sample = run['sample']
        self.ncores = run['ncores']
        self.plot = run['plot']
        self.overwrite = run['overwrite']
        
        self.RIN, self.ROUT = profile['rin'], profile['rout'] #Mpc/h
        self.NBINS = profile['nbins']
        self.NJK = profile['njk']
        self.NSIDE = profile['nside']
        self.addnoise = profile['addnoise']
        self.binning = profile['binning']
        self.nback = profile['nback']

        self.zbins = self._edges_to_bins(cat['lenses']['zedges'], 'zedges')
        self.rvbins = self._edges_to_bins(cat['lenses']['rvedges'], 'rvedges')
        self.voidtype = cat['lenses']['voidtype']
        self.flag = cat['lenses']['flag']
        self.fullshape = cat['lenses']['fullshape']
        self.is_MICE = cat['lenses']['is_mice']

    def _edges_to_bins(self, edges, name):
        if not isinstance(edges, list) or len(edges) < 2:
            raise ValueError(f'[LENSES] {name} must be a list with at least 2 values.')
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo >= hi:
                raise ValueError(f'[LENSES] {name} must be strictly increasing, got {lo} >= {hi}.')
        return list(zip(edges[:-1], edges[1:]))



