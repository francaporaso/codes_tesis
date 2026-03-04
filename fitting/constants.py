import numpy as np
from astropy.cosmology import FlatLambdaCDM

h = 1.0
cosmo = FlatLambdaCDM(H0=100*h, Om0=0.25)

SQPI = np.sqrt(np.pi)

def rho_mean(z):
    '''densidad media en Msun/(pc**2 Mpc)'''
    p_cr0 = cosmo.critical_density(0).to('Msun/(pc**2 Mpc)').value
    a = cosmo.scale_factor(z)
    out = p_cr0*cosmo.Om0/a**3
    return out