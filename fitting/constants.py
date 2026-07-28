import numpy as np
from astropy.cosmology import LambdaCDM, FlatLambdaCDM

SQPI = np.sqrt(np.pi)

def get_cosmo(h=1.0, Om0=0.25, Ode0=0.75, is_flat=True):
    if is_flat:
        return FlatLambdaCDM(H0=100.0*h, Om0=Om0)
    return LambdaCDM(H0=100*h, Om0=Om0, Ode0=Ode0)

def rho_mean(cosmo, z):
    '''densidad media en Msun/(pc**2 Mpc)'''
    p_cr0 = cosmo.critical_density(0).to('Msun/(pc**2 Mpc)').value
    a = cosmo.scale_factor(z)
    out = p_cr0*cosmo.Om0/a**3
    return out
