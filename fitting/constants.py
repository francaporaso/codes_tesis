import numpy as np
from astropy.cosmology import LambdaCDM, FlatLambdaCDM
from astropy.constants import G, c, M_sun, pc

SQPI: float = np.sqrt(np.pi)
Z_CMB: float = 1090.0  # redshift to the last scattering surface
SC_CONSTANT: float = (
    (c.value**2.0 / (4.0 * np.pi * G.value)) * (pc.value / M_sun.value) * 1e-6
)


def get_cosmo(h=1.0, Om0=0.25, Ode0=0.75, is_flat=True):
    if is_flat:
        return FlatLambdaCDM(H0=100.0 * h, Om0=Om0)
    return LambdaCDM(H0=100 * h, Om0=Om0, Ode0=Ode0)


def rho_mean(cosmo, z):
    """densidad media en Msun/(pc**2 Mpc)"""
    p_cr0 = cosmo.critical_density(0).to("Msun/(pc**2 Mpc)").value
    a = cosmo.scale_factor(z)
    out = p_cr0 * cosmo.Om0 / a**3
    return out


def sigma_crit(cosmo, z_l, z_s=Z_CMB):
    d_l = cosmo.angular_diameter_distance(z_l).value
    d_s = cosmo.angular_diameter_distance(z_s).value
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    return SC_CONSTANT * d_s / (d_l * d_ls)
