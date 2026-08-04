import numpy as np
from astropy.cosmology import LambdaCDM, FlatLambdaCDM, w0waCDM, Flatw0waCDM, wCDM, FlatwCDM
from astropy.constants import G, c, M_sun, pc

SQPI: float = np.sqrt(np.pi)
Z_CMB: float = 1090.0  # redshift to the last scattering surface
SC_CONSTANT: float = (
    (c.value**2.0 / (4.0 * np.pi * G.value)) * (pc.value / M_sun.value) * 1e-6
)

def get_cosmo(model='lcdm', h=1.0, Om0=0.25, Ode0=0.75, is_flat=True, 
              w0=-1.0, wa=1.0, Ob0=0.05, Tcmb0=0.0, m_nu=0.0):
    '''
    h (float) :: reduced hubble constant at z=0
    Om0 (float) :: matter density parameter at z=0
    Ode0 (float) :: dark energy density parameter at z=0
    m_nu (float | array(3)[float]) :: neutrino mass in eV. If scalar, all species have the same mass, otherwise the mass of each species.
    Ob0 (float) :: barion density parameter at z=0
    Tcmb0 (float) :: temperature of the CMB at z=0 in K
    is_flat (bool) :: flat cosmology
    w0 (float) :: dark energy parameter in the EOS (wCDM/w0waCDM)
    wa (float) :: dark energy parameter in the EOS (w0waCDM)
    '''

    model = model.lower() #to avoid problems if capitalized
    H0 = 100.0*h

    if model

    if model == 'lcdm':
        if is_flat:
            return FlatLambdaCDM(H0=H0, Om0=Om0, Tcmb0=Tcmb0, Ob0=Ob0, m_nu=m_nu)
        return LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0, Tcmb0=Tcmb0, Ob0=Ob0, m_nu=m_nu)

    if model == 'wcdm':
        if is_flat:
            return FlatwCDM(H0=H0, Om0=Om0, w0=w0, Tcmb0=Tcmb0, Ob0=Ob0, m_nu=m_nu)
        return wCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0, Tcmb0=Tcmb0, Ob0=Ob0, m_nu=m_nu)

    if model == 'w0wacdm':
        if is_flat:
            return Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa, Tcmb0=Tcmb0, Ob0=Ob0, m_nu=m_nu)
        return w0waCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0, wa=wa, Tcmb0=Tcmb0, Ob0=Ob0, m_nu=m_nu)
    
    raise ValueError(f'model must be one of ["lcdm", "wcmd", "w0wacdm"]. got {model!r}.')

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
