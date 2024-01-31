''' Funciones para GRAFICAR: NO USAR COMO AJUSTE XQ TIENE UN PARÁMETRO EXTRA (z)'''

import numpy as np
from astropy.cosmology import LambdaCDM
from scipy.integrate import quad, quad_vec

def pm(z):
    '''densidad media en Msun/(pc**2 Mpc)'''
    h = 1.
    cosmo = LambdaCDM(H0 = 100.*h, Om0=0.3, Ode0=0.7)
    p_cr0 = cosmo.critical_density(0).to('Msun/(pc**2 Mpc)').value
    a = cosmo.scale_factor(z)
    out = p_cr0*cosmo.Om0/a**3
    return out


def hamaus(r , rs, rv, delta, a, b):
        
    d = delta*(1. - (r/rs)**a)/(1. + (r/rv)**b)
    return d

def clampitt(r, Rv, R2, dc, d2):
    R_V = np.full_like(r, Rv)
    R2s  = np.full_like(r,R2)
    
    delta = (r<=R_V)*(dc + (d2-dc)*(r/Rv)**3) + ((r>R_V)&(r<=R2s))*d2 + (r>R2s)*0
    
    return delta

def higuchi(r,Rv,R2,dc,d2):
    unos = np.full_like(r,Rv)
    R2s  = np.full_like(r,R2)
    
    delta = (r<=unos)*dc + ((r>unos)&(r<=R2s))*d2 + (r>R2s)*0
    
    return delta


### Densidades proyectadas para cada función

def sigma_higuchi(R,reds,R2,dc,d2,x):
    Rv = 1.
    if Rv>R2:
        return np.inf
        
    Rv = np.full_like(R,Rv)
    R2 = np.full_like(R,R2)
    
    m1 = (R<=Rv)
    m2 = (R>Rv)&(R<=R2)
    
    den_integrada = np.zeros_like(R)
    den_integrada[m1] = (np.sqrt(Rv[m1]**2-R[m1]**2)*(dc-d2) + d2*np.sqrt(R2[m1]**2-R[m1]**2))
    den_integrada[m2] = d2*np.sqrt(R2[m2]**2-R[m2]**2)

    rho_mean = pm(reds)
    sigma = rho_mean*den_integrada/Rv + x
    return sigma

def sigma_clampitt(R,reds,R2,dc,d2,x):
    Rv=1
    if Rv>R2:
        return np.inf

    Rv = np.full_like(R,Rv)
    R2 = np.full_like(R,R2)
    
    den_integrada = np.zeros_like(R)
    
    m1 = (R<=Rv)
    m2 = (R>Rv)&(R<=R2)
    
    s2 = np.sqrt(R2[m1]**2 - R[m1]**2)
    sv = np.sqrt(Rv[m1]**2 - R[m1]**2)
    arg = np.sqrt((Rv[m1]/R[m1])**2 - 1)

    den_integrada[m1] = 2*(dc*s2 + (d2-dc)*(sv*(5/8*(R[m1]/Rv[m1])**2 - 1) + s2 + 3/8*(R[m1]**4/Rv[m1]**3)*np.arcsinh(arg)))   
    den_integrada[m2] = 2*(d2*np.sqrt(R2[m2]**2-R[m2]**2))

    rho_mean = pm(reds)

    sigma = rho_mean*den_integrada/Rv + x
    return sigma

def sigma_hamaus(r,reds,rs,dc,a,b,x):
    rv=1
    def integrand(z,R):
        return hamaus(r=np.sqrt(z**2+R**2),rv=rv,rs=rs,delta=dc,a=a,b=b)
  
    den_integrada = quad_vec(integrand, -1e3, 1e3, args=(r,), epsrel=1e-3)[0]

    rho_mean = pm(reds)

    sigma = rho_mean*den_integrada/rv + x
    
    return sigma


## Contraste de Densidad Proyectada de cada función

def Scl(y,reds,R2,dc,d2,x):
    '''
    funcion sigma_clampitt pero solo admite como entrada un float,
    ideal para integrar
    '''
    Rv=1
    rho_mean = pm(reds)

    if y<=Rv:
        sv = np.sqrt(Rv**2 - y**2)
        s2 = np.sqrt(R2**2 - y**2)
        arg = np.sqrt((Rv/y)**2 - 1)
        f1 = 2*(dc*s2 + (d2-dc)*(sv*(5/8*(y/Rv)**2 - 1) + s2 + 3/8*(y**4/Rv**3)*np.arcsinh(arg)))
        return rho_mean*f1/Rv+x
    elif y>R2:
        return x
    else:
        f2 = 2*(d2*np.sqrt(R2**2-y**2))
        return rho_mean*f2/Rv+x

def delta_sigma_clampitt(R,reds,R2,dc,d2):
    
    def integrand(y):
        return Scl(y,reds,R2,dc,d2,0)*y

    anillo = sigma_clampitt(R,reds,R2,dc,d2,0)
    disco = np.zeros_like(R)
    for i,Ri in enumerate(R):
        disco[i] = (2/Ri**2)*quad(integrand, 0, Ri)[0]
    
    return disco-anillo

def Shi(y,reds,R2,dc,d2,x):
    '''
    funcion sigma_higuchi pero solo admite como entrada un float,
    ideal para integrar
    '''
    Rv=1
    rho_mean = pm(reds)
    
    if y<=Rv:
        f1 = (np.sqrt(Rv**2-y**2)*(dc-d2) + d2*np.sqrt(R2**2-y**2))
        return rho_mean*f1/Rv+x
    elif y>R2:
        return x
    else:
        f2 = d2*np.sqrt(R2**2-y**2)
        return rho_mean*f2/Rv+x

    
def delta_sigma_higuchi(R,reds,R2,dc,d2):

    def integrand(y):
        return Shi(y,reds,R2,dc,d2,0)*y

    anillo = sigma_higuchi(R,reds,R2,dc,d2,0)
    disco = np.zeros_like(R)
    for i,Ri in enumerate(R):
        disco[i] = (2/Ri**2)*quad(integrand, 0, Ri)[0]
    
    return disco-anillo


def delta_sigma_hamaus(r,reds,rs,dc,a,b):

    # Rv = 1.
    
    def integrand(y):
        return sigma_hamaus(y,reds,rs,dc,a,b,0)*y

    anillo = sigma_hamaus(r,reds,rs,dc,a,b,0)
    disco = np.zeros_like(r)
    for j,p in enumerate(r):
        disco[j] = 2./p**2 * quad(integrand, 0., p)[0]

    return disco-anillo

if __name__ == '__main__':
    
    z = input('Redshift?',)
    z = float(z)
    rho_mean = pm(z)