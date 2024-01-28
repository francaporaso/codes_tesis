import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import emcee
import time
import corner
from astropy.io import fits
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord, Angle
from astropy.constants import G,c,M_sun,pc
from scipy.integrate import quad, romberg, fixed_quad, quad_vec
from multiprocessing import Pool

h = 1.
Om0 = 0.3
cosmo = LambdaCDM(H0=100*h, Om0=Om0, Ode0=0.7)

### funciones de delta, sigma y delta sigma

def pm(z):
    '''densidad media en Msun/(pc**2 Mpc)'''
    h = 1.
    cosmo = LambdaCDM(H0 = 100.*h, Om0=0.3, Ode0=0.7)
    p_cr0 = cosmo.critical_density(0).to('Msun/(pc**2 Mpc)').value
    a = cosmo.scale_factor(z)
    out = p_cr0*cosmo.Om0/a**3
    return out

def hamaus(r, rs, rv, delta, a, b):
        
    d = delta*(1. - (r/rs)**a)/(1. + (r/rv)**b)
    return d

def clampitt(r,Rv,R2,dc,d2):
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
def sigma_higuchi(R,R2,dc,d2,x):
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

    sigma = rho_mean*den_integrada/Rv + x
    return sigma

def sigma_clampitt(R,R2,dc,d2,x):
    Rv = 1.
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

    sigma = rho_mean*den_integrada/Rv + x
    return sigma

def sigma_hamaus(r,rs,dc,a,b,x):
    rv = 1.
    def integrand(z,R):
        return hamaus(r=np.sqrt(z**2+R**2),rv=rv,rs=rs,delta=dc,a=a,b=b)
  
    den_integrada = quad_vec(integrand, -1e3, 1e3, args=(r,), epsrel=1e-3)[0]

    sigma = rho_mean*den_integrada/rv + x
    
    return sigma


## Contraste de Densidad Proyectada de cada función
def Scl(y,Rv,R2,dc,d2,x):
    '''
    funcion sigma_clampitt pero solo admite como entrada un float,
    ideal para integrar
    '''
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

def delta_sigma_clampitt(R,R2,dc,d2):
    Rv = 1.
    def integrand(y):
        return Scl(y,Rv,R2,dc,d2,0)*y

    anillo = sigma_clampitt(R,R2,dc,d2,0)
    disco = np.zeros_like(R)
    for i,Ri in enumerate(R):
        disco[i] = (2/Ri**2)*quad(integrand, 0, Ri)[0]
    
    return disco-anillo

def Shi(y,Rv,R2,dc,d2,x):
    '''
    funcion sigma_higuchi pero solo admite como entrada un float,
    ideal para integrar
    '''
    
    if y<=Rv:
        f1 = (np.sqrt(Rv**2-y**2)*(dc-d2) + d2*np.sqrt(R2**2-y**2))
        return rho_mean*f1/Rv+x
    elif y>R2:
        return x
    else:
        f2 = d2*np.sqrt(R2**2-y**2)
        return rho_mean*f2/Rv+x
    
def delta_sigma_higuchi(R,R2,dc,d2):
    Rv = 1.
    def integrand(y):
        return Shi(y,Rv,R2,dc,d2,0)*y

    anillo = sigma_higuchi(R,R2,dc,d2,0)
    disco = np.zeros_like(R)
    for i,Ri in enumerate(R):
        disco[i] = (2/Ri**2)*quad(integrand, 0, Ri)[0]
    
    return disco-anillo

def delta_sigma_hamaus(r,rs,dc,a,b):
    
    rv = 1.
    def integrand(y):
        return sigma_hamaus(y,rs,dc,a,b,x=0)*y

    anillo = sigma_hamaus(r,rs,dc,a,b,x=0)
    disco = np.zeros_like(r)
    for j,p in enumerate(r):
        disco[j] = 2./p**2 * quad(integrand, 0., p)[0]

    return disco-anillo

### ----

### likelihoods

def log_likelihood(theta, r, y, yerr):
    
    rs,dc,a,b,x = theta
    model = sigma_hamaus(r, rs, dc, a, b, x)
    sigma2 = yerr**2
    return -0.5 * np.sum(((y - model)**2 )/sigma2 + np.log(sigma2))
    # return -0.5 * np.sum(((y - model)**2 )/sigma2)

def log_prior(theta):
    rs,dc,a,b,x = theta
    if (0. <= rs <= 3.)&(-1. <= dc <= 0.)&(0. <= a <= 10.)&(1. <= b <= 10.)&(-10<=x<=10):
        return 0.0
    return -np.inf

def log_probability(theta, r, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, r, y, yerr)

### ---
## datos

# for j,carpeta in enumerate(['Rv_6-10/rvchico_','Rv_10-50/rvalto_']):
#     for k, archivo in enumerate(['tot', 'R', 'S']):

carpeta = 'Rv_6-10/rvchico_'
archivo = 'tot'

with fits.open(f'../profiles/voids/{carpeta}{archivo}.fits') as dat:
   h = dat[0].header
   Rp = dat[1].data.Rp
   B = dat[2].data
   C = dat[3].data

rho_mean = pm(h['z_mean'])

S = B.Sigma.reshape(101,60)[0]
DSt = B.DSigma_T.reshape(101,60)[0]
covS = C.covS.reshape(60,60)
eS = np.sqrt(np.diag(covS))
covDSt = C.covDSt.reshape(60,60)
eDSt = np.sqrt(np.diag(covDSt))

### --- 
## ajuste
nw = 15
pos = np.array([
    np.random.uniform(0.8, 1.2, nw),     # rs
    np.random.uniform(-0.7, -0.5, nw),   # dc
    np.random.uniform(1., 5., nw),       # a
    np.random.uniform(5., 9., nw),       # b
    np.random.uniform(-1, 1., nw),       # x
    ]).T     

nwalkers, ndim = pos.shape

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(Rp,S,eS), pool=pool)
    start = time.time()
    sampler.run_mcmc(pos, 1000, progress=True)
    end = time.time()
    multi_time = end - start
print(multi_time)

mcmc_out = sampler.get_chain(flat=True).T

### ---
## guardado
rs = np.percentile(mcmc_out[0][100:], [16, 50, 84])
dc = np.percentile(mcmc_out[1][100:], [16, 50, 84])
a = np.percentile(mcmc_out[2][100:], [16, 50, 84])
b = np.percentile(mcmc_out[3][100:], [16, 50, 84])
x = np.percentile(mcmc_out[4][100:], [16, 50, 84])

table_opt = np.array([
    fits.Column(name='rs',format='D',array=mcmc_out[0]),
    fits.Column(name='dc',format='D',array=mcmc_out[1]),
    fits.Column(name='a',format='D',array=mcmc_out[2]),
    fits.Column(name='b',format='D',array=mcmc_out[3]),
    fits.Column(name='x',format='D',array=mcmc_out[4])
    ])

hdu = fits.Header()
hdu.append(('rs',np.round(rs[1],4)))
hdu.append(('dc',np.round(dc[1],4)))
hdu.append(('a',np.round(a[1],4)))
hdu.append(('b',np.round(b[1],4)))
hdu.append(('x',np.round(x[1],4)))

sample = 'prueba'
primary_hdu = fits.PrimaryHDU(header=hdu)
hdul = fits.HDUList([primary_hdu])
carpeta_out = carpeta.split('/')[0]
outfile = f'{carpeta_out}/fit/fit_mcmc_{archivo}_hamaus_{sample}.fits'

print(f'Guardado en {outfile}')

hdul.writeto(outfile, overwrite=True)