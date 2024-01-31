import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import emcee
import time
import corner
import argparse
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

## chi reducido
def chi_red(ajuste,data,err,gl):
	'''
	Reduced chi**2
	------------------------------------------------------------------
	INPUT:
	ajuste       (float or array of floats) fitted value/s
	data         (float or array of floats) data used for fitting
	err          (float or array of floats) error in data
	gl           (float) grade of freedom (number of fitted variables)
	------------------------------------------------------------------
	OUTPUT:
	chi          (float) Reduced chi**2 	
	'''
		
	BIN=len(data)
	chi=((((ajuste-data)**2)/(err**2)).sum())/float(BIN-1-gl)
	return chi

### ----
### likelihoods sigma

def log_likelihood_sigma_higuchi(theta, r, y, yerr):
    '''
    r : eje x
    y : datos eje y
    yerr: error en los datos -> L_S utiliza yerr como la inversa de la mat de cov
    '''
    R2,dc,d2,x = theta
    modelo = sigma_higuchi(r, R2, dc, d2, x)
    
    # sigma2 = yerr**2
    # return -0.5 * np.sum(((y - model)**2 )/sigma2 + np.log(sigma2))
    # return -0.5 * np.sum(((y - model)**2 )/sigma2)

    L_S = -np.dot((y-modelo),np.dot(yerr,(y-modelo)))/2.0
        
    return L_S    

def log_prior(theta):
    R2,dc,d2,x = theta
    if (2. <= R2 <= 3.)&(-1. <= dc <= 0.)&(-0.5 <= d2 <= 0.5)&(-5<=x<=5):
        return 0.0
    return -np.inf

def log_probability_sigma_higuchi(theta, r, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_sigma_higuchi(theta, r, y, yerr)



def log_likelihood_sigma_clampitt(theta, r, y, yerr):
    '''
    r : eje x
    y : datos eje y
    yerr: error en los datos -> L_S utiliza yerr como la inversa de la mat de cov
    '''
    R2,dc,d2,x = theta
    modelo = sigma_clampitt(r, R2, dc, d2, x)
    
    # sigma2 = yerr**2
    # return -0.5 * np.sum(((y - model)**2 )/sigma2 + np.log(sigma2))
    # return -0.5 * np.sum(((y - model)**2 )/sigma2)

    L_S = -np.dot((y-modelo),np.dot(yerr,(y-modelo)))/2.0
        
    return L_S    

def log_probability_sigma_clampitt(theta, r, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_sigma_clampitt(theta, r, y, yerr)



def log_likelihood_sigma_hamaus(theta, r, y, yerr):
    '''
    r : eje x
    y : datos eje y
    yerr: error en los datos -> L_S utiliza yerr como la inversa de la mat de cov
    '''
    rs,dc,a,b,x = theta
    modelo = sigma_hamaus(r, rs, dc, a, b, x)
    
    # sigma2 = yerr**2
    # return -0.5 * np.sum(((y - model)**2 )/sigma2 + np.log(sigma2))
    # return -0.5 * np.sum(((y - model)**2 )/sigma2)

    L_S = -np.dot((y-modelo),np.dot(yerr,(y-modelo)))/2.0
        
    return L_S    

def log_prior_hamaus(theta):
    rs,dc,a,b,x = theta
    if (0. <= rs <= 3.)&(-1. <= dc <= 0.)&(0. <= a <= 10.)&(1. <= b <= 20.)&(-10<=x<=10):
        return 0.0
    return -np.inf

def log_probability_sigma_hamaus(theta, r, y, yerr):
    lp = log_prior_hamaus(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_sigma_hamaus(theta, r, y, yerr)

### ----
### likelihoods delta sigma

def log_likelihood_DSt_higuchi(theta, r, y, yerr):
    '''
    r : eje x
    y : datos eje y
    yerr: error en los datos -> L_S utiliza yerr como la inversa de la mat de cov
    '''
    R2,dc,d2 = theta
    modelo = delta_sigma_higuchi(r, R2, dc, d2)
    
    # sigma2 = yerr**2
    # return -0.5 * np.sum(((y - model)**2 )/sigma2 + np.log(sigma2))
    # return -0.5 * np.sum(((y - model)**2 )/sigma2)

    L_S = -np.dot((y-modelo),np.dot(yerr,(y-modelo)))/2.0
        
    return L_S    

def log_prior_delta(theta):
    R2,dc,d2 = theta
    if (2. <= R2 <= 3.)&(-1. <= dc <= 0.)&(-0.5 <= d2 <= 0.5):
        return 0.0
    return -np.inf

def log_probability_DSt_higuchi(theta, r, y, yerr):
    lp = log_prior_delta(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_DSt_higuchi(theta, r, y, yerr)



def log_likelihood_DSt_clampitt(theta, r, y, yerr):
    '''
    r : eje x
    y : datos eje y
    yerr: error en los datos -> L_S utiliza yerr como la inversa de la mat de cov
    '''
    R2,dc,d2 = theta
    modelo = delta_sigma_clampitt(r, R2, dc, d2)
    
    # sigma2 = yerr**2
    # return -0.5 * np.sum(((y - model)**2 )/sigma2 + np.log(sigma2))
    # return -0.5 * np.sum(((y - model)**2 )/sigma2)

    L_S = -np.dot((y-modelo),np.dot(yerr,(y-modelo)))/2.0
        
    return L_S    

def log_probability_DSt_clampitt(theta, r, y, yerr):
    lp = log_prior_delta(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_DSt_clampitt(theta, r, y, yerr)



def log_likelihood_DSt_hamaus(theta, r, y, yerr):
    '''
    r : eje x
    y : datos eje y
    yerr: error en los datos -> L_S utiliza yerr como la inversa de la mat de cov
    '''
    rs,dc,a,b = theta
    modelo = delta_sigma_hamaus(r, rs, dc, a, b)
    
    # sigma2 = yerr**2
    # return -0.5 * np.sum(((y - model)**2 )/sigma2 + np.log(sigma2))
    # return -0.5 * np.sum(((y - model)**2 )/sigma2)

    L_S = -np.dot((y-modelo),np.dot(yerr,(y-modelo)))/2.0
        
    return L_S    

def log_prior_DSt_hamaus(theta):
    rs,dc,a,b = theta
    if (0. <= rs <= 3.)&(-1. <= dc <= 0.)&(0. <= a <= 10.)&(1. <= b <= 20.):
        return 0.0
    return -np.inf

def log_probability_DSt_hamaus(theta, r, y, yerr):
    lp = log_prior_DSt_hamaus(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_DSt_hamaus(theta, r, y, yerr)

### ---
## helpers funciones para dejar fijo algun parametro
def h1S(r,dc,b,x):
    return sigma_hamaus(r=r,rs=np.inf,dc=dc,a=1,b=b,x=x)

def log_likelihood_h1S(theta, r, y, yerr):
    '''
    r : eje x
    y : datos eje y
    yerr: error en los datos -> L_S utiliza yerr como la inversa de la mat de cov
    '''
    dc,b,x = theta
    modelo = h1S(r, dc=dc, b=b, x=x)
    
    # sigma2 = yerr**2
    # return -0.5 * np.sum(((y - model)**2 )/sigma2 + np.log(sigma2))
    # return -0.5 * np.sum(((y - model)**2 )/sigma2)

    L_S = -np.dot((y-modelo),np.dot(yerr,(y-modelo)))/2.0
        
    return L_S    

def log_prior_h1S(theta):
    dc,b,x = theta
    if (-1. <= dc <= 0.)&(1.<=b<=20.)&(-2<=x<=2):
        return 0.0
    return -np.inf

def log_probability_h1S(theta, r, y, yerr):
    lp = log_prior_h1S(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_h1S(theta, r, y, yerr)

### --- 
## ajuste

def ajuste(xdata, ydata, ycov, pos, log_probability,
           nit=1000, ncores=32):
    
    '''
    ajuste con mcmc
    xdata: datos en el eje x
    ydata: datos en el eje y
    ycov: error de ydata, pueden ser errores de la diagonal o la matriz completa
    '''   

    nwalkers, ndim = pos.shape

    if ycov.shape == ydata.shape:
        yerr = ycov
        print('Usando diagonal')
    else:
        yerr = np.linalg.inv(ycov)
        print('Usando matriz de covarianza')


    with Pool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata,ydata,yerr), pool=pool)
        sampler.run_mcmc(pos, nit, progress=True)

    mcmc_out = sampler.get_chain()

    return mcmc_out

### ---
## guardado
def guardar_perfil_sigma(mcmc_out, xdata, ydata, yerr, func,
                        tirar=0.2, carpeta='Rv_6-10/rvchico_', archivo='tot', sample='pru'):

    '''
    guardado del mcmc
    tirar: porcentaje de iteraciones iniciales descartadas (default 20% de las iteraciones)
    '''

    nit, nw, ndim = mcmc_out.shape
    tirar = int(tirar*nit)
    print(f'{tirar} iteraciones descartadas')

    if func.__name__ == 'sigma_hamaus':
        rs = np.percentile(mcmc_out[tirar:,:,0], [16, 50, 84])
        dc = np.percentile(mcmc_out[tirar:,:,1], [16, 50, 84])
        a  = np.percentile(mcmc_out[tirar:,:,2], [16, 50, 84])
        b  = np.percentile(mcmc_out[tirar:,:,3], [16, 50, 84])
        x  = np.percentile(mcmc_out[tirar:,:,4], [16, 50, 84])

        chi = chi_red(sigma_hamaus(xdata, rs=rs[1], dc=dc[1], a=a[1], b=b[1], x=x[1]), ydata, yerr, 5)

        table_opt = np.array([
                                fits.Column(name='rs',format='D',array=mcmc_out[:,:,0].flatten()),
                                fits.Column(name='dc',format='D',array=mcmc_out[:,:,1].flatten()),
                                fits.Column(name='a' ,format='D',array=mcmc_out[:,:,2].flatten()),
                                fits.Column(name='b' ,format='D',array=mcmc_out[:,:,3].flatten()),
                                fits.Column(name='x' ,format='D',array=mcmc_out[:,:,4].flatten()),
                            ])

        hdu = fits.Header()

        hdu.append(('rs',rs[1]))
        hdu.append(('dc',dc[1]))
        hdu.append(('a',a[1]))
        hdu.append(('b',b[1]))
        hdu.append(('x',x[1]))

    else:
        
        R2 = np.percentile(mcmc_out[tirar:,:,0], [16, 50, 84])
        dc = np.percentile(mcmc_out[tirar:,:,1], [16, 50, 84])
        d2  = np.percentile(mcmc_out[tirar:,:,2], [16, 50, 84])
        x  = np.percentile(mcmc_out[tirar:,:,3], [16, 50, 84])

        params = np.array([R2[1], dc[1], d2[1], x[1]])

        chi = chi_red(func(xdata,*params), ydata, yerr, 4)

        table_opt = np.array([
                                fits.Column(name='R2',format='D',array=mcmc_out[:,:,0].flatten()),
                                fits.Column(name='dc',format='D',array=mcmc_out[:,:,1].flatten()),
                                fits.Column(name='d2',format='D',array=mcmc_out[:,:,2].flatten()),
                                fits.Column(name='x' ,format='D',array=mcmc_out[:,:,3].flatten()),
                            ])

        hdu = fits.Header()

        hdu.append(('R2',R2[1]))
        hdu.append(('dc',dc[1]))
        hdu.append(('d2',d2[1]))
        hdu.append(('x',x[1]))

    hdu.append(('nw',nw))
    hdu.append(('ndim',ndim))
    hdu.append(('nit',nit))
    hdu.append(('chi_red',chi))

    primary_hdu = fits.PrimaryHDU(header=hdu)
    tbhdu1 = fits.BinTableHDU.from_columns(table_opt)
    hdul = fits.HDUList([primary_hdu, tbhdu1])
    carpeta_out = carpeta.split('/')[0]

    outfile = f'../profiles/voids/{carpeta_out}/fit/fit_mcmc_{archivo}_{func.__name__}_{sample}.fits'

    print(f'Guardado en {outfile}')
    hdul.writeto(outfile, overwrite=True)

def guardar_perfil_deltasigma(mcmc_out, xdata, ydata, yerr, func,
                              tirar=0.2, carpeta='Rv_6-10/rvchico_', archivo='tot', sample='pru'):

    '''
    guardado del mcmc
    tirar: porcentaje de iteraciones iniciales descartadas (default 20% de las iteraciones)
    '''

    nit, nw, ndim = mcmc_out.shape
    tirar = int(tirar*nit)
    print(f'{tirar} iteraciones descartadas')

    if func.__name__ == 'delta_sigma_hamaus':
        rs = np.percentile(mcmc_out[tirar:,:,0], [16, 50, 84])
        dc = np.percentile(mcmc_out[tirar:,:,1], [16, 50, 84])
        a  = np.percentile(mcmc_out[tirar:,:,2], [16, 50, 84])
        b  = np.percentile(mcmc_out[tirar:,:,3], [16, 50, 84])

        chi = chi_red(delta_sigma_hamaus(xdata, rs=rs[1], dc=dc[1], a=a[1], b=b[1]), ydata, yerr, 4)

        table_opt = np.array([
                                fits.Column(name='rs',format='D',array=mcmc_out[:,:,0].flatten()),
                                fits.Column(name='dc',format='D',array=mcmc_out[:,:,1].flatten()),
                                fits.Column(name='a' ,format='D',array=mcmc_out[:,:,2].flatten()),
                                fits.Column(name='b' ,format='D',array=mcmc_out[:,:,3].flatten()),
                            ])

        hdu = fits.Header()

        hdu.append(('rs',rs[1]))
        hdu.append(('dc',dc[1]))
        hdu.append(('a',a[1]))
        hdu.append(('b',b[1]))

    else:
        
        R2 = np.percentile(mcmc_out[tirar:,:,0], [16, 50, 84])
        dc = np.percentile(mcmc_out[tirar:,:,1], [16, 50, 84])
        d2  = np.percentile(mcmc_out[tirar:,:,2], [16, 50, 84])

        params = np.array([R2[1], dc[1], d2[1]])

        chi = chi_red(func(xdata,*params), ydata, yerr, 3)

        table_opt = np.array([
                                fits.Column(name='R2',format='D',array=mcmc_out[:,:,0].flatten()),
                                fits.Column(name='dc',format='D',array=mcmc_out[:,:,1].flatten()),
                                fits.Column(name='d2',format='D',array=mcmc_out[:,:,2].flatten()),
                            ])

        hdu = fits.Header()

        hdu.append(('R2',R2[1]))
        hdu.append(('dc',dc[1]))
        hdu.append(('d2',d2[1]))

    hdu.append(('nw',nw))
    hdu.append(('ndim',ndim))
    hdu.append(('nit',nit))
    hdu.append(('chi_red',chi))

    primary_hdu = fits.PrimaryHDU(header=hdu)
    tbhdu1 = fits.BinTableHDU.from_columns(table_opt)
    hdul = fits.HDUList([primary_hdu, tbhdu1])
    carpeta_out = carpeta.split('/')[0]

    outfile = f'../profiles/voids/{carpeta_out}/fit/fit_mcmc_{archivo}_{func.__name__}_{sample}.fits'

    print(f'Guardado en {outfile}')
    hdul.writeto(outfile, overwrite=True)


def pos_makerS(func, nw=32):

    # comunes
    xpos = np.random.uniform(-1, 1., nw)
    dcpos = np.random.uniform(-0.95, -0.05, nw)
    d2pos = np.random.uniform(-0.3, 0.3, nw)
    r2pos = np.random.uniform(2.1, 2.9, nw)

    # hamaus
    rspos = np.random.uniform(0.1, 2.9, nw)
    apos = np.random.uniform(0.1, 4.9, nw)
    bpos = np.random.uniform(2., 9., nw)

    if func=='sigma_hamaus':
        pos = np.array([
                        rspos,     # rs
                        dcpos,     # dc
                        apos,      # a
                        bpos,      # b
                        xpos,      # x
                    ]).T

    else:
        pos = np.array([
                        r2pos,      # r2
                        dcpos,      # dc
                        d2pos,      # d2
                        xpos,       # x
                    ]).T

    return pos

def pos_makerDSt(func, nw=32):

    # comunes
    dcpos = np.random.uniform(-0.9, -0.1, nw)
    d2pos = np.random.uniform(-0.5, 0.5, nw)
    r2pos = np.random.uniform(1.02, 2.9, nw)

    # hamaus
    rspos = np.random.uniform(0.2, 2.8, nw)
    apos = np.random.uniform(0.5, 4.9, nw)
    bpos = np.random.uniform(5., 9., nw)

    if func=='delta_sigma_hamaus':
        pos = np.array([
                        rspos,     # rs
                        dcpos,     # dc
                        apos,      # a
                        bpos,      # b
                    ]).T

    else:
        pos = np.array([
                        r2pos,      # r2
                        dcpos,      # dc
                        d2pos,      # d2
                    ]).T

    return pos

if __name__ == '__main__':

    ## datos
    # sample = 'pru_cov2'
    # nit = 100
    # ncores = 32
    # nw = 32 # emcee usa la mitad de nucleos q de walkers

    ### ---
    ## parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', action='store', dest='sample',default='pru')
    parser.add_argument('-nit', action='store', dest='nit',default=1000)
    parser.add_argument('-ncores', action='store', dest='ncores',default=10)
    parser.add_argument('-nw', action='store', dest='nw',default=32)
    parser.add_argument('-tirar', action='store', dest='tirar',default=0.1)
    parser.add_argument('-Sig', action='store', dest='Sig',default='True')
    parser.add_argument('-DSig', action='store', dest='DSig',default='False')
    args = parser.parse_args()

    sample = args.sample
    ncores = int(args.ncores)
    nit    = int(args.nit)
    nw     = int(args.nw)
    tirar  = float(args.tirar)

    if args.Sig == 'True':
        Sig = True
    else:
        Sig = False

    if args.DSig == 'True':
        DSig = True
    else:
        DSig = False


    funcs_S = np.array([
                        (sigma_higuchi, log_probability_sigma_higuchi),
                        (sigma_clampitt, log_probability_sigma_clampitt),
                        # (sigma_hamaus, log_probability_sigma_hamaus),
                      ])
    
    funcs_hS = np.array([
                         (h1S, log_probability_h1S),
                       ])
    
    funcs_DSt = np.array([
                          (delta_sigma_higuchi, log_probability_DSt_higuchi),
                          (delta_sigma_clampitt, log_probability_DSt_clampitt),
                          (delta_sigma_hamaus, log_probability_DSt_hamaus),
                        ])

    # for j,carpeta in enumerate(['Rv_6-10/rvchico_','Rv_10-50/rvalto_']):
    # # for j,carpeta in enumerate(['Rv_10-50/rvalto_']):
    #     # for k, archivo in enumerate(['tot', 'R', 'S']):
    #     for k, archivo in enumerate(['R']):

    #         # if (f'{carpeta}{archivo}'=='Rv_6-10/rvchico_tot') or (f'{carpeta}{archivo}'=='Rv_6-10/rvchico_R') or (f'{carpeta}{archivo}'=='Rv_6-10/rvchico_S'):
    #         #     print(f'Salteado {carpeta}{archivo}')
    #         #     continue

    #         with fits.open(f'../profiles/voids/{carpeta}{archivo}.fits') as dat:
    #            h = dat[0].header
    #            Rp = (dat[1].data.Rp).astype(float)
    #            B = dat[2].data
    #            C = dat[3].data

    #         rho_mean = pm(h['z_mean'])

    #         S = (B.Sigma.reshape(101,60)[0]).astype(float)
    #         covS = (C.covS.reshape(60,60)).astype(float)
    #         eS = np.sqrt(np.diag(covS))

    #         DSt = (B.DSigma_T.reshape(101,60)[0]).astype(float)
    #         covDSt = (C.covDSt.reshape(60,60)).astype(float)
    #         eDSt = np.sqrt(np.diag(covDSt))
            
    #         print(f'Ajustando perfil {carpeta}{archivo}')

    #         # ajustando sigma
    #         if Sig:
    #             for fu, logp in funcs_S:

    #                 pos = pos_makerS(fu.__name__, nw=nw) 

    #                 try:
    #                     print(f'Usando {fu.__name__}')
    #                     mcmc_out = ajuste(xdata=Rp, ydata=S, ycov=covS, pos=pos,log_probability=logp,
    #                                       nit=nit, ncores=ncores)

    #                     print('Guardando...')
    #                     guardar_perfil_sigma(mcmc_out=mcmc_out, xdata=Rp, ydata=S, yerr=eS, func=fu,
    #                                     tirar=tirar, carpeta=carpeta, archivo=archivo, sample=sample)
    #                 except ValueError:
    #                     print('Error en la funcion log probability')
    #                     print(F'{carpeta}{archivo} no ajustó para {fu.__name__}')
    #                     print('CONTINUANDO')
    #                     print('----o----')

    #         # ajustando delta sigma
    #         if DSig:
    #             for fu, logp in funcs_DSt:
                
    #                 pos = pos_makerDSt(fu.__name__, nw=nw) 

    #                 try:
    #                     print(f'Usando {fu.__name__}')
    #                     mcmc_out = ajuste(xdata=Rp, ydata=DSt, ycov=covDSt, pos=pos,log_probability=logp,
    #                                       nit=nit, ncores=ncores)

    #                     print('Guardando...')
    #                     guardar_perfil_deltasigma(mcmc_out=mcmc_out, xdata=Rp, ydata=DSt, yerr=eDSt, func=fu,
    #                                     tirar=tirar, carpeta=carpeta, archivo=archivo, sample=sample)
                        
    #                 except ValueError:
    #                     print('Error en la funcion log probability')
    #                     print(F'{carpeta}{archivo} no ajustó para {fu.__name__}')
    #                     print('CONTINUANDO')
    #                     print('----o----')


#### para ajustar los helper...
    for j,carpeta in enumerate(['Rv_6-10/rvchico_','Rv_10-50/rvalto_']):
        for k, archivo in enumerate(['R']):

            with fits.open(f'../profiles/voids/{carpeta}{archivo}.fits') as dat:
               h = dat[0].header
               Rp = (dat[1].data.Rp).astype(float)
               B = dat[2].data
               C = dat[3].data

            rho_mean = pm(h['z_mean'])

            S = (B.Sigma.reshape(101,60)[0]).astype(float)
            covS = (C.covS.reshape(60,60)).astype(float)
            eS = np.sqrt(np.diag(covS))

            DSt = (B.DSigma_T.reshape(101,60)[0]).astype(float)
            covDSt = (C.covDSt.reshape(60,60)).astype(float)
            eDSt = np.sqrt(np.diag(covDSt))

            print(f'Ajustando perfil {carpeta}{archivo}')

            pos = np.array([
                np.random.uniform(-0.9, -0.1, nw),  # dc
                np.random.uniform(5., 9., nw),      # b
                np.random.uniform(-1, 1., nw),      # x
            ]).T

            for fu, logp in funcs_hS:
                try:
                    print(f'Usando {fu.__name__}')
                    mcmc_out = ajuste(xdata=Rp, ydata=S, ycov=covS, pos=pos,log_probability=logp,
                                      nit=nit, ncores=ncores)
                    
                    nit, nw, ndim = mcmc_out.shape
                    t = int(tirar*nit)
                    print(f'{t} iteraciones descartadas')

                    dc = np.percentile(mcmc_out[t:,:,0], [16, 50, 84])
                    b  = np.percentile(mcmc_out[t:,:,1], [16, 50, 84])
                    x  = np.percentile(mcmc_out[t:,:,2], [16, 50, 84])
                    chi = chi_red(h1S(Rp, dc=dc[1], b=b[1], x=x[1]), S, eS, 3)
                    table_opt = np.array([
                                            fits.Column(name='dc',format='D',array=mcmc_out[:,:,0].flatten()),
                                            fits.Column(name='b' ,format='D',array=mcmc_out[:,:,1].flatten()),
                                            fits.Column(name='x' ,format='D',array=mcmc_out[:,:,2].flatten()),
                                        ])
                    hdu = fits.Header()
                    hdu.append(('dc',dc[1]))
                    hdu.append(('b',b[1]))
                    hdu.append(('x',x[1]))

                    
                except ValueError:
                    print('Error en la funcion log probability')
                    print(F'{carpeta}{archivo} no ajustó para {fu.__name__}')
                    print('CONTINUANDO')
                    print('----o----')                


    
    print('Terminado!')