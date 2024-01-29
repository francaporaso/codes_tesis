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
### likelihoods

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
    if (1. <= R2 <= 3.)&(-1. <= dc <= 0.)&(-1. <= d2 <= 5.)&(-10<=x<=10):
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
    if (0. <= rs <= 3.)&(-1. <= dc <= 0.)&(0. <= a <= 10.)&(1. <= b <= 10.)&(-10<=x<=10):
        return 0.0
    return -np.inf

def log_probability_sigma_hamaus(theta, r, y, yerr):
    lp = log_prior_hamaus(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_sigma_hamaus(theta, r, y, yerr)


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
                                fits.Column(name='rs',format='D',array=mcmc_out[:,:,0]),
                                fits.Column(name='dc',format='D',array=mcmc_out[:,:,1]),
                                fits.Column(name='a',format='D',array=mcmc_out[:,:,2]),
                                fits.Column(name='b',format='D',array=mcmc_out[:,:,3]),
                                fits.Column(name='x',format='D',array=mcmc_out[:,:,4]),
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
                                fits.Column(name='R2',format='D',array=mcmc_out[:,:,0]),
                                fits.Column(name='dc',format='D',array=mcmc_out[:,:,1]),
                                fits.Column(name='d2',format='D',array=mcmc_out[:,:,2]),
                                fits.Column(name='x',format='D',array=mcmc_out[:,:,3]),
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


def pos_maker(func, nw=32):

    # comunes
    xpos = np.random.uniform(-1, 1., nw)
    dcpos = np.random.uniform(-0.9, -0.1, nw)
    d2pos = np.random.uniform(-0.5, 0.5, nw)
    r2pos = np.random.uniform(1.2, 2.8, nw)

    # hamaus
    rspos = np.random.uniform(0.2, 2.8, nw)
    apos = np.random.uniform(0.5, 4.9, nw)
    bpos = np.random.uniform(5., 9., nw)

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

if __name__ == '__main__':

    ### ---
    ## datos

    # carpeta = 'Rv_6-10/rvchico_'
    # archivo = 'tot'
    sample = 'pru_cov2'
    nit = 100
    nw = 32
    ncores = 32

    funcs = np.array([
                        (sigma_higuchi, log_probability_sigma_higuchi),
                        (sigma_clampitt, log_probability_sigma_clampitt),
                        (sigma_hamaus, log_probability_sigma_hamaus),
                    ])

    for j,carpeta in enumerate(['Rv_6-10/rvchico_','Rv_10-50/rvalto_']):
        for k, archivo in enumerate(['tot', 'R', 'S']):

            # if f'{carpeta}{archivo}'=='Rv_6-10/rvchico_tot':
            #     print(f'Salteado {carpeta}{archivo}')
            #     continue

            with fits.open(f'../profiles/voids/{carpeta}{archivo}.fits') as dat:
               h = dat[0].header
               Rp = dat[1].data.Rp
               B = dat[2].data
               C = dat[3].data

            rho_mean = pm(h['z_mean'])

            S = B.Sigma.reshape(101,60)[0]
            covS = C.covS.reshape(60,60)
            eS = np.sqrt(np.diag(covS))
            # DSt = B.DSigma_T.reshape(101,60)[0]
            # covDSt = C.covDSt.reshape(60,60)
            # eDSt = np.sqrt(np.diag(covDSt))

            # pos inicial
            for fu, logp in funcs:

                pos = pos_maker(fu.__name__, nw=nw) 

                print(f'Ajustando perfil {carpeta}{archivo}')
                print(f'Usando {fu.__name__}')
                mcmc_out = ajuste(xdata=Rp, ydata=S, ycov=covS, pos=pos,log_probability=logp,
                                  nit=nit, ncores=ncores)

                print('Guardando...')
                guardar_perfil_sigma(mcmc_out=mcmc_out, xdata=Rp, ydata=S, yerr=eS, func=fu,
                                tirar=0.2, carpeta=carpeta, archivo=archivo, sample=sample)

    print('Terminado!')