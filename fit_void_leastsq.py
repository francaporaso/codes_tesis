'''Ajuste de perfiles de voids mediante cuadrados minimos. Por defecto ajusta ambos Sigma y DSigma'''
import numpy as np
from scipy.integrate import quad, quad_vec
import matplotlib.pyplot as plt
from multiprocessing import Pool
from astropy.io import fits
from scipy.optimize import curve_fit

## Para que funcione scipy las funciones deben tomar arrays

def clampitt(r,Rv,A3):
    '''Clampitt et al (2016); eq 12'''
    A0 = 1-A3
    return np.piecewise(r,[r<Rv],[lambda r: A0-1+A3*(r/Rv)**3,A0+A3-1]) 

def krause(r,Rv,A3,c):
    pass

#fitS = True

def projected_density(rvals, *params, rho=clampitt, rmax=100):
    '''perfil de densidad proyectada dada la densidad 3D
    rvals   (array) : puntos de r a evaluar el perfil
    *params (float) : parametros de la funcion rho (los que se ajustan)
    rho     (func)  : densidad 3D
    rmax    (int)   : valor maximo de r para integrar
    
    return
    density  (float) : densidad proyectada en rvals'''
    


    def integrand(z, r, *params):
        return rho(np.sqrt(np.square(r) + np.square(z)), *params)
    
    density = np.array([quad(integrand, -rmax, rmax, args=(r,)+params)[0] for r in rvals])
    density = np.array([quad(integrand, -rmax, rmax, args=(r,)+params)[0] for r in rvals])

    
    return density

#fitDS = True

def aux(R, *params, rho=clampitt, rmax=100):
    '''perfil de densidad proyectada dada la densidad 3D
    r       (float) : puntos de r a evaluar el perfil
    *params (float) : parametros de la funcion rho (los que se ajustan)
    rho     (func)  : densidad 3D
    rmax    (int)   : valor maximo de r para integrar
    
    return
    density  (float) : densidad proyectada en rvals'''
    
    def integrand(z, r, *params):
        return rho(np.sqrt(np.square(r) + np.square(z)), *params)
    
    density = quad(integrand, -rmax, rmax, args=(R,)+params)[0]
    
    return density

def projected_density_contrast(rvals, *params, rho=clampitt, rmax=100):
    
    '''perfil de densidad proyectada dada la densidad 3D
    rvals   (float) : punto de r a evaluar el perfil
    *params (float) : parametros de la funcion rho (los que se ajustan)
    rho     (func)  : densidad 3D
    rmax    (int)   : valor maximo de r para integrar
    
    contrast (float): contraste de densidad proy en rvals'''

    def integrand(x,*p): 
        return x*projected_density([x], *p, rho=rho, rmax=rmax)
    
    # def integrand(x,*p): 
    #     return x*aux(x, *p, rho=rho, rmax=rmax)
    
    anillo = projected_density(rvals,*params, rho=rho, rmax=rmax)
    disco  = np.array([2./(np.square(r))*quad(integrand, 0, r, args=(r,)+params)[0] for r in rvals])

    contrast = disco - anillo
    return contrast




p_S, pcov_S = curve_fit(projected_density, Rp, p.Sigma.reshape(101,60)[0], sigma=eS, p0=(1,1))
perr_S = np.sqrt(np.diag(pcov_S))


if __name__ == '__main__':
    #ncores = 32

    profile = fits.open('../profiles/voids/Rv_15-18/Rv1518.fits')
    p = profile[1].data
    cov = profile[2].data

    Rv_min, Rv_max = profile[0].header['RV_MIN'], profile[0].header['RV_MAX']

    eS = np.sqrt(np.diag(cov.cov_S.reshape(40,40)))
    eDSt = np.sqrt(np.diag(cov.cov_DSt.reshape(40,40)))

    p_S, pcov_S = curve_fit(parallel_S, p.Rp, p.Sigma, sigma=eS)
    p_DSt, pcov_DSt = curve_fit(parallel_DS, p.Rp, p.DSigmaT, sigma=eDSt)

    perr_S = np.sqrt(np.diag(pcov_S))
    perr_DSt = np.sqrt(np.diag(pcov_DSt))

    hdu = fits.Header()
    hdu.append(('Nvoids',profile[0].header['NVOIDS']))
    hdu.append(('Rv_min',profile[0].header['RV_MIN']))
    hdu.append(('Rv_max',profile[0].header['RV_MAX']))
    
    table_opt = [fits.Column(name='p_S',format='D',array=p_S),
                 fits.Column(name='p_S',format='D',array=p_DSt)]
    
    table_err = [fits.Column(name='p_eS',format='D',array=perr_S),
                 fits.Column(name='p_eS',format='D',array=perr_DSt)]

    tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_opt))
    tbhdu_cov = fits.BinTableHDU.from_columns(fits.ColDefs(table_err))
            
    primary_hdu = fits.PrimaryHDU(header=hdu)
            
    hdul = fits.HDUList([primary_hdu, tbhdu_pro, tbhdu_cov])
    
    try:
            os.mkdir(f'../profiles/voids/Rv_{int(Rv_min)}-{int(Rv_max)}/fitting')
    except FileExistsError:
            pass

    hdul.writeto(f'../profiles/voids/fitting/fitLS_Rv{int(Rv_min)}{int(Rv_max)}',overwrite=True)
    
