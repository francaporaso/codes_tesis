import numpy as np
from astropy.io import fits
from multiprocessing import Pool, Process
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import os

def clampitt(r,Rv,A0,A3):
    '''3D density from Clampitt et al 2016'''
    return np.piecewise(r,[r<Rv],[A0+A3*(r/Rv)**3,A0+A3])

def sigma_clampitt(r,Rv,A0,A3):
    '''projected density from the 3D density clampitt'''
    try:
        integral = np.array([])
        for m in r:
            arg = lambda z: clampitt(np.sqrt(m**2+z**2),Rv,A0,A3)
            integral = np.append(integral,integrate.quad(arg,-100,100)[0])
    except:
        arg = lambda z: clampitt(np.sqrt(r**2+z**2),Rv,A0,A3)
        integral = integrate.quad(arg,-100,100)[0]
        
    return integral

def sigma_clampitt_unpack(karg):
    return sigma_clampitt(*karg)

def parallel_S(r,Rv,A0,A3,ncores=int(40)):
    '''projected density calculated in parallel'''
    ncores = int(40)
    partial = sigma_clampitt_unpack

    #Rv,A0,A3 = p_clampitt

    rv = np.full_like(r,Rv)
    a0 = np.full_like(r,A0)
    a3 = np.full_like(r,A3)

    entrada = np.array([r,rv,a0,a3]).T

    with Pool(processes=ncores) as pool:
        salida = np.array(pool.map(partial,entrada))
        pool.close()
        pool.join()
    return salida

def Dsigma_clampitt(r,Rv,A0,A3):
    '''projected density contrast from the 3D density clampitt'''
    try:
        Dsigma = np.array([])
        for m in r:
            s_disco = integrate.quad(sigma_clampitt,0,m,args=(Rv,A0,A3))[0]/m
            s_anillo = sigma_clampitt(m,Rv,A0,A3)
        
            Dsigma = np.append(Dsigma, s_disco-s_anillo)
    except:
        s_disco = integrate.quad(sigma_clampitt,0,r,args=(Rv,A0,A3))[0]/r
        s_anillo = sigma_clampitt(r,Rv,A0,A3)
        
        Dsigma = s_disco-s_anillo
    return Dsigma

def Dsigma_clampitt_unpack(kargs):
    return Dsigma_clampitt(*kargs)

def parallel_DS(r,Rv,A0,A3,ncores=40):
    '''projected density contrast calculated in parallel'''
    #ncores = 32
    partial = Dsigma_clampitt_unpack

    #Rv,A0,A3 = p_clampitt

    rv = np.full_like(r,Rv)
    a0 = np.full_like(r,A0)
    a3 = np.full_like(r,A3)

    entrada = np.array([r,rv,a0,a3]).T

    with Pool(processes=ncores) as pool:
        salida = np.array(pool.map(partial,entrada))
        pool.close()
        pool.join()
    return salida

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
    hdu.append(('Nvoids',profile[0].header['N_VOIDS']))
    hdu.append(('Rv_min',profile[0].header['RV_MIN']))
    hdu.append(('Rv_max',profile[0].header['RV_MAX']))
    
    table_opt = [fits.Column(name='p_S',format='D',array=p_S),
                 fits.Column(name='p_S',format='D',array=p_DSt)]
    
    table_err = [fits.Column(name='p_S',format='D',array=perr_S),
                 fits.Column(name='p_S',format='D',array=perr_DSt)]

    tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_opt))
    tbhdu_cov = fits.BinTableHDU.from_columns(fits.ColDefs(table_err))
            
    primary_hdu = fits.PrimaryHDU(header=hdu)
            
    hdul = fits.HDUList([primary_hdu, tbhdu_pro, tbhdu_cov])
    
    try:
            os.mkdir(f'../profiles/voids/Rv_{int(Rv_min)}-{int(Rv_max)}/fitting')
    except FileExistsError:
            pass

    hdul.writeto(f'../profiles/voids/fitting/fitLS_Rv{int(Rv_min)}{int(Rv_max)}',overwrite=True)
    
