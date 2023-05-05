'''Ajuste de perfiles de voids mediante cuadrados minimos. Por defecto ajusta ambos Sigma y DSigma'''
import numpy as np
from scipy.integrate import quad, quad_vec
import matplotlib.pyplot as plt
from multiprocessing import Pool
from astropy.io import fits
from scipy.optimize import curve_fit
import argparse

'''rho: ( en realidad son las fluctuaciones rho/rhomean - 1 )
    clampitt -> Clampitt et al 2016 (cuadratica adentro de Rv, constante afuera) eq 12
    krause   -> Krause et al 2012 (como clampitt pero con compensacion) eq 1 pero leer texto
    higuchi  -> Higuchi et al 2013 (conocida como top hat, 3 contantes) eq 23
    hamaus   -> Hamaus et al 2014 (algo similar a una ley de potencias) eq 2'''

def clampitt(r,Rv,A3):
    '''Clampitt et al (2016); eq 12'''
    A0 = 1-A3
    if r<Rv:
        return A0-1+A3*(r/Rv)**3
    else:
        return A0+A3-1
    # return np.piecewise(r,[r<Rv],[lambda r: A0-1+A3*(r/Rv)**3,A0+A3-1]) 

def krause(r,Rv,A3,A0):
    '''Krause et al (2012); eq 1 (see text)'''
    if r<Rv:
        return A0-1+A3*(r/Rv)**3
    elif r>2*Rv: #ultima parte
        return 0
    else: #centro
        return A0+A3-1

    # return np.piecewise(r,[r<Rv, 2*Rv>r>Rv],[lambda r: A0+A3*(r/Rv)**3, A0+A3, 1])

def higuchi(r,R1,R2,rho1,rho2):
    '''Higuchi et al (2013); eq 23'''
    if r<R1:
        return rho1-1 
    elif r>R2:
        return 0
    else:
        return rho2-1

def hamaus(r, delta, rs, Rv, a, b):
    '''Hamaus et al (2014); eq 2'''
    return delta*(1-(r/rs)**a)/(1+(r/Rv)**b)

def projected_density(rvals, *params, rho=clampitt, rmax=np.inf):
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


# def aux(R, *params, rho=clampitt, rmax=100):
#     '''perfil de densidad proyectada dada la densidad 3D
#     r       (float) : puntos de r a evaluar el perfil
#     *params (float) : parametros de la funcion rho (los que se ajustan)
#     rho     (func)  : densidad 3D
#     rmax    (int)   : valor maximo de r para integrar
    
#     return
#     density  (float) : densidad proyectada en rvals'''
    
#     def integrand(z, r, *params):
#         return rho(np.sqrt(np.square(r) + np.square(z)), *params)
    
#     density = quad(integrand, -rmax, rmax, args=(R,)+params)[0]
    
#     return density

def projected_density_contrast(rvals, *params, rho=clampitt, rmax=np.inf):
    
    '''perfil de contraste de densidad proyectada dada la densidad 3D
    rvals   (float) : punto de r a evaluar el perfil
    *params (float) : parametros de la funcion rho (los que se ajustan)
    rho     (func)  : densidad 3D
    rmax    (int)   : valor maximo de r para integrar
    
    contrast (float): contraste de densidad proy en rvals'''

    def integrand(x,*p): 
        return x*projected_density([x], *p, rho=rho, rmax=rmax)
    
    # def integrand(x,*p): 
    #     return x*aux(x, *p, rho=rho, rmax=rmax)
    
    anillo = projected_density([rvals],*params, rho=rho, rmax=rmax)
    disco  = np.array([2./(np.square(r))*quad(integrand, 0, r, args=(r,)+params)[0] for r in [rvals]])

    contrast = disco - anillo
    return contrast

def projected_density_contrast_unpack(kargs):
    return projected_density_contrast(*kargs)

def projected_density_contrast_parallel(rvals, *params, rho=clampitt, rmax=np.inf, ncores=10):
    
    partial = projected_density_contrast_unpack
    
    lbins = int(round(len(rvals)/float(ncores), 0))
    slices = ((np.arange(lbins)+1)*ncores).astype(int)
    slices = slices[(slices < len(rvals))]
    Rsplit = np.split(rvals,slices)

    dsigma = np.zeros(len(Rsplit))
    nparams = len(params)

    for j,r_j in enumerate(Rsplit):
        
        par_a   = np.array([np.full_like(r_j,params[k]) for k in np.arange(nparams)])
        entrada = np.append(np.array([r_j]), np.array([par_a[k] for k in np.arange(nparams)]),axis=0).T


        with Pool(processes=ncores) as pool:
            salida = np.array(pool.map(partial,entrada))
            pool.close()
            pool.join()

        dsigma[j] = salida

    return dsigma



# p_S, pcov_S = curve_fit(projected_density, Rp, p.Sigma.reshape(101,60)[0], sigma=eS, p0=(1,1))
# perr_S = np.sqrt(np.diag(pcov_S))


if __name__ == '__main__':
    #ncores = 32

    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', action='store', dest='sample',default='Rv_6-9')
    parser.add_argument('-name', action='store', dest='sample',default='smallz_6-9')
    parser.add_argument('-ncores', action='store', dest='sample',default=10)
    parser.add_argument('-rmax', action='store', dest='sample',default=100)
    parser.add_argument('-fitS', action='store', dest='sample',default=True)
    parser.add_argument('-fitDS', action='store', dest='sample',default=True)
    parser.add_argument('-rho', action='store', dest='sample',default='clampitt')
    parser.add_argument('-p0', action='store', dest='sample',default=1)
    args = parser.parse_args()
    
    sample = args.sample
    name   = args.name
    ncores = int(args.ncores)
    fitS   = bool(args.fitS)
    fitDS  = bool(args.fitDS)
    rho    = args.rho   
    p0     = float(args.p0)

    '''rho:
    clampitt -> Clampitt et al 2016 (cuadratica adentro de Rv, constante afuera) eq 12
    krause   -> Krause et al 2012 (como clampitt pero con compensacion) eq 1 pero leer texto
    higuchi  -> Higuchi et al 2013 (conocida como top hat, 3 contantes) eq 23
    hamaus   -> Hamaus et al 2014 (algo similar a una ley de potencias) eq 2'''

    directory = f'../profiles/voids/{sample}/{name}.fits'
    header = fits.open(directory)[0]
    Rp     = fits.open(directory)[1].data.Rp
    p      = fits.open(directory)[2].data
    covar = fits.open(directory)[3].data


    covDSt = covar.covDSt.reshape(60,60)
    covDSx = covar.covDSx.reshape(60,60)

    if fitS:
        covS   = covar.covS.reshape(60,60)
        
        f_S, fcov_S = curve_fit(projected_density, Rp, p.Sigma.reshape(101,60)[0], sigma=covS, p0=(1,1))




    # profile = fits.open('../profiles/voids/Rv_15-18/Rv1518.fits')
    # p = profile[1].data
    # cov = profile[2].data

    # Rv_min, Rv_max = profile[0].header['RV_MIN'], profile[0].header['RV_MAX']

    # eS = np.sqrt(np.diag(cov.cov_S.reshape(40,40)))
    # eDSt = np.sqrt(np.diag(cov.cov_DSt.reshape(40,40)))

    # p_S, pcov_S = curve_fit(parallel_S, p.Rp, p.Sigma, sigma=eS)
    # p_DSt, pcov_DSt = curve_fit(parallel_DS, p.Rp, p.DSigmaT, sigma=eDSt)

    # perr_S = np.sqrt(np.diag(pcov_S))
    # perr_DSt = np.sqrt(np.diag(pcov_DSt))

    # hdu = fits.Header()
    # hdu.append(('Nvoids',profile[0].header['NVOIDS']))
    # hdu.append(('Rv_min',profile[0].header['RV_MIN']))
    # hdu.append(('Rv_max',profile[0].header['RV_MAX']))
    
    # table_opt = [fits.Column(name='p_S',format='D',array=p_S),
    #              fits.Column(name='p_S',format='D',array=p_DSt)]
    
    # table_err = [fits.Column(name='p_eS',format='D',array=perr_S),
    #              fits.Column(name='p_eS',format='D',array=perr_DSt)]

    # tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_opt))
    # tbhdu_cov = fits.BinTableHDU.from_columns(fits.ColDefs(table_err))
            
    # primary_hdu = fits.PrimaryHDU(header=hdu)
            
    # hdul = fits.HDUList([primary_hdu, tbhdu_pro, tbhdu_cov])
    
    # try:
    #         os.mkdir(f'../profiles/voids/Rv_{int(Rv_min)}-{int(Rv_max)}/fitting')
    # except FileExistsError:
    #         pass

    # hdul.writeto(f'../profiles/voids/fitting/fitLS_Rv{int(Rv_min)}{int(Rv_max)}',overwrite=True)
    
