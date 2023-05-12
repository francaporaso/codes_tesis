'''Ajuste de perfiles de voids mediante cuadrados minimos. Por defecto ajusta ambos Sigma y DSigma'''
import numpy as np
from scipy.integrate import quad, quad_vec
import matplotlib.pyplot as plt
from multiprocessing import Pool
from astropy.io import fits
from scipy.optimize import curve_fit
import argparse
import os

'''rho: ( en realidad son las fluctuaciones rho/rhomean - 1 )
    clampitt -> Clampitt et al 2016 (cuadratica adentro de Rv, constante afuera) eq 12
    krause   -> Krause et al 2012 (como clampitt pero con compensacion) eq 1 pero leer texto
    higuchi  -> Higuchi et al 2013 (conocida como top hat, 3 contantes) eq 23
    hamaus   -> Hamaus et al 2014 (algo similar a una ley de potencias) eq 2'''

def clampitt(r,Rv,A3):
    '''Clampitt et al (2016); eq 12
       id = 0'''
    A0 = 1-A3
    if r<Rv:
        return A0-1+A3*(r/Rv)**3
    else:
        return A0+A3-1
    # return np.piecewise(r,[r<Rv],[lambda r: A0-1+A3*(r/Rv)**3,A0+A3-1]) 

def krause(r,Rv,A3,A0):
    '''Krause et al (2012); eq 1 (see text)
       id = 1'''
    if r<Rv:
        return A0-1+A3*(r/Rv)**3
    elif r>2*Rv: #ultima parte
        return 0
    else: #centro
        return A0+A3-1

    # return np.piecewise(r,[r<Rv, 2*Rv>r>Rv],[lambda r: A0+A3*(r/Rv)**3, A0+A3, 1])

def higuchi(r,R1,R2,rho1,rho2):
    '''Higuchi et al (2013); eq 23
       id = 2'''
    if r<R1:
        return rho1-1 
    elif r>R2:
        return 0
    else:
        return rho2-1

def hamaus(r, delta, rs, Rv, a, b):
    '''Hamaus et al (2014); eq 2
       id = 3'''
    return delta*(1-(r/rs)**a)/(1+(r/Rv)**b)


rho_id = {0: clampitt, 1: krause, 2: higuchi, 3: hamaus}

def projected_density(data, *params, rmax=np.inf):
    '''perfil de densidad proyectada dada la densidad 3D
    rvals   (array) : puntos de r a evaluar el perfil
    *params (float) : parametros de la funcion rho (los que se ajustan)
    rho     (func)  : densidad 3D
    rmax    (int)   : valor maximo de r para integrar
    
    return
    density  (float) : densidad proyectada en rvals'''

    rvals, rho = data[:-1], data[-1]
    if isinstance(rho,float):
        rho = rho_id.get(rho)
        
    
    def integrand(z, r, *params):
        return rho(np.sqrt(np.square(r) + np.square(z)), *params)
    
    density = np.array([quad(integrand, -rmax, rmax, args=(r,)+params)[0] for r in rvals])
    density = np.array([quad(integrand, -rmax, rmax, args=(r,)+params)[0] for r in rvals])

    return density


def projected_density_contrast(data, *params, rmax=np.inf):
    
    '''perfil de contraste de densidad proyectada dada la densidad 3D
    rvals   (float) : punto de r a evaluar el perfil
    *params (float) : parametros de la funcion rho (los que se ajustan)
    rho     (func)  : densidad 3D
    rmax    (int)   : valor maximo de r para integrar
    
    contrast (float): contraste de densidad proy en rvals'''
    
    rvals, rho = data[:-1], data[-1]
    # rho = rho_id.get(rho)

    def integrand(x,*p): 
        return x*projected_density([x], rho, *p, rmax=rmax)
    
    # def integrand(x,*p): 
    #     return x*aux(x, *p, rho=rho, rmax=rmax)
    
    anillo = projected_density([rvals], rho, *params, rmax=rmax)
    disco  = np.array([2./(np.square(r))*quad(integrand, 0., r, args=(r,)+params)[0] for r in [rvals]])

    contrast = disco - anillo
    return contrast

def projected_density_contrast_unpack(kargs):
    return projected_density_contrast(*kargs)

def projected_density_contrast_parallel(data, *params, rmax=np.inf):

    rvals, rho, ncores = data[:-2], data[-2], int(data[-1])
    rho = rho_id.get(rho)
    partial = projected_density_contrast_unpack
    
    lbins = int(round(len(rvals)/float(ncores), 0))
    slices = ((np.arange(lbins)+1)*ncores).astype(int)
    slices = slices[(slices < len(rvals))]
    Rsplit = np.split(rvals,slices)

    dsigma = np.zeros(len(Rsplit))
    nparams = len(params)

    for j,r_j in enumerate(Rsplit):

        rhos   = np.full_like(r_j,data[-2])
        par_a   = np.array([np.full_like(r_j,params[k]) for k in np.arange(nparams)])
        entrada = np.append(np.array([r_j,rhos]), np.array([par_a[k] for k in np.arange(nparams)]),axis=0).T

        with Pool(processes=ncores) as pool:
            salida = np.array(pool.map(partial,entrada))
            pool.close()
            pool.join()

        dsigma[j] = salida

    return dsigma

def fitear(sample,name):

    if fitS & fitDS:
        raise ValueError('No es compatible fitS y fitDS = True, dejar sin especificar para fitear ambos')

    variables = np.append(Rp,rho)
    var_wcores = np.append(variables,ncores)
    p0 = np.ones(nparams)
    
    if fitS:
        covS   = covar.covS.reshape(60,60)
        
        if usecov:
            out = f'S_cov'
            print(f'Fitting Sigma, using covariance matrix')
            f_S, fcov_S = curve_fit(projected_density, variables, p.Sigma.reshape(101,60)[0], sigma=covS, p0=p0)

            table_opt = [fits.Column(name='f_S',format='D',array=f_S)]
            table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten())]

        else:
            out = f'S_diag'

            print(f'Fitting Sigma, using covariance diagonal only')

            eS   = np.sqrt(np.diag(covS))
            f_S, fcov_S = curve_fit(projected_density, variables, p.Sigma.reshape(101,60)[0], sigma=eS, p0=p0)

            table_opt = [fits.Column(name='f_S',format='D',array=f_S)]
            table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten())]

    elif fitDS:#REVISAR
        covDSt = covar.covDSt.reshape(60,60)

        if usecov:
            out = f'DS_cov'

            print(f'Fitting Delta Sigma, using covariance matrix')

            f_DS, fcov_DS = curve_fit(projected_density, variables, p.DSigma_T.reshape(101,60)[0], sigma=covDSt, p0=p0)
            
            table_opt = [fits.Column(name='f_DSt',format='D',array=f_DS)]
            table_err = [fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]

        else: 
            out = f'DS_diag'

            print(f'Fitting Delta Sigma, using covariance diagonal only')

            eDSt = np.sqrt(np.diag(covDSt))
            f_DS, fcov_DS = curve_fit(projected_density_contrast, variables, p.DSigma_T.reshape(101,60)[0], sigma=eDSt, p0=p0)
            print('FUNCO 2!')
            
            table_opt = [fits.Column(name='f_DSt',format='D',array=f_DS)]
            table_err = [fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]
    
    else:
        covS   = covar.covS.reshape(60,60)
        covDSt = covar.covDSt.reshape(60,60)

        if usecov:
            out =f'full_cov'
            print(f'Fitting Sigma and Delta Sigma, using covariance matrix')

            f_S, fcov_S   = curve_fit(projected_density, variables, p.Sigma.reshape(101,60)[0], sigma=covS, p0=p0)
            # f_DS, fcov_DS = curve_fit(projected_density, variables, p.DSigma_T.reshape(101,60)[0], sigma=covDSt, p0=p0)
            
            table_opt = [fits.Column(name='f_S',format='D',array=f_S),
                         fits.Column(name='f_DSt',format='D',array=f_DS)]
            
            table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten()),
                         fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]

        else:
            out =f'full_diag'
            print(f'Fitting Sigma, using covariance diagonal only')

            eS   = np.sqrt(np.diag(covS))
            eDSt = np.sqrt(np.diag(covDSt))
            f_S, fcov_S   = curve_fit(projected_density, variables, p.Sigma.reshape(101,60)[0], sigma=eS, p0=p0)

            print(f'Fitting Delta Sigma, using covariance diagonal only')

            f_DS, fcov_DS = curve_fit(projected_density_contrast_parallel, var_wcores, p.DSigma_T.reshape(101,60)[0], sigma=eDSt, p0=p0)

            table_opt = [fits.Column(name='f_S',format='D',array=f_S),
                         fits.Column(name='f_DSt',format='D',array=f_DS)]
            
            table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten()),
                         fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]

    print(f'Saved in ../profiles/voids/{sample}/fit/lsq_{name}_{rho_str}_{out}.fits !')

    hdu = fits.Header()
    hdu.append(('Nvoids',header.header['N_VOIDS']))
    hdu.append(('Rv_min',header.header['RV_MIN']))
    hdu.append(('Rv_max',header.header['RV_MAX']))
    hdu.append(f'using {rho_str}')
    

    tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_opt))
    tbhdu_cov = fits.BinTableHDU.from_columns(fits.ColDefs(table_err))
            
    primary_hdu = fits.PrimaryHDU(header=hdu)
            
    hdul = fits.HDUList([primary_hdu, tbhdu_pro, tbhdu_cov])
    
    try:
        os.mkdir(f'../profiles/voids/{sample}/fit')
    except FileExistsError:
        pass

    hdul.writeto(f'../profiles/voids/{sample}/fit/lsq_{name}_{rho_str}_{out}.fits',overwrite=True)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', action='store', dest='sample',default='Rv_6-9')
    parser.add_argument('-name', action='store', dest='name',default='smallz_6-9')
    parser.add_argument('-ncores', action='store', dest='ncores',default=10)
    parser.add_argument('-rmax', action='store', dest='rmax',default='inf')
    parser.add_argument('-fitS', action='store', dest='fitS',default=False)
    parser.add_argument('-fitDS', action='store', dest='fitDS',default=False)
    parser.add_argument('-usecov', action='store', dest='usecov',default=False)
    parser.add_argument('-rho', action='store', dest='rho',default='clampitt')
    parser.add_argument('-p0', action='store', dest='p0',default=1)
    args = parser.parse_args()
    
    sample = args.sample
    name   = args.name
    ncores = int(args.ncores)
    fitS   = bool(args.fitS)
    fitDS  = bool(args.fitDS)
    usecov  = bool(args.usecov)
    rho    = args.rho   
    p0     = float(args.p0)
    
    if args.rmax == 'inf':
        rmax = np.inf
    else:
        rmax = float(args.rmax)

    '''rho:
    clampitt -> Clampitt et al 2016 (cuadratica adentro de Rv, constante afuera) eq 12
    krause   -> Krause et al 2012 (como clampitt pero con compensacion) eq 1 pero leer texto
    higuchi  -> Higuchi et al 2013 (conocida como top hat, 3 contantes) eq 23
    hamaus   -> Hamaus et al 2014 (algo similar a una ley de potencias) eq 2'''

    rho_dict = {'clampitt': (0,2), 'krause': (1,3), 'higuchi': (2,4), 'hamaus': (3,5)} #(id_func, nparams)
    nparams  = rho_dict.get(rho)[1]
    rho_str  = np.copy(rho)
    rho      = rho_dict.get(rho)[0]
    

    directory = f'../profiles/voids/{sample}/{name}.fits'
    header = fits.open(directory)[0]
    Rp     = fits.open(directory)[1].data.Rp
    p      = fits.open(directory)[2].data
    covar = fits.open(directory)[3].data

    print(f'Fitting from {directory}')
    print(f'Using {ncores} cores')
    print(f'Model: {rho_str}')

    fitear(sample,name)

    # if fitS & fitDS:
#         raise ValueError('No es compatible fitS y fitDS = True, dejar sin especificar para fitear ambos')
# 
    # variables = np.append(Rp,rho)
    # var_wcores = np.append(variables,ncores)
    # p0 = np.ones(nparams)
    # 
    # if fitS:
        # covS   = covar.covS.reshape(60,60)
        # 
        # if usecov:
            # out = f'S_cov'
            # print(f'Fitting Sigma, using covariance matrix')
            # f_S, fcov_S = curve_fit(projected_density, variables, p.Sigma.reshape(101,60)[0], sigma=covS, p0=p0)
# 
            # table_opt = [fits.Column(name='f_S',format='D',array=f_S)]
            # table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten())]
# 
        # else:
            # out = f'S_diag'
# 
            # print(f'Fitting Sigma, using covariance diagonal only')
# 
            # eS   = np.sqrt(np.diag(covS))
            # f_S, fcov_S = curve_fit(projected_density, variables, p.Sigma.reshape(101,60)[0], sigma=eS, p0=p0)
# 
            # table_opt = [fits.Column(name='f_S',format='D',array=f_S)]
            # table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten())]
# 
    # elif fitDS:#REVISAR
        # covDSt = covar.covDSt.reshape(60,60)
# 
        # if usecov:
            # out = f'DS_cov'
# 
            # print(f'Fitting Delta Sigma, using covariance matrix')
# 
            # f_DS, fcov_DS = curve_fit(projected_density, variables, p.DSigma_T.reshape(101,60)[0], sigma=covDSt, p0=p0)
            # 
            # table_opt = [fits.Column(name='f_DSt',format='D',array=f_DS)]
            # table_err = [fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]
# 
        # else: 
            # out = f'DS_diag'
# 
            # print(f'Fitting Delta Sigma, using covariance diagonal only')
# 
            # eDSt = np.sqrt(np.diag(covDSt))
            # f_DS, fcov_DS = curve_fit(projected_density_contrast, variables, p.DSigma_T.reshape(101,60)[0], sigma=eDSt, p0=p0)
            # print('FUNCO 2!')
            # 
            # table_opt = [fits.Column(name='f_DSt',format='D',array=f_DS)]
            # table_err = [fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]
    # 
    # else:
        # covS   = covar.covS.reshape(60,60)
        # covDSt = covar.covDSt.reshape(60,60)
# 
        # if usecov:
            # out =f'full_cov'
            # print(f'Fitting Sigma and Delta Sigma, using covariance matrix')
# 
            # f_S, fcov_S   = curve_fit(projected_density, variables, p.Sigma.reshape(101,60)[0], sigma=covS, p0=p0)
            # f_DS, fcov_DS = curve_fit(projected_density, variables, p.DSigma_T.reshape(101,60)[0], sigma=covDSt, p0=p0)
            # 
            # table_opt = [fits.Column(name='f_S',format='D',array=f_S),
                        #  fits.Column(name='f_DSt',format='D',array=f_DS)]
            # 
            # table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten()),
                        #  fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]
# 
        # else:
            # out =f'full_diag'
            # print(f'Fitting Sigma, using covariance diagonal only')
# 
            # eS   = np.sqrt(np.diag(covS))
            # eDSt = np.sqrt(np.diag(covDSt))
            # f_S, fcov_S   = curve_fit(projected_density, variables, p.Sigma.reshape(101,60)[0], sigma=eS, p0=p0)
# 
            # print(f'Fitting Delta Sigma, using covariance diagonal only')
# 
            # f_DS, fcov_DS = curve_fit(projected_density_contrast_parallel, var_wcores, p.DSigma_T.reshape(101,60)[0], sigma=eDSt, p0=p0)
# 
            # table_opt = [fits.Column(name='f_S',format='D',array=f_S),
                        #  fits.Column(name='f_DSt',format='D',array=f_DS)]
            # 
            # table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten()),
                        #  fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]
# 
    # print(f'Saved in ../profiles/voids/{sample}/fit/lsq_{name}_{rho_str}_{out}.fits !')
# 
    # hdu = fits.Header()
    # hdu.append(('Nvoids',header.header['N_VOIDS']))
    # hdu.append(('Rv_min',header.header['RV_MIN']))
    # hdu.append(('Rv_max',header.header['RV_MAX']))
    # hdu.append(f'using {rho_str}')
    # 
# 
    # tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_opt))
    # tbhdu_cov = fits.BinTableHDU.from_columns(fits.ColDefs(table_err))
            # 
    # primary_hdu = fits.PrimaryHDU(header=hdu)
            # 
    # hdul = fits.HDUList([primary_hdu, tbhdu_pro, tbhdu_cov])
    # 
    # try:
        # os.mkdir(f'../profiles/voids/{sample}/fit')
    # except FileExistsError:
        # pass
# 
    # hdul.writeto(f'../profiles/voids/{sample}/fit/lsq_{name}_{rho_str}_{out}.fits',overwrite=True)
