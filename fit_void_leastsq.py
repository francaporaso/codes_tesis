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

def clampitt(r,A3,Rv):
    '''Clampitt et al (2016); eq 12
       id = 0'''
    A0 = 1-A3
    if r<Rv:
        return A0-1+A3*(r/Rv)**3
    else:
        return A0+A3-1
    # return np.piecewise(r,[r<Rv],[lambda r: A0-1+A3*(r/Rv)**3,A0+A3-1]) 

def krause(r,A3,A0,Rv):
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

def hamaus(r, rs, delta, Rv, a, b):
    '''Hamaus et al (2014); eq 2
       id = 3'''
    return delta*(1-(r/rs)**a)/(1+(r/Rv)**b)


### Densidades proyectadas para cada funcion

def sigma_clampitt(r,A3,Rv):
    
    def integrand(z,R,A3,Rv):
        return clampitt(np.sqrt(np.square(z)+np.square(R)),A3,Rv)

    sigma = np.zeros_like(r)
    for j,x in enumerate(r):
        sigma[j] = quad(integrand, -np.inf, np.inf, args=(x,A3,Rv))[0]
    return sigma

def sigma_krause(r,A3,A0,Rv):
    
    def integrand(z,R,A3,A0,Rv):
        return krause(np.sqrt(np.square(z)+np.square(R)),A3,A0,Rv)
  
    sigma = np.zeros_like(r)
    for j,x in enumerate(r):
        sigma[j] = quad(integrand, -np.inf, np.inf, args=(x,A3,A0,Rv))[0]
    return sigma

def sigma_higuchi(r,A3,A0,Rv):
    pass

def sigma_hamaus(r,rs,delta,a,b):
    Rv=1.
    def integrand(z,R,rs,delta,a,b):
        return hamaus(np.sqrt(np.square(z)+np.square(R)),rs,delta,Rv,a,b)
  
    sigma = np.zeros_like(r)
    for j,x in enumerate(r):
        sigma[j] = quad(integrand, -np.inf, np.inf, args=(x,rs,delta,a,b))[0]
    return sigma

# ----- o -----

def delta_sigma_clampitt(r,A3,Rv):
    def integrand(x,A3,Rv):
        return sigma_clampitt([x],A3,Rv)*x

    anillo = sigma_clampitt(r,A3,Rv)
    disco = np.zeros_like(r)
    for j,p in enumerate(r):
        disco[j] = 2./p**2 * quad(integrand, 0., p, args=(A3,Rv))[0]

    return disco - anillo

def delta_sigma_krause(r,A3,A0,Rv):
    
    def integrand(x):
        return sigma_krause([x],A3,A0,Rv)*x

    anillo = sigma_krause(r,A3,A0,Rv)
    disco = np.zeros_like(r)
    for j,p in enumerate(r):
        disco[j] = 2./p**2 * quad(integrand, 0., p)[0]
        
    return disco - anillo

def delta_sigma_higuchi():
    pass

def delta_sigma_hamaus(data,rs,delta,a,b):
    
    #r, Rv = data[:-1], data[-1]
    r = [data]
    # Rv = 1.
    
    def integrand(x,rs,delta,a,b):
        return sigma_hamaus([x],rs,delta,a,b)*x

    anillo = sigma_hamaus(r,rs,delta,a,b)
    disco = np.zeros_like(r)
    for j,p in enumerate(r):
        disco[j] = 2./p**2 * quad(integrand, 0., p, args=(rs,delta,a,b))[0]

    return disco-anillo

## ----- o -----

def DSt_clampitt_unpack(kargs):
    return delta_sigma_clampitt(*kargs)

def DSt_clampitt_parallel(data,A3,Rv):
    
    r, ncores = data[:-1], int(data[-1])
    partial = DSt_clampitt_unpack
    
    if ncores > len(r):
        ncores = len(r)
    
    lbins = int(round(len(r)/float(ncores), 0))
    slices = ((np.arange(lbins)+1)*ncores).astype(int)
    slices = slices[(slices < len(r))]
    Rsplit = np.split(r,slices)

    dsigma = np.zeros((len(Rsplit),ncores))
    nparams = 2

    for j,r_j in enumerate(Rsplit):
        
        num = len(r_j)
        
        A3_arr    = np.full_like(r_j,A3)
        Rv_arr = np.full_like(r_j,Rv)
        
        entrada = np.array([r_j.T,A3_arr, Rv_arr]).T
                
        with Pool(processes=num) as pool:
            salida = np.array(pool.map(partial,entrada))
            pool.close()
            pool.join()
        
        dsigma[j] = salida.flatten()

    return dsigma.flatten()


def DSt_hamaus_unpack(kargs):
    return delta_sigma_hamaus(*kargs)

def DSt_hamaus_parallel(data,rs,delta,a,b):
    
    r, ncores = data[:-1], int(data[-1])
    partial = DSt_hamaus_unpack
    
    if ncores > len(r):
        ncores = len(r)
    
    lbins = int(round(len(r)/float(ncores), 0))
    slices = ((np.arange(lbins)+1)*ncores).astype(int)
    slices = slices[(slices < len(r))]
    Rsplit = np.split(r,slices)

    dsigma = np.zeros((len(Rsplit),ncores))
    nparams = 5

    for j,r_j in enumerate(Rsplit):
        
        num = len(r_j)
        
        rs_arr    = np.full_like(r_j,rs)
        delta_arr = np.full_like(r_j,delta)
        a_arr     = np.full_like(r_j,a)
        b_arr     = np.full_like(r_j,b)
        
        entrada = np.array([r_j.T,rs_arr,delta_arr,a_arr,b_arr]).T
                
        with Pool(processes=num) as pool:
            salida = np.array(pool.map(partial,entrada))
            pool.close()
            pool.join()
        
        dsigma[j] = salida.flatten()

    return dsigma.flatten()



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-sample', action='store', dest='sample',default='Rv_6-9')
  parser.add_argument('-name', action='store', dest='name',default='smallz_6-9')
  parser.add_argument('-out', action='store', dest='out',default='pru')
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
  output   = args.out
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

    
  if rho=='clampitt':
      projected_density = sigma_clampitt
      projected_density_contrast = delta_sigma_clampitt
      nparams = 2
  elif rho=='krause':
      projected_density = sigma_krause
      projected_density_contrast = delta_sigma_krause
      nparams = 3
  elif rho=='higuchi':
      projected_density = sigma_higuchi
      projected_density_contrast = delta_sigma_higuchi
      nparams = 4
  elif rho=='hamaus':
      projected_density = sigma_hamaus
      projected_density_contrast = DSt_hamaus_parallel
      nparams = 4
  else:
      raise TypeError(f'rho: "{rho}" no es ninguna de las funciones definidas.')
      
  directory = f'../profiles/voids/{sample}/{name}.fits'
  header    = fits.open(directory)[0]
  Rp        = fits.open(directory)[1].data.Rp
  p         = fits.open(directory)[2].data
  covar     = fits.open(directory)[3].data
  
  print(f'Fitting from {directory}')
  print(f'Using {ncores} cores')
  print(f'Model: {rho}')
  
  
  if fitS & fitDS:
      raise ValueError('No es compatible fitS y fitDS = True, dejar sin especificar para fitear ambos')
  
  p0 = np.ones(nparams)
  bounds = (-np.inf,np.inf)
      
  if rho == 'hamaus':
      p0 = np.array([ 0.9, -0.9,  1.4,  4.])
      var = np.append(Rp,ncores)
      bounds = (np.array([0.,0., -np.inf, -np.inf, -np.inf]),np.array([np.inf,np.inf, np.inf, np.inf, np.inf]))
  
  
  if fitS:
      covS   = covar.covS.reshape(60,60)
          
      if usecov:
          out = f'S_cov'
          print(f'Fitting Sigma, using covariance matrix')
          f_S, fcov_S = curve_fit(projected_density, Rp, p.Sigma.reshape(101,60)[0], sigma=covS, p0=p0)
          
          print(f'parametros ajustados: {f_S}')
          print(f'errores: {np.diag(fcov_S)}')
          table_opt = [fits.Column(name='f_S',format='D',array=f_S)]
          table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten())]
  
      else:   
          out = f'S_diag'
  
          print(f'Fitting Sigma, using covariance diagonal only')
  
          eS   = np.sqrt(np.diag(covS))
          f_S, fcov_S = curve_fit(projected_density, Rp, p.Sigma.reshape(101,60)[0], sigma=eS, p0=p0)
  
          print(f'parametros ajustados: {f_S}')
          print(f'errores: {np.diag(fcov_S)}')
          table_opt = [fits.Column(name='f_S',format='D',array=f_S)]
          table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten())]
  
  elif fitDS:
      covDSt = covar.covDSt.reshape(60,60)
  
      if usecov:
          out = f'DS_cov'
  
          print(f'Fitting Delta Sigma, using covariance matrix')
  
          f_DS, fcov_DS = curve_fit(projected_density_contrast, Rp, p.DSigma_T.reshape(101,60)[0], sigma=covDSt, p0=p0)
              
          print(f'parametros ajustados: {f_DS}')
          print(f'errores: {np.diag(fcov_DS)}')
          table_opt = [fits.Column(name='f_DSt',format='D',array=f_DS)]
          table_err = [fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]
  
      else: 
          out = f'DS_diag'
  
          print(f'Fitting Delta Sigma, using covariance diagonal only')
  
          eDSt = np.sqrt(np.diag(covDSt))
          f_DS, fcov_DS = curve_fit(projected_density_contrast, Rp, p.DSigma_T.reshape(101,60)[0], sigma=eDSt, p0=p0)
              
          print(f'parametros ajustados: {f_DS}')
          print(f'errores: {np.diag(fcov_DS)}')
          table_opt = [fits.Column(name='f_DSt',format='D',array=f_DS)]
          table_err = [fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]
      
  else:
      covS   = covar.covS.reshape(60,60)
      covDSt = covar.covDSt.reshape(60,60)
  
      if usecov:
          out =f'full_cov'
          print(f'Fitting Sigma and Delta Sigma, using covariance matrix')
  
          f_S, fcov_S   = curve_fit(projected_density, Rp, p.Sigma.reshape(101,60)[0], sigma=covS, p0=p0)
          f_DS, fcov_DS = curve_fit(projected_density, Rp, p.DSigma_T.reshape(101,60)[0], sigma=covDSt, p0=p0)
              
          table_opt = [fits.Column(name='f_S',format='D',array=f_S),
                      fits.Column(name='f_DSt',format='D',array=f_DS)]
              
          table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten()),
                      fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]
  
      else:
          out =f'full_diag'
          print(f'Fitting Sigma, using covariance diagonal only')
  
          eS   = np.sqrt(np.diag(covS))
          eDSt = np.sqrt(np.diag(covDSt))
          f_S, fcov_S   = curve_fit(projected_density, Rp, p.Sigma.reshape(101,60)[0], sigma=eS, p0=p0)
  
          print(f'Fitting Delta Sigma, using covariance diagonal only')
  
          f_DS, fcov_DS = curve_fit(projected_density_contrast, var, p.DSigma_T.reshape(101,60)[0], sigma=eDSt, p0=p0)
  
          table_opt = [fits.Column(name='f_S',format='D',array=f_S),
                      fits.Column(name='f_DSt',format='D',array=f_DS)]
              
          table_err = [fits.Column(name='fcov_S',format='D',array=fcov_S.flatten()),
                      fits.Column(name='fcov_DSt',format='D',array=fcov_DS.flatten())]
  
  
  hdu = fits.Header()
  hdu.append(('Nvoids',header.header['N_VOIDS']))
  hdu.append(('Rv_min',header.header['RV_MIN']))
  hdu.append(('Rv_max',header.header['RV_MAX']))
  hdu.append(f'using {rho}')
  
  tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_opt))
  tbhdu_cov = fits.BinTableHDU.from_columns(fits.ColDefs(table_err))
          
  primary_hdu = fits.PrimaryHDU(header=hdu)
          
  hdul = fits.HDUList([primary_hdu, tbhdu_pro, tbhdu_cov])
  
  out = out+output
  try:
      os.mkdir(f'../profiles/voids/{sample}/fit')
  except FileExistsError:
      pass
  hdul.writeto(f'../profiles/voids/{sample}/fit/lsq_{name}_{rho}_{out}.fits',overwrite=True)
  print(f'Saved in ../profiles/voids/{sample}/fit/lsq_{name}_{rho}_{out}.fits !')
