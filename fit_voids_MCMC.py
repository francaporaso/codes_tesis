import sys
sys.path.append('/home/fcaporaso/lens_codes_v3.7')
import time
import numpy as np
from astropy.io import fits
from multiprocessing import Pool
from multiprocessing import Process
import argparse
from astropy.constants import G,c,M_sun,pc
import emcee
import os
from fit_void_leastsq import *

# ARGUMENTOS

'''
sample = 'Rv_6-9'
name   = 'smallz_6-9'
output = 'pru'
fitS   = False
fitDS  = False
ncores = 32
nit    = 5
pos    = 'uniform'
rho    = 'hamaus'
RIN    = 0.01
ROUT   = 3.01
'''

parser = argparse.ArgumentParser()
parser.add_argument('-sample', action='store', dest='sample',default='Rv_6-9')
parser.add_argument('-name', action='store', dest='name',default='smallz_6-9')
parser.add_argument('-out', action='store', dest='out',default='pru')
parser.add_argument('-fitS', action='store', dest='fitS',default=False)
parser.add_argument('-fitDS', action='store', dest='fitDS',default=False)
parser.add_argument('-ncores', action='store', dest='ncores',default=32)
parser.add_argument('-nit', action='store', dest='nit',default=50)
parser.add_argument('-pos', action='store', dest='pos',default='uniform')
parser.add_argument('-rho', action='store', dest='rho',default='clampitt')
parser.add_argument('-RIN', action='store', dest='RIN', default=0.01)
parser.add_argument('-ROUT', action='store', dest='ROUT', default=3.01)
args = parser.parse_args()

sample = args.sample
name   = args.name
output = args.out
fitS   = bool(args.fitS)
fitDS  = bool(args.fitDS)
ncores = int(args.ncores)
nit    = int(args.nit)
pos    = args.pos
rho    = args.rho
RIN    = float(args.RIN)
ROUT   = float(args.ROUT)

pos_name = pos
'''pos: distribucion de los walkers
- uniform : distribucion uniforme
- gaussian: distribucion gaussiana'''

'''rho: 
- clampitt -> Clampitt et al 2016 (cuadratica adentro de Rv, constante afuera) eq 12
- krause   -> Krause et al 2012 (como clampitt pero con compensacion) eq 1 pero leer texto
- higuchi  -> Higuchi et al 2013 (conocida como top hat, 3 contantes) eq 23
- hamaus   -> Hamaus et al 2014 (algo similar a una ley de potencias) eq 2'''


directory = f'../profiles/voids/{sample}/{name}.fits'
header    = fits.open(directory)[0]
Rp        = fits.open(directory)[1].data.Rp
p         = fits.open(directory)[2].data
covar     = fits.open(directory)[3].data

outfolder = f'../profiles/voids/{sample}/fit/'

try:
    os.mkdir(outfolder)
except FileExistsError:
    pass

if fitS & fitDS:
    raise ValueError('No es compatible fitS y fitDS = True. Dejar sin especificar para ajustar ambos')

# CLAMPITT
# ----- 0 ------

def log_likelihoodDS_clampitt(data, R, DS, iCds):  
    
    A3, Rv = data
    variables = R, ncores

    ds = DSt_clampitt_parallel(variables,A3,Rv)

    return -np.dot((ds-DS),np.dot(iCds,(ds-DS)))/2.

def log_probabilityDS_clampitt(data, R, DS, eDS):
    
    A3, Rv = data
    variables = R, ncores

    if (-10. < A3 < 10.) and (0. < Rv < 3.):
        return log_likelihoodDS_clampitt(data, R, DS, eDS)
    return -np.inf

def log_likelihoodS_clampitt(data, R, S, iCs): 
    
    A3, Rv = data
    s = sigma_clampitt(R, A3, Rv)
    
    return -np.dot((s-S),np.dot(iCs,(s-S)))/2.

def log_probabilityS_clampitt(data, R, S, eS):
    
    A3, Rv = data
    if (-10. < A3 < 10.) and (0. < Rv < 3.):
        return log_likelihoodS_clampitt(data, R, S, eS)
    return -np.inf

# HAMAUS
# ----- 0 -----

def log_likelihoodDS_hamaus(data, R, DS, iCds):  
    
    rs,delta,Rv,a,b = data
    # variables = np.append(R, ncores)

    ds = DSt_hamaus_parallel(R,rs,delta,Rv,a,b)

    return -np.dot((ds-DS),np.dot(iCds,(ds-DS)))/2.

def log_probabilityDS_hamaus(data, R, DS, eDS):
    
    rs,delta,Rv,a,b = data

    if (0. < rs < 50.) and (-10. < delta < 0.) and (.5 < Rv < 2.) and (0. < a < 10.) and (0. < b < 10.) and (a < b):
        return log_likelihoodDS_hamaus(data, R, DS, eDS)
    return -np.inf

def log_likelihoodS_hamaus(data, R, S, iCs): 
    
    rs,delta,Rv,a,b = data
    s = sigma_hamaus(R,rs,delta,Rv,a,b)
    
    return -np.dot((s-S),np.dot(iCs,(s-S)))/2.

def log_probabilityS_hamaus(data, R, S, eS):
    
    rs,delta,Rv,a,b = data

    if (0. < rs < 50.) and (-10. < delta < 0.) and (.5 < Rv < 2.)  and (0. < a < 10.) and (0. < b < 10.) and (a < b):
        try:
            l = log_likelihoodS_hamaus(data, R, S, eS)
            return l
        except:
            print('hubo un error calculando la probabilidad (quiza un NaN), devolviendo -inf)')
            return -np.inf
        
    return -np.inf


# INICIALIZANDO
# ----- 0 -----

print(f'Fitting from {directory}')
print(f'Using {ncores} cores') #esta mal usa hasta la cantidad de puntos que hay
print(f'Model: {rho}')
print(f'Distribucion: {pos}')


if rho=='clampitt':
    log_probability_DS = log_probabilityDS_clampitt
    log_probability_S  = log_probabilityS_clampitt
    variables = Rp
    if pos=='uniform':
        pos = np.array([np.random.uniform(-10.,10.,15),     #A3
                        np.random.uniform(0.5,2.,15)]).T     #Rv
    elif pos=='gaussian':
        pos = np.array([np.random.normal(1.,0.8,15),
                        np.random.normal(1.,0.8,15)]).T
elif rho=='krause':
    raise ValueError(f'{rho} no implementado')
elif rho=='higuchi':
    raise ValueError(f'{rho} no implementado')
elif rho=='hamaus':
    log_probability_DS = log_probabilityDS_hamaus
    log_probability_S  = log_probabilityS_hamaus
    variables = np.append(Rp,ncores)
    if pos=='uniform':
        pos = np.array([np.random.uniform(0.,50.,15),       #rs
                        np.random.uniform(-5.,0.,15),       #delta
                        np.random.uniform(0.5,2.,15),       #Rv
                        np.random.uniform(0.,10.,15),       #alpha
                        np.random.uniform(0.,10.,15)]).T    #beta
    elif pos=='gaussian':
        pos = np.array([np.random.normal(2.,0.8,15),        #rs
                        np.random.normal(-1.5,0.8,15),       #delta
                        np.random.normal(1.,0.6,15),       #Rv
                        np.random.normal(3.,0.8,15),        #alpha
                        np.random.normal(5.,0.8,15)]).T     #beta
else:
    raise TypeError(f'rho: "{rho}" no es ninguna de las funciones definidas.')

nwalkers, ndim = pos.shape

# RUNNING EMCEE
# ----- 0 -----

maskr   = (Rp > RIN)&(Rp < ROUT)
mr = np.meshgrid(maskr,maskr)[1]*np.meshgrid(maskr,maskr)[0]

# p = p[maskr]

t1 = time.time()

if fitDS:

    outname = f'DSt'

    CovDS = covar.covDSt.reshape(len(Rp),len(Rp))[mr]
    CovDS = CovDS.reshape(maskr.sum(),maskr.sum())
    iCds  =  np.linalg.inv(CovDS)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_DS, args=(variables, p.DSigma_T.reshape(101,60)[0], iCds))

    print('Fitting Delta Sigma')
    sampler.run_mcmc(pos, nit, progress=True)
    print('TOTAL TIME FIT')    
    print(f'{np.round((time.time()-t1)/60., 2)} min')

    mcmc_out = sampler.get_chain(flat=True).T
    if rho=='clampitt':

        A3 = np.percentile(mcmc_out[0][1500:], [16, 50, 84])
        Rv = np.percentile(mcmc_out[1][1500:], [16, 50, 84])

        hdu = fits.Header()
        hdu.append(('A3',np.round(A3[1],4)))
        hdu.append(('eA3_min',np.round(np.diff(A3)[0],4)))
        hdu.append(('eA3_max',np.round(np.diff(A3)[1],4)))
        hdu.append(('Rv',np.round(Rv[1],4)))
        hdu.append(('eRv_min',np.round(np.diff(Rv)[0],4)))
        hdu.append(('eRv_max',np.round(np.diff(Rv)[1],4)))
        
        table_opt = np.array([fits.Column(name='A3',format='D',array=mcmc_out[0]),
                              fits.Column(name='Rv',format='D',array=mcmc_out[1])])
        
    elif rho=='hamaus':
        rs    = np.percentile(mcmc_out[0][1500:], [16, 50, 84])
        delta = np.percentile(mcmc_out[1][1500:], [16, 50, 84])
        Rv    = np.percentile(mcmc_out[2][1500:], [16, 50, 84])
        a     = np.percentile(mcmc_out[3][1500:], [16, 50, 84])
        b     = np.percentile(mcmc_out[4][1500:], [16, 50, 84])

        hdu = fits.Header()
        hdu.append(('rs',np.round(rs[1],4)))
        hdu.append(('ers_min',np.round(np.diff(rs)[0],4)))
        hdu.append(('ers_max',np.round(np.diff(rs)[1],4)))
        hdu.append(('delta',np.round(delta[1],4)))
        hdu.append(('edelta_min',np.round(np.diff(delta)[0],4)))
        hdu.append(('edelta_max',np.round(np.diff(delta)[1],4)))
        hdu.append(('Rv',np.round(Rv[1],4)))
        hdu.append(('eRv_min',np.round(np.diff(Rv)[0],4)))
        hdu.append(('eRv_max',np.round(np.diff(Rv)[1],4)))
        hdu.append(('a',np.round(a[1],4)))
        hdu.append(('ea_min',np.round(np.diff(a)[0],4)))
        hdu.append(('ea_max',np.round(np.diff(a)[1],4)))
        hdu.append(('b',np.round(b[1],4)))
        hdu.append(('eb_min',np.round(np.diff(b)[0],4)))
        hdu.append(('eb_max',np.round(np.diff(b)[1],4)))

        table_opt = np.array([fits.Column(name='rs',format='D',array=mcmc_out[0]),
                              fits.Column(name='delta',format='D',array=mcmc_out[1]),
                              fits.Column(name='Rv',format='D',array=mcmc_out[2]),
                              fits.Column(name='alpha',format='D',array=mcmc_out[3]),
                              fits.Column(name='beta',format='D',array=mcmc_out[4])])
    else:
        raise ValueError(f'{rho} No implementado')

elif fitS:

    outname = f'S'

    CovS = covar.covS.reshape(len(Rp),len(Rp))[mr]
    CovS = CovS.reshape(sum(maskr),sum(maskr))
    iCs  =  np.linalg.inv(CovS)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_S, args=(Rp, p.Sigma.reshape(101,60)[0], iCs))
    print('Fitting Sigma')
    sampler.run_mcmc(pos, nit, progress=True)
    print('TOTAL TIME FIT')
    print(f'{np.round((time.time()-t1)/60., 2)} min')

    mcmc_out = sampler.get_chain(flat=True).T
    if rho=='clampitt':
        A3 = np.percentile(mcmc_out[0][1500:], [16, 50, 84])
        Rv = np.percentile(mcmc_out[1][1500:], [16, 50, 84])

        hdu = fits.Header()
        hdu.append(('A3',np.round(A3[1],4)))
        hdu.append(('eA3_min',np.round(np.diff(A3)[0],4)))
        hdu.append(('eA3_max',np.round(np.diff(A3)[1],4)))
        hdu.append(('Rv',np.round(Rv[1],4)))
        hdu.append(('eRv_min',np.round(np.diff(Rv)[0],4)))
        hdu.append(('eRv_max',np.round(np.diff(Rv)[1],4)))

        table_opt = np.array([fits.Column(name='A3',format='D',array=mcmc_out[0]),
                              fits.Column(name='Rv',format='D',array=mcmc_out[1])])
        
    elif rho=='hamaus':
        rs    = np.percentile(mcmc_out[0][1500:], [16, 50, 84])
        delta = np.percentile(mcmc_out[1][1500:], [16, 50, 84])
        Rv    = np.percentile(mcmc_out[2][1500:], [16, 50, 84])
        a     = np.percentile(mcmc_out[3][1500:], [16, 50, 84])
        b     = np.percentile(mcmc_out[4][1500:], [16, 50, 84])

        hdu = fits.Header()
        hdu.append(('rs',np.round(rs[1],4)))
        hdu.append(('ers_min',np.round(np.diff(rs)[0],4)))
        hdu.append(('ers_max',np.round(np.diff(rs)[1],4)))
        hdu.append(('delta',np.round(delta[1],4)))
        hdu.append(('edelta_min',np.round(np.diff(delta)[0],4)))
        hdu.append(('edelta_max',np.round(np.diff(delta)[1],4)))
        hdu.append(('Rv',np.round(Rv[1],4)))
        hdu.append(('eRv_min',np.round(np.diff(Rv)[0],4)))
        hdu.append(('eRv_max',np.round(np.diff(Rv)[1],4)))
        hdu.append(('a',np.round(a[1],4)))
        hdu.append(('ea_min',np.round(np.diff(a)[0],4)))
        hdu.append(('ea_max',np.round(np.diff(a)[1],4)))
        hdu.append(('b',np.round(b[1],4)))
        hdu.append(('eb_min',np.round(np.diff(b)[0],4)))
        hdu.append(('eb_max',np.round(np.diff(b)[1],4)))

        table_opt = np.array([fits.Column(name='rs',format='D',array=mcmc_out[0]),
                              fits.Column(name='delta',format='D',array=mcmc_out[1]),
                              fits.Column(name='Rv',format='D',array=mcmc_out[2]),
                              fits.Column(name='alpha',format='D',array=mcmc_out[3]),
                              fits.Column(name='beta',format='D',array=mcmc_out[4])])
    else:
        raise ValueError(f'{rho} No implementado')

else:

    outname = f'full'

    
    CovDS = covar.covDSt.reshape(len(Rp),len(Rp))[mr]
    CovDS = CovDS.reshape(maskr.sum(),maskr.sum())
    iCds  =  np.linalg.inv(CovDS)

    CovS  = covar.covS.reshape(len(Rp),len(Rp))[mr]
    CovS  = CovS.reshape(sum(maskr),sum(maskr))
    iCs   =  np.linalg.inv(CovS)

    samplerDS = emcee.EnsembleSampler(nwalkers, ndim, log_probability_DS, args=(variables, p.DSigma_T.reshape(101,60)[0], iCds))
    samplerS  = emcee.EnsembleSampler(nwalkers, ndim, log_probability_S, args=(Rp, p.Sigma.reshape(101,60)[0], iCs))

    print('Fitting Delta Sigma')
    samplerDS.run_mcmc(pos, nit, progress=True)
    print('TIME FIT DELTA SIGMA')
    print(f'{np.round((time.time()-t1)/60., 2)} min')
    print('Fitting Sigma')
    t2 = time.time()    
    samplerS.run_mcmc(pos, nit, progress=True)
    print('TIME FIT SIGMA')
    print(f'{np.round((time.time()-t2)/60., 2)} min')

    print('TOTAL TIME FIT')
    print(f'{np.round((time.time()-t1)/60., 2)} min')

    mcmc_outDS = samplerDS.get_chain(flat=True).T
    mcmc_outS  = samplerS.get_chain(flat=True).T

    if rho=='clampitt':
        A3_DS = np.percentile(mcmc_outDS[0][1500:], [16, 50, 84])
        Rv_DS = np.percentile(mcmc_outDS[1][1500:], [16, 50, 84])

        A3_S = np.percentile(mcmc_outS[0][1500:], [16, 50, 84])
        Rv_S = np.percentile(mcmc_outS[1][1500:], [16, 50, 84])

        hdu = fits.Header()
        hdu.append(('A3_DS',np.round(A3_DS[1],4)))
        hdu.append(('eA3_DS_min',np.round(np.diff(A3_DS)[0],4)))
        hdu.append(('eA3_DS_max',np.round(np.diff(A3_DS)[1],4)))
        hdu.append(('Rv_DS',np.round(Rv_DS[1],4)))
        hdu.append(('eRv_DS_min',np.round(np.diff(Rv_DS)[0],4)))
        hdu.append(('eRv_DS_max',np.round(np.diff(Rv_DS)[1],4)))

        hdu.append(('A3_S',np.round(A3_S[1],4)))
        hdu.append(('eA3_S_min',np.round(np.diff(A3_S)[0],4)))
        hdu.append(('eA3_S_max',np.round(np.diff(A3_S)[1],4)))
        hdu.append(('Rv_S',np.round(Rv_S[1],4)))
        hdu.append(('eRv_S_min',np.round(np.diff(Rv_S)[0],4)))
        hdu.append(('eRv_S_max',np.round(np.diff(Rv_S)[1],4)))


        table_opt = np.array([fits.Column(name='A3_DS',format='D',array=mcmc_outDS[0]),
                              fits.Column(name='A3_S',format='D',array=mcmc_outS[0]),

                              fits.Column(name='Rv_DS',format='D',array=mcmc_outDS[1]),
                              fits.Column(name='Rv_S',format='D',array=mcmc_outS[1])])
        
    elif rho=='hamaus':
        rs_DS    = np.percentile(mcmc_outDS[0][1500:], [16, 50, 84])
        delta_DS = np.percentile(mcmc_outDS[1][1500:], [16, 50, 84])
        Rv_DS    = np.percentile(mcmc_outDS[2][1500:], [16, 50, 84])
        a_DS     = np.percentile(mcmc_outDS[3][1500:], [16, 50, 84])
        b_DS     = np.percentile(mcmc_outDS[4][1500:], [16, 50, 84])

        rs_S    = np.percentile(mcmc_outS[0][1500:], [16, 50, 84])
        delta_S = np.percentile(mcmc_outS[1][1500:], [16, 50, 84])
        Rv_S    = np.percentile(mcmc_outS[2][1500:], [16, 50, 84])
        a_S     = np.percentile(mcmc_outS[3][1500:], [16, 50, 84])
        b_S     = np.percentile(mcmc_outS[4][1500:], [16, 50, 84])

        hdu = fits.Header()
        hdu.append(('rs_DS',np.round(rs_DS[1],4)))
        hdu.append(('ers_DS_min',np.round(np.diff(rs_DS)[0],4)))
        hdu.append(('ers_DS_max',np.round(np.diff(rs_DS)[1],4)))
        hdu.append(('delta_DS',np.round(delta_DS[1],4)))
        hdu.append(('edelta_DS_min',np.round(np.diff(delta_DS)[0],4)))
        hdu.append(('edelta_DS_max',np.round(np.diff(delta_DS)[1],4)))
        hdu.append(('Rv_DS',np.round(Rv_DS[1],4)))
        hdu.append(('eRv_DS_min',np.round(np.diff(Rv_DS)[0],4)))
        hdu.append(('eRv_DS_max',np.round(np.diff(Rv_DS)[1],4)))
        hdu.append(('a_DS',np.round(a_DS[1],4)))
        hdu.append(('ea_DS_min',np.round(np.diff(a_DS)[0],4)))
        hdu.append(('ea_DS_max',np.round(np.diff(a_DS)[1],4)))
        hdu.append(('b_DS',np.round(b_DS[1],4)))
        hdu.append(('eb_DS_min',np.round(np.diff(b_DS)[0],4)))
        hdu.append(('eb_DS_max',np.round(np.diff(b_DS)[1],4)))
        
        hdu.append(('rs_S',np.round(rs_S[1],4)))
        hdu.append(('ers_S_min',np.round(np.diff(rs_S)[0],4)))
        hdu.append(('ers_S_max',np.round(np.diff(rs_S)[1],4)))
        hdu.append(('delta_S',np.round(delta_S[1],4)))
        hdu.append(('edelta_S_min',np.round(np.diff(delta_S)[0],4)))
        hdu.append(('edelta_S_max',np.round(np.diff(delta_S)[1],4)))        
        hdu.append(('Rv_S',np.round(Rv_S[1],4)))
        hdu.append(('eRv_S_min',np.round(np.diff(Rv_S)[0],4)))
        hdu.append(('eRv_S_max',np.round(np.diff(Rv_S)[1],4)))
        hdu.append(('a_S',np.round(a_S[1],4)))
        hdu.append(('ea_S_min',np.round(np.diff(a_S)[0],4)))
        hdu.append(('ea_S_max',np.round(np.diff(a_S)[1],4)))
        hdu.append(('b_S',np.round(b_S[1],4)))
        hdu.append(('eb_S_min',np.round(np.diff(b_S)[0],4)))
        hdu.append(('eb_S_max',np.round(np.diff(b_S)[1],4)))

        table_opt = np.array([fits.Column(name='rs_DS',format='D',array=mcmc_outDS[0]),
                              fits.Column(name='delta_DS',format='D',array=mcmc_outDS[1]),
                              fits.Column(name='Rv_DS',format='D',array=mcmc_outDS[2]),
                              fits.Column(name='alpha_DS',format='D',array=mcmc_outDS[3]),
                              fits.Column(name='beta_DS',format='D',array=mcmc_outDS[4]),

                              fits.Column(name='rs_S',format='D',array=mcmc_outS[0]),
                              fits.Column(name='delta_S',format='D',array=mcmc_outS[1]),
                              fits.Column(name='Rv_S',format='D',array=mcmc_outS[2]),
                              fits.Column(name='alpha_S',format='D',array=mcmc_outS[3]),
                              fits.Column(name='beta_S',format='D',array=mcmc_outS[4])])
    else:
        raise ValueError(f'{rho} No implementado')


hdu = fits.Header()
hdu.append(f'using {rho}')

tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_opt))
        
primary_hdu = fits.PrimaryHDU(header=hdu)
        
hdul = fits.HDUList([primary_hdu, tbhdu_pro])

outfile = f'{outfolder}mcmc_{name}_{rho}_{pos_name}_{outname}.fits'

print(f'SAVING FIT IN {outfile}')

hdul.writeto(outfile, overwrite=True)

