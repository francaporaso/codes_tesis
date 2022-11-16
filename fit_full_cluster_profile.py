import sys
import time
import numpy as np
from astropy.io import fits
from multiprocessing import Pool
from multiprocessing import Process
import argparse
from astropy.constants import G,c,M_sun,pc
import emcee
from models_profiles import *
# import corner
import os
from colossus.cosmology import cosmology  
from colossus.halo import concentration
from colossus.lss import bias
from colossus.halo import profile_nfw
from colossus.halo import profile_outer
from colossus.cosmology import cosmology  
params = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MyCosmo', params)
cosmo = cosmology.setCosmology('MyCosmo')

cmodel = 'diemer19'


'''
# folder = '/home/eli/Documentos/Astronomia/proyectos/PARES-PAU/profiles/'
folder = '/home/elizabeth/PARES-PAU/profiles/'
file_name = 'profile_w1w2w3.fits'
ncores = 32
nit = 250
RIN = 0.
ROUT =5000.
# '''

parser = argparse.ArgumentParser()
parser.add_argument('-folder', action='store', dest='folder', default='../profiles/')
parser.add_argument('-file', action='store', dest='file_name', default='profile.fits')
parser.add_argument('-ncores', action='store', dest='ncores', default=2)
parser.add_argument('-RIN', action='store', dest='RIN', default=0)
parser.add_argument('-ROUT', action='store', dest='ROUT', default=5000)
parser.add_argument('-nit', action='store', dest='nit', default=400)
args = parser.parse_args()


folder    = args.folder
file_name = args.file_name

	
nit       = int(args.nit)
ncores    = args.ncores
ncores    = int(ncores)
RIN       = float(args.RIN)
ROUT      = float(args.ROUT)


outfile     = 'fitresults_boost_'+str(int(RIN))+'_'+str(int(ROUT))+'_'+file_name



print('fitting profiles')
print(folder)
print(file_name)
print('ncores = ',ncores)
print('RIN ',RIN)
print('ROUT ',ROUT)
print('nit', nit)
print('outfile',outfile)

profile = fits.open(folder+file_name)
h       = profile[1].header
p       = profile[1].data
zmean   = h['Z_MEAN'] 
lmean   = h['L_MEAN']

Rl = (lmean/100.)**(0.2)
s_off = 0.2 * Rl

def log_likelihood(data_model, R, DS, eDS):
    
    logM,pcc = data_model
    
    c200 = concentration.concentration(10**logM, '200c', zmean, model = cmodel)
    
    ds   = DELTA_SIGMA_full_parallel(R,zmean,10**logM,c200,s_off = s_off, pcc = pcc, P_Roff = Gamma, cosmo_params=params,ncores=ncores)
    
    sigma2 = eDS**2
    return -0.5 * np.sum((DS - ds)**2 / sigma2 + np.log(2.*np.pi*sigma2))


def log_probability(data_model, R, DS, eDS):
    
    logM,pcc = data_model
    
    if 13. < logM < 15. and 0.5 < pcc < 1.:
        return log_likelihood(data_model, R, DS, eDS)
        
    return -np.inf

# initializing

pos = np.array([np.random.uniform(13.5,14.5,15),
                np.random.uniform(0.7,0.9,15)]).T

nwalkers, ndim = pos.shape

#-------------------
# running emcee

maskr   = (p.Rp > (RIN/1000.))*(p.Rp < (ROUT/1000.))
p  = p[maskr]


t1 = time.time()


# pool = Pool(processes=(ncores))    
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                args=(p.Rp,p.DSigma_T,p.error_DSigma_T))
                                # pool = pool)
				
sampler.run_mcmc(pos, nit, progress=True)
# pool.terminate()
    
    
print('TOTAL TIME FIT')    
print((time.time()-t1)/60.)

#-------------------
# saving mcmc out

mcmc_out = sampler.get_chain(flat=True).T

logM     = np.percentile(mcmc_out[0][1500:], [16, 50, 84])
pcc      = np.percentile(mcmc_out[1][1500:], [16, 50, 84])
c200 = concentration.concentration(10**logM[1], '200c', zmean, model = cmodel)

table = [fits.Column(name='logM', format='E', array=mcmc_out[0]),
         fits.Column(name='pcc', format='E', array=mcmc_out[1])]

tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(table))


h = fits.Header()
h.append(('c200',np.round(c200,4)))
h.append(('lM200',np.round(logM[1],4)))
h.append(('elM200_min',np.round(np.diff(logM)[0],4)))
h.append(('elM200_max',np.round(np.diff(logM)[1],4)))
h.append(('pcc',np.round(pcc[1],4)))
h.append(('epcc_min',np.round(np.diff(pcc)[0],4)))
h.append(('epecc_max',np.round(np.diff(pcc)[1],4)))
primary_hdu = fits.PrimaryHDU(header=h)

hdul = fits.HDUList([primary_hdu, tbhdu])
hdul.writeto(folder+outfile,overwrite=True)

print('SAVED FILE')

