import sys
sys.path.append('/mnt/projects/lensing/lens_codes_v3.7')
sys.path.append('/home/eli/lens_codes_v3.7')
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import time
import numpy as np
from astropy.io import fits
from multiprocessing import Pool
from multiprocessing import Process
import argparse
from astropy.constants import G,c,M_sun,pc
import emcee
from models_profiles import *
from fit_profiles_curvefit import *
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
parser.add_argument('-folder', action='store', dest='folder', default='/home/eli/Documentos/Astronomia/proyectos/PARES-PAU/profiles/')
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


outfile     = 'fitresults_'+str(int(RIN))+'_'+str(int(ROUT))+'_'+file_name



print('fitting profiles')
print(folder)
print(file_name)
print('ncores = ',ncores)
print('RIN ',RIN)
print('ROUT ',ROUT)
print('nit', nit)
print('outfile',outfile)

profile = fits.open(folder+file_name)
h       = profile[0].header
p       = profile[1].data
cov     = profile[2].data
zmean   = h['Z_MEAN'] 


### compute dilution
bines = np.logspace(np.log10(h['RIN']),np.log10(h['ROUT']),num=len(p)+1)
area = np.pi*np.diff(bines**2)

ngal = p.NGAL_w

d = ngal/area

fcl = ((d - np.mean(d[-2:]))*area)/ngal

bcorr = 1./(1-fcl)
p.DSigma_T = bcorr*p.DSigma_T
p.DSigma_X = bcorr*p.DSigma_X
p.error_DSigma_T = bcorr*p.error_DSigma_T
p.error_DSigma_X = bcorr*p.error_DSigma_X


def log_likelihood(logM, R, DS, iCds):
    
    c200 = concentration.concentration(10**logM, '200c', zmean, model = cmodel)
    
    ds   = Delta_Sigma_NFW_2h(R,zmean,M200 = 10**logM,c200=c200,cosmo_params=params,terms='1h+2h')    
    
    L_DS = -np.dot((ds-DS),np.dot(iCds,(ds-DS)))/2.0
    
    return L_DS


def log_probability(logM, R, DS, iCds):
    
    
    if 11. < logM < 15.:
        return log_likelihood(logM, R, DS, iCds)
        
    return -np.inf

# initializing

pos = np.array([np.random.uniform(11.5,13.5,20)]).T

nwalkers, ndim = pos.shape

#-------------------
# running emcee

maskr   = (p.Rp > (RIN/1000.))*(p.Rp < (ROUT/1000.))
mr = np.meshgrid(maskr,maskr)[1]*np.meshgrid(maskr,maskr)[0]

CovDS  = cov['COV_ST'].reshape(len(p.Rp),len(p.Rp))[mr]
CovDS  = CovDS.reshape(maskr.sum(),maskr.sum())
iCds   =  np.linalg.inv(CovDS)
p  = p[maskr]

t1 = time.time()


pool = Pool(processes=(ncores))    
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                args=(p.Rp,p.DSigma_T,iCds),
                                pool = pool)
				
sampler.run_mcmc(pos, nit, progress=True)
pool.terminate()
    
    
print('TOTAL TIME FIT')    
print((time.time()-t1)/60.)

#-------------------
# saving mcmc out

mcmc_out = sampler.get_chain(flat=True)

table = [fits.Column(name='logM', format='E', array=mcmc_out)]

tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(table))

logM    = np.percentile(mcmc_out[2500:], [16, 50, 84])
c200 = concentration.concentration(10**logM[1], '200c', zmean, model = cmodel)



h = fits.Header()
h.append(('c200',np.round(c200,4)))

h.append(('lM200',np.round(logM[1],4)))
h.append(('elM200_min',np.round(np.diff(logM)[0],4)))
h.append(('elM200_max',np.round(np.diff(logM)[1],4)))
primary_hdu = fits.PrimaryHDU(header=h)

hdul = fits.HDUList([primary_hdu, tbhdu])
hdul.writeto(folder+outfile,overwrite=True)

print('SAVED FILE')

