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
from fit_void_leastsq import parallel_DS, parallel_S


def log_likelihoodDS(data, R, DS, iCds): 
    ds = parallel_DS(R,*data)
    return -np.dot((ds-DS),np.dot(iCds,(ds-DS)))/2.0

def log_probabilityDS(data, R, DS, eDS):
    Rv,A0,A3 = data
    
    if (.5 < Rv < 1.5) and (-5 < A0 < 5) and (-5 < A3 < 5): #and 0.5 < pcc < 1. and 0.1 < tau < 0.5:
        return log_likelihoodDS(data, R, DS, eDS)    
    return -np.inf

def log_likelihoodS(data, R, S, iCs): 
    s = parallel_S(R,*data)
    return -np.dot((s-S),np.dot(iCs,(s-S)))/2.0

def log_probabilityS(data, R, S, eS):
    Rv,A0,A3 = data
    
    if (.5 < Rv < 1.5) and (-5 < A0 < 5) and (-5 < A3 < 5): #and 0.5 < pcc < 1. and 0.1 < tau < 0.5:
        return log_likelihoodS(data, R, S, eS)    
    return -np.inf

# initializing

#distribucion uniforme
pos = np.array([np.random.uniform(.5,1.5,15),
                np.random.uniform(-5,5,15),
                np.random.uniform(-5,5,15)]).T
 
#distribucion gausiana
#pos = np.array([np.random.normal(1.,0.5,15),
#                np.random.normal(0.,0.5,15),
#                np.random.normal(0.,0.5,15)]).T

nwalkers, ndim = pos.shape

# running emcee
def run_emcee(RIN,ROUT):
    maskr   = (p.Rp > (RIN))*(p.Rp < (ROUT))
    mr = np.meshgrid(maskr,maskr)[1]*np.meshgrid(maskr,maskr)[0]

    CovDS  = cov.cov_DSt.reshape(len(p.Rp),len(p.Rp))[mr]
    CovDS  = CovDS.reshape(sum(maskr),sum(maskr))
    iCds   =  np.linalg.inv(CovDS)

    CovS  = cov.cov_S.reshape(len(p.Rp),len(p.Rp))[mr]
    CovS  = CovS.reshape(sum(maskr),sum(maskr))
    iCs   =  np.linalg.inv(CovS)

    p  = p[maskr]

    t1 = time.time()

    #Delta Sigma
    samplerDS = emcee.EnsembleSampler(nwalkers, ndim, log_probabilityDS, args=(p.Rp,p.DSigmaT,iCds))
    samplerDS.run_mcmc(pos, nit, progress=True)
    #Sigma
    samplerS = emcee.EnsembleSampler(nwalkers, ndim, log_probabilityS, args=(p.Rp,p.Sigma,iCs))
    samplerS.run_mcmc(pos, nit, progress=True)

    # saving mcmc out
    #Delta Sigma
    mcmc_outDS = samplerDS.get_chain(flat=True).T

    RvDS = np.percentile(mcmc_outDS[0][1500:], [16, 50, 84])
    A0DS = np.percentile(mcmc_outDS[1][1500:], [16, 50, 84])
    A3DS = np.percentile(mcmc_outDS[2][1500:], [16, 50, 84])
    
    tableDS = [fits.Column(name='RvDS', format='D', array=mcmc_outDS[0]),
               fits.Column(name='A0DS', format='D', array=mcmc_outDS[1]),
               fits.Column(name='A3DS', format='D', array=mcmc_outDS[2])]

    tbhduDS = fits.BinTableHDU.from_columns(fits.ColDefs(tableDS))

    #Sigma
    mcmc_outS = samplerS.get_chain(flat=True).T

    RvS = np.percentile(mcmc_outS[0][1500:], [16, 50, 84])
    A0S = np.percentile(mcmc_outS[1][1500:], [16, 50, 84])
    A3S = np.percentile(mcmc_outS[2][1500:], [16, 50, 84])

    tableS = [fits.Column(name='RvS', format='D', array=mcmc_outS[0]),
              fits.Column(name='A0S', format='D', array=mcmc_outS[1]),
              fits.Column(name='A3S', format='D', array=mcmc_outS[2])]
    
    tbhduS = fits.BinTableHDU.from_columns(fits.ColDefs(tableS))

    h = fits.Header()
    h.append('DS')
    h.append(('Rv',np.round(RvDS[1],4)))
    h.append(('eRv_min',np.round(np.diff(RvDS)[0],4)))
    h.append(('eRv_max',np.round(np.diff(RvDS)[1],4)))
    h.append(('A0',np.round(A0DS[1],4)))
    h.append(('eA0_min',np.round(np.diff(A0DS)[0],4)))
    h.append(('eA0_max',np.round(np.diff(A0DS)[1],4)))
    h.append(('A3',np.round(A3DS[1],4)))
    h.append(('eA3_min',np.round(np.diff(A3DS)[0],4)))
    h.append(('eA3_max',np.round(np.diff(A3DS)[1],4)))

    h.append('S')
    h.append(('Rv',np.round(RvS[1],4)))
    h.append(('eRv_min',np.round(np.diff(RvS)[0],4)))
    h.append(('eRv_max',np.round(np.diff(RvS)[1],4)))
    h.append(('A0',np.round(A0S[1],4)))
    h.append(('eA0_min',np.round(np.diff(A0S)[0],4)))
    h.append(('eA0_max',np.round(np.diff(A0S)[1],4)))
    h.append(('A3',np.round(A3S[1],4)))
    h.append(('eA3_min',np.round(np.diff(A3S)[0],4)))
    h.append(('eA3_max',np.round(np.diff(A3S)[1],4)))
    
    primary_hdu = fits.PrimaryHDU(header=h)

    hdul = fits.HDUList([primary_hdu, tbhduDS, tbhduS])
    hdul.writeto(folder+outfile,overwrite=True)

    print('SAVED FILE')

if __name__ == '__main__':
    profile = fits.open('../profiles/voids/Rv_15-18/Rv1518.fits')
    p = profile[1].data
    cov = profile[2].data

    Rv_min, Rv_max = profile[0].header['RV_MIN'], profile[0].header['RV_MAX']

    ncores = 128
    RIN,ROUT = 0.005, 3.
    nit = 100

    try:
        os.mkdir(f'../profiles/voids/Rv_{int(Rv_min)}-{int(Rv_max)}/fitting')
    except FileExistsError:
        pass

    folder = f'../profiles/voids/fitting'
    outfile = f'fitMCMC_Rv{int(Rv_min)}{int(Rv_max)}.fits'

    run_emcee(RIN,ROUT)