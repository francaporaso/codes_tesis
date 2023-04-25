import numpy as np
from astropy.io import fits
import os
import argparse

def cov_matrix(array):
        
        K = len(array)
        Kmean = np.average(array,axis=0)
        bins = array.shape[1]
        
        COV = np.zeros((bins,bins))
        
        for k in range(K):
            dif = (array[k]- Kmean)
            COV += np.outer(dif,dif)        
        
        COV *= (K-1)/K
        return COV


def paste(sample, name, n=10):
    '''
    COMO DEFLATENAR LOS ARRAYS:
    Sigma, DSigma_T/X y Ninbin son matrices 101xN, donde N es el numero de puntos del perfil entero. En este caso N=60
    => se guardan como arreglos planos 101*N, para utilizarlos se debe 'deflatenar' usando np.reshape(101,N)

    Las matrices de covarianza tienen dimensiones NxN (son cuadradas), y se guardan como arrgelos planos. 
    Para usarlas usar np.reshape(N,N) 
    '''

    directory = f'../profiles/voids/{sample}/'

    prof = np.array([name+f'rbin_{j}' for j in np.arange(n)])

    headers  = np.array([fits.open(directory+f'{p}.fits')[0] for p in prof])
    perfiles = np.array([fits.open(directory+f'{p}.fits')[1] for p in prof])

    R        = np.array([perfiles[j].data.Rp.reshape(101,6)[0] for j in np.arange(n)]).flatten()
    Sigma    = np.concatenate(np.array([perfiles[j].data.Sigma.reshape(101,6) for j in np.arange(n)]),axis=1)
    DSigma_T = np.concatenate(np.array([perfiles[j].data.DSigma_T.reshape(101,6) for j in np.arange(n)]),axis=1)
    DSigma_X = np.concatenate(np.array([perfiles[j].data.DSigma_X.reshape(101,6) for j in np.arange(n)]),axis=1)
    Ninbin   = np.concatenate(np.array([perfiles[j].data.Ninbin.reshape(101,6) for j in np.arange(n)]),axis=1)

    covS   = cov_matrix(Sigma[1:,:]).flatten()
    covDSt = cov_matrix(DSigma_T[1:,:]).flatten()
    covDSx = cov_matrix(DSigma_X[1:,:]).flatten()

    Nvoids   = headers[0].header['N_voids']
    Rv_min   = headers[0].header['Rv_min']
    Rv_max   = headers[0].header['Rv_max']
    rho2_min = headers[0].header['rho2_min']
    rho2_max = headers[0].header['rho2_max']
    z_min    = headers[0].header['z_min']
    z_max    = headers[0].header['z_max']
    ndots    = np.sum([headers[j].header['ndots'] for j in np.arange(n)])


    hdu = fits.Header()
    hdu.append(('N_VOIDS',int(Nvoids)))
    hdu.append(('Rv_min',np.round(Rv_min,2)))
    hdu.append(('Rv_max',np.round(Rv_max,2)))
    hdu.append(('rho2_min',np.round(rho2_min,2)))
    hdu.append(('rho2_max',np.round(rho2_max,2)))
    hdu.append(('z_min',np.round(z_min,2)))
    hdu.append(('z_max',np.round(z_max,2)))
    hdu.append(('ndots',int(ndots)))

    table_r = [fits.Column(name='Rp', format='E', array=R)]

    table_p = [fits.Column(name='Sigma',    format='E', array=Sigma.flatten()),
               fits.Column(name='DSigma_T', format='E', array=DSigma_T.flatten()),
               fits.Column(name='DSigma_X', format='E', array=DSigma_X.flatten()),
               fits.Column(name='Ninbin', format='E', array=Ninbin.flatten())]
    
    table_c = [fits.Column(name='covS', format='E', array = covS),
               fits.Column(name='covDSt', format='E', array = covDSt),
               fits.Column(name='covDSx', format='E', array = covDSx)]
    
    tbhdu_r = fits.BinTableHDU.from_columns(fits.ColDefs(table_r))
    tbhdu_p = fits.BinTableHDU.from_columns(fits.ColDefs(table_p))
    tbhdu_c = fits.BinTableHDU.from_columns(fits.ColDefs(table_c))
    primary_hdu = fits.PrimaryHDU(header=hdu)
    hdul = fits.HDUList([primary_hdu, tbhdu_r, tbhdu_p, tbhdu_c])

    hdul.writeto(f'{directory}{name}_{int(Rv_min)}-{int(Rv_max)}.fits',overwrite=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', action='store', dest='sample',default='Rv_6-9')
    parser.add_argument('-name', action='store', dest='name',default='smallz')
    parser.add_argument('-n', action='store', dest='n',default=10)
    args = parser.parse_args()

    sample = args.sample
    name   = args.name
    n      = int(args.n)

    paste(sample,name,n)  


    