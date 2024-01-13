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


def paste(sample, name,nslices=10):
    '''
    COMO DEFLATENAR LOS ARRAYS:
    Sigma, DSigma_T/X y Ninbin son matrices 101xN, donde N es el numero de puntos del perfil entero. En este caso N=60
    => se guardan como arreglos planos 101*N, para utilizarlos se debe 'deflatenar' usando np.reshape(101,N)

    Las matrices de covarianza tienen dimensiones NxN (son cuadradas), y se guardan como arrgelos planos. 
    Para usarlas usar np.reshape(N,N) 
    '''

    # n = ndts//nslices

    directory = f'../profiles/voids/{sample}/'

    prof = np.array([name+f'rbin_{j}' for j in np.arange(nslices)])

    headers  = np.array([fits.open(directory+f'{p}.fits')[0] for p in prof])
    perfiles = np.array([fits.open(directory+f'{p}.fits')[1] for p in prof])

    n = np.array([headers[i]['ndots']//nslices for i in range(nslices)])

    R        = np.array([perfiles[j].data.Rp.reshape(101,n[j])[0] for j in np.arange(nslices)]).flatten()
    Sigma    = np.concatenate(np.array([perfiles[j].data.Sigma.reshape(101,n[j]) for j in np.arange(nslices)]),axis=1)
    DSigma_T = np.concatenate(np.array([perfiles[j].data.DSigma_T.reshape(101,n[j]) for j in np.arange(nslices)]),axis=1)
    DSigma_X = np.concatenate(np.array([perfiles[j].data.DSigma_X.reshape(101,n[j]) for j in np.arange(nslices)]),axis=1)
    Ninbin   = np.concatenate(np.array([perfiles[j].data.Ninbin.reshape(101,n[j]) for j in np.arange(nslices)]),axis=1)

    covS   = cov_matrix(Sigma[1:,:]).flatten()
    covDSt = cov_matrix(DSigma_T[1:,:]).flatten()
    covDSx = cov_matrix(DSigma_X[1:,:]).flatten()

    Nvoids    = headers[0].header['N_voids']
    Rv_min    = headers[0].header['Rv_min']
    Rv_max    = headers[0].header['Rv_max']
    rho2_min  = headers[0].header['rho2_min']
    rho2_max  = headers[0].header['rho2_max']
    z_min     = headers[0].header['z_min']
    z_max     = headers[0].header['z_max']
    ndots     = np.sum([headers[j].header['ndots'] for j in np.arange(nslices)])

    try:
        Rv_mean   = headers[0].header['Rv_mean']
        rho2_mean = headers[0].header['rho2_mean']
        z_mean    = headers[0].header['z_mean']
    except:
        print('Calculando valores medios')
        L = np.loadtxt(f'/mnt/simulations/MICE/voids_MICE.dat').T
        Rv, z, rho_1, rho_2, flag = L[1], L[4], L[8], L[9], L[11]
        mvoids = ((Rv >= Rv_min)&(Rv < Rv_max))&((z >= z_min)&(z < z_max))&((rho_1 >= -1.)&(rho_1 < 1.))&(
                  (rho_2 >= rho2_min)&(rho_2 < rho2_max))&(flag >= 2)
        L = L[:,mvoids]
        z_mean    = np.mean(L[4])
        Rv_mean   = np.mean(L[1])
        rho2_mean = np.mean(L[9])


    hdu = fits.Header()
    hdu.append(('N_VOIDS',int(Nvoids)))
    hdu.append(('Rv_min',np.round(Rv_min,2)))
    hdu.append(('Rv_max',np.round(Rv_max,2)))
    hdu.append(('Rv_mean',np.round(Rv_mean,4)))
    hdu.append(('rho2_min',np.round(rho2_min,2)))
    hdu.append(('rho2_max',np.round(rho2_max,2)))
    hdu.append(('rho2_mean',np.round(rho2_mean,4)))
    hdu.append(('z_min',np.round(z_min,2)))
    hdu.append(('z_max',np.round(z_max,2)))
    hdu.append(('z_mean',np.round(z_mean,4)))
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
    print(f'Guardado en {directory}{name}_{int(Rv_min)}-{int(Rv_max)}.fits')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', action='store', dest='sample',default='Rv_6-9')
    parser.add_argument('-name', action='store', dest='name',default='smallz')
    parser.add_argument('-nslices', action='store', dest='nslices',default=10)

    args = parser.parse_args()
    sample  = args.sample
    name    = args.name
    nslices = int(args.nslices)

    print('Pegando...')
    paste(sample,name,nslices)  
    print('Terminado!')
    