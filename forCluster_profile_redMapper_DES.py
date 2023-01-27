'''
calculates density contrasts with weak lensing stacking. returns the density profile for DES data (profiles are 
calculated differently from others surveys)
'''
import sys
sys.path.append('../lens_codes_v3.7')
import time
import numpy as np
from astropy.io import fits
from astropy.cosmology import LambdaCDM
from maria_func import *
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from multiprocessing import Pool
from multiprocessing import Process
import argparse
from astropy.constants import G,c,M_sun,pc
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample

#parameters
cvel = c.value;   # Speed of light (m.s-1)
G    = G.value;   # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)
Dg   = 0.02       #delta gamma for calculating Rsel

'''
sample='pruDES2'
z_min = 0.1
z_max = 0.3
lmin = 30.
lmax = 32.
pcc_min = 0.
RIN = 300.
ROUT =1000.
ndots= 20
ncores=128
hcosmo=1.
h = hcosmo
main(sample,z_min,z_max,lmin,lmax,pcc_min,RIN,ROUT,nbins,ncores,hcosmo)
'''

#catalogo DES Y1
# mean_z, z_mc de 0.3 a 1.4 y z_sigma68 < 1.2
S = fits.open('../cats/DES/DESy1_shape_mof_mcal.fits')[1].data


def SigmaCrit(zl, zs, h=1.):
    '''Calcula el Sigma_critico dados los redshifts. 
    Debe ser usada con astropy.cosmology y con astropy.constants
    
    zl:   (float) redshift de la lente (lens)
    zs:   (float) redshift de la fuente (source)
    h :   (float) H0 = 100.*h
    '''

    cosmo = LambdaCDM(H0=100*h, Om0=0.3, Ode0=0.7)

    dl  = cosmo.angular_diameter_distance(zl).value
    Dl = dl*1.e6*pc #en Mpc
    ds  = cosmo.angular_diameter_distance(zs).value              #dist ang diam de la fuente
    dls = cosmo.angular_diameter_distance_z1z2(zl, zs).value      #dist ang diam entre fuente y lente
                
    BETA_array = dls / ds
        
    sigma_crit = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)
    return sigma_crit


def partial_profile(RA0,DEC0,Z,
                    RIN,ROUT,ndots,h):

        cosmo = LambdaCDM(H0=100*h, Om0=0.3, Ode0=0.7)
        ndots = int(ndots)
        
        dl  = cosmo.angular_diameter_distance(Z).value        #dist angular diametral de la lente-> depende de Z
        KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0   #kpc que subtiende un dist ang diam
        
        delta = ROUT/(3600*KPCSCALE)

        mask_region = (S.ra < (RA0+delta))&(S.ra > (RA0-delta))&(S.dec > (DEC0-delta))&(S.dec < (DEC0+delta))
        
        mask = mask_region&(S.mcal_mean_z > (Z + 0.1))&(S.mof_z_mc > (Z + 0.1))
        
        catdata = S[mask]

        mS1p = (catdata.flags_select_1p == 0)
        mS1m = (catdata.flags_select_1m == 0)
        mS2p = (catdata.flags_select_2p == 0)
        mS2m = (catdata.flags_select_2m == 0)

        del mask
        del mask_region
        
        #Metacalibration (ec 11 paper maria)
        sigma_c_mcal = SigmaCrit(Z, catdata.mcal_mean_z) 

        #MOF_BPZ (ec 10 paper maria)
        sigma_c_mof = SigmaCrit(Z, catdata.mof_z_mc)

        rads, theta, test1,test2 = eq2p2(np.deg2rad(catdata.ra),
                                         np.deg2rad(catdata.dec),
                                         np.deg2rad(RA0),
                                         np.deg2rad(DEC0))    #sale de maria_func
               
        #Correct polar angle for e1, e2
        theta = theta+np.pi/2.
        
        e1     = catdata.e1
        e2     = catdata.e2
    
        #R_gamma y R_gamma_T ec (4) de McClintock 2017
        Rg_11 = catdata.R11
        Rg_12 = catdata.R12
        Rg_21 = catdata.R21
        Rg_22 = catdata.R22

        Rg_T = Rg_11 * (np.cos(2*theta))**2 + Rg_22 * (np.sin(2*theta))**2 +(Rg_12+Rg_21)*np.sin(2*theta)*np.cos(2*theta)
        
        #get tangential ellipticities 
        et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))
        #get cross ellipticities
        ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))

        r = np.rad2deg(rads)*3600*KPCSCALE

        peso = 1./(sigma_c_mcal) #ec 15 McClintock 2019
        
        Ntot = len(catdata)   
        
        bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1)
        dig = np.digitize(r,bines)
        
        DSIGMAwsum_T = np.empty(ndots)
        DSIGMAwsum_X = np.empty(ndots)
        WEIGHT_RTsum = np.empty(ndots)
        WEIGHTwsum   = np.empty(ndots)
        E1_P         = np.empty(ndots) #elipticidad promedio de ec 5 McClintock para Rsel
        E1_M         = np.empty(ndots)
        E2_P         = np.empty(ndots)
        E2_M         = np.empty(ndots)

        NGAL         = np.empty(ndots)
        NS1P         = np.empty(ndots)
        NS1M         = np.empty(ndots)
        NS2P         = np.empty(ndots)
        NS2M         = np.empty(ndots)

        for nbin in range(ndots):
                mbin = dig == nbin+1              
                
                DSIGMAwsum_T[nbin] = (et[mbin]*peso[mbin]).sum()     #numerador ec 12 McClintock
                DSIGMAwsum_X[nbin] = (ex[mbin]*peso[mbin]).sum() 
                WEIGHT_RTsum[nbin] = ((1/sigma_c_mof[mbin])*peso[mbin]*Rg_T[mbin]).sum()  #1mer termino denominador ec 12 McClintock
                WEIGHTwsum[nbin]   = ((1/sigma_c_mof[mbin])*peso[mbin]).sum()        #parentesis 2do termnino denominador 
                E1_P[nbin]         = e1[mbin & mS1p].sum()
                E1_M[nbin]         = e1[mbin & mS1m].sum()
                E2_P[nbin]         = e2[mbin & mS2p].sum()
                E2_M[nbin]         = e2[mbin & mS2m].sum()
                NGAL[nbin]         = mbin.sum()
                
                NS1P[nbin]         = (mbin & mS1p).sum() #cantidad de galaxias en el bin para poder hacer el promedio
                NS1M[nbin]         = (mbin & mS1p).sum()
                NS2P[nbin]         = (mbin & mS1p).sum()
                NS2M[nbin]         = (mbin & mS1p).sum()
                
        
        output = {'DSIGMAwsum_T':DSIGMAwsum_T, 'DSIGMAwsum_X':DSIGMAwsum_X, 
                  'WEIGHT_RTsum':WEIGHT_RTsum, 'WEIGHTwsum':WEIGHTwsum, 
                  'E1_P':E1_P, 'E1_M':E1_M, 'E2_P':E2_P, 'E2_M':E2_M, 
                  'NS1P':NS1P, 'NS1M':NS1M, 'NS2P':NS2P, 'NS2M':NS2M,
                  'Ntot':Ntot, 'NGAL':NGAL}
        
        return output

def partial_profile_unpack(minput):
        return partial_profile(*minput)


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


def main(sample='pru',z_min = 0.1, z_max = 0.4,
                lmin = 20., lmax = 150., pcc_min = 0.,
                RIN = 300., ROUT =10000.,
                ndots= 20,ncores=10,hcosmo=1.):

        '''
        
        INPUT
        ---------------------------------------------------------
        sample         (str) sample name
        z_min          (float) lower limit for z - >=
        z_max          (float) higher limit for z - <
        RIN            (float) Inner bin radius of profile
        ROUT           (float) Outer bin radius of profile
        ndots          (int) Number of bins of the profile
        ncores         (int) to run in parallel, number of cores
        hcosmo         (float) H0 = 100.*h
        '''

        cosmo = LambdaCDM(H0=100*hcosmo, Om0=0.3, Ode0=0.7)
        tini = time.time()
        
        print('Selecting clusters with:')
        print(f'{z_min} <= z < {z_max}')
        print(f'{lmin} <= lambda < {lmax}')
        print(f'pcc > {pcc_min}')
        print('Background galaxies with:')
        print(f'Profile has {ndots} bins')
        print(f'from {RIN} kpc to {ROUT} kpc')
        print(f'h = {hcosmo}')
              
        # Defining radial bins
        bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1)
        R = (bines[:-1] + np.diff(bines)*0.5)*1.e-3
        
        #reading redmapper
        cat = fits.open('../cats/DES/redmapper_y1a1_public_v6.4_catalog.fits')[1].data
        pcc = cat.P_CEN[:,0]
        
        mws82 = (cat.DEC < 2.)&(cat.DEC > -2.)&(cat.RA < 360.)&(cat.RA > 315.)
        mwspt = (cat.DEC < -35.)&(cat.DEC > -61.)&((cat.RA > 0.)&(cat.RA < 100.)+(cat.RA > 301.)&(cat.RA < 360.))
        
        RA  = np.concatenate((cat.RA[mws82],cat.RA[mwspt]))
        DEC = np.concatenate((cat.DEC[mws82],cat.DEC[mwspt]))
        z   = np.concatenate((cat.Z_LAMBDA[mws82],cat.Z_LAMBDA[mwspt]))
        LAMBDA = np.concatenate((cat.LAMBDA[mws82],cat.LAMBDA[mwspt]))
        pcc  = np.concatenate((pcc[mws82],pcc[mwspt]))
         
        mz  = (z >= z_min)&(z < z_max)
        ml  = (LAMBDA >= lmin)&(LAMBDA < lmax)
        mpcc = (pcc > pcc_min)

        mlenses = mz&ml&mpcc
        
        del mz
        del ml
        del mpcc

        L = np.array([RA[mlenses],DEC[mlenses],z[mlenses]])
        z = z[mlenses]
        Nlenses = mlenses.sum()

        if Nlenses < ncores:
                ncores = Nlenses
        
        print(f'Nlenses {Nlenses}')
        print(f'CORRIENDO EN {ncores} CORES')

        # Define K masks   
        ncen = 50
        X    = np.array([RA[mlenses],DEC[mlenses]]).T
        
        km = kmeans_sample(X, ncen, maxiter=100, tol=1.0e-5)
        labels = km.find_nearest(X)
        kmask = np.zeros((ncen+1,len(X)))
        kmask[0] = np.ones(len(X)).astype(bool)
        
        for j in np.arange(1,ncen+1):
            kmask[j] = ~(labels == j-1)
        
        # SPLIT LENSING CAT
        
        lbins = int(round(Nlenses/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < Nlenses)]
        Lsplit = np.split(L.T,slices)
        Ksplit = np.split(kmask.T,slices)
                
        # WHERE THE SUMS ARE GOING TO BE SAVED
        
        DSIGMAwsum_T = np.zeros((ncen+1,ndots))
        DSIGMAwsum_X = np.zeros((ncen+1,ndots))
        WEIGHT_RTsum = np.zeros((ncen+1,ndots))
        NGALsum      = np.zeros((ncen+1,ndots))
        WEIGHTwsum   = np.zeros((ncen+1,ndots))

        E1_P         = np.zeros((ncen+1,ndots))
        E1_M         = np.zeros((ncen+1,ndots))
        E2_P         = np.zeros((ncen+1,ndots))
        E2_M         = np.zeros((ncen+1,ndots))
        NS1P         = np.zeros((ncen+1,ndots)) #cantidad de galaxias en el bin para poder hacer el promedio de e_i
        NS1M         = np.zeros((ncen+1,ndots))
        NS2P         = np.zeros((ncen+1,ndots))
        NS2M         = np.zeros((ncen+1,ndots))

        Ntot         = np.array([])
        tslice       = np.array([])
        
        for l, Lsplit_l in enumerate(Lsplit):
                
                print(f'RUN {l+1} OF {len(Lsplit)}')
                
                t1 = time.time()
                
                num = len(Lsplit_l)
                
                rin  = RIN*np.ones(num)
                rout = ROUT*np.ones(num)
                nd   = ndots*np.ones(num)
                h_a  = hcosmo*np.ones(num)
                
                if num == 1:
                        entrada = [Lsplit_l.T[0], Lsplit_l.T[1],
                                   Lsplit_l.T[2],
                                   RIN,ROUT,ndots,hcosmo]
                        
                        salida = np.array(partial_profile_unpack(entrada))
                else:          
                        entrada = np.array([Lsplit_l.T[0], Lsplit_l.T[1],
                                           Lsplit_l.T[2],
                                           rin,rout,nd,h_a]).T
                        
                        pool = Pool(processes=(num))
                        salida = np.array(pool.map(partial_profile_unpack, entrada))
                        pool.terminate()

                #esta parte separa el dict 'salida' de partial_profile en varios arrays                
                for j, profilesums in enumerate(salida):

                        km      = np.tile(Ksplit[l][j],(3,ndots,1)).T

                        DSIGMAwsum_T += np.tile(profilesums['DSIGMAwsum_T'],(ncen+1,1))*km[:,:,0]
                        DSIGMAwsum_X += np.tile(profilesums['DSIGMAwsum_X'],(ncen+1,1))*km[:,:,0]
                        WEIGHT_RTsum += np.tile(profilesums['WEIGHT_RTsum'],(ncen+1,1))*km[:,:,0]
                        NGALsum      += np.tile(profilesums['NGAL'],(ncen+1,1))*km[:,:,0]
                        WEIGHTwsum   += np.tile(profilesums['WEIGHT_RTsum'],(ncen+1,1))*km[:,:,0]
                        
                        E1_P         += np.tile(profilesums['E1_P'],(ncen+1,1))*km[:,:,0]
                        E1_M         += np.tile(profilesums['E1_M'],(ncen+1,1))*km[:,:,0]
                        E2_P         += np.tile(profilesums['E2_P'],(ncen+1,1))*km[:,:,0]
                        E2_M         += np.tile(profilesums['E2_M'],(ncen+1,1))*km[:,:,0]
                        NS1P         += np.tile(profilesums['NS1P'],(ncen+1,1))*km[:,:,0]
                        NS1M         += np.tile(profilesums['NS1M'],(ncen+1,1))*km[:,:,0]
                        NS2P         += np.tile(profilesums['NS2P'],(ncen+1,1))*km[:,:,0]
                        NS2M         += np.tile(profilesums['NS2M'],(ncen+1,1))*km[:,:,0]
                        
                        Ntot         = np.append(Ntot,profilesums['Ntot'])
                
                t2 = time.time()
                ts = np.round((t2-t1)/60. ,2)
                tslice = np.append(tslice,ts)
                print('TIME SLICE')
                print(f'{ts} min')
                print('Estimated remaining time')
                print(np.round((np.mean(tslice)*(len(Lsplit)-(l+1))),2))
        
        # COMPUTING PROFILE

        E1_P_mean = E1_P / NS1P
        E1_M_mean = E1_M / NS1M
        E2_P_mean = E2_P / NS2P
        E2_M_mean = E2_M / NS2M

        Rsel_T    = 0.5 * ((E1_P_mean - E1_M_mean) + (E2_P_mean - E2_M_mean)) / Dg
        DSigma_T  = (DSIGMAwsum_T / (WEIGHT_RTsum + WEIGHTwsum*Rsel_T))
        DSigma_X  = (DSIGMAwsum_X / (WEIGHT_RTsum + WEIGHTwsum*Rsel_T))

        # COMPUTE COVARIANCE
        
        COV_DSt  = cov_matrix(DSigma_T[1:,:])
        COV_DSx  = cov_matrix(DSigma_X[1:,:])

        # AVERAGE LENS PARAMETERS
        
        zmean        = np.average(z,weights=Ntot)
        lmean        = np.average(LAMBDA[mlenses],weights=Ntot)
 
        # WRITING OUTPUT FITS FILE
        h = fits.Header()
        h.append(('N_LENSES',int(Nlenses)))
        h.append(('z_min',np.round(z_min,4)))
        h.append(('z_max',np.round(z_max,4)))
        h.append(('l_min',np.round(lmin,4)))
        h.append(('l_max',np.round(lmax,4)))
        h.append(('pcc_min',np.round(pcc_min,4)))
        h.append(('z_mean',np.round(zmean,4)))
        h.append(('l_mean',np.round(lmean,4)))

        table_pro = [fits.Column(name='Rp', format='D', array=R),
                     fits.Column(name='DSigma_T', format='D', array=DSigma_T),
                     fits.Column(name='DSigma_X', format='D', array=DSigma_X),
                     fits.Column(name='NGAL', format='D', array=NGALsum)]

        table_cov = [fits.Column(name='COV_DST', format='E', array=COV_DSt.flatten()),
                     fits.Column(name='COV_DSX', format='E', array=COV_DSx.flatten())]       
        
        tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_pro))
        tbhdu_cov = fits.BinTableHDU.from_columns(fits.ColDefs(table_cov))

        primary_hdu = fits.PrimaryHDU(header=h)

        hdul = fits.HDUList([primary_hdu, tbhdu_pro, tbhdu_cov])
                
        outfile = f'../profiles/des/profile_{sample}.fits'
        hdul.writeto(outfile,overwrite=True)
                
        tfin = time.time()
        print(f'File saved... {outfile}')
        print(f'TOTAL TIME {(tfin-tini)/60.}')
        


if __name__ == '__main__':
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-sample', action='store', dest='sample',default='pru')
        parser.add_argument('-z_min', action='store', dest='z_min', default=0.1)
        parser.add_argument('-z_max', action='store', dest='z_max', default=0.4)
        parser.add_argument('-lmin', action='store', dest='lmin', default=20.)
        parser.add_argument('-lmax', action='store', dest='lmax', default=150.)
        parser.add_argument('-pcc_min', action='store', dest='pcc_min', default=0.)
        parser.add_argument('-RIN', action='store', dest='RIN', default=300.)
        parser.add_argument('-ROUT', action='store', dest='ROUT', default=10000.)
        parser.add_argument('-nbins', action='store', dest='nbins', default=20)
        parser.add_argument('-ncores', action='store', dest='ncores', default=10)
        parser.add_argument('-h_cosmo', action='store', dest='h_cosmo', default=1.)
        args = parser.parse_args()
        
        sample     = args.sample
        z_min      = float(args.z_min) 
        z_max      = float(args.z_max) 
        lmin       = float(args.lmin) 
        lmax       = float(args.lmax) 
        pcc_min    = float(args.pcc_min) 
        RIN        = float(args.RIN)
        ROUT       = float(args.ROUT)
        nbins      = int(args.nbins)
        ncores     = int(args.ncores)
        h          = float(args.h_cosmo)
        
        main(sample,z_min,z_max,lmin,lmax,pcc_min,RIN,ROUT,nbins,ncores,h)
