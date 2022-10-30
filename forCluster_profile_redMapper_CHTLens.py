import sys
import time
#calcula tiempo que demora un proceso
import numpy as np
from astropy.io import fits
#lee los catalogos
from astropy.cosmology import LambdaCDM
from maria_func import *
#calcula separaciones angulares
from astropy.stats import bootstrap
#calculo de errores a traves de bootstraping
from astropy.utils import NumpyRNGContext
from multiprocessing import Pool
from multiprocessing import Process
#para calcular en paralelo
import argparse
#? averiguar!
from astropy.constants import G,c,M_sun,pc

#parameters
cvel = c.value;   # Speed of light (m.s-1)
G    = G.value;   # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

#catalogo CFHTLens, separado en las cuatro areas del cielo

#HAY QUE CAMBIAR EL PATH, Y PENSAR COMO SEPARAR EL ARCHIVO EN LAS 4 AREAS (usar una mascara de ascension recta o declinacion)
w4 = fits.open('/mnt/projects/lensing/CFHTLens/CFHTLens_W4.fits')[1].data
w3 = fits.open('/mnt/projects/lensing/CFHTLens/CFHTLens_W3.fits')[1].data
w2 = fits.open('/mnt/projects/lensing/CFHTLens/CFHTLens_W2.fits')[1].data
w1 = fits.open('/mnt/projects/lensing/CFHTLens/CFHTLens_W1.fits')[1].data

#mascaras para los datos, tiramos los que no cumplan los requisitos
m1 = (w1.ODDS >= 0.5)*(w1.Z_B > 0.2)*(w1.Z_B < 1.2)*(w1.weight > 0)*(w1.fitclass == 0)*(w1.MASK <= 1)
m2 = (w2.ODDS >= 0.5)*(w2.Z_B > 0.2)*(w2.Z_B < 1.2)*(w2.weight > 0)*(w2.fitclass == 0)*(w2.MASK <= 1)
m3 = (w3.ODDS >= 0.5)*(w3.Z_B > 0.2)*(w3.Z_B < 1.2)*(w3.weight > 0)*(w3.fitclass == 0)*(w3.MASK <= 1)
m4 = (w4.ODDS >= 0.5)*(w4.Z_B > 0.2)*(w4.Z_B < 1.2)*(w4.weight > 0)*(w4.fitclass == 0)*(w4.MASK <= 1)

#datos enmascarados (xd)
w1_sources = w1[m1]
w2_sources = w2[m2]
w3_sources = w3[m3]
w4_sources = w4[m4]


def partial_profile(RA0,DEC0,Z,field,
                    RIN,ROUT,ndots,h,nboot=100):

        if field == 1:
            S = w1_sources
        if field == 2:
            S = w2_sources
        if field == 3:
            S = w3_sources
        if field == 4:
            S = w4_sources

        cosmo = LambdaCDM(H0=100*h, Om0=0.3, Ode0=0.7)
        ndots = int(ndots)
        
        dl  = cosmo.angular_diameter_distance(Z).value
        KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
        
        delta = ROUT/(3600*KPCSCALE)

        
        mask_region = (S.RAJ2000 < (RA0+delta))&(S.RAJ2000 > (RA0-delta))&(S.DECJ2000 > (DEC0-delta))&(S.DECJ2000 < (DEC0+delta))
               
        mask = mask_region*(S.Z_B > (Z + 0.1))*(S.ODDS >= 0.5)*(S.Z_B > 0.2)
        
        catdata = S[mask]

        ds  = cosmo.angular_diameter_distance(catdata.Z_B).value
        dls = cosmo.angular_diameter_distance_z1z2(Z, catdata.Z_B).value
                
        BETA_array = dls/ds
        
        
        Dl = dl*1.e6*pc
        sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)



        rads, theta, test1,test2 = eq2p2(np.deg2rad(catdata.RAJ2000),
                                        np.deg2rad(catdata.DECJ2000),
                                        np.deg2rad(RA0),
                                        np.deg2rad(DEC0))
               
        #Correct polar angle for e1, e2
        theta = theta+np.pi/2.
        
        e1     = catdata.e1
        e2     = catdata.e2-catdata.c2
        
        #get tangential ellipticities 
        et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
        #get cross ellipticities
        ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
        
        del(e1)
        del(e2)
        
        r=np.rad2deg(rads)*3600*KPCSCALE
        del(rads)
        
        peso = catdata.weight
        peso = peso/(sigma_c**2) 
        m    = catdata.m
        
        Ntot = len(catdata)
        # del(catdata)    
        
        bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1)
        dig = np.digitize(r,bines)
        
        DSIGMAwsum_T = []
        DSIGMAwsum_X = []
        WEIGHTsum    = []
        Mwsum        = []
        BOOTwsum_T   = np.zeros((nboot,ndots))
        BOOTwsum_X   = np.zeros((nboot,ndots))
        BOOTwsum     = np.zeros((nboot,ndots))
        NGAL         = []
        
        
        for nbin in range(ndots):
                mbin = dig == nbin+1              
                
                DSIGMAwsum_T = np.append(DSIGMAwsum_T,(et[mbin]*peso[mbin]).sum())
                DSIGMAwsum_X = np.append(DSIGMAwsum_X,(ex[mbin]*peso[mbin]).sum())
                WEIGHTsum    = np.append(WEIGHTsum,(peso[mbin]).sum())
                Mwsum        = np.append(Mwsum,(m[mbin]*peso[mbin]).sum())
                NGAL         = np.append(NGAL,mbin.sum())
                
                index = np.arange(mbin.sum())
                if mbin.sum() == 0:
                        continue
                else:
                        with NumpyRNGContext(1):
                                bootresult = bootstrap(index, nboot)
                        INDEX=bootresult.astype(int)
                        BOOTwsum_T[:,nbin] = np.sum(np.array(et[mbin]*peso[mbin])[INDEX],axis=1)
                        BOOTwsum_X[:,nbin] = np.sum(np.array(ex[mbin]*peso[mbin])[INDEX],axis=1)
                        BOOTwsum[:,nbin]   = np.sum(np.array(peso[mbin])[INDEX],axis=1)
        
        output = {'DSIGMAwsum_T':DSIGMAwsum_T,'DSIGMAwsum_X':DSIGMAwsum_X,
                   'WEIGHTsum':WEIGHTsum, 'Mwsum':Mwsum, 
                   'BOOTwsum_T':BOOTwsum_T, 'BOOTwsum_X':BOOTwsum_X, 'BOOTwsum':BOOTwsum, 
                   'Ntot':Ntot,'NGAL':NGAL}
        
        return output

def partial_profile_unpack(minput):
	return partial_profile(*minput)
        

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
        print(z_min,' <= z < ',z_max)
        print(lmin,' <= lambda < ',lmax)
        print('pcc > ',pcc_min)
        print('Background galaxies with:')
        #print('ODDS > ',odds_min)
        print('Profile has ',ndots,'bins')
        print('from ',RIN,'kpc to ',ROUT,'kpc')
        print('h = ',hcosmo)
              
        # Defining radial bins
        bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1)
        R = (bines[:-1] + np.diff(bines)*0.5)*1.e-3
        
        #reading cats
        
        cat = fits.open('/mnt/projects/lensing/redMaPPer/redmapper_dr8_public_v6.3_catalog.fits')[1].data
        pcc = cat.P_CEN[:,0]
        
        mw1 = (cat.RA < 39)*(cat.RA > 30.)*(cat.DEC < -3.5)*(cat.DEC > -11.5)
        mw3 = (cat.RA < 221)*(cat.RA > 208)*(cat.DEC < 58)*(cat.DEC > 51)
        mw2 = (cat.RA < 137)*(cat.RA > 132)*(cat.DEC < -0.9)*(cat.DEC > -5.7)
        mw4 = (cat.RA < 336)*(cat.RA > 329)*(cat.DEC < 4.7)*(cat.DEC > -1.1)
        
        RA  = np.concatenate((cat.RA[mw1],cat.RA[mw2],cat.RA[mw3],cat.RA[mw4]))
        DEC = np.concatenate((cat.DEC[mw1],cat.DEC[mw2],cat.DEC[mw3],cat.DEC[mw4]))
        z   = np.concatenate((cat.Z_LAMBDA[mw1],cat.Z_LAMBDA[mw2],cat.Z_LAMBDA[mw3],cat.Z_LAMBDA[mw4]))
        LAMBDA = np.concatenate((cat.LAMBDA[mw1],cat.LAMBDA[mw2],cat.LAMBDA[mw3],cat.LAMBDA[mw4]))
        field = np.concatenate((np.ones(mw1.sum())*1.,np.ones(mw2.sum())*2.,np.ones(mw3.sum())*3.,np.ones(mw4.sum())*4.))
        pcc  = np.concatenate((pcc[mw1],pcc[mw2],pcc[mw3],pcc[mw4]))
         
        mz  = (z >= z_min)*(z < z_max)
        ml  = (LAMBDA >= lmin)*(LAMBDA < lmax)
        mpcc = (pcc > pcc_min)

        mlenses = mz*ml*mpcc
        
        L = np.array([RA[mlenses],DEC[mlenses],z[mlenses],field[mlenses]])
        z = z[mlenses]
        Nlenses = mlenses.sum()

        if Nlenses < ncores:
                ncores = Nlenses
        
        print('Nlenses',Nlenses)
        print('CORRIENDO EN ',ncores,' CORES')

        
        # SPLIT LENSING CAT
        
        lbins = int(round(Nlenses/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < Nlenses)]
        Lsplit = np.split(L.T,slices)
                
        # WHERE THE SUMS ARE GOING TO BE SAVED
        
        DSIGMAwsum_T = np.zeros(ndots) 
        DSIGMAwsum_X = np.zeros(ndots)
        WEIGHTsum    = np.zeros(ndots)
        NGALsum      = np.zeros(ndots)
        Mwsum        = np.zeros(ndots)
        BOOTwsum_T   = np.zeros((100,ndots))
        BOOTwsum_X   = np.zeros((100,ndots))
        BOOTwsum     = np.zeros((100,ndots))
        Ntot         = []
        tslice       = np.array([])
        
        for l in range(len(Lsplit)):
                
                print('RUN ',l+1,' OF ',len(Lsplit))
                
                t1 = time.time()
                
                num = len(Lsplit[l])
                
                rin  = RIN*np.ones(num)
                rout = ROUT*np.ones(num)
                nd   = ndots*np.ones(num)
                h_a  = hcosmo*np.ones(num)
                
                if num == 1:
                        entrada = [Lsplit[l].T[0][0], Lsplit[l].T[1][0],
                                   Lsplit[l].T[2][0],Lsplit[l].T[-1][0],
                                   RIN,ROUT,ndots,hcosmo]
                        
                        salida = [partial_profile_unpack(entrada)]
                else:          
                        entrada = np.array([Lsplit[l].T[0], Lsplit[l].T[1],
                                   Lsplit[l].T[2],Lsplit[l].T[-1],
                                        rin,rout,nd,h_a]).T
                        
                        pool = Pool(processes=(num))
                        salida = np.array(pool.map(partial_profile_unpack, entrada))
                        pool.terminate()
                                
                for profilesums in salida:
                        DSIGMAwsum_T += profilesums['DSIGMAwsum_T']
                        DSIGMAwsum_X += profilesums['DSIGMAwsum_X']
                        WEIGHTsum    += profilesums['WEIGHTsum']
                        NGALsum      += profilesums['NGAL']
                        Mwsum        += profilesums['Mwsum']
                        BOOTwsum_T   += profilesums['BOOTwsum_T']
                        BOOTwsum_X   += profilesums['BOOTwsum_X']
                        BOOTwsum     += profilesums['BOOTwsum']
                        Ntot         = np.append(Ntot,profilesums['Ntot'])
                
                t2 = time.time()
                ts = (t2-t1)/60.
                tslice = np.append(tslice,ts)
                print('TIME SLICE')
                print(ts)
                print('Estimated ramaining time')
                print((np.mean(tslice)*(len(Lsplit)-(l+1))))
        
        # COMPUTING PROFILE        
                
        Mcorr     = Mwsum/WEIGHTsum
        DSigma_T  = (DSIGMAwsum_T/WEIGHTsum)/(1+Mcorr)
        DSigma_X  = (DSIGMAwsum_X/WEIGHTsum)/(1+Mcorr)
        eDSigma_T =  np.std((BOOTwsum_T/BOOTwsum),axis=0)/(1+Mcorr)
        eDSigma_X =  np.std((BOOTwsum_X/BOOTwsum),axis=0)/(1+Mcorr)
        
        # AVERAGE LENS PARAMETERS
        
        zmean        = np.average(z,weights=Ntot)
        
 
        # WRITING OUTPUT FITS FILE
        
        
        tbhdu = fits.BinTableHDU.from_columns(
                [fits.Column(name='Rp', format='D', array=R),
                fits.Column(name='DSigma_T', format='D', array=DSigma_T),
                fits.Column(name='error_DSigma_T', format='D', array=eDSigma_T),
                fits.Column(name='DSigma_X', format='D', array=DSigma_X),
                fits.Column(name='error_DSigma_X', format='D', array=eDSigma_X),
                fits.Column(name='NGAL', format='D', array=NGALsum),
                fits.Column(name='NGAL_w', format='D', array=WEIGHTsum)])
        
        h = tbhdu.header
        h.append(('N_LENSES',np.int(Nlenses)))
        h.append(('z_min',np.round(z_min,4)))
        h.append(('z_max',np.round(z_max,4)))
        h.append(('lmin',np.round(lmin,4)))
        h.append(('lmax',np.round(lmax,4)))
        h.append(('pcc_min',np.round(pcc_min,4)))
        h.append(('z_mean',np.round(zmean,4)))

                
        outfile = '../profiles/profile_'+sample+'.fits'
        tbhdu.writeto(outfile,overwrite=True)
                
        tfin = time.time()
        print('File saved... ',outfile)
        print('TOTAL TIME ',(tfin-tini)/60.)
        


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
