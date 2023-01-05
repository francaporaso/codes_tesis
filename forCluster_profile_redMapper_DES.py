import sys
sys.path.append('../../lens_codes_v3.7')
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

#parameters
cvel = c.value;   # Speed of light (m.s-1)
G    = G.value;   # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

'''
uso los catalogos : redmapper_y1a1_public_v6.4_catalog.fits para clusters
                  : mcal-y1a1-combined-riz-unblind-v4-matched para formas/fuentes
                  : mcal-y1a1-combined-griz-blind-v3-matched_BPZbase.fits para redshifts de pesos (ec 15 McClintock 2017)
                  : y1a1-gold-mof-badregion_BPZ.fits para redshifts de la normalizacion (ec 14 McClintock 2017)
'''

#catalogo DES Y1
w = fits.open('../cats/DES/DES_y1_shape_mof_mcal.fits')[1].data
# mean_z de 0.3 a 1.4 y z_sigma68 <1.2
m_sources = (w.mcal_mean_z > 0.3)&(w.mcal_mean_z < 1.4)&(w.mcal_z_sigma68 < 1.2)
#mascara para las distintas aeras: Stripe82 y South Pole Telescope
mas82 = (w.dec < 2.)&(w.dec > -2.)&(w.ra < 360.)&(w.ra > 315.)
maspt = (w.dec < -35.)&(w.dec > -61.)&((w.ra > 0.)&(w.ra < 100.)+(w.ra > 301.)&(w.ra < 360.))
#mask total
m1 = m_sources & mas82
m2 = m_sources & maspt
#datos
w1_sources = w[m1]
w2_sources = w[m2]

def partial_profile(RA0,DEC0,Z,field,
                    RIN,ROUT,ndots,h,nboot=100):

        if field == 1:
            S = w1_sources
        if field == 2:
            S = w2_sources

        cosmo = LambdaCDM(H0=100*h, Om0=0.3, Ode0=0.7)
        ndots = int(ndots)
        
        dl  = cosmo.angular_diameter_distance(Z).value        #dist angular diametral de la lente-> depende de Z
        KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0   #kpc que subtiende un dist ang diam
        
        delta = ROUT/(3600*KPCSCALE)

        Dl = dl*1.e6*pc

        mask_region = (S.ra < (RA0+delta))&(S.ra > (RA0-delta))&(S.dec > (DEC0-delta))&(S.dec < (DEC0+delta))
        
        mask = mask_region*(S.mcal_mean_z > (Z + 0.1))&(S.mcal_mean_z > 0.3)
        
        mS0  = mask & (S.flags_select == 0)
        mS1p = mask & (S.flags_select_1p == 0)
        mS1m = mask & (S.flags_select_1m == 0)
        mS2p = mask & (S.flags_select_2p == 0)
        mS2m = mask & (S.flags_select_2m == 0)

        catdata = S[mS0]
        #S1p = S[mS1p]
        #S1m = S[mS1m]
        #S2p = S[mS2p]
        #S2m = S[mS2m]
        
        #Metacalibration (ec 11 paper maria)
        ds_mcal  = cosmo.angular_diameter_distance(catdata.mcal_mean_z).value              #dist ang diam de la fuente
        dls_mcal = cosmo.angular_diameter_distance_z1z2(Z, catdata.mcal_mean_z).value      #dist ang diam entre fuente y lente
                
        BETA_array_mcal = dls_mcal / ds_mcal
        
        sigma_c_mcal = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array_mcal))*(pc**2/Msun)

        
        #MOF_BPZ (ec 10 paper maria)
        ds_mof  = cosmo.angular_diameter_distance(catdata.mof_z_mc).value              #dist ang diam de la fuente
        dls_mof = cosmo.angular_diameter_distance_z1z2(Z, catdata.mof_z_mc).value      #dist ang diam entre fuente y lente
                
        BETA_array_mof = dls_mof / ds_mof
        
        sigma_c_mof = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array_mof))*(pc**2/Msun)

        rads, theta, test1,test2 = eq2p2(np.deg2rad(catdata.ra),
                                         np.deg2rad(catdata.dec),
                                         np.deg2rad(RA0),
                                         np.deg2rad(DEC0))    #sale de maria_func
               
        #Correct polar angle for e1, e2
        theta = theta+np.pi/2.
        
        e1     = catdata.e1
        e2     = catdata.e2

        Dg_j = 2 * 0.01
    
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
        del(rads)

        peso = 1./(sigma_c_mcal) #ec 15 McClintock 2019
        
        Ntot = len(catdata)   
        
        bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1)
        dig = np.digitize(r,bines)
        
        DSIGMAwsum_T = []
        #DSIGMAwsum_X = []
        WEIGHT_RTsum = []
        WEIGHTwsum    = []
        E1_P         = [] #elipticidad promedio de ec 5 McClintock para Rsel
        E1_M         = []
        E2_P         = []
        E2_M         = []
        #Mwsum        = []
        #BOOTwsum_T   = np.zeros((nboot,ndots))
        #BOOTwsum_X   = np.zeros((nboot,ndots))
        #BOOTwsum     = np.zeros((nboot,ndots))
        NGAL         = []
                
        
        for nbin in range(ndots):
                mbin = dig == nbin+1              
                
                DSIGMAwsum_T = np.append(DSIGMAwsum_T,(et[mbin]*peso[mbin]).sum())     #numerador ec 12 McClintock
                #DSIGMAwsum_X = np.append(DSIGMAwsum_X,(ex[mbin]*peso[mbin]).sum()) 
                WEIGHT_RTsum = np.append(WEIGHT_RTsum, (sigma_c_mof[mbin]*peso[mbin]*Rg_T[mbin]).sum())  #1mer termino denominador ec 12 McClintock
                WEIGHTwsum   = np.append(WEIGHTwsum,(sigma_c_mof[mbin]*peso[mbin]).sum())        #parentesis 2do termnino denominador 
                E1_P         = np.append(E1_P,S[mS1p].e1[mbin].mean())
                E1_M         = np.append(E1_M,S[mS1m].e1[mbin].mean())
                E2_P         = np.append(E1_P,S[mS2p].e2[mbin].mean())
                E2_M         = np.append(E2_M,S[mS2m].e2[mbin].mean())
                NGAL         = np.append(NGAL,mbin.sum())
                
                '''
                index = np.arange(mbin.sum())
                if mbin.sum() == 0:
                        continue
                else:
                        with NumpyRNGContext(1):
                                bootresult = bootstrap(index, nboot)
                        INDEX=bootresult.astype(int)
                        BOOTwsum_T[:,nbin] = np.sum(np.array(et[mbin]*peso[mbin])[INDEX],axis=1)
                        BOOTwsum_X[:,nbin] = np.sum(np.array(ex[mbin]*peso[mbin])[INDEX],axis=1)
                        BOOTwsum[:,nbin]   = np.sum(np.array(peso[mbin])[INDEX],axis=1)'''
        
        output = {'DSIGMAwsum_T':DSIGMAwsum_T,
                  #'DSIGMAwsum_X':DSIGMAwsum_X, 
                  'WEIGHT_RTsum':WEIGHT_RTsum, 
                  'E1_P':E1_P, 'E1_M':E1_M, 'E2_P':E2_P, 'E2_M':E2_M, 
                  'WEIGHTwsum':WEIGHTwsum, 
                  #'BOOTwsum_T':BOOTwsum_T, 'BOOTwsum_X':BOOTwsum_X, 'BOOTwsum':BOOTwsum, 
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
        print('Profile has ',ndots,'bins')
        print('from ',RIN,'kpc to ',ROUT,'kpc')
        print('h = ',hcosmo)
              
        # Defining radial bins
        bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1)
        R = (bines[:-1] + np.diff(bines)*0.5)*1.e-3
        
        #reading cats
        #redmapper
        cat = fits.open('../cats/DES/redmapper_y1a1_public_v6.4_catalog.fits')[1].data
        pcc = cat.P_CEN[:,0]
        
        mws82 = (cat.DEC < 2.)*(cat.DEC > -2.)*(cat.RA < 360.)*(cat.RA > 315.)
        mwspt = (cat.DEC < -35.)*(cat.DEC > -61.)*((cat.RA > 0.)*(cat.RA < 100.)+(cat.RA > 301.)*(cat.RA < 360.))
        
        RA  = np.concatenate((cat.RA[mws82],cat.RA[mwspt]))
        DEC = np.concatenate((cat.DEC[mws82],cat.DEC[mwspt]))
        z   = np.concatenate((cat.Z_LAMBDA[mws82],cat.Z_LAMBDA[mwspt]))
        LAMBDA = np.concatenate((cat.LAMBDA[mws82],cat.LAMBDA[mwspt]))
        field = np.concatenate((np.ones(mws82.sum())*1.,np.ones(mwspt.sum())*2.))
        pcc  = np.concatenate((pcc[mws82],pcc[mwspt]))
         
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
        #DSIGMAwsum_X = np.zeros(ndots)
        WEIGHT_RTsum = np.zeros(ndots)
        NGALsum      = np.zeros(ndots)
        WEIGHTwsum   = np.zeros(ndots)
        E1_P         = np.zeros(ndots)
        E1_M         = np.zeros(ndots)
        E2_P         = np.zeros(ndots)
        E2_M         = np.zeros(ndots)
        #BOOTwsum_T   = np.zeros((100,ndots))
        #BOOTwsum_X   = np.zeros((100,ndots))
        #BOOTwsum     = np.zeros((100,ndots))
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

                #esta parte separa el dict 'salida' de partial_profile en varios arrays                
                for profilesums in salida:
                        DSIGMAwsum_T += profilesums['DSIGMAwsum_T']
                        #DSIGMAwsum_X += profilesums['DSIGMAwsum_X']
                        WEIGHT_RTsum += profilesums['WEIGHT_RTsum']
                        NGALsum      += profilesums['NGAL']
                        WEIGHTwsum   += profilesums['WEIGHT_RTsum']
                        E1_P         += profilesums['E1_P']
                        E1_M         += profilesums['E1_M']
                        E2_P         += profilesums['E2_P']
                        E2_M         += profilesums['E2_M']
                        #BOOTwsum_T   += profilesums['BOOTwsum_T']
                        #BOOTwsum_X   += profilesums['BOOTwsum_X']
                        #BOOTwsum     += profilesums['BOOTwsum']
                        Ntot         = np.append(Ntot,profilesums['Ntot'])
                
                t2 = time.time()
                ts = (t2-t1)/60.
                tslice = np.append(tslice,ts)
                print('TIME SLICE')
                print(ts)
                print('Estimated ramaining time')
                print((np.mean(tslice)*(len(Lsplit)-(l+1))))
        
        # COMPUTING PROFILE        
                
        Dg = 0.02

        Rsel_T      = 0.5 * ((E1_P - E1_M) + (E2_P - E2_M)) / Dg
        DSigma_T  = (DSIGMAwsum_T/(WEIGHT_RTsum+WEIGHTwsum*Rsel_T))
        #DSigma_X  = (DSIGMAwsum_X/WEIGHTsum)/Mcorr
        #eDSigma_T =  np.std((BOOTwsum_T/BOOTwsum),axis=0)/Mcorr
        #eDSigma_X =  np.std((BOOTwsum_X/BOOTwsum),axis=0)/Mcorr
        
        # AVERAGE LENS PARAMETERS
        
        zmean        = np.average(z,weights=Ntot)
        
        lmean        = np.average(LAMBDA[mlenses],weights=Ntot)
 
        # WRITING OUTPUT FITS FILE
        
        
        tbhdu = fits.BinTableHDU.from_columns(
                [fits.Column(name='Rp', format='D', array=R),
                 fits.Column(name='DSigma_T', format='D', array=DSigma_T),
                 #fits.Column(name='error_DSigma_T', format='D', array=eDSigma_T),
                 #fits.Column(name='DSigma_X', format='D', array=DSigma_X),
                 #fits.Column(name='error_DSigma_X', format='D', array=eDSigma_X),
                 #fits.Column(name='NGAL_w', format='D', array=WEIGHTsum),
                 fits.Column(name='NGAL', format='D', array=NGALsum)
                ])
        
        h = tbhdu.header
        h.append(('N_LENSES',int(Nlenses)))
        h.append(('z_min',np.round(z_min,4)))
        h.append(('z_max',np.round(z_max,4)))
        h.append(('l_min',np.round(lmin,4)))
        h.append(('l_max',np.round(lmax,4)))
        h.append(('pcc_min',np.round(pcc_min,4)))
        h.append(('z_mean',np.round(zmean,4)))
        h.append(('l_mean',np.round(lmean,4)))
                
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
