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

#parameters
cvel = c.value;   # Speed of light (m.s-1)
G    = G.value;   # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

'''
sample='pruDES'
z_min = 0.1
z_max = 0.3
lmin = 30.
lmax = 32.
pcc_min = 0.
RIN = 300.
ROUT =1000.
ndots= 20
ncores=10
hcosmo=1.
h = hcosmo
main(sample,z_min,z_max,lmin,lmax,pcc_min,RIN,ROUT,nbins,ncores,hcosmo)
'''

#catalogo DES Y1
w = fits.open('../cats/DES/DES_y1_shape_mof_mcal.fits')[1].data
# mean_z de 0.3 a 1.4 y z_sigma68 <1.2
m_sources = (w.flags_select == 0)&(w.mcal_mean_z > 0.3)&(w.mcal_mean_z < 1.4)&(w.mcal_z_sigma68 < 1.2)&(w.mcal_mean_z > 0.3)&(w.mof_z_mc < 1.4)&(w.mof_z_sigma68 < 1.2)
#datos
S = w[m_sources]
del w
del m_sources

def SigmaCrit(zl, zs, h=1.):
    '''Calcula el Sigma_critico dados los redshifts. 
    Debe ser usada con astropy.cosmology y con astropy.constants
    
    zl:   (float) redshift de la lente (lens)
    zs:   (float) redshift de la fuente (source)
    h :   (float) H0 = 100.*h
    '''
    #if zl > zs:
    #    raise ValueError('Redshift de la fuente es menor al de la lente')

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

        #Dl = dl*1.e6*pc

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

        Dg_j = 2 * 0.01
    
        #R_gamma y R_gamma_T ec (4) de McClintock 2017
        Rg_11 = catdata.R11
        Rg_12 = catdata.R12
        Rg_21 = catdata.R21
        Rg_22 = catdata.R22

        Rg_T = Rg_11 * (np.cos(2*theta))**2 + Rg_22 * (np.sin(2*theta))**2 +(Rg_12+Rg_21)*np.sin(2*theta)*np.cos(2*theta)
        
        del Rg_11
        del Rg_22
        del Rg_12
        del Rg_21
        
        #get tangential ellipticities 
        et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))
        #get cross ellipticities
        #ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))

        r = np.rad2deg(rads)*3600*KPCSCALE
        del rads
        del theta

        peso = 1./(sigma_c_mcal) #ec 15 McClintock 2019
        
        Ntot = len(catdata)   
        
        bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1)
        dig = np.digitize(r,bines)
        
        DSIGMAwsum_T = []
        #DSIGMAwsum_X = []
        WEIGHT_RTsum = []
        WEIGHTwsum   = []
        E1_P         = [] #elipticidad promedio de ec 5 McClintock para Rsel
        E1_M         = []
        E2_P         = []
        E2_M         = []
        #Mwsum        = []
        #BOOTwsum_T   = np.zeros((nboot,ndots))
        #BOOTwsum_X   = np.zeros((nboot,ndots))
        #BOOTwsum     = np.zeros((nboot,ndots))
        NGAL         = []
        NS1P         = []
        NS1M         = []
        NS2P         = []
        NS2M         = []

        for nbin in range(ndots):
                mbin = dig == nbin+1              
                
                DSIGMAwsum_T = np.append(DSIGMAwsum_T,(et[mbin]*peso[mbin]).sum())     #numerador ec 12 McClintock
                #DSIGMAwsum_X = np.append(DSIGMAwsum_X,(ex[mbin]*peso[mbin]).sum()) 
                WEIGHT_RTsum = np.append(WEIGHT_RTsum, (sigma_c_mof[mbin]*peso[mbin]*Rg_T[mbin]).sum())  #1mer termino denominador ec 12 McClintock
                WEIGHTwsum   = np.append(WEIGHTwsum,(sigma_c_mof[mbin]*peso[mbin]).sum())        #parentesis 2do termnino denominador 
                E1_P         = np.append(E1_P,e1[mbin & mS1p].sum())
                E1_M         = np.append(E1_M,e1[mbin & mS1m].sum())
                E2_P         = np.append(E1_P,e2[mbin & mS2p].sum())
                E2_M         = np.append(E2_M,e2[mbin & mS2m].sum())
                NGAL         = np.append(NGAL,mbin.sum())
                
                NS1P         = np.append(NS1P,(mbin & mS1p).sum()) #cantidad de galaxias en el bin para poder hacer el promedio
                NS1M         = np.append(NS1M,(mbin & mS1p).sum())
                NS2P         = np.append(NS2P,(mbin & mS1p).sum())
                NS2M         = np.append(NS2M,(mbin & mS1p).sum())
                
        
        output = {'DSIGMAwsum_T':DSIGMAwsum_T,
                  #'DSIGMAwsum_X':DSIGMAwsum_X, 
                  'WEIGHT_RTsum':WEIGHT_RTsum,'WEIGHTwsum':WEIGHTwsum, 
                  'E1_P':E1_P, 'E1_M':E1_M, 'E2_P':E2_P, 'E2_M':E2_M, 
                  'NS1P':NS1P, 'NS1M':NS1M, 'NS2P':NS2P, 'NS2M':NS2M,
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
        
        L = np.array([RA[mlenses],DEC[mlenses],z[mlenses]])
        #L = L[]:2]
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
        NS1P         = np.zeros(ndots) #cantidad de galaxias en el bin para poder hacer el promedio de e_i
        NS1M         = np.zeros(ndots)
        NS2P         = np.zeros(ndots)
        NS2M         = np.zeros(ndots)

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
                                   Lsplit[l].T[2][0],
                                   RIN,ROUT,ndots,hcosmo]
                        
                        salida = [partial_profile_unpack(entrada)]
                else:          
                        entrada = np.array([Lsplit[l].T[0], Lsplit[l].T[1],
                                           Lsplit[l].T[2],
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
                        NS1P         += profilesums['NS1P']
                        NS1M         += profilesums['NS1M']
                        NS2P         += profilesums['NS2P']
                        NS2M         += profilesums['NS2M']
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

        E1_P_mean = E1_P / NS1P
        E1_M_mean = E1_M / NS1M
        E2_P_mean = E2_P / NS2P
        E2_M_mean = E2_M / NS2M
        Rsel_T    = 0.5 * ((E1_P_mean - E1_M_mean) + (E2_P_mean - E2_M_mean)) / Dg
        DSigma_T  = (DSIGMAwsum_T / (WEIGHT_RTsum + WEIGHTwsum*Rsel_T))
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
