import sys
sys.path.append('../lens_codes_v3.7')
import os
from maria_func import *
import numpy as np
from astropy.cosmology import LambdaCDM
from astropy.constants import G,c,M_sun,pc
from astropy.io import fits
import time
import argparse
from multiprocessing import Pool, Process

#parameters
cvel = c.value;    # Speed of light (m.s-1)
G    = G.value;    # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value    # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

def gal_inbin(RA0,DEC0,Z,Rv,
              RIN,ROUT,ndots,h=1):

        ndots = int(ndots)

        Rv   = Rv/h
        cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)
        dl  = cosmo.angular_diameter_distance(Z).value
        KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
        
        delta = ROUT*(Rv*1000.)*5./(3600*KPCSCALE)
        
        mask = (S.ra_gal < (RA0+delta))&(S.ra_gal > (RA0-delta))&(S.dec_gal > (DEC0-delta))&(
                S.dec_gal < (DEC0+delta))&(S.z_cgal_v > (Z+0.1))
        catdata = S[mask]

        del mask
        del delta

        rads, _, _, _ = eq2p2(np.deg2rad(catdata.ra_gal), np.deg2rad(catdata.dec_gal),
                                        np.deg2rad(RA0), np.deg2rad(DEC0))
        
        r = (np.rad2deg(rads)*3600*KPCSCALE)/(Rv*1000.)
     
        Ntot = len(catdata)
        
        bines = np.linspace(RIN,ROUT,num=ndots+1)
        dig   = np.digitize(r, bines)

        N_inbin = np.array([np.count_nonzero(dig==nbin+1) for nbin in range(0,ndots)])
        
        return N_inbin, Ntot

def gal_inbin_unpack(a):
    return gal_inbin(*a)

def main(lcat, sample='pru',
         Rv_min=0.,Rv_max=50.,
         rho1_min=-1.,rho1_max=0.,
         rho2_min=-1.,rho2_max=100.,
         z_min = 0.1, z_max = 1.0,
         RIN = 0.5, ROUT =10.,
         ndots= 40, ncores=10, 
         hcosmo=1.0, FLAG = 2.):

        cosmo = LambdaCDM(H0=100*hcosmo, Om0=0.25, Ode0=0.75)
        tini = time.time()
        
        print(f'Voids catalog {lcat}')
        print(f'Sample {sample}')
        print(f'RIN : {RIN}')
        print(f'ROUT: {ROUT}')
        print(f'ndots: {ndots}')
        print('Selecting voids with:')
        print(f'{Rv_min}    <=  Rv  < {Rv_max}')
        print(f'{z_min}     <=  Z   < {z_max}')
        print(f'{rho1_min}   <= rho1 < {rho1_max}')
        print(f'{rho2_min}   <= rho2 < {rho2_max}')
        
        L = np.loadtxt(folder+lcat).T

        Rv, ra, dec, z, rho_1, rho_2, flag = L[1], L[2], L[3], L[4], L[8], L[9], L[11]

        mradio  = (Rv >= Rv_min)&(Rv < Rv_max)
        mz      = (z >= z_min)&(z < z_max)
        mrho1   = (rho_1 >= rho1_min)&(rho_1 < rho2_max)
        mrho2   = (rho_2 >= rho2_min)&(rho_2 < rho2_max)
        mflag   = flag >= FLAG
        mvoids = mradio & mz & mrho1 & mrho2 & mflag

        del mradio, mz, mrho1, mrho2, mflag
        
        # SELECT RELAXED HALOS
                
        Nvoids = mvoids.sum()

        if Nvoids < ncores:
                ncores = Nvoids
        
        print(f'Nvoids {Nvoids}')
        print(f'CORRIENDO EN {ncores} CORES')
        
        L = L[:,mvoids]
                                       
        # SPLIT LENSING CAT
        
        lbins = int(round(Nvoids/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < Nvoids)]
        Lsplit = np.split(L.T,slices)
        
        del L

        print(f'Profile has {ndots} bins')
        print(f'from {RIN} Rv to {ROUT} Rv')
        os.system('mkdir ../tests')
        output_file = f'tests/count_{sample}.fits'

        # Defining radial bins
        bines = np.linspace(RIN,ROUT,num=ndots+1)
        R = (bines[:-1] + np.diff(bines)*0.5)

        # WHERE THE SUMS ARE GOING TO BE SAVED
        
        Ninbin = np.zeros(ndots)

        # FUNCTION TO RUN IN PARALLEL
        partial = gal_inbin_unpack
        
        print(f'Saved in ../{output_file}')
                
        Ntot         = np.array([])
        tslice       = np.array([])
        
        for l, Lsplit_l in enumerate(Lsplit):
                
                print(f'RUN {l+1} OF {len(Lsplit)}')
                
                t1 = time.time()
                
                num = len(Lsplit_l)

                if num == 1:
                        entrada = [Lsplit_l[2], Lsplit_l[3],
                                   Lsplit_l[4],Lsplit_l[1],
                                   RIN,ROUT,ndots]
                        
                        salida = [partial(entrada)]
                else:
                        rin            = np.full(num, RIN)
                        rout           = np.full(num, ROUT)
                        nd             = np.full(num, ndots)
                        
                        entrada = np.array([Lsplit_l.T[2],Lsplit_l.T[3],
                                            Lsplit_l.T[4],Lsplit_l.T[1],
                                            rin,rout,nd]).T
                        
                        with Pool(processes=num) as pool:
                            salida = np.array(pool.map(partial, entrada))
                            pool.close()
                            pool.join()
                                                
                for j, profilesums in enumerate(salida):
                        
                        Ntot   = np.append(Ntot,profilesums['Ntot'])
                        Ninbin = np.append(Ninbin, profilesums['N_inbin'])
                        
                t2 = time.time()
                ts = (t2-t1)/60.
                tslice = np.append(tslice,ts)
                print('TIME SLICE')
                print(f'{np.round(ts,2)} min')
                print('Estimated remaining time')
                print(f'{np.round(np.mean(tslice)*(len(Lsplit)-(l+1)),2)} min')

        # AVERAGE VOID PARAMETERS AND SAVE IT IN HEADER

        H = fits.Header()
        H.append(('N_VOIDS',np.int32(Nvoids)))
        H.append(('Rv_min',np.round(Rv_min,2)))
        H.append(('Rv_max',np.round(Rv_max,2)))
        H.append(('rho1_min',np.round(rho1_min,2)))
        H.append(('rho1_max',np.round(rho1_max,2)))
        H.append(('rho2_min',np.round(rho2_min,2)))
        H.append(('rho2_max',np.round(rho2_max,2)))
        H.append(('z_min',np.round(z_min,2)))
        H.append(('z_max',np.round(z_max,2)))

        table_pro = [fits.Column(name='Rp', format='E', array=R),
                    fits.Column(name='Ninbin', format='E', array=Ninbin)]

        tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_pro))
        primary_hdu = fits.PrimaryHDU(header=H)

        hdul = fits.HDUList([primary_hdu, tbhdu_pro])#, tbhdu_cov])
        
        hdul.writeto(f'../{output_file}',overwrite=True)
                
        tfin = time.time()
        
        print(f'TOTAL TIME {(tfin-tini)/60.}')


if __name__ == '__main__':
    print('Testeando...')

    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', action='store', dest='sample', default='pru')
    parser.add_argument('-lens_cat', action='store', dest='lcat',default='voids_MICE.dat')
    parser.add_argument('-Rv_min', action='store', dest='Rv_min', default=10.)
    parser.add_argument('-Rv_max', action='store', dest='Rv_max', default=20.)
    parser.add_argument('-rho1_min', action='store', dest='rho1_min', default=-1.)
    parser.add_argument('-rho1_max', action='store', dest='rho1_max', default=1.)
    parser.add_argument('-rho2_min', action='store', dest='rho2_min', default=-1.)
    parser.add_argument('-rho2_max', action='store', dest='rho2_max', default=100.)
    parser.add_argument('-FLAG', action='store', dest='FLAG', default=2.)
    parser.add_argument('-z_min', action='store', dest='z_min', default=0.1)
    parser.add_argument('-z_max', action='store', dest='z_max', default=0.4)
    parser.add_argument('-RIN', action='store', dest='RIN', default=0.05)
    parser.add_argument('-ROUT', action='store', dest='ROUT', default=10.)
    parser.add_argument('-nbins', action='store', dest='nbins', default=20)
    parser.add_argument('-ncores', action='store', dest='ncores', default=10)
    parser.add_argument('-h_cosmo', action='store', dest='h_cosmo', default=1.)
    args = parser.parse_args()
    
    sample     = args.sample
    lcat       = args.lcat
    Rv_min     = float(args.Rv_min)
    Rv_max     = float(args.Rv_max) 
    rho1_min   = float(args.rho1_min)
    rho1_max   = float(args.rho1_max) 
    rho2_min   = float(args.rho2_min)
    rho2_max   = float(args.rho2_max) 
    FLAG       = float(args.FLAG) 
    z_min      = float(args.z_min) 
    z_max      = float(args.z_max) 
    RIN        = float(args.RIN)
    ROUT       = float(args.ROUT)
    ndots      = int(args.nbins)
    ncores     = int(args.ncores)
    hcosmo     = float(args.h_cosmo)

    folder = '/mnt/simulations/MICE/'
    S      = fits.open(folder+'MICE_sources_HSN_withextra.fits')[1].data

    main(lcat, sample, Rv_min, Rv_max, rho1_min,rho1_max, rho2_min, rho2_max,
         z_min, z_max, RIN, ROUT, ndots, ncores, hcosmo, FLAG)

    S.close()



