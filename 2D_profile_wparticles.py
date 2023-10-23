'''calcula el perfil proyectado en el plano del cielo utilizando la pocision y masa de las particulas en el box seleccionado'''

import sys
import os
sys.path.append('home/fcaporaso/lens_codes_v3.7/')
from maria_func import *
import time
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import LambdaCDM, z_at_value
from astropy.wcs import WCS
# from fit_profiles_curvefit import *
# from astropy.stats import bootstrap
# from astropy.utils import NumpyRNGContext
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from multiprocessing import Pool, Process
import argparse
from astropy.constants import G,c,M_sun,pc
from scipy import stats
# from models_profiles import Gamma
# For map
wcs = WCS(naxis=2)
wcs.wcs.crpix = [0., 0.]
wcs.wcs.cdelt = [1./3600., 1./3600.]
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]    

#parameters
cvel = c.value;    # Speed of light (m.s-1)
G    = G.value;    # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value    # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

'''
sample     = 'pruDES'
idlist     = None
lcat       = 'voids_MICE.dat'
Rv_min     = 15.
Rv_max     = 25.
rho1_min   = -1.
rho1_max   = 2.
rho2_min   = -1.
rho2_max   = 100.
FLAG       = 2.
z_min      = 0.1
z_max      = 0.3
RIN        = 0.05
ROUT       = 5.0
ndots      = 40
ncores     = 128
hcosmo     = 1.
nback      = 30.
domap      = False
addnoise   = False
'''

def partial_profile(RA0,DEC0,Z,Rv,
                    RIN,ROUT,ndots,h=1.,
                    addnoise=False):

        '''
        RA0,DEC0 (float): posicion del centro del void
        Z: redshift del void
        RIN,ROUT: bordes del perfil
        ndots: cantidad de puntos del perfil
        h: cosmologia
        addnoise(bool): agregar ruido (forma intrinseca) a las galaxias de fondo
        devuelve la MASA tot por anillo, la cant de galaxias por bin (Ninbin) 
        y las totales (Ntot)'''
        
        ndots = int(ndots)

        Rv   = Rv/h *u.Mpc
        cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)
        
        DEGxMPC = cosmo.arcsec_per_kpc_proper(Z).to('deg/Mpc')
        delta = (DEGxMPC*(ROUT*Rv))

        dec_h, ra_h, z_h = comoving2ecuatiorial(S.xhalo, S.yhalo, S.zhalo)

        delta_z = z_at_value(cosmo.comoving_distance, Rv*ROUT) # caja cortada en redshift

        pos_angles = 0*u.deg, 90*u.deg, 180*u.deg, 270*u.deg
        c1 = SkyCoord(RA0*u.deg, DEC0*u.deg)
        c2 = np.array([c1.directional_offset_by(pos_angle, delta) for pos_angle in pos_angles])

        mask = (dec_h < c2[0].dec.deg)&(dec_h > c2[2].dec.deg)&(ra_h < c2[1].ra.deg)&(
                ra_h > c2[3].ra.deg)&(np.abs(z_h - Z) <= delta_z)
        
        catdata = S[mask]

        del mask, delta, delta_z

        dec_h, ra_h, z_h = comoving2ecuatiorial(catdata.xhalo, catdata.yhalo, catdata.zhalo)

        rads, theta, *_ = eq2p2(np.deg2rad(ra_h), np.deg2rad(dec_h),
                                  np.deg2rad(RA0), np.deg2rad(DEC0))
                                       
        
        # sacamos masas de las particulas
        mhalo = 10 **(catdata.lmhalo)

        r = (np.rad2deg(rads)/DEGxMPC.value)/(Rv.value)
        Ntot = len(catdata)        

        del catdata
        del theta, rads

        bines = np.linspace(RIN,ROUT,num=ndots+1)
        dig   = np.digitize(r,bines)
                
        MASAsum = np.ones(ndots)
        N_inbin = np.ones(ndots)
                                             
        for nbin in range(ndots):
                mbin = dig == nbin+1              

                MASAsum[nbin] = mhalo[mbin].sum()
                N_inbin[nbin] = np.count_nonzero(mbin)
        
        output = np.array([MASAsum, N_inbin, Ntot], dtype=object)
        
        return output

def partial_profile_unpack(minput):
	return partial_profile(*minput)


def comoving2ecuatiorial(xc_rc, yc_rc, zc_rc, h=1.):
        '''
        transforma de coordenadas cartesianas comoviles a coord esfericas ecuatoriales
        '''

        cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)

        ra_rc = np.rad2deg(np.arctan(xc_rc/yc_rc))
        ra_rc[yc_rc==0] = 90.
        dec_rc = np.rad2deg(np.arcsin(zc_rc/np.sqrt(xc_rc**2 + yc_rc**2 + zc_rc**2)))
        D = np.sqrt(xc_rc**2 + yc_rc**2 + zc_rc**2)

        z = z_at_value(cosmo.comoving_distance, D)

        return ra_rc, dec_rc, z


def main(lcat, sample='pru', output_file=None,
         Rv_min=0., Rv_max=50.,
         rho1_min=-1., rho1_max=0.,
         rho2_min=-1., rho2_max=100.,
         z_min = 0.1, z_max = 1.0,
         domap = False, RIN = .05, ROUT =5.,
         ndots= 40, ncores=10, 
         hcosmo=1.0, addnoise = False, FLAG = 2.):

        '''
        
        INPUT
        ---------------------------------------------------------
        sample         (str) sample name
        Rv_min         (float) lower limit for void radii - >=
        Rv_max         (float) higher limit for void radii - <
        rho1_min       (float) lower limit for inner density - >=
        rho1_max       (float) higher limit for inner density - <
        rho2_min       (float) lower limit for outer density - >=
        rho2_max       (float) higher limit for outer density - <
        FLAG           (float) higher limit for flag - <
        z_min          (float) lower limit for z - >=
        z_max          (float) higher limit for z - <
        domap          (bool) Instead of computing a profile it 
                       will compute a map with 2D bins ndots lsize
        RIN            (float) Inner bin radius of profile
        ROUT           (float) Outer bin radius of profile
        ndots          (int) Number of bins of the profile
        ncores         (int) to run in parallel, number of cores
        h              (float) H0 = 100.*h
        addnoise       (bool) add shape noise
        '''

        cosmo = LambdaCDM(H0=100*hcosmo, Om0=0.25, Ode0=0.75)
        tini = time.time()
        
        print(f'Voids catalog {lcat}')
        print(f'Sample {sample}')
        print(f'RIN : {RIN}')
        print(f'ROUT: {ROUT}')
        print(f'ndots: {ndots}')
        print('Selecting voids with:')
        print(f'{Rv_min}   <=  Rv  < {Rv_max}')
        print(f'{z_min}    <=  Z   < {z_max}')
        print(f'{rho1_min}  <= rho1 < {rho1_max}')
        print(f'{rho2_min}  <= rho2 < {rho2_max}')

        
        if addnoise:
            print('ADDING SHAPE NOISE')
        
        #reading Lens catalog
                
        L = np.loadtxt(folder+lcat).T

        Rv    = L[1]
        ra    = L[2]
        dec   = L[3]
        z     = L[4]
        rho_1 = L[8] #Sobredensidad integrada a un radio de void 
        rho_2 = L[9] #Sobredensidad integrada máxima entre 2 y 3 radios de void 
        flag  = L[11]

        mvoids = ((Rv >= Rv_min)&(Rv < Rv_max))&((z >= z_min)&(z < z_max))&(
                 (rho_1 >= rho1_min)&(rho_1 < rho1_max))&((rho_2 >= rho2_min)&(rho_2 < rho2_max))&(flag >= FLAG)        
        # SELECT RELAXED HALOS
                
        Nvoids = np.count_nonzero(mvoids)

        if Nvoids < ncores:
                ncores = Nvoids
        
        print(f'Nvoids {Nvoids}')
        print(f'CORRIENDO EN {ncores} CORES')
        
        L = L[:,mvoids]
        
        zmean    = np.mean(L[4])
        Rvmean   = np.mean(L[1])
        rho2mean = np.mean(L[9])

        # Define K masks   
        
        ncen = 100
        
        kmask    = np.zeros((ncen+1,len(ra)))
        kmask[0] = np.ones(len(ra)).astype(bool)
        
        ramin  = np.min(ra)
        cdec   = np.sin(np.deg2rad(dec))
        decmin = np.min(cdec)
        dra    = ((np.max(ra)+1.e-5)  - ramin)/10.
        ddec   = ((np.max(cdec)+1.e-5) - decmin)/10.
        
        c = 1
        
        for a in range(10): 
                for d in range(10): 
                        mra  = (ra  >= ramin + a*dra)&(ra < ramin + (a+1)*dra) 
                        mdec = (cdec >= decmin + d*ddec)&(cdec < decmin + (d+1)*ddec) 
                        # plt.plot(ra[(mra*mdec)],dec[(mra*mdec)],'C'+str(c+1)+',')
                        kmask[c] = ~(mra&mdec)
                        c += 1
        
        ind_rand0 = np.arange(Nvoids)
        np.random.shuffle(ind_rand0)



        # SPLIT LENSING CAT
        
        lbins = int(round(Nvoids/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < Nvoids)]
        Lsplit = np.split(L.T,slices)
        Ksplit = np.split(kmask.T,slices)

        del L

        print(f'Profile has {ndots} bins')
        print(f'from {RIN} Rv to {ROUT} Rv')
        
        try:
            os.mkdir('../profiles')
        except FileExistsError:
            pass
        
        if not output_file:
            output_file = f'../profiles/voids/'
        
        # Defining radial bins
        bines = np.linspace(RIN,ROUT,num=ndots+1)
        R = (bines[:-1] + np.diff(bines)*0.5)
        # WHERE THE SUMS ARE GOING TO BE SAVED
        
        Ninbin = np.zeros((ncen+1,ndots))
        MASAsum    = np.zeros((ncen+1,ndots)) 
                                
        # FUNCTION TO RUN IN PARALLEL
        partial = partial_profile_unpack
        
        print(f'Saved in ../{output_file+sample}.fits')

        LARGO = len(Lsplit)
        tslice = np.array([])
        
        for l, Lsplit_l in enumerate(Lsplit):
                
                print(f'RUN {l+1} OF {LARGO}')
                
                t1 = time.time()

                num = len(Lsplit_l)
                
                if num == 1:
                        entrada = [Lsplit_l[2], Lsplit_l[3],
                                   Lsplit_l[4],Lsplit_l[1],
                                   RIN,ROUT,ndots,hcosmo,
                                   addnoise]
                        
                        salida = [partial(entrada)]
                else:                
                        rin       = np.full(num, RIN)
                        rout      = np.full(num, ROUT)
                        nd        = np.full(num, ndots, dtype=int)
                        h_array   = np.full(num, hcosmo)
                        addnoise_array = np.full(num, addnoise, dtype=bool)
                        
                        entrada = np.array([Lsplit_l.T[2],Lsplit_l.T[3],
                                            Lsplit_l.T[4],Lsplit_l.T[1],
                                            rin,rout,nd,h_array,
                                            addnoise_array]).T

                        with Pool(processes=num) as pool:
                                salida = np.array(pool.map(partial,entrada))
                                pool.close()
                                pool.join()
                
                for j, profilesums in enumerate(salida):
                        
                        if domap:
                                print('Sin mapa')
                            
                        else:

                            km      = np.tile(Ksplit[l][j],(ndots,1)).T
                            Ninbin += np.tile(profilesums[1],(ncen+1,1))*km
                                                
                            MASAsum    += np.tile(profilesums[0],(ncen+1,1))*km
                            
                Ntot   = np.array([profilesums[-1] for profilesums in salida])

                t2 = time.time()
                ts = (t2-t1)/60.
                tslice = np.append(tslice, ts)
                print('TIME SLICE')
                print(f'{np.round(ts,4)} min')
                print('Estimated remaining time')
                print(f'{np.round(np.mean(tslice)*(LARGO-(l+1)), 3)} min')

        # AVERAGE VOID PARAMETERS AND SAVE IT IN HEADER

        h = fits.Header()
        h.append(('N_VOIDS',int(Nvoids)))
        h.append(('Lens_cat',lcat))
        #h.append(('MICE version sources 2.0'))
        h.append(('Rv_min',np.round(Rv_min,2)))
        h.append(('Rv_max',np.round(Rv_max,2)))
        h.append(('Rv_mean',np.round(Rvmean,4)))
        h.append(('rho1_min',np.round(rho1_min,2)))
        h.append(('rho1_max',np.round(rho1_max,2)))
        h.append(('rho2_min',np.round(rho2_min,2)))
        h.append(('rho2_max',np.round(rho2_max,2)))
        h.append(('rho2_mean',np.round(rho2mean,4)))
        h.append(('z_min',np.round(z_min,2)))
        h.append(('z_max',np.round(z_max,2)))
        h.append(('z_mean',np.round(zmean,4)))
        h.append(('hcosmo',np.round(hcosmo,4)))
        
        h.append(('---SLICES_INFO---'))
        h.append(('Rp_min',np.round(RIN,4)))
        h.append(('Rp_max',np.round(ROUT,4)))
        h.append(('ndots',np.round(ndots,4)))
        
        
        # COMPUTING PROFILE        
        Ninbin[MASAsum == 0] = 1.
                
        MASA     = MASAsum/Ninbin        
        
        table_p = np.array([fits.Column(name='Rp', format='E', array=R),
                            fits.Column(name='MASA',    format='E', array=MASA.flatten()),
                            fits.Column(name='Ninbin', format='E', array=Ninbin.flatten())])

        tbhdu_p = fits.BinTableHDU.from_columns(fits.ColDefs(table_p))
        
        primary_hdu = fits.PrimaryHDU(header=h)
        
        hdul = fits.HDUList([primary_hdu, tbhdu_p])
        
        hdul.writeto(f'{output_file+sample}.fits',overwrite=True)

        print(f'File saved... {output_file+sample}.fits')
                
        tfin = time.time()
        
        print(f'Partial time: {np.round((tfin-tini)/60. , 3)} mins')
        


if __name__=='__main__':
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-sample', action='store', dest='sample',default='pru')
        parser.add_argument('-lens_cat', action='store', dest='lcat',default='voids_MICE.dat')
        parser.add_argument('-Rv_min', action='store', dest='Rv_min', default=0.)
        parser.add_argument('-Rv_max', action='store', dest='Rv_max', default=50.)
        parser.add_argument('-rho1_min', action='store', dest='rho1_min', default=-1.)
        parser.add_argument('-rho1_max', action='store', dest='rho1_max', default=1.)
        parser.add_argument('-rho2_min', action='store', dest='rho2_min', default=-1.)
        parser.add_argument('-rho2_max', action='store', dest='rho2_max', default=100.)
        parser.add_argument('-FLAG', action='store', dest='FLAG', default=2.)
        parser.add_argument('-z_min', action='store', dest='z_min', default=0.1)
        parser.add_argument('-z_max', action='store', dest='z_max', default=0.5)
        parser.add_argument('-domap', action='store', dest='domap', default='False')
        parser.add_argument('-addnoise', action='store', dest='addnoise', default='False')
        parser.add_argument('-RIN', action='store', dest='RIN', default=0.05)
        parser.add_argument('-ROUT', action='store', dest='ROUT', default=5.)
        parser.add_argument('-nbins', action='store', dest='nbins', default=40)
        parser.add_argument('-ncores', action='store', dest='ncores', default=10)
        parser.add_argument('-h_cosmo', action='store', dest='h_cosmo', default=1.)
        parser.add_argument('-ides_list', action='store', dest='idlist', default=None)
        parser.add_argument('-nback', action='store', dest='nback', default=30)
        parser.add_argument('-nslices', action='store', dest='nslices', default=1.)
        args = parser.parse_args()

        sample     = args.sample
        idlist     = args.idlist
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
        nback      = float(args.nback)
        nslices      = int(args.nslices)

        if args.domap == 'True':
            domap = True
        elif args.domap == 'False':
            domap = False

        if args.addnoise == 'True':
            addnoise = True
        elif args.addnoise == 'False':
            addnoise = False

        folder = '/mnt/simulations/MICE/'
        S      = fits.open('/home/fcaporaso/cats/MICE/micecat2_halos.fits')[1].data
        
        if nback < 30.:
            nselec = int(nback*5157*3600.)
            j      = np.random.choice(np.array(len(S)),nselec)
            S  = S[j]

        #Sgal_coord = SkyCoord(S.ra_gal, S.dec_gal, unit='deg', frame='icrs')

        print('BACKGROUND GALAXY DENSINTY',len(S)/(5157*3600))

        tin = time.time()

        main(lcat, sample='pru2D', output_file=None, Rv_min=Rv_min, Rv_max=Rv_max,
                rho1_min=rho1_min, rho1_max=rho1_max,
                rho2_min=rho2_min, rho2_max=rho2_max,
                z_min=z_min, z_max=z_max,
                domap=False, RIN=RIN, ROUT=ROUT,
                ndots=ndots, ncores=ncores, 
                hcosmo=hcosmo, addnoise=False, FLAG=2.)

        tfin = time.time()

        print(f'TOTAL TIME: {np.round((tfin-tin)/60.,2)} min')

