import sys
sys.path.append('../lens_codes_v3.7')
import time
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import LambdaCDM
from astropy.wcs import WCS
from maria_func import *
from fit_profiles_curvefit import *
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from multiprocessing import Pool
from multiprocessing import Process
import argparse
from astropy.constants import G,c,M_sun,pc
from scipy import stats
import astropy.units as u
from astropy.coordinates import SkyCoord
from models_profiles import Gamma
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


if args.domap == 'True':
    domap = True
elif args.domap == 'False':
    domap = False

if args.addnoise == 'True':
    addnoise = True
elif args.addnoise == 'False':
    addnoise = False


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


folder = '/mnt/simulations/MICE/'
S      = fits.open(folder+'MICE_sources_HSN_withextra.fits')[1].data

if nback < 30.:
    nselec = int(nback*5157*3600.)
    j      = np.random.choice(np.array(len(S)),nselec)
    S  = S[j]

print('BACKGROUND GALAXY DENSINTY',len(S)/(5157*3600))

def partial_map(RA0,DEC0,Z,Rv,
                RIN,ROUT,ndots,h,
                addnoise):

        
        lsize = int(np.sqrt(ndots))
        ndots = int(ndots)
        
        cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)
        dl  = cosmo.angular_diameter_distance(Z).value
        KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
        
        delta = (ROUT*1.5)/(3600*KPCSCALE)
        rarray = [[-1.*(ROUT*1.e-3),(ROUT*1.e-3)],[-1.*(ROUT*1.e-3),(ROUT*1.e-3)]]
        
        t0 = time.time()
        # mask = (S.ra < (RA0+delta))&(S.ra > (RA0-delta))&(S.dec > (DEC0-delta))&(S.dec < (DEC0+delta))&(S.z_v > (Z+0.1))
        mask = (S.ra_gal < (RA0+delta))&(S.ra_gal > (RA0-delta))&(S.dec_gal > (DEC0-delta))&(S.dec_gal < (DEC0+delta))&(S.z_cgal_v > (Z+0.1))
                       
        catdata = S[mask]

        ds  = cosmo.angular_diameter_distance(catdata.z_cgal_v).value
        dls = cosmo.angular_diameter_distance_z1z2(Z, catdata.z_cgal_v).value
                
        
        BETA_array = dls/ds
        
        Dl = dl*1.e6*pc
        sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)

        # Add miscentring
        c1 = SkyCoord(RA0*u.degree, DEC0*u.degree)
        roff_deg = roff/(KPCSCALE*3600.)
        c2 = c1.directional_offset_by(phi_off*u.degree,roff_deg*u.degree)


        rads, theta, test1,test2 = eq2p2(np.deg2rad(catdata.ra_gal),
                                        np.deg2rad(catdata.dec_gal),
                                        np.deg2rad(c2.ra.value),
                                        np.deg2rad(c2.dec.value))


        theta2 = (2.*np.pi - theta) +np.pi/2.
        theta_ra = theta2
        theta_ra[theta2 > 2.*np.pi] = theta2[theta2 > 2.*np.pi] - 2.*np.pi
                       
        e1     = catdata.gamma1
        e2     = -1.*catdata.gamma2
        
        # Add shape noise due to intrisic galaxy shapes        
        if addnoise:
            es1 = -1.*catdata.eps1
            es2 = catdata.eps2
            e1 += es1
            e2 += es2
       
        #get tangential ellipticities 
        et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
        #get cross ellipticities
        ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
        # '''
        k  = catdata.kappa*sigma_c
                
        wcs.wcs.crval = [RA0,DEC0]
        dx, dy = wcs.wcs_world2pix(catdata.ra,catdata.dec, 0)

        dx = dx*KPCSCALE*1.e-3
        dy = dy*KPCSCALE*1.e-3
          
        del(e1)
        del(e2)
        
        r = np.rad2deg(rads)*3600*KPCSCALE
        
        del(rads)
        
        
        Ntot = len(catdata)
        del(catdata)    
        
        
        t = np.tile(theta_ra,(3,1)).T
        an = np.tile(angles,(len(theta_ra),1))
        at     = t - an
        
        GTsum = np.zeros((ndots,3))
        GXsum = np.zeros((ndots,3))
        Ksum  = np.zeros((ndots,3))
        Ninbin  = np.zeros((ndots,3))
        
        
        for j in range(3):
                x_t  = (dx*np.cos(an[:,j]) + dy*np.sin(an[:,j])) 
                y_t = (-1.*dx*np.sin(an[:,j]) + dy*np.cos(an[:,j])) 
                m   =  (abs(x_t) < (ROUT*1.e-3))*(abs(y_t) < (ROUT*1.e-3)) 
                out = stats.binned_statistic_2d(x_t[m], y_t[m], et[m], 
                                               'count',bins=lsize, range = rarray)
                Ninbin[:,j] = out.statistic.flatten()
                out = stats.binned_statistic_2d(x_t[m], y_t[m], et[m], 
                                               'sum',bins=lsize, range = rarray)
                GTsum[:,j] = out.statistic.flatten()
                out = stats.binned_statistic_2d(x_t[m], y_t[m], ex[m], 
                                               'sum',bins=lsize, range = rarray)
                GXsum[:,j] = out.statistic.flatten()
                out = stats.binned_statistic_2d(x_t[m], y_t[m], k[m], 
                                               'sum',bins=lsize, range = rarray)
                Ksum[:,j] = out.statistic.flatten()

        Ntot = m.sum()
   
        output = {'GTsum':GTsum,'GXsum':GXsum,
                   'Ksum': Ksum, 'N_inbin':Ninbin,
                   'Ntot': Ntot}
        
        return output

def partial_map_unpack(minput):
	return partial_map(*minput)

def partial_profile(RA0,DEC0,Z,Rv,
                    RIN,ROUT,ndots,h,
                    addnoise):

        ndots = int(ndots)


        Rv   = Rv/h
        cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)
        dl  = cosmo.angular_diameter_distance(Z).value
        KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
        
        delta = ROUT*(Rv*1000.)*5./(3600*KPCSCALE)
        
        mask = (S.ra_gal < (RA0+delta))&(S.ra_gal > (RA0-delta))&(S.dec_gal > (DEC0-delta))&(S.dec_gal < (DEC0+delta))&(S.z_cgal_v > (Z+0.1))
        catdata = S[mask]

        del mask
        del delta

        ds  = cosmo.angular_diameter_distance(catdata.z_cgal_v).value
        dls = cosmo.angular_diameter_distance_z1z2(Z, catdata.z_cgal_v).value
                
        BETA_array = dls/ds
        
        Dl = dl*1.e6*pc
        sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)

        del ds
        del dls
        del BETA_array

        rads, theta, test1,test2 = eq2p2(np.deg2rad(catdata.ra_gal),
                                        np.deg2rad(catdata.dec_gal),
                                        np.deg2rad(RA0),
                                        np.deg2rad(DEC0))
                               
        del test1
        del test2
        
        e1     = catdata.gamma1
        e2     = -1.*catdata.gamma2

        # Add shape noise due to intrisic galaxy shapes        
        if addnoise:
            es1 = -1.*catdata.eps1
            es2 = catdata.eps2
            e1 += es1
            e2 += es2
        
        #get tangential ellipticities 
        et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
        #get cross ellipticities
        ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
        # '''
        k  = catdata.kappa*sigma_c
          
        del e1
        del e2
        del theta

        r = (np.rad2deg(rads)*3600*KPCSCALE)/(Rv*1000.)
        del rads
        
        
        Ntot = len(catdata)
        del catdata    
        
        bines = np.linspace(RIN,ROUT,num=ndots+1)
        dig = np.digitize(r,bines)
                
        
        SIGMAwsum    = []
        DSIGMAwsum_T = []
        DSIGMAwsum_X = []
        N_inbin      = []
                                             
        for nbin in range(ndots):
                mbin = dig == nbin+1              
                
                SIGMAwsum    = np.append(SIGMAwsum,k[mbin].sum())
                DSIGMAwsum_T = np.append(DSIGMAwsum_T,et[mbin].sum())
                DSIGMAwsum_X = np.append(DSIGMAwsum_X,ex[mbin].sum())
                               
                N_inbin = np.append(N_inbin,len(et[mbin]))
                
        
        output = {'SIGMAwsum':SIGMAwsum,'DSIGMAwsum_T':DSIGMAwsum_T,
                   'DSIGMAwsum_X':DSIGMAwsum_X,
                   'N_inbin':N_inbin,'Ntot':Ntot}
        
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
        
def main(lcat, sample='pru',
         Rv_min=0.,Rv_max=50.,
         rho1_min=-1.,rho1_max=0.,
         rho2_min=-1.,rho2_max=100.,
         z_min = 0.1, z_max = 1.0,
         domap = False, RIN = 400., ROUT =5000.,
         ndots= 40, ncores=10, 
         idlist= None, hcosmo=1.0, 
         addnoise = False, FLAG = 2.):

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
        print('Selecting voids with:')
        print(f'{Rv_min} <= Rv < {Rv_max}')
        print(f'{z_min}  <= Z < {z_max}')
        print(f'{rho1_min} <= rho1 < {rho1_max}')
        print(f'{rho2_min} <= rho2 < {rho2_max}')
        
        if idlist:
                print('From id list '+idlist)
        # else:
                # print(lM_min,' <= log(M) < ',lM_max)
                # print(z_min,' <= z < ',z_max)
                # print(q_min,' <= q < ',q_max)
                # print(rs_min,' <= rs < ',rs_max)
                #print(R5s_min,' <= R5s < ',R5s_max)
                # print('resNFW_S < ',resNFW_max)
                # print('h ',hcosmo)
                # print('misalign '+str(misalign))
        
        if addnoise:
            print('ADDING SHAPE NOISE')
        
        #reading cats
                
        L = np.loadtxt(folder+lcat).T

        Rv    = L[1]
        ra    = L[2]
        dec   = L[3]
        z     = L[4]
        rho_1 = L[8] #Sobredensidad integrada a un radio de void 
        rho_2 = L[9] #Sobredensidad integrada máxima entre 2 y 3 radios de void 
        flag  = L[11]
        
        if idlist:
                ides = np.loadtxt(idlist).astype(int)
                mlenses = np.in1d(L[0],ides)
        else:                
                mradio  = (Rv >= Rv_min)&(Rv < Rv_max)
                mz      = (z >= z_min)&(z < z_max)
                mrho1   = (rho_1 >= rho1_min)&(rho_1 < rho2_max)
                mrho2   = (rho_2 >= rho2_min)&(rho_2 < rho2_max)
                mflag   = flag >= FLAG
                mlenses = mradio & mz & mrho1 & mrho2 & mflag
        
        # SELECT RELAXED HALOS
                
        Nlenses = mlenses.sum()

        if Nlenses < ncores:
                ncores = Nlenses
        
        print(f'Nvoids {Nlenses}')
        print(f'CORRIENDO EN {ncores} CORES')
        
        L = L[:,mlenses]
                        
        # Define K masks   
        ncen = 100
        
        kmask = np.zeros((ncen+1,len(ra)))
        kmask[0] = np.ones(len(ra)).astype(bool)
        
        ramin  = np.min(ra)
        cdec   = np.sin(np.deg2rad(dec))
        decmin = np.min(cdec)
        dra  = ((np.max(ra)+1.e-5)  - ramin)/10.
        ddec = ((np.max(cdec)+1.e-5) - decmin)/10.
        
        c    = 1
        
        for a in range(10): 
                for d in range(10): 
                        mra  = (ra  >= ramin + a*dra)*(ra < ramin + (a+1)*dra) 
                        mdec = (cdec >= decmin + d*ddec)*(cdec < decmin + (d+1)*ddec) 
                        # plt.plot(ra[(mra*mdec)],dec[(mra*mdec)],'C'+str(c+1)+',')
                        kmask[c] = ~(mra*mdec)
                        c += 1
        
        ind_rand0 = np.arange(Nlenses)
        np.random.shuffle(ind_rand0)
        
        
                                
        # SPLIT LENSING CAT
        
        lbins = int(round(Nlenses/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < Nlenses)]
        Lsplit = np.split(L.T,slices)
        Ksplit = np.split(kmask.T,slices)
        
        if domap:

            print('Maps have ',ndots,'pixels')
            print('up to ',ROUT,'kpc')
            
            os.system('mkdir ../maps')
            output_file = 'maps/map_'+sample+'.fits'

            # Defining 2D bins
            
            lsize = int(np.sqrt(ndots))
            
            ndots = int(lsize**2)
            
            Rmpc = 1.e-3*ROUT
            
            xb = np.linspace(-1.*Rmpc+(Rmpc/lsize),Rmpc-(Rmpc/lsize),lsize)
            
            ybin, xbin = np.meshgrid(xb,xb)
            
            ybin = ybin.flatten()
            xbin = xbin.flatten()

            # WHERE THE SUMS ARE GOING TO BE SAVED
            GTsum = np.zeros((ncen+1,ndots,3))
            GXsum = np.zeros((ncen+1,ndots,3))
            Ksum  = np.zeros((ncen+1,ndots,3))        
            
            Ninbin = np.zeros((ncen+1,ndots,3))
            
            # FUNCTION TO RUN IN PARALLEL
            partial = partial_map_unpack
            

        else:

            print(f'Profile has {ndots} bins')
            print(f'from {RIN} Mpc to {ROUT} Mpc')
            os.system('mkdir ../profiles')
            output_file = f'profiles/voids/profile_{sample}.fits'

            # Defining radial bins
            bines = np.linspace(RIN,ROUT,num=ndots+1)
            R = (bines[:-1] + np.diff(bines)*0.5)

            # WHERE THE SUMS ARE GOING TO BE SAVED
            
            Ninbin = np.zeros((ncen+1,ndots))
            
            SIGMAwsum    = np.zeros((ncen+1,ndots)) 
            DSIGMAwsum_T = np.zeros((ncen+1,ndots)) 
            DSIGMAwsum_X = np.zeros((ncen+1,ndots))
                            
            # FUNCTION TO RUN IN PARALLEL
            partial = partial_profile_unpack
            

        print(f'Saved in ../{output_file}')
                                   
                
        Ntot         = np.array([])
        tslice       = np.array([])
        
        for l in range(len(Lsplit)):
                
                print(f'RUN {l+1} OF {len(Lsplit)}')
                
                t1 = time.time()
                
                num = len(Lsplit[l])
                
                rin  = RIN*np.ones(num)
                rout = ROUT*np.ones(num)
                nd   = ndots*np.ones(num)
                h_array   = hcosmo*np.ones(num)
                addnoise_array   = np.array([addnoise]*np.ones(num))
                
                if num == 1:
                        entrada = [Lsplit[l][2], Lsplit[l][3],
                                   Lsplit[l][4],Lsplit[l][1],
                                   RIN,ROUT,ndots,hcosmo,
                                   addnoise]
                        
                        salida = [partial(entrada)]
                else:          
                        entrada = np.array([Lsplit[l].T[2],Lsplit[l].T[3],
                                        Lsplit[l].T[4],Lsplit[l].T[1],
                                        rin,rout,nd,h_array,
                                        addnoise_array]).T
                        
                        pool = Pool(processes=(num))
                        salida = np.array(pool.map(partial, entrada))
                        pool.terminate()
                                
                for j in range(len(salida)):
                        
                        profilesums = salida[j]
                        Ntot         = np.append(Ntot,profilesums['Ntot'])
                        
                        if domap:
                            
                            km      = np.tile(Ksplit[l][j],(3,ndots,1)).T
                            Ninbin += np.tile(profilesums['N_inbin'],(ncen+1,1,1))*km

                            GTsum += np.tile(profilesums['GTsum'],(ncen+1,1,1))*km
                            GXsum += np.tile(profilesums['GXsum'],(ncen+1,1,1))*km
                            Ksum  += np.tile(profilesums['Ksum'],(ncen+1,1,1))*km
                            
                        else:

                            km      = np.tile(Ksplit[l][j],(ndots,1)).T

                            Ninbin += np.tile(profilesums['N_inbin'],(ncen+1,1))*km
                                                
                            SIGMAwsum    += np.tile(profilesums['SIGMAwsum'],(ncen+1,1))*km
                            DSIGMAwsum_T += np.tile(profilesums['DSIGMAwsum_T'],(ncen+1,1))*km
                            DSIGMAwsum_X += np.tile(profilesums['DSIGMAwsum_X'],(ncen+1,1))*km
                            
                        
                
                t2 = time.time()
                ts = (t2-t1)/60.
                tslice = np.append(tslice,ts)
                print('TIME SLICE')
                print(ts)
                print('Estimated remaining time')
                print(np.mean(tslice)*(len(Lsplit)-(l+1)))

        # AVERAGE LENS PARAMETERS AND SAVE IT IN HEADER
        
        # zmean        = np.average(L,weights=Ntot)
        # lM_mean      = np.log10(np.average(10**L.lgMEin_rho,weights=Ntot))
        # c200_mean    = np.average(L.cNFW_S,weights=Ntot)
        # lM200_mean   = np.log10(np.average(10**L.lgMNFW_S,weights=Ntot))
        

        h = fits.Header()
        h.append(('N_VOIDS',np.int(Nlenses)))
        h.append(('Lens cat',lcat))
        h.append(('MICE version sources 2.0'))
        h.append(('Rv_min',np.round(Rv_min,2)))
        h.append(('Rv_max',np.round(Rv_max,2)))
        h.append(('rho1_max',np.round(rho1_max,2)))
        h.append(('rho1_min',np.round(rho1_min,2)))
        h.append(('rho2_max',np.round(rho2_max,2)))
        h.append(('rho2_min',np.round(rho2_min,2)))
        h.append(('z_min',np.round(z_min,2)))
        h.append(('z_max',np.round(z_max,2)))
        h.append(('hcosmo',np.round(hcosmo,4)))
        # h.append(('lM_mean',np.round(lM_mean,4)))
        # h.append(('lM200_mean',np.round(lM200_mean,4)))
        # h.append(('c200_mean',np.round(c200_mean,4)))
        # h.append(('z_mean',np.round(zmean,4)))

        if domap:
            
            # COMPUTING PROFILE        
                
            GT  = (GTsum/Ninbin)
            GX  = (GXsum/Ninbin)
            K   = (Ksum/Ninbin)
                    
            # COMPUTE COVARIANCE
            
            COV_Gtc = cov_matrix(GT[1:,:,0])
            COV_Gt  = cov_matrix(GT[1:,:,1])
            COV_Gtr = cov_matrix(GT[1:,:,2])
            
            COV_Gxc = cov_matrix(GX[1:,:,0])
            COV_Gx  = cov_matrix(GX[1:,:,1])
            COV_Gxr = cov_matrix(GX[1:,:,2])
    
            COV_Kc = cov_matrix(GX[1:,:,0])
            COV_K  = cov_matrix(GX[1:,:,1])
            COV_Kr = cov_matrix(GX[1:,:,2])
            
            table_pro = [fits.Column(name='xmpc', format='E', array=xbin),
                    fits.Column(name='ympc', format='E', array=ybin),
                    fits.Column(name='GT_control', format='E', array=GT[0,:,0]),
                    fits.Column(name='GT', format='E', array=GT[0,:,1]),
                    fits.Column(name='GT_reduced', format='E', array=GT[0,:,2]),
                    fits.Column(name='GX_control', format='E', array=GX[0,:,0]),
                    fits.Column(name='GX', format='E', array=GX[0,:,1]),
                    fits.Column(name='GX_reduced', format='E', array=GX[0,:,2]),
                    fits.Column(name='K_control', format='E', array=K[0,:,0]),
                    fits.Column(name='K', format='E', array=K[0,:,1]),
                    fits.Column(name='K_reduced', format='E', array=K[0,:,2])]
                    
                        
            table_cov = [fits.Column(name='COV_GT_control', format='E', array=COV_Gtc.flatten()),
                        fits.Column(name='COV_GT', format='E', array=COV_Gt.flatten()),
                        fits.Column(name='COV_GT_reduced', format='E', array=COV_Gtr.flatten()),
                        fits.Column(name='COV_GX_control', format='E', array=COV_Gxc.flatten()),
                        fits.Column(name='COV_GX', format='E', array=COV_Gx.flatten()),
                        fits.Column(name='COV_GX_reduced', format='E', array=COV_Gxr.flatten()),
                        fits.Column(name='COV_K_control', format='E', array=COV_Kc.flatten()),
                        fits.Column(name='COV_K', format='E', array=COV_K.flatten()),
                        fits.Column(name='COV_K_reduced', format='E', array=COV_Kr.flatten())]
    
    
        else:
            # COMPUTING PROFILE        
            Ninbin[DSIGMAwsum_T == 0] = 1.
                    
            Sigma     = (SIGMAwsum/Ninbin)
            DSigma_T  = (DSIGMAwsum_T/Ninbin)
            DSigma_X  = (DSIGMAwsum_X/Ninbin)
            
            
            # COMPUTE COVARIANCE
    
            COV_S   = cov_matrix(Sigma[1:,:])
            COV_DSt  = cov_matrix(DSigma_T[1:,:])
            COV_DSx  = cov_matrix(DSigma_X[1:,:])

            # WRITING OUTPUT FITS FILE
            
            table_pro = [fits.Column(name='Rp', format='E', array=R),
                    fits.Column(name='Sigma', format='E', array=Sigma[0]),
                    fits.Column(name='DSigma_T', format='E', array=DSigma_T[0]),
                    fits.Column(name='DSigma_X', format='E', array=DSigma_X[0]),
                    fits.Column(name='Ninbin', format='E', array=Ninbin[0])]
                    
                        
            table_cov = [fits.Column(name='COV_DST', format='E', array=COV_DSt.flatten()),
                        fits.Column(name='COV_S', format='E', array=COV_S.flatten()),
                        fits.Column(name='COV_DSX', format='E', array=COV_DSx.flatten())]
        
            #table_cov = [fits.Column(name='Sigma', format='E', array=Sigma.flatten()),
            #            fits.Column(name='DSigma_T', format='E', array=DSigma_T.flatten()),
            #            fits.Column(name='DSIgma_X', format='E', array=DSIgma_X.flatten())]

        tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_pro))
        tbhdu_cov = fits.BinTableHDU.from_columns(fits.ColDefs(table_cov))
        
        
        primary_hdu = fits.PrimaryHDU(header=h)
        
        hdul = fits.HDUList([primary_hdu, tbhdu_pro, tbhdu_cov])
        
        hdul.writeto(f'../{output_file}',overwrite=True)
                
        tfin = time.time()
        
        print(f'TOTAL TIME {(tfin-tini)/60.}')
        


main(lcat, sample, Rv_min, Rv_max, rho1_min, rho1_max, rho2_min, rho2_max, z_min, z_max, domap, RIN, ROUT, ndots, ncores, idlist, hcosmo, addnoise, FLAG)
