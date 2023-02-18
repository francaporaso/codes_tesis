import sys
sys.path.append('../lens_codes_v3.7')
from maria_func import *
import timeit
import numpy as np
from astropy.cosmology import LambdaCDM
from astropy.constants import G,c,M_sun,pc
from astropy.io import fits

#parameters
cvel = c.value;    # Speed of light (m.s-1)
G    = G.value;    # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value    # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

RIN      = 0.05
ROUT     = 5.0
ndots    = 40
hcosmo   = 1
addnoise = False

folder = '/mnt/simulations/MICE/'
S      = fits.open(folder+'MICE_sources_HSN_withextra.fits')[1].data

L = np.loadtxt(folder+'/voids_MICE.dat')

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

    return (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)


RA0,DEC0,Z,Rv,RIN,ROUT,ndots,h,addnoise = L[2][0], L[3][0],L[4][0],L[1][0],RIN,ROUT,ndots,hcosmo,addnoise

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
sigma_c = SigmaCrit(Z, catdata.z_cgal_v)

rads, theta, tes1,tes2 = eq2p2(np.deg2rad(catdata.ra_gal),
                                np.deg2rad(catdata.dec_gal),
                                np.deg2rad(RA0),
                                np.deg2rad(DEC0))
                       
del tes1, tes2

e1     = catdata.gamma1
e2     = -1.*catdata.gamma2

#get tangential ellipticities 
et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
#get cross ellipticities
ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
# '''
k  = catdata.kappa*sigma_c
  
del e1, e2, theta, sigma_c
r = (np.rad2deg(rads)*3600*KPCSCALE)/(Rv*1000.)
del rads

Ntot = len(catdata)
del catdata    

bines = np.linspace(RIN,ROUT,num=ndots+1)
dig = np.digitize(r, bines)


def test1():
    SIGMAwsum = np.array([k[dig==nbin+1].sum() for nbin in range(0,ndots)])


def test2():
    SIGMAwsum_2 = np.zeros(ndots)
    for nbin in range(0,ndots):
        mbin = dig==nbin+1
        SIGMAwsum_2[nbin] = k[mbin].sum()

if __name__ == '__main__':
    print('Testeando...')

    list_comp = min(timeit.repeat(test1, repeat=10, number=10_000))
    for_loop  = min(timeit.repeat(test2, repeat=10, number=10_000))

    print(f'Metodo de list comprehension: {round(list_comp, 3)}')
    print(f'Metodo de for loop: {round(for_loop, 3)}')

    percent_faster = (1 - (for_loop/list_comp))*100
    print(f'for loop es {round(percent_faster, 2):,}% mas rapido')
     

