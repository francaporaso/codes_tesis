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

RIN      = 0.1
ROUT     = 3.0
ndots    = 10
hcosmo   = 1.
addnoise = False

folder = '/mnt/simulations/MICE/'
S      = fits.open(folder+'MICE_sources_HSN_withextra.fits')[1].data

L = np.loadtxt(folder+'/voids_MICE.dat').T

""" def SigmaCrit(zl, zs, h=1.):
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

    return (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun) """


RA0,DEC0,Z,Rv,h = L[2][0], L[3][0],L[4][0],L[1][0],hcosmo

#ndots = int(ndots)
Rv   = Rv/h
cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)
dl  = cosmo.angular_diameter_distance(Z).value
KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0

delta = ROUT*(Rv*1000.)*5./(3600*KPCSCALE)


""" del mask
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
 """

def test1():
    mask = (S.ra_gal < (RA0+delta))&(S.ra_gal > (RA0-delta))&(S.dec_gal > (DEC0-delta))&(S.dec_gal < (DEC0+delta))&(S.z_cgal_v > (Z+0.1))
    catdata = S[mask]


# def test2():
#     N_inbin = np.array([len(et[dig==nbin+1]) for nbin in range(0,ndots)])

if __name__ == '__main__':
    print('Testeando...')

    mascara = min(timeit.repeat(test1, repeat=10, number=10_000))
    #ref  = min(timeit.repeat(test2, repeat=10, number=10_000))

    print(f'Con np.count_nonzero: {round(otra_ver, 5)}')
    # print(f'Referencia len(et[mbin]): {round(ref, 5)}')

    # percent_faster = (1 - (ref/otra_ver))*100
    # print(f'Referencia {round(percent_faster, 3):,}% mas rapido')
     

