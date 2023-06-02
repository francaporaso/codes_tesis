#tenemos el centro, queremos los halos en una esfera cetrada ahi y hacer el perfil 3D

#una forma facil es pedir que sqrrt(x²+y²+z²) este entre R1 y R2 -> readaptar para quedarse con
#las galaxias en el cubo -> en vez de ra,dec,reds en xyz y cuente halos en cascarones esfericos
#para sacar una dist de densidad

# rho = masa(r1<r<r2)/V_cascaron

#Masa = # particulas*masa_particula 

# #particulas = \sum n_part_en_cada_halo

#necesitamos si es central o satelite, pos del halo, masa del halo

#vamos sumando masa de halos en cada region
#-> Masa = \sum masa_halo

#para sumar una sola vez vamos sumando los halos de galaxias centrales

import sys
sys.path.append('../lens_codes_v3.7')
import time
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import LambdaCDM
from astropy.wcs import WCS
from maria_func import *
# from fit_profiles_curvefit import *
# from astropy.stats import bootstrap
# from astropy.utils import NumpyRNGContext
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from multiprocessing import Pool, Process
import argparse
from astropy.constants import G,c,M_sun,pc
# from scipy import stats
# from models_profiles import Gamma

def partial_profile(x_void, y_void, z_void, Rv,
                    RIN, ROUT, ndots, h):


        ndots = int(ndots)      
        Rv   = Rv/h
        cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)
        
        delta = 2.*ROUT*Rv    
        
        mask = (np.abs(M.xhalo - x_void) < delta) & (np.abs(M.yhalo - y_void) < delta) & (np.abs(M.zhalo - z_void) < delta) & (M.flag_central == 0)
        catdata = M[mask]

        #masas de los halos
        M_halo = 10. **(catdata.lmhalo)
        
        #distancia al centro de cada halo
        r_halo = np.sqrt(np.square(x_void - catdata.xhalo) + np.square(y_void - catdata.yhalo) + np.square(z_void - catdata.zhalo)) 
        
        Ntot = len(catdata)
        bines = np.linspace(RIN*Rv, ROUT*Rv, num=ndots+1)
        dig = np.digitize(r_halo, bines)
                
        Mshell  = np.empty(ndots)
        N_inbin = np.empty(ndots)
                                             
        for nbin in np.arange(ndots):
                mbin = dig == nbin+1                    
                
                Mshell[nbin] = np.sum(M_halo[mbin])
                N_inbin[nbin] = np.count_nonzero(mbin)

        Vshell = np.array([(bines[j+1]**3-bines[j]**3)*4*np.pi/3 for j in np.arange(ndots)])
        
        output = np.array([Mshell, Vshell, N_inbin, Ntot], dtype=object)        
        return output


if __name__ == '__main__':
        folder = '/mnt/simulations/MICE/'
        S      = fits.open(folder+'MICE_sources_HSN_withextra.fits')[1].data #propiedades de galaxias fuente

        M =  fits.open('../cats/MICE/micecat2_halos.fits')[1].data #propiedades de halos