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
from fit_profiles_curvefit import *
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from multiprocessing import Pool, Process
import argparse
from astropy.constants import G,c,M_sun,pc
from scipy import stats
from models_profiles import Gamma

def partial_profile(x,y,z,Rv,
                    RIN,ROUT,ndots,h):


        ndots = int(ndots)

        Rv   = Rv/h
        cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)
        
        delta = ROUT*Rv    
        
        mask = (np.abs(x_halo - x_void) < delta) & (np.abs(y_halo - y_void) < delta) & (np.abs(z_halo - z_void) < delta)        

        catdata = S[mask]

        #sacamos las masas de los halos
        M = 10**S.halo_mass_exponent


        Ntot = len(catdata)        

        bines = np.linspace(RIN,ROUT,num=ndots+1)
        dig = np.digitize(r,bines)
                
        RHOsum  = np.empty(ndots)
        N_inbin = np.empty(ndots)
        V = np.empty(ndots) #vol de la cascara
                                             
        for nbin in range(ndots):
                mbin = dig == nbin+1              

                SIGMAwsum[nbin]    = M[mbin].sum()
                 
                N_inbin[nbin]      = np.count_nonzero(mbin)
        
        output = np.array([SIGMAwsum, DSIGMAwsum_T, DSIGMAwsum_X, N_inbin, Ntot], dtype=object)
        #output = (SIGMAwsum, DSIGMAwsum_T, DSIGMAwsum_X, N_inbin, Ntot)
        
        return output
