import sys, os
import numpy as np
import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate
from models_profiles import *
from astropy import units as u
from astropy.constants import G,c,M_sun, pc

cvel = c.value;   # Speed of light (m.s-1)
G    = G.value;   # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

class Delta_Sigma_fit:
	# R en Mpc, D_Sigma M_Sun/pc2
	#Ecuacion 15 (g(x)/2)

    def __init__(self,R,D_Sigma,err,z, cosmo,fitc = False):

        roc_mpc = cosmo.critical_density(z).to(u.kg/(u.Mpc)**3).value

        xplot   = np.arange(0.001,R.max()+1.,0.001)
        if fitc:
            def NFW_profile(R,R200,c200):
                M200 = M200_NFW(R200,z,cosmo)
                return Delta_Sigma_NFW(R,z,M200,c200,cosmo=cosmo)
                
            try:
            
                NFW_out = curve_fit(NFW_profile,R,D_Sigma,sigma=err,absolute_sigma=True)
                pcov    = NFW_out[1]
                perr    = np.sqrt(np.diag(pcov))
                e_R200  = perr[0]
                e_c200  = perr[1]
                R200    = NFW_out[0][0]
                c200    = NFW_out[0][1]
                M200    = M200_NFW(R200,z,cosmo)
                
                ajuste  = NFW_profile(R,R200,c200)
                chired  = chi_red(ajuste,D_Sigma,err,2)	
                
                yplot   = NFW_profile(xplot,R200,c200)
                
            except:
                
                
                print('WARNING: Fit was not performed')
                
                e_R200  = -999.
                e_c200  = -999.
                R200    = -999.
                c200    = -999.
                M200    = -999.                
                chired  = -999.
                
                yplot   = xplot

    
        else:
            
            def NFW_profile(R,R200):
                M200 = M200_NFW(R200,z,cosmo)
                return Delta_Sigma_NFW(R,z=z,M200=M200,cosmo=cosmo)
            
            try:
                NFW_out = curve_fit(NFW_profile,R,D_Sigma,sigma=err,absolute_sigma=True)
                e_R200  = np.sqrt(NFW_out[1][0][0])
                R200    = NFW_out[0][0]
                
                ajuste  = NFW_profile(R,R200)
                
                chired  = chi_red(ajuste,D_Sigma,err,1)	
                
                yplot   = NFW_profile(xplot,R200)
                
                #calculo de c usando la relacion de Duffy et al 2008
                M200   = M200_NFW(R200,z,cosmo)
                
                c200   = c200_duffy(M200*cosmo.h,z)
                e_c200 = 0.
            
            except:
                
                print('WARNING: Fit was not performed')
                
                e_R200  = -999.
                e_c200  = -999.
                R200    = -999.
                c200    = -999.
                M200    = -999.                
                chired  = -999.
                
                yplot   = xplot

        
        e_M200 =((800.0*np.pi*roc_mpc*(R200**2))/(Msun))*e_R200

        self.xplot = xplot
        self.yplot = yplot
        self.chi2  = chired
        self.R200 = R200
        self.error_R200 = e_R200
        self.M200 = M200
        self.error_M200 = e_M200
        self.c200 = c200
        self.error_c200 = e_c200



class Sigma_fit:
	# R en Mpc, D_Sigma M_Sun/pc2
	#Ecuacion 15 (g(x)/2)

    def __init__(self,R,Sigma,err,z, cosmo,fitc = False):

        roc_mpc = cosmo.critical_density(z).to(u.kg/(u.Mpc)**3).value

        xplot   = np.arange(0.001,R.max()+1.,0.001)
        if fitc:
            def NFW_profile(R,R200,c200):
                M200 = M200_NFW(R200,z,cosmo)
                return Sigma_NFW(R,z,M200,c200,cosmo=cosmo)
            
            try:
                NFW_out = curve_fit(NFW_profile,R,Sigma,sigma=err,absolute_sigma=True,bounds=([0,0],[4,50]))
                pcov    = NFW_out[1]
                perr    = np.sqrt(np.diag(pcov))
                e_R200  = perr[0]
                e_c200  = perr[1]
                R200    = NFW_out[0][0]
                c200    = NFW_out[0][1]
                M200    = M200_NFW(R200,z,cosmo)
                
                ajuste  = NFW_profile(R,R200,c200)
                chired  = chi_red(ajuste,Sigma,err,2)	
                
                # compute residuals Eq6-Meneghetti et al 2014
                res     = np.sqrt(chi_red(np.log10(ajuste),np.log10(Sigma),np.ones(len(Sigma)),1))

                
                yplot   = NFW_profile(xplot,R200,c200)
            
            except:
                
                print('WARNING: Fit was not performed')
                
                e_R200  = -999.
                e_c200  = -999.
                R200    = -999.
                c200    = -999.
                M200    = -999.                
                chired  = -999.
                res     = -999.
                
                yplot   = xplot
                
    
        else:
            
            def NFW_profile(R,R200):
                M200 = M200_NFW(R200,z,cosmo)
                return Sigma_NFW(R,z=z,M200=M200,cosmo=cosmo)

            try:
                NFW_out = curve_fit(NFW_profile,R,Sigma,sigma=err,absolute_sigma=True)
                e_R200  = np.sqrt(NFW_out[1][0][0])
                R200    = NFW_out[0][0]
                
                ajuste  = NFW_profile(R,R200)
                
                chired  = chi_red(ajuste,Sigma,err,1)	
                
                yplot   = NFW_profile(xplot,R200)
                
                #calculo de c usando la relacion de Duffy et al 2008
                M200   = M200_NFW(R200,z,cosmo)
                
                c200   = c200_duffy(M200*cosmo.h,z)
                e_c200 = 0.
                
                # compute residuals Eq6-Meneghetti et al 2014
                res     = np.sqrt(chi_red(np.log10(ajuste),np.log10(Sigma),np.ones(len(Sigma)),0))

            
            except:
                
                print('WARNING: Fit was not performed')
                
                e_R200  = -999.
                e_c200  = -999.
                R200    = -999.
                c200    = -999.
                M200    = -999.                
                chired  = -999.
                res     = -999.
                
                yplot   = xplot

        
        e_M200 =((800.0*np.pi*roc_mpc*(R200**2))/(Msun))*e_R200

        self.xplot = xplot
        self.yplot = yplot
        self.chi2  = chired
        self.R200 = R200
        self.res  = res
        self.error_R200 = e_R200
        self.M200 = M200
        self.error_M200 = e_M200
        self.c200 = c200
        self.error_c200 = e_c200


class rho_fit:
	# R en Mpc, D_Sigma M_Sun/pc2
	#Ecuacion 15 (g(x)/2)

    def __init__(self,R,rho,err,z, cosmo,fitc = False):

        roc_mpc = cosmo.critical_density(z).to(u.kg/(u.Mpc)**3).value

        xplot   = np.arange(0.001,R.max()+1.,0.001)
        if fitc:
            def NFW_profile(R,R200,c200):
                M200 = M200_NFW(R200,z,cosmo)
                return rho_NFW(R,z,M200,c200,cosmo=cosmo)
            
            
            try:
                
                NFW_out = curve_fit(NFW_profile,R,rho,sigma=err,absolute_sigma=True,bounds=([0,0],[4,50]))
                pcov    = NFW_out[1]
                perr    = np.sqrt(np.diag(pcov))
                e_R200  = perr[0]
                e_c200  = perr[1]
                R200    = NFW_out[0][0]
                c200    = NFW_out[0][1]
                M200    = M200_NFW(R200,z,cosmo)
                
                ajuste  = NFW_profile(R,R200,c200)
                chired  = chi_red(ajuste,rho,err,2)	
                
                # compute residuals Eq6-Meneghetti et al 2014
                res     = np.sqrt(chi_red(np.log10(ajuste),np.log10(rho),np.ones(len(rho)),1))
                
                yplot   = NFW_profile(xplot,R200,c200)
            
            except:
                
                print('WARNING: Fit was not performed')
                e_R200  = -999.
                e_c200  = -999.
                R200    = -999.
                c200    = -999.
                M200    = -999.
                
                chired  = -999.
                res     = -999.
                
                yplot   = xplot
    
        else:
            
            def NFW_profile(R,R200):
                M200 = M200_NFW(R200,z,cosmo)
                return rho_NFW(R,z=z,M200=M200,cosmo=cosmo)
                
            try:
            
                NFW_out = curve_fit(NFW_profile,R,rho,sigma=err,absolute_sigma=True)
                e_R200  = np.sqrt(NFW_out[1][0][0])
                R200    = NFW_out[0][0]
                
                ajuste  = NFW_profile(R,R200)
                
                chired  = chi_red(ajuste,rho,err,1)	
                # compute residuals Eq6-Meneghetti et al 2014
                res     = np.sqrt(chi_red(np.log10(ajuste),np.log10(rho),np.ones(len(rho)),0))
                yplot   = NFW_profile(xplot,R200)
                
                #calculo de c usando la relacion de Duffy et al 2008
                M200   = M200_NFW(R200,z,cosmo)
                c200   = c200_duffy(M200*cosmo.h,z)
                e_c200 = 0.
                
            except:
                
                print('WARNING: Fit was not performed')
                
                e_R200  = -999.
                e_c200  = -999.
                R200    = -999.
                c200    = -999.
                M200    = -999.
                
                chired  = -999.
                res     = -999.
                
                yplot   = xplot
                
        
        e_M200 =((800.0*np.pi*roc_mpc*(R200**2))/(Msun))*e_R200

        self.xplot = xplot
        self.yplot = yplot
        self.chi2  = chired
        self.R200 = R200
        self.error_R200 = e_R200
        self.M200 = M200
        self.error_M200 = e_M200
        self.c200 = c200
        self.res  = res
        self.error_c200 = e_c200
        
        
class log_rho_fit:
	# R en Mpc, D_Sigma M_Sun/pc2
	#Ecuacion 15 (g(x)/2)

    def __init__(self,R,logrho,err,z, cosmo,fitc = False):

        roc_mpc = cosmo.critical_density(z).to(u.kg/(u.Mpc)**3).value

        xplot   = np.arange(0.001,R.max()+1.,0.001)
        if fitc:
            def NFW_profile(R,R200,c200):
                M200 = M200_NFW(R200,z,cosmo)
                return np.log10(rho_NFW(R,z,M200,c200,cosmo=cosmo))
            
            
            try:
                
                NFW_out = curve_fit(NFW_profile,R,logrho,sigma=err,absolute_sigma=True,bounds=([0,0],[4,50]))
                pcov    = NFW_out[1]
                perr    = np.sqrt(np.diag(pcov))
                e_R200  = perr[0]
                e_c200  = perr[1]
                R200    = NFW_out[0][0]
                c200    = NFW_out[0][1]
                M200    = M200_NFW(R200,z,cosmo)
                
                ajuste  = NFW_profile(R,R200,c200)
                chired  = chi_red(ajuste,logrho,err,2)	
                
                yplot   = NFW_profile(xplot,R200,c200)
            
            except:
                
                print('WARNING: Fit was not performed')
                e_R200  = -999.
                e_c200  = -999.
                R200    = -999.
                c200    = -999.
                M200    = -999.
                
                chired  = -999.
                
                yplot   = xplot
    
        else:
            
            def NFW_profile(R,R200):
                M200 = M200_NFW(R200,z,cosmo)
                return np.log10(rho_NFW(R,z=z,M200=M200,cosmo=cosmo))
                
            try:
            
                NFW_out = curve_fit(NFW_profile,R,logrho,sigma=err,absolute_sigma=True)
                e_R200  = np.sqrt(NFW_out[1][0][0])
                R200    = NFW_out[0][0]
                
                ajuste  = NFW_profile(R,R200)
                
                chired  = chi_red(ajuste,logrho,err,1)	
                yplot   = NFW_profile(xplot,R200)
                
                #calculo de c usando la relacion de Duffy et al 2008
                M200   = M200_NFW(R200,z,cosmo)
                c200   = c200_duffy(M200*cosmo.h,z)
                e_c200 = 0.
                
            except:
                
                print('WARNING: Fit was not performed')
                
                e_R200  = -999.
                e_c200  = -999.
                R200    = -999.
                c200    = -999.
                M200    = -999.
                
                chired  = -999.
                
                yplot   = xplot
                
        
        e_M200 =((800.0*np.pi*roc_mpc*(R200**2))/(Msun))*e_R200

        self.xplot = xplot
        self.yplot = yplot
        self.chi2  = chired
        self.R200 = R200
        self.error_R200 = e_R200
        self.M200 = M200
        self.error_M200 = e_M200
        self.c200 = c200
        self.error_c200 = e_c200        
        
class rho_fit_colossus:
	# R en Mpc, D_Sigma M_Sun/pc2
	#Ecuacion 15 (g(x)/2)

    def __init__(self,r,rho,err,z, cosmo,otype = 'critical'):

        def NFW_profile(R,lMass,con):
            return np.array(NFW.NFW(10**lMass,con,z,cosmology=cosmo,overdensity_type=otype).density(R))
            
        NFW_out = curve_fit(NFW_profile,r,rho,sigma=err,absolute_sigma=True, bounds=([11,0],[16,10]))
        lM_200    = NFW_out[0][0]
        c_200   = NFW_out[0][1]
            
        self.M = 10**lM_200
        self.c = c_200

