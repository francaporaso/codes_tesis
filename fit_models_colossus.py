import sys, os
import numpy as np
import sys
from scipy.optimize import curve_fit
from colossus.cosmology import cosmology  
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MICE', params)
cosmo = cosmology.setCosmology('MICE')
from colossus.halo import profile_nfw
from colossus.halo import profile_einasto


class Sigma_fit:
	# R en Mpc, Sigma M_Sun/Mpc2
	

    def __init__(self,R,Sigma,err,z,model='NFW',Min=1.e13,cin=3.,fit_alpha=True):

        Min = np.max([1.e11,Min])
        Min = np.min([1.e15,Min])
        cin = np.max([1,cin])
        cin = np.min([10,cin])


        bines = np.round(np.logspace(np.log10(100),np.log10(10000),num=41),0)
        xplot = (bines[:-1] + np.diff(bines)*0.5)*1.e-3
        
        if model == 'NFW':

            pmodel = profile_nfw.NFWProfile(M = Min, c = cin, z = z, mdef = '200c')
            
        elif model == 'Einasto':
            
            pmodel = profile_einasto.EinastoProfile(M = Min, c = cin, z = z, mdef = '200c')
            

        try:
            
            BIN= len(Sigma)
            
            if model == 'NFW':
                
                out = pmodel.fit(R*1000., Sigma*(1.e3**2), 'Sigma', q_err = err*(1.e3**2), tolerance = 1.e-04,verbose=False)
                
                rhos,rs = out['x']
                prof = profile_nfw.NFWProfile(rhos = rhos, rs = rs)
                alpha = -999.
                Ndoff = float(BIN - 2)
                
            elif model == 'Einasto':
                
                if fit_alpha:
                    out = pmodel.fit(R*1000., Sigma*(1.e3**2), 'Sigma', q_err = err*(1.e3**2), tolerance = 1.e-04,verbose=False)
                    rhos,rs,alpha = out['x']
                else:
                    out = pmodel.fit(R*1000., Sigma*(1.e3**2), 'Sigma', q_err = err*(1.e3**2), tolerance = 1.e-04,verbose=False,mask=[True,True,False])
                    rhos,rs = out['x']
                    alpha = pmodel.par['alpha'] #alpha = p.par['alpha'] , idea como en ln 183 a 212
                
                
                prof = profile_einasto.EinastoProfile(rhos = rhos, rs = rs, alpha = alpha)
                Ndoff = float(BIN - 3)
            
            ajuste = prof.surfaceDensity(R*1000.)/(1.e3**2)
            yplot  = prof.surfaceDensity(xplot*1000.)/(1.e3**2)
            
            
            
            res=np.sqrt(((((np.log10(ajuste)-np.log10(Sigma))**2)).sum())/Ndoff)

            M200 = prof.MDelta(z,'200c')
            c200 = prof.RDelta(z,'200c')/rs
        
        except:
            yplot = xplot
            res   = -999.
            M200  = -999.
            c200  = -999.
            alpha = -999.
        

        self.xplot = xplot
        self.yplot = yplot
        self.res  = res
        self.M200 = M200
        self.c200 = c200
        self.alpha = alpha


class Delta_Sigma_fit:
	# R en Mpc, Sigma M_Sun/Mpc2
	

    def __init__(self,R,DSigma,err,z,model='NFW',Min=1.e13,cin=3.,fit_alpha=True):

        Min = np.max([1.e11,Min])
        Min = np.min([1.e15,Min])
        cin = np.max([1,cin])
        cin = np.min([10,cin])

        bines = np.round(np.logspace(np.log10(100),np.log10(10000),num=41),0)
        xplot = (bines[:-1] + np.diff(bines)*0.5)*1.e-3
        
        if model == 'NFW':

            pmodel = profile_nfw.NFWProfile(M = Min, c = cin, z = z, mdef = '200c')
            
        elif model == 'Einasto':
            
            pmodel = profile_einasto.EinastoProfile(M = Min, c = cin, z = z, mdef = '200c')
            

        try:
            
            BIN = len(DSigma)
            
            if model == 'NFW':
                
                out = pmodel.fit(R*1000., DSigma*(1.e3**2), 'DeltaSigma', q_err = err*(1.e3**2), tolerance = 1.e-04,verbose=False)
                
                rhos,rs = out['x']
                prof = profile_nfw.NFWProfile(rhos = rhos, rs = rs)
                alpha = -999.
                Ndoff = float(BIN - 2)
                
            elif model == 'Einasto':
                
                if fit_alpha:
                    out = pmodel.fit(R*1000., DSigma*(1.e3**2), 'DeltaSigma', q_err = err*(1.e3**2), tolerance = 1.e-04,verbose=False)
                    rhos,rs,alpha = out['x']
                    Ndoff = float(BIN - 2)
                else:
                    out = pmodel.fit(R*1000., DSigma*(1.e3**2), 'DeltaSigma', q_err = err*(1.e3**2), tolerance = 1.e-04,verbose=False,mask=[True,True,False])
                    rhos,rs = out['x']
                    alpha = pmodel.par['alpha'] #alpha = p.par['alpha'] , idea como en ln 183 a 212
                    Ndoff = float(BIN - 3)
                
                prof = profile_einasto.EinastoProfile(rhos = rhos, rs = rs, alpha = alpha)
                
            
            ajuste = prof.deltaSigma(R*1000.)/(1.e3**2)
            yplot  = prof.deltaSigma(xplot*1000.)/(1.e3**2)
            
            
            
            res=np.sqrt(((((np.log10(ajuste)-np.log10(DSigma))**2)).sum())/Ndoff)

            M200 = prof.MDelta(z,'200c')
            c200 = prof.RDelta(z,'200c')/rs
        
        except:
            yplot = xplot
            res   = -999.
            M200  = -999.
            c200  = -999.
            alpha = -999.
        

        self.xplot = xplot
        self.yplot = yplot
        self.res  = res
        self.M200 = M200
        self.c200 = c200
        self.alpha = alpha


class rho_fit:
	# R en Mpc, rho M_Sun/Mpc3
	

    def __init__(self,R,rho,err,z,model='NFW',Min=1.e13,cin=3.,fit_alpha=True):

        Min = np.max([1.e11,Min])
        Min = np.min([1.e15,Min])
        cin = np.max([1,cin])
        cin = np.min([10,cin])


        xplot   = np.arange(0.001,R.max()+1.,0.001)

        if model == 'NFW':

            p = profile_nfw.NFWProfile(M = Min, c = cin, z = z, mdef = '200c')
            
        elif model == 'Einasto':
            
            p = profile_einasto.EinastoProfile(M = Min, c = cin, z = z, mdef = '200c')
            

        try:
        
            BIN= len(rho)
            
            if model == 'NFW':
                
                out = p.fit(R*1000., rho/(1.e3**3), 'rho', q_err = err/(1.e3**3), tolerance = 1.e-04,verbose=False)
                
                rhos,rs = out['x']
                prof = profile_nfw.NFWProfile(rhos = rhos, rs = rs)
                alpha = -999.
                Ndoff = float(BIN - 2)
                
            elif model == 'Einasto':
                
                
                if fit_alpha:
                    out = p.fit(R*1000., rho/(1.e3**3), 'rho', q_err = err/(1.e3**3), tolerance = 1.e-04,verbose=False)
                    rhos,rs,alpha = out['x']
                else:
                    out = p.fit(R*1000., rho/(1.e3**3), 'rho', q_err = err/(1.e3**3), tolerance = 1.e-04,verbose=False,mask=[True,True,False])
                    rhos,rs = out['x']
                    alpha = p.par['alpha']
                    
                prof = profile_einasto.EinastoProfile(rhos = rhos, rs = rs, alpha = alpha)
                Ndoff = float(BIN - 3)
            
            ajuste = prof.density(R*1000.)*(1.e3**3)
            yplot  = prof.density(xplot*1000.)*(1.e3**3)
            
            res=np.sqrt(((((np.log10(ajuste)-np.log10(rho))**2)).sum())/Ndoff)
            
            M200 = prof.MDelta(z,'200c')
            c200 = prof.RDelta(z,'200c')/rs
        
        except:
            yplot = xplot
            res   = -999.
            M200  = -999.
            c200  = -999.
            alpha = -999.
        

        self.xplot = xplot
        self.yplot = yplot
        self.res  = res
        self.M200 = M200
        self.c200 = c200
        self.alpha = alpha

