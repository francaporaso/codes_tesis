import numpy as np
from scipy.integrate import simpson, quad, cumulative_trapezoid
from scipy.special import erf

from fitting.constants import *

def logistic(x, x0=1, k=10):
    return (1.0+np.exp(-2.0*k*(x-x0)))**(-1)

# ==================== 
# Base models: sigma, delta_sigma integration
# ==================== 

class BaseModelFast:

    def density_contrast(self):
        ''' density contrast delta(r) = rho(x)/rho_mean - 1 '''
        raise NotImplementedError('Must be defined in child class')
    
    # TODO:  - agregar parametro ctte Sigma_0 a sigma
    def sigma(self, R, *params):

        u_grid = np.linspace(0.0, 100.0, 500)
        radius_grid = np.hypot(u_grid[None, :], R[:, None])
        integrand_grid = self.density_contrast(radius_grid, *params)
        result = 2.0 * simpson(integrand_grid, u_grid, axis=1)
        
        return result

    def delta_sigma(self, R, *params):

        num_theta=200
        num_x=1000
        
        x_grid = np.linspace(1e-5, R.max(), num_x)
        #x_grid = np.geomspace(1e-5, R.max(), num_x)
        integrand_x = x_grid**2 * self.density_contrast(x_grid, *params)
        cumulative = cumulative_trapezoid(integrand_x, x_grid, initial=0.0)
        I1_interp = np.interp(R, x_grid, cumulative)
        
        theta = np.linspace(0.0, np.pi/2.0 - 1e-6, num_theta)
        denom = 4.0 * np.sin(theta) + 3.0 - np.cos(2.0 * theta)
        
        r_mesh = R[:, None] / np.cos(theta[None, :])
        
        integrand_theta = self.density_contrast(r_mesh, *params) / denom[None, :]
        I2 = simpson(integrand_theta, theta, axis=1)

        return (4.0 / R**2) * I1_interp - 4.0 * R * I2
    
class BaseModelQuad:

    def delta_sigma(self, R, *params):

        x_grid = np.linspace(0.0, R.max(), 1000)
        integrand = x_grid**2 * self.density_contrast(x_grid, *params)
        cumulative = cumulative_trapezoid(integrand, x_grid, initial=0.0)

        I1_interp = np.interp(R, x_grid, cumulative)
        
        result = np.zeros_like(R)

        for i, Ri in enumerate(R):
            def integrand2(theta):
                return self.density_contrast(Ri/np.cos(theta), *params) / (4.0*np.sin(theta) + 3 - np.cos(2.0*theta))

            I2,_ = quad(integrand2, 0.0, np.pi/2.0 - 1e-6)
            result[i] = (4.0/Ri**2)*I1_interp[i] - 4.0*Ri*I2

        return result

# ==================== 
# Density Models
# ==================== 

class HSW(BaseModelFast):
    def density_contrast(self, r, dc, rs, a, b):
        return dc*(1-(r/rs)**a)/(1+r**b)

class B15(BaseModelFast):
    def density_contrast(self, r, dc, rs, rv, a, b):
        return dc*(1-(r/rs)**a)/(1+(r/rv)**b)

class ModifiedLW(BaseModelFast):
    def density_contrast(self, r, dc, dw, rw):
        rv = 1.0
        return np.where(r < rv, dc+(dw-dc)*(r/rv)**3, np.where(r > rw, 0.0, dw))

class TopHat(BaseModelFast):
    def density_contrast(self, r, dc, dw, rw):
        rv = 1.0
        return np.where(r < rv, dc, np.where(r > rw, 0.0, dw))
    
class Paz13(BaseModelFast):
    def density_contrast(self, r, S, Rs, P, W):
        x = np.log10(r/Rs)
        asym_gauss = np.where(r<Rs, np.exp(-S*x**2), np.exp(-W*x**2))

        Delta = 0.5*(erf(S*x)-1) + P*asym_gauss
        
        t1 = S*np.exp(-(S*x)**2)/(SQPI*r)
        t2 = (-2.0*P*x/r) * asym_gauss
        Delta_prime = t1+t2

        return Delta+1/3*r*Delta_prime

class THLogistic(BaseModelFast):
    def density_contrast(self, r, dc, rw, dw):
        k=15
        return dc*(1.0-logistic(r, x0=1, k=k)) + dw*(logistic(r, x0=1, k=k) - logistic(r, x0=rw, k=k))

models_dict = {
    'HSW':HSW(),
    'TH':TopHat(),
    'mLW':ModifiedLW(),
    'B15':B15(),
}
default_limits = {
    'HSW':{'dc':(-1.0,0.0),'rs':(0.5,5.0),'a':(1.0,15.0),'b':(1.0,15.0)},
    'B15':{'dc':(-1.0,0.0),'rs':(0.5,5.0),'rv':(0.5,5.0),'a':(1.0,15.0),'b':(1.0,15.0)},
    'TH':{'dc':(-1.0,0.0),'dw':(-0.5,0.5),'rw':(1.0,5.0)},
    'mLW':{'dc':(-1.0,0.0),'dw':(-0.5,0.5),'rw':(1.0,5.0)},
}
default_guess = {
    'HSW':(-0.7,0.9,3.0,6.0),
    'B15':(-0.7,0.9,1.0,3.0,6.0),
    'TH':(-0.7,0.2,2.5),
    'mLW':(-0.7,0.2,2.5),
}
