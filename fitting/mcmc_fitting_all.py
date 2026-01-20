import numpy as np
import pandas as pd
from scipy.integrate import simpson
import emcee
from astropy.io import fits
from astropy.cosmology import LambdaCDM
import time
from multiprocessing import Pool
from scipy.special import erf
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# from scipy.optimize import curve_fit
# import corner
### TODO
# implementar el modelo de Dante, ver notas de BCN

h = 1.0
cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)

def chi_red(ajuste,data,err,gl):
	'''
	Reduced chi**2
	------------------------------------------------------------------
	INPUT:
	ajuste       (float or array of floats) fitted value/s
	data         (float or array of floats) data used for fitting
	err          (float or array of floats) error in data
	gl           (float) grade of freedom (number of fitted variables)
	------------------------------------------------------------------
	OUTPUT:
	chi          (float) Reduced chi**2
	'''

	BIN=len(data)
	chi=((((ajuste-data)**2)/(err**2)).sum())/float(BIN-1-gl)
	return chi

def rho_mean(z):
    '''densidad media en Msun/(pc**2 Mpc)'''
    global cosmo
    p_cr0 = cosmo.critical_density(0).to('Msun/(pc**2 Mpc)').value
    a = cosmo.scale_factor(z)
    out = p_cr0*cosmo.Om0/a**3
    return out

class Likelihood:
    def __init__(self, func, r, y, yerr, limits, redshift):
        self.func = func
        self.r = r
        self.y = y
        self.yerr = yerr
        self.params = list(limits.keys())
        self.limits = limits
        self.rhomean = rho_mean(redshift)

    def log_likelihood(self, theta):
        model = self.func(self.r, *theta)*self.rhomean
        dist = self.y - model
        return -0.5*np.dot(dist, np.dot(self.yerr, dist))
        #return -0.5 * np.sum(((self.y - model)**2 )/self.yerr**2)

    def log_prior(self, theta):
        ### tener cuidado con el orden de lims!
        if np.prod(
            [self.limits[self.params[j]][0] < theta[j] < self.limits[self.params[j]][1] for j in range(len(self.params))],
            dtype=bool
        ): return 0
        return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

class Profile:

    def sigma(self, R, *params):
        chi = np.linspace(0.001, 200.0, 1000)
        vals = self.model(R[None, :], chi[:,None], *params)
        return 2.0*simpson(vals, x=chi, axis=0)

    def mean_sigma(self, R, *params):
        x_grid = np.linspace(1e-5, R.max(), 1000)
        f_grid = self.sigma(x_grid, *params)
        F_vals = np.array([
            simpson(x_grid[x_grid <= Ri] * f_grid[x_grid <= Ri], x=x_grid[x_grid <= Ri])
            for Ri in R
        ])
        return F_vals

    def delta_sigma(self, R, *params):
        anillo = self.sigma(R, *params)
        disco = self.mean_sigma(R, *params)
        return (2 / R**2) * disco - anillo

class newHSW(Profile):
    def __init__(self, limits=None):
        #self.limits_S = {'dc':(-0.99,-0.01), 'rs':(0.1,4.99), 'a':(1.01,9.99), 'b':(1.01,14.99), 'off':(-0.5,0.5)}
        if limits==None:
            self.limits = {'dc':(-0.99,-0.01), 'rs':(0.1,4.99), 'a':(1.01,9.99), 'b':(1.01,14.99)}
        else:
            self.limits = limits

    def model(self, R, chi, dc, rs, a, b):
        r = np.hypot(R, chi)
        return dc*(1-(r/rs)**a)/(1+r**b)

class ErrFunc(Profile):
    def __init__(self, limits=None):
        #self.limits_S = {'S':(0.0,5.0), 'Rs':(0.0,5.0), 'P':(0.0,5.0), 'W':(0.0, 5.0), 'off':(-0.5,0.5)}
        if limits==None:
            self.limits = {'S':(0.0,5.0), 'Rs':(0.0,5.0), 'P':(0.0,5.0), 'W':(0.0, 5.0)}
        else:
            self.limits = limits

    def model(self, R, chi, S, Rs, P, W):
        "chequear notas en cuadernito de cosmosur"

        r = np.hypot(R, chi)
        Theta_sq = np.where(r<Rs, 1/(2*S), 1/(2*W))
        # Theta_cube = np.where(r<Rs, (2*S)**(-3/2), (2*W)**(-3/2))
        # Theta_prime = np.where(r==Rs, 2**(-1/2)*(np.sqrt(S)+np.sqrt(W))/np.sqrt(S*W), 0)
        x = np.log(r/Rs)
        t1 = S*np.exp(-(S*x)**2)/(np.sqrt(np.pi)*r)
        t2 = -P*np.exp(-x**2/(2*Theta_sq))*(x/(r*Theta_sq)) # - x**2*Theta_prime/Theta_cube dentro del parentesis... pero tiene una delta de dirac...
        Delta_prime = t1+t2
        Delta = 0.5*(erf(S*x)-1) + P*np.exp(-0.5*x**2/Theta_sq)
        return Delta+1/3*r*Delta_prime

class HSW:
    def __init__(self, limits=None):
        #self.limits_S = {'dc':(-0.99,-0.01), 'rs':(0.1,4.99), 'a':(1.01,9.99), 'b':(1.01,14.99), 'off':(-0.5,0.5)}

        if limits==None:
            self.limits = {'dc':(-0.99,-0.01), 'rs':(0.1,4.99), 'a':(1.01,9.99), 'b':(1.01,14.99)}
        else:
            self.limits = limits

    def h_integrand(self, chi, R, dc, rs, a, b):
        r = np.sqrt(R**2 + chi**2)
        return dc * (1 - (r / rs)**a) / (1 + r**b)

    def sigma(self, R, dc, rs, a, b, off=0.0):
        if self.fix_off:
            off = 0.0
        chi = np.linspace(0.001, 200.0, 700)
        h_vals = self.h_integrand(chi=chi[:, None], R=R[None, :], dc=dc, rs=rs, a=a, b=b)
        return 2.0*simpson(h_vals, x=chi, axis=0) + off

    def mean_sigma(self, R_vals, dc, rs, a, b):
        x_grid = np.linspace(0.001, R_vals.max(), 700)
        f_grid = self.sigma(x_grid, dc, rs, a, b, off=0.0)
        F_vals = np.array([
            simpson(x_grid[x_grid <= R] * f_grid[x_grid <= R], x=x_grid[x_grid <= R])
            for R in R_vals
        ])
        return F_vals

    def delta_sigma(self, R, dc, rs, a, b):
        anillo = self.sigma(R, dc, rs, a, b, off=0.0)
        disco = self.mean_sigma(R, dc, rs, a, b)
        return (2 / R**2) * disco - anillo

class TopHat:
    def __init__(self, limits):
        #self.limits_S = {'rs':(1.0,5.0), 'dc':(-1.0,0.0), 'd2':(-0.5,0.5), 'off':(-0.5,0.5)}
        if limits==None:
            self.limits = {'rs':(1.0,5.0), 'dc':(-1.0,0.0), 'd2':(-0.5,0.5)}
        else:
            self.limits = limits

    def tophat(self, r, rv, rs, dc, d2):
        '''
        top-hat model. (FALTA CITA PAPER)
        '''
        return np.where(r < rv, dc, np.where(r > rs, 0.0, d2))

    def sigma(self, R, rs, dc, d2, off):
        Rv = 1.
        if Rv>rs:
            return np.inf
        den_integrada = np.where(
            R<Rv,
            np.sqrt(np.abs(Rv**2-R**2))*(dc-d2) + d2*np.sqrt(np.abs(rs**2-R**2)),
            np.where(
                R>rs,
                0.0,
                d2*np.sqrt(np.abs(rs**2-R**2)),
            )
        )
        sigma = den_integrada/Rv + off
        return sigma

    def mean_sigma(self, R_vals, rs, dc, d2):
        x_grid = np.linspace(0.001, R_vals.max(), 700)
        f_grid = self.sigma(x_grid, rs, dc, d2, off=0.0)
        F_vals = np.array([
            simpson(x_grid[x_grid <= R] * f_grid[x_grid <= R], x=x_grid[x_grid <= R])
            for R in R_vals
        ])
        return F_vals

    def delta_sigma(self, R, rs, dc, d2):
        anillo = self.sigma(R, rs, dc, d2, off=0.0)
        disco = self.mean_sigma(R, rs, dc, d2)
        return (2 / R**2) * disco - anillo

class modifiedLW:
    def __init__(self, limits=None):
        #self.limits_S = {'rs':(1.0,5.0), 'dc':(-1.0,0.0), 'd2':(-0.5,0.5), 'off':(-0.5,0.5)}
        if limits==None:
            self.limits = {'rs':(1.0,5.0), 'dc':(-1.0,0.0), 'd2':(-0.5,0.5)}
        else:
            self.limits = limits

    def mLW(r, rv, rs, dc, d2):
        '''
        modified Lavaux-Wandelt model. (arXiv:1110.0345)
        '''
        return np.where(r < rv, dc+(d2-dc)*(r/rv)**3, np.where(r > rs, 0.0, d2))

    def sigma(self, R, rs, dc, d2, off):
        Rv = 1.
        if Rv>rs:
            return np.inf
        sq_diff = lambda r,r2: np.sqrt(np.abs(r2**2 - r**2))
        argument = lambda r,rv: np.sqrt(np.abs((rv/r)**2 - 1))
        den_integrada = 2*np.where(
            R<Rv,
            dc*sq_diff(R,rs) + (d2-dc)*(sq_diff(R,Rv)*(5/8*(R/Rv)**2 - 1) + sq_diff(R,rs) + 3/8*(R**4/Rv**3)*np.arcsinh(argument(R,Rv))),
            np.where(
                R>rs,
                0.0,
                d2*sq_diff(R,rs)
            )
        )
        sigma = den_integrada/Rv + off
        return sigma

    def mean_sigma(self, R_vals, rs, dc, d2):
        x_grid = np.linspace(0.001, R_vals.max(), 700)
        f_grid = self.sigma(x_grid, rs, dc, d2, off=0.0)
        F_vals = np.array([
            simpson(x_grid[x_grid <= R] * f_grid[x_grid <= R], x=x_grid[x_grid <= R])
            for R in R_vals
        ])
        return F_vals

    def delta_sigma(self, R, rs, dc, d2):
        anillo = self.sigma(R, rs, dc, d2, off=0.0)
        disco = self.mean_sigma(R, rs, dc, d2)
        return (2 / R**2) * disco - anillo



if __name__ == '__main__':
    folder = '/home/fcaporaso/profiles/voids/'
    f = {
        '6-10':{
            'ALL': fits.open(folder+'Rv_6-10/lensing_Rv6-10_z02-04_typeall_RMAX5.fits'),
            'S':   fits.open(folder+'Rv_6-10/lensing_Rv6-10_z02-04_typeS_RMAX5.fits'),
            'R':   fits.open(folder+'Rv_6-10/lensing_Rv6-10_z02-04_typeR_RMAX5.fits'),
        },
        '10-50':{
            'ALL': fits.open(folder+'Rv_10-50/lensing_Rv10-50_z02-04_typeall_RMAX5.fits'),
            'S':   fits.open(folder+'Rv_10-50/lensing_Rv10-50_z02-04_typeS_RMAX5.fits'),
            'R':   fits.open(folder+'Rv_10-50/lensing_Rv10-50_z02-04_typeR_RMAX5.fits'),
        },
    }

    nk = 100
    p={}
    cov = {}
    for radius, ff in f.items():
        p[radius] = {}
        cov[radius] = {}
        for tipo, value in ff.items():
            ndots = value[0].header['ndots']
            # print(value[2].data.Sigma.shape,flush=True)
            # print('')
            p[radius][tipo] = pd.DataFrame({
                'Rp':value[1].data.Rp,
                'S':value[2].data.Sigma.reshape(nk+1,ndots)[0],
                'DSt':value[2].data.DSigma_T.reshape(nk+1,ndots)[0],
                'DSx':value[2].data.DSigma_X.reshape(nk+1,ndots)[0],
                'eS':np.sqrt(np.diag(value[3].data.covS.reshape(ndots,ndots))),
                'eDSt':np.sqrt(np.diag(value[3].data.covDSt.reshape(ndots,ndots))),
                'eDSx':np.sqrt(np.diag(value[3].data.covDSx.reshape(ndots,ndots))),
            })

            cov[radius][tipo] = {
                'covS':value[3].data.covS.reshape(ndots,ndots),
                'covDSt':value[3].data.covDSt.reshape(ndots,ndots),
                'covDSx':value[3].data.covDSx.reshape(ndots,ndots),
            }

    ## ---------------------------------------------
    ncores = 32
    nwalkers = 32
    # ndim = len(l.params)
    nit = 1000
    sample = 'TEST'

    ## ---------------- FIT ------------------------
    hsw = HSW()
    for radius, pp in p.items():
        for tipo, profile in pp.items():

            print('Fitting sigma for:')
            print('Rv: '.ljust(10,'.'),radius)
            print('Tipo: '.ljust(10,'.'),tipo)

            l = Likelihood(
                func=hsw.sigma,
                r = p[radius][tipo].Rp.to_numpy(),
                y=p[radius][tipo].S.to_numpy(),
                # yerr=np.linalg.inv(cov[radius][tipo]['covS']), ## No fittea con la cov completa
                yerr=p[radius][tipo].eS.to_numpy(),
                limits=hsw.limits_S,
                redshift=f[radius][tipo][0].header['Z_MEAN']
            )
            ## el orden es importante! -> chequear con hsw.sigma
            pos = np.array([
                np.random.uniform(-0.9,-0.4,nwalkers),
                np.random.uniform(0.8,4.5,nwalkers),
                np.random.uniform(1.5,4.5,nwalkers),
                np.random.uniform(4.5,9.0,nwalkers),
                np.random.uniform(-0.1,0.1,nwalkers),
            ]).T

            with Pool(processes=ncores) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, len(l.params), l.log_probability, pool=pool)
                sampler.run_mcmc(pos, nit, progress=True)
            mcmc_out = sampler.get_chain()

            fitted_params = {}
            error_params = {}
            for i,param in enumerate(l.params):
                percentil = np.percentile(mcmc_out[int(nit*0.5),:,i], [16,50,84])
                fitted_params[param] = percentil[1]
                error_params[param] = tuple(percentil[[0,2]]-percentil[1])

            red_chisq = chi_red(l.func(l.r, **fitted_params)*l.rhomean, l.y, l.yerr, len(l.params))
            print('Reduced chi: '.ljust(10,'.'), red_chisq)

            table_opt = np.array([
                fits.Column(name=param,format='D',array=mcmc_out[:,:,i].flatten()) for i,param in enumerate(l.params)
            ])

            hdu = fits.Header()
            for param, value in fitted_params.items():
                hdu.append((param, value))
            hdu.append(('nw',nwalkers))
            hdu.append(('ndim',len(l.params)))
            hdu.append(('nit',nit))
            hdu.append(('chi_red',red_chisq))
            hdu['HISTORY'] = f'{time.asctime()}'
            hdu['COMMENT'] = f[radius][tipo].filename()

            primary_hdu = fits.PrimaryHDU(header=hdu)
            tbhdu1 = fits.BinTableHDU.from_columns(table_opt)
            hdul = fits.HDUList([primary_hdu, tbhdu1])

            outfile = folder+f'Rv_{radius}/fit/{sample}_MCMC_lensing_Sigma_Rv{radius}_type{tipo}_5RMAX_dim{nit}x{nwalkers}.fits'
            print(f'Guardado en {outfile}')
            hdul.writeto(outfile, overwrite=True)

            ### ====================================================================================

            print('Fitting delta_sigma for:')
            print('Rv: '.ljust(10,'.'),radius)
            print('Tipo: '.ljust(10,'.'),tipo)

            l = Likelihood(
                func=hsw.delta_sigma,
                r = p[radius][tipo].Rp.to_numpy(),
                y=p[radius][tipo].DSt.to_numpy(),
                # yerr=np.linalg.inv(cov[radius][tipo]['covS']), ## No fittea con la cov completa
                yerr=p[radius][tipo].eDSt.to_numpy(),
                limits=hsw.limits_DSt,
                redshift=f[radius][tipo][0].header['Z_MEAN']
            )
            ## el orden es importante! -> chequear con hsw.sigma
            pos = np.array([
                np.random.uniform(-0.9,-0.4,nwalkers),
                np.random.uniform(0.8,4.5,nwalkers),
                np.random.uniform(1.5,4.5,nwalkers),
                np.random.uniform(4.5,9.0,nwalkers),
            ]).T

            with Pool(processes=ncores) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, len(l.params), l.log_probability, pool=pool)
                sampler.run_mcmc(pos, nit, progress=True)
            mcmc_out = sampler.get_chain()

            fitted_params = {}
            error_params = {}
            for i,param in enumerate(l.params):
                percentil = np.percentile(mcmc_out[int(nit*0.5),:,i], [16,50,84])
                fitted_params[param] = percentil[1]
                error_params[param] = tuple(percentil[[0,2]]-percentil[1])

            red_chisq = chi_red(hsw.delta_sigma(l.r, **fitted_params), l.y, l.yerr, len(l.params))
            print('Reduced chi: '.ljust(10,'.'), red_chisq)

            table_opt = np.array([
                fits.Column(name=param,format='D',array=mcmc_out[:,:,i].flatten()) for i,param in enumerate(l.params)
            ])

            hdu = fits.Header()
            for param, value in fitted_params.items():
                hdu.append((param, value))
            hdu.append(('nw',nwalkers))
            hdu.append(('ndim',len(l.params)))
            hdu.append(('nit',nit))
            hdu.append(('chi_red',red_chisq))
            hdu['HISTORY'] = f'{time.asctime()}'
            hdu['COMMENT'] = f[radius][tipo].filename()

            primary_hdu = fits.PrimaryHDU(header=hdu)
            tbhdu1 = fits.BinTableHDU.from_columns(table_opt)
            hdul = fits.HDUList([primary_hdu, tbhdu1])

            outfile = folder+f'Rv_{radius}/fit/{sample}_MCMC_lensing_DeltaSigma_Rv{radius}_type{tipo}_5R.fits'
            print(f'Guardado en {outfile}')
            hdul.writeto(outfile, overwrite=True)
