'''Ajuste de perfiles de voids mediante cuadrados minimos. Por defecto ajusta ambos Sigma y DSigma'''
import numpy as np
import argparse
import os
import time
from multiprocessing import Pool
from scipy.integrate import quad, quad_vec
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.cosmology import LambdaCDM
from astropy.constants import c, G


def pm(z):
    '''densidad media en Msun/(pc**2 Mpc)'''
    h = 1.
    cosmo = LambdaCDM(H0 = 100.*h, Om0=0.3, Ode0=0.7)
    p_cr0 = cosmo.critical_density(0).to('Msun/(pc**2 Mpc)').value
    a = cosmo.scale_factor(z)
    out = p_cr0*cosmo.Om0/a**3
    return out


def hamaus(r, rs, rv, delta, a, b):
        
    d = delta*(1. - (r/rs)**a)/(1. + (r/rv)**b)
    return d

def clampitt(r,Rv,R2,dc,d2):
    R_V = np.full_like(r, Rv)
    R2s  = np.full_like(r,R2)
    
    delta = (r<=R_V)*(dc + (d2-dc)*(r/Rv)**3) + ((r>R_V)&(r<=R2s))*d2 + (r>R2s)*0
    
    return delta

def higuchi(r,Rv,R2,dc,d2):
    unos = np.full_like(r,Rv)
    R2s  = np.full_like(r,R2)
    
    delta = (r<=unos)*dc + ((r>unos)&(r<=R2s))*d2 + (r>R2s)*0
    
    return delta


### Densidades proyectadas para cada función

def sigma_higuchi(R,R2,dc,d2,x):
    Rv = 1.
    if Rv>R2:
        return np.inf
        
    Rv = np.full_like(R,Rv)
    R2 = np.full_like(R,R2)
    
    m1 = (R<=Rv)
    m2 = (R>Rv)&(R<=R2)
    
    den_integrada = np.zeros_like(R)
    den_integrada[m1] = (np.sqrt(Rv[m1]**2-R[m1]**2)*(dc-d2) + d2*np.sqrt(R2[m1]**2-R[m1]**2))
    den_integrada[m2] = d2*np.sqrt(R2[m2]**2-R[m2]**2)

    sigma = rho_mean*den_integrada/Rv + x
    return sigma

def sigma_clampitt(R,R2,dc,d2,x):
    Rv = 1.
    if Rv>R2:
        return np.inf

    Rv = np.full_like(R,Rv)
    R2 = np.full_like(R,R2)
    
    den_integrada = np.zeros_like(R)
    
    m1 = (R<=Rv)
    m2 = (R>Rv)&(R<=R2)
    
    s2 = np.sqrt(R2[m1]**2 - R[m1]**2)
    sv = np.sqrt(Rv[m1]**2 - R[m1]**2)
    arg = np.sqrt((Rv[m1]/R[m1])**2 - 1)

    den_integrada[m1] = 2*(dc*s2 + (d2-dc)*(sv*(5/8*(R[m1]/Rv[m1])**2 - 1) + s2 + 3/8*(R[m1]**4/Rv[m1]**3)*np.arcsinh(arg)))   
    den_integrada[m2] = 2*(d2*np.sqrt(R2[m2]**2-R[m2]**2))

    sigma = rho_mean*den_integrada/Rv + x
    return sigma

def sigma_hamaus(r,rs,dc,a,b,x):
    rv = 1.
    def integrand(z,R):
        return hamaus(r=np.sqrt(z**2+R**2),rv=rv,rs=rs,delta=dc,a=a,b=b)
  
    den_integrada = quad_vec(integrand, -1e3, 1e3, args=(r,), epsrel=1e-3)[0]

    sigma = rho_mean*den_integrada/rv + x
    
    return sigma


## Contraste de Densidad Proyectada de cada función

def Scl(y,Rv,R2,dc,d2,x):
    '''
    funcion sigma_clampitt pero solo admite como entrada un float,
    ideal para integrar
    '''
    if y<=Rv:
        sv = np.sqrt(Rv**2 - y**2)
        s2 = np.sqrt(R2**2 - y**2)
        arg = np.sqrt((Rv/y)**2 - 1)
        f1 = 2*(dc*s2 + (d2-dc)*(sv*(5/8*(y/Rv)**2 - 1) + s2 + 3/8*(y**4/Rv**3)*np.arcsinh(arg)))
        return rho_mean*f1/Rv+x
    elif y>R2:
        return x
    else:
        f2 = 2*(d2*np.sqrt(R2**2-y**2))
        return rho_mean*f2/Rv+x

def delta_sigma_clampitt(R,R2,dc,d2):
    Rv = 1.
    def integrand(y):
        return Scl(y,Rv,R2,dc,d2,0)*y

    anillo = sigma_clampitt(R,R2,dc,d2,0)
    disco = np.zeros_like(R)
    for i,Ri in enumerate(R):
        disco[i] = (2/Ri**2)*quad(integrand, 0, Ri)[0]
    
    return disco-anillo

def Shi(y,Rv,R2,dc,d2,x):
    '''
    funcion sigma_higuchi pero solo admite como entrada un float,
    ideal para integrar
    '''
    
    if y<=Rv:
        f1 = (np.sqrt(Rv**2-y**2)*(dc-d2) + d2*np.sqrt(R2**2-y**2))
        return rho_mean*f1/Rv+x
    elif y>R2:
        return x
    else:
        f2 = d2*np.sqrt(R2**2-y**2)
        return rho_mean*f2/Rv+x

    
def delta_sigma_higuchi(R,R2,dc,d2):
    Rv = 1.
    def integrand(y):
        return Shi(y,Rv,R2,dc,d2,0)*y

    anillo = sigma_higuchi(R,R2,dc,d2,0)
    disco = np.zeros_like(R)
    for i,Ri in enumerate(R):
        disco[i] = (2/Ri**2)*quad(integrand, 0, Ri)[0]
    
    return disco-anillo


def delta_sigma_hamaus(r,rs,dc,a,b):
    
    rv = 1.
    def integrand(y):
        return sigma_hamaus(y,rs,dc,a,b)*y

    anillo = sigma_hamaus(r,rs,dc,a,b)
    disco = np.zeros_like(r)
    for j,p in enumerate(r):
        disco[j] = 2./p**2 * quad(integrand, 0., p)[0]

    return disco-anillo


## ----

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

def gl(func):
    if func.__name__ == 'sigma_hamaus':
        return 6
    else:
        return 5

def ajuste(func, xdata, y, ey, p0, b):
    
    try:
        popt, cov = curve_fit(f=func, xdata=xdata, ydata=y, sigma=ey,
                              p0=p0, bounds=b)
        
        chi2 = chi_red(func(xdata,*popt), y, ey, gl(func))

    except RuntimeError:
        print(f'El perfil no ajustó para la funcion {func.__name__}')
        popt = np.ones_like(p0)
        cov = np.ones((len(p0),len(p0)))
        chi2 = 1000

    return chi2, popt, cov


## --- 
if __name__ == '__main__':

    funcs = np.array([
                        sigma_hamaus, 
                        # sigma_clampitt, 
                        # sigma_higuchi,
                    ])
    # funcs = np.array([
                        # delta_sigma_hamaus
                        # delta_sigma_clampitt, 
                        # delta_sigma_higuchi,  
                    # ]) 
    
    p0 = np.array(
                    [
                        [2.8,-0.9,0.6,3.5,-0.01],
                        # [1.,1.5,-0.5,0.1],   
                        # [1.,1.5,-0.5,0.1,0],   
                    ],
                    dtype=object
                  )
    
    bounds = np.array([
                        ([0.,-1,0.,1.,-10],[3.,0,10.,20,10]),
                        # ([0.,0.1,-1,-1.],[3.,3.,10.,100.]),
                        # ([0.,0.1,-1,-1.,-10],[3.,3.,10.,100.,10]),
                      ], dtype=object
                     )
    orden = np.array([
                        'rs, dc, a, b, x', 
                        # 'Rv, R2, dc, d2',
                        # 'Rv, R2, dc, d2, x',   
                     ])


    ## PARA LOS PERFILES NUEVOS
    i = 0
    tslice = np.array([])

    # for j,carpeta in enumerate(['Rv_6-10/rvchico_','Rv_10-50/rvalto_']):}
    for j,carpeta in enumerate(['Rv_10-50/rvalto_']):
        for k, archivo in enumerate(['R']):
            t1 = time.time()
            print(f'Ajustando el perfil: {carpeta}{archivo}.fits')

            with fits.open(f'../profiles/voids/{carpeta}{archivo}.fits') as dat:
                h = dat[0].header
                A = dat[1].data
                B = dat[2].data
                C = dat[3].data

            rho_mean = pm(h['z_mean'])

            S = B.Sigma.reshape(101,60)[0]
            # DSt = B.DSigma_T.reshape(101,60)[0]
            # DSx = B.DSigma_X.reshape(101,60)[0]
            covS = C.covS.reshape(60,60)
            eS = np.sqrt(np.diag(covS))
            # covDSt = C.covDSt.reshape(60,60)
            # eDSt = np.sqrt(np.diag(covDSt))
            # covDSx = C.covDSx.reshape(60,60)
            # eDSx = np.sqrt(np.diag(covDSx))

            for fu,P0,Bo,Or in zip(funcs,p0,bounds,orden):
                print(f'con {fu.__name__}')
                chi2, popt, pcov = ajuste(fu ,xdata=A.Rp, y=S, ey=eS, p0=P0, b=Bo)            
                # chi2, popt, pcov = ajuste(fu ,xdata=A.Rp, y=DSt, ey=eDSt, p0=P0, b=Bo)            

                h = fits.Header()
                h.append(('orden', Or))
                h.append(('chi_red', chi2))

                params = fits.ColDefs([fits.Column(name='param', format='E', array=popt)])
                covs   = fits.ColDefs([fits.Column(name='cov', format='E', array=pcov.flatten())])

                tbhdu1 = fits.BinTableHDU.from_columns(params)
                tbhdu2 = fits.BinTableHDU.from_columns(covs)
                primary_hdu = fits.PrimaryHDU(header=h)
                hdul = fits.HDUList([primary_hdu, tbhdu1, tbhdu2])


                try:
                    aaa = carpeta.split('/')[0]
                    carpeta_out = f'../profiles/voids/{aaa}/fit'    
                    os.mkdir(carpeta_out)
                except FileExistsError:
                    pass        
                
                output = f'{carpeta_out}/fit_{fu.__name__}_{archivo}.fits'
                hdul.writeto(output,overwrite=True)

            t2 = time.time()
            ts = (t2-t1)/60
            tslice = np.append(tslice,ts)
            i+=1
            print(f'Tardó {np.round(ts,4)} min')
            # print(f' ')
        print('Tiempo restante estimado')
        print(f'{np.round(np.mean(tslice)*(6-(i)), 3)} min')

    print(f'Terminado en {np.round(np.sum(tslice),3)} min!')



''' ## PARA LOS PERFILES QUE SACAMOS PRIMERO
radios = np.array(['6-9', '9-12', '12-15', '15-18', '18-50'])
files = np.array(['smallz', 'highz', 'sz_S', 'hz_S', 'sz_R', 'hz_R'])
nombres = np.array(['tot_lowz', 'tot_highz', 'S_lowz', 'S_highz', 'R_lowz', 'R_highz'])

tslice = np.array([])
i=0
for rad in radios:
    print('----')
    print(f'Ajustando para los radios {rad}')
    print('----')
    d = f'/home/fcaporaso/profiles/voids/Rv_{rad}'
    for j,f in enumerate(files):
        print(f'Ajustando el perfil: {f}_{rad}.fits')
        t1 = time.time()

        ##
        if f'{f}_{rad}' == 'smallz_6-9':
            print('Salteado!')
            continue
        if f'{f}_{rad}' == 'highz_6-9':
            print('Salteado!')
            continue
        if f'{f}_{rad}' == 'sz_S':
            print('Salteado!')
            continue
        if f'{f}_{rad}' == 'hz_S':
            print('Salteado!')
            continue
        ##


        with fits.open(f'{d}/{f}_{rad}.fits') as hdu:
            h = hdu[0].header
            r = hdu[1].data.Rp
            p = hdu[2].data
            c = hdu[3].data

            rv_medio = h['Rv_mean']
            z_medio  = h['z_mean']

            rho_mean = pm(z_medio)

            Sigma = p.Sigma.reshape(101,60)[0]/rv_medio
            covS = c.covS.reshape(60,60)
            eSigma = np.sqrt(np.diag(covS))/rv_medio

            for fu,P,B,O in zip(funcs,p0,bounds,orden):

                print(f'con {fu.__name__}')
                ajuste(fu ,xdata=r, y=Sigma, ey=eSigma, p0=P, b=B, orden=O, f=nombres[j], d=d)

        t2 = time.time()
        ts = (t2-t1)/60
        tslice = np.append(tslice,ts)
        i+=1
        print(f'Tardó {np.round(ts,4)} min')
        # print(f' ')
    print('Tiempo restante estimado')
    print(f'{np.round(np.mean(tslice)*(30-(i)), 3)} min')
'''
