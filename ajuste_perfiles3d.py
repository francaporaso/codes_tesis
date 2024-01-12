from scipy.integrate import quad_vec
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit


## definicion de funciones de ajuste
def hamaus(r, rs, rv, delta, a, b,x):
        
    d = delta*(1. - (r/rs)**a)/(1. + (r/rv)**b)
    return d+x

def int_hamaus(r,rs,rv,delta,a,b,x):
    
    def integrando(q):
        return hamaus(q,rs,rv,delta,a,b,0)*(q**2)
    
    integral = np.array([quad_vec(integrando, 0, ri)[0] for ri in r])
    return (3/r**3)*integral+x

def clampitt(r,Rv,R2,dc,d2,x):
    R_V = np.full_like(r, Rv)
    R2s  = np.full_like(r,R2)
    
    delta = (r<=R_V)*(dc + (d2-dc)*(r/Rv)**3) + ((r>R_V)&(r<=R2s))*d2 + (r>R2s)*0
    
    return delta+x

def int_clampitt(r,Rv,R2,dc,d2,x):
    
    def integrando(q):
        return clampitt(q,Rv,R2,dc,d2,0)*(q**2)
    
    integral = np.array([quad_vec(integrando, 0, ri)[0] for ri in r])
    return (3/r**3)*integral+x

def higuchi(r,Rv,R2,dc,d2,x):
    unos = np.full_like(r,Rv)
    R2s  = np.full_like(r,R2)
    
    delta = (r<=unos)*dc + ((r>unos)&(r<=R2s))*d2 + (r>R2s)*0
    
    return delta+x

def int_higuchi(r,Rv,R2,dc,d2,x):
    
    def integrando(q):
        return higuchi(q,Rv,R2,dc,d2,0)*(q**2)
    
    integral = np.array([quad_vec(integrando, 0, ri)[0] for ri in r])
    return (3/r**3)*integral+x

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
    if func.__name__ == 'hamaus':
        return 5
    else:
        return 4

## funcion de ajuste

def ajuste(func_dif, func_int, xdata, ydif, edif, yint, eint, p0, b, orden, f, d):
    
    try:
        a_dif, cov_dif = curve_fit(f=func_dif, xdata=xdata, ydata=ydif, sigma=edif,
                                p0=p0, bounds=b)
        
        chi2_dif = chi_red(func_dif(xdata,*a_dif), ydif, edif, gl(func_dif))

    except RuntimeError:
        print(f'El perfil {f} no ajustó para la funcion {func_dif.__name__}')
        a_dif = np.ones_like(p0)
        cov_dif = np.ones((len(p0),len(p0)))
        chi2_dif = 1000

    try:    
        a_int, cov_int = curve_fit(f=func_int, xdata=xdata, ydata=yint, sigma=eint,
                                p0=p0, bounds=b)
        
        chi2_int = chi_red(func_int(xdata,*a_dif), yint, eint, gl(func_dif))
        
    except RuntimeError:
        print(f'El perfil {f} no ajustó para la función {func_int.__name__}')    
        a_int = np.ones_like(p0)
        cov_int = np.ones_like((len(p0),len(p0)))
        chi2_int = 1000

    h = fits.Header()
    h.append(('orden', orden))
    h.append(('chi_red_dif', chi2_dif))
    h.append(('chi_red_int', chi2_int))
    params = [fits.Column(name='param_dif', format='E', array=a_dif),
             fits.Column(name='param_int', format='E', array=a_int)]

    covs   = [fits.Column(name='cov_dif', format='E', array=cov_dif.flatten()),
             fits.Column(name='cov_int', format='E', array=cov_int.flatten())]

    tbhdu1 = fits.BinTableHDU.from_columns(fits.ColDefs(params))
    tbhdu2 = fits.BinTableHDU.from_columns(fits.ColDefs(covs))
    primary_hdu = fits.PrimaryHDU(header=h)
    hdul = fits.HDUList([primary_hdu, tbhdu1, tbhdu2])

    f = f.split('/')[-1]
    d = d+'/'+f.split('/')[0]+'/'+f.split('/')[1]
    output = f'{d}/fit_{func_dif.__name__}_{f}'
    hdul.writeto(output,overwrite=True)

##
f1 = np.array([hamaus, clampitt, higuchi])
# f1 = np.array([hamaus])
f2 = np.array([int_hamaus, int_clampitt, int_higuchi])
# f2 = np.array([int_hamaus])

## p0 de cada func
p0 = np.array([[2.,0.6,-0.6,1.5,2.,0],
               [0.5,0.5,-0.5,0.1,0],   
               [0.5,0.5,-0.5,0.1,0]], dtype=object)   
b = np.array([([0.,0.,-1,1,1,-10],[3.,3.,0,10.,10,10]),
              ([0.,0.,-1,-1.,-10],[3.,3.,10.,10.,10]),
              ([0.,0.,-1,-1.,-10],[3.,3.,10.,10.,10])], dtype=object)
orden = np.array(['rs, rv, dc, a, b, x', 
                  'Rv, R2, dc, d2, x',
                  'Rv, R2, dc, d2, x'])

## leyendo datos
# radios = np.array(['6-9', '9-12', '12-15', '15-18', '18-50'])
radios = ['6-10','10-50']

# tipos = np.array(['Total', 'S', 'R'])
# redshift = np.array(['lowz', 'highz'])

# file = np.array([f'3d_{t}_{z}.fits' for t in tipos for z in redshift])
file = np.array(['Rv_6-10/3D/rvchico_tot.fits', 'Rv_6-10/3D/rvchico_R.fits', 'Rv_6-10/3D/rvchico_S.fits',
                 'Rv_10-50/3D/rvalto_tot.fits', 'Rv_10-50/3D/rvalto_R.fits', 'Rv_10-50/3D/rvalto_S.fits'])

d = '../profiles/voids'
for f in file:
    print(f'Ajustando perfil {f}')

    with fits.open(f'{d}/{f}') as dat:
        p = dat[1].data
        h = dat[0].header
    
        xdata = p.r
        ydif=p.den_dif
        edif=p.e_den_dif
        yint=p.den_int
        eint=p.e_den_int
        for f_dif,f_int,p,v,o in zip(f1,f2,p0,b,orden):
            print(f'usando {f_dif.__name__}')
            ajuste(f_dif,f_int,xdata=xdata, ydif=ydif, edif=edif, yint=yint, eint=eint,
                   p0=p, b=v, orden=o, f=f, d=d)
