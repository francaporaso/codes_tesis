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


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import LambdaCDM
import argparse
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.io import fits
import os


h=1
cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)


def step_densidad(xv, yv, zv, rv_j,
              NBINS=10,RMIN=0.01,RMAX=3., LOGM=12.):
    '''calcula la masa en funcion de la distancia al centro para 1 void
    
    j (int): # void
    xv,yv,zv (float): posicion comovil del void en Mpc
    rv_j (float): radio del void en Mpc
    M_halos (array): catalogo de halos de MICE
    z (float): redshift del centro
    NBINS (int): cantidad de puntos a calcular del perfil
    RMIN,RMAX (float): radio minimo y maximo donde calcular el perfil'''
    
    
    #seleccionamos los halos dentro de la caja
    mask_j = ((np.abs(M_halos.xhalo-xv)<=RMAX*rv_j)&(np.abs(M_halos.yhalo-yv)<=RMAX*rv_j)&(np.abs(M_halos.zhalo-zv)<=RMAX*rv_j)&(
               M_halos.lmhalo >= LOGM))
    halos_vj = M_halos[mask_j]
    
    xh = halos_vj.xhalo
    yh = halos_vj.yhalo
    zh = halos_vj.zhalo
    mhalo = 10**(halos_vj.lmhalo)
    
    r_halos_v = np.sqrt((xh-xv)**2+(yh-yv)**2+(zh-zv)**2)/rv_j # distancia radial del centro del void a los halos en unidades reducidas
    
    #calculamos el perfil M(r)
    step = (RMAX-RMIN)/NBINS # en Mpc
    rin = RMIN               # en Mpc
    MASAsum = np.zeros(NBINS)  # en M_sun/ Mpc^3
    Ninbin  = np.zeros(NBINS)  # en M_sun/ Mpc^3
    nhalos = len(halos_vj)

    for cascara in range(NBINS):
        
        mk = (r_halos_v > rin)&(r_halos_v <= rin+step)    
        
        MASAsum[cascara] = np.sum(mhalo[mk])
        Ninbin[cascara] = np.sum(mk)
        rin += step

    MASAacum = np.cumsum(MASAsum)

    return np.array([MASAsum, MASAacum, Ninbin, nhalos], dtype=object)


def perfil_rho(NBINS, RMIN, RMAX, LOGM = 12.,
              Rv_min = 12., Rv_max=15., z_min=0.2, z_max=0.3, rho1_min=-1., rho1_max=1., rho2_min=-1., rho2_max=100., FLAG=2,
              lcat = 'voids_MICE.dat', folder = '/mnt/simulations/MICE/', nboot=100, interpolar=False):
    
    ## cargamos el catalogo de voids identificados
    L = np.loadtxt(folder+lcat).T
    
    Rv    = L[1]
    z     = L[4]
    rho_1 = L[8] #Sobredensidad integrada a un radio de void 
    rho_2 = L[9] #Sobredensidad integrada máxima entre 2 y 3 radios de void 
    flag  = L[11]

    MASKvoids = ((Rv >= Rv_min)&(Rv < Rv_max)&(z >= z_min)&(z < z_max)&(rho_1 >= rho1_min)&(rho_1 < rho1_max)&(
                  rho_2 >= rho2_min)&(rho_2 < rho2_max)&(flag >= FLAG))
    
    L = L[:,MASKvoids]
    
    del z, rho_1, rho_2, flag
    
    # radio medio del ensemble
    rv_mean = np.mean(L[1])
    
    bines = np.linspace(RMIN,RMAX,NBINS+1)
    R = bines[:-1] + 0.5*np.diff(bines)

    #calculamos los perfiles de cada void
    Nvoids = len(L.T)
    print(f'# de voids: {Nvoids}')
    MASAsum  = np.zeros((Nvoids, NBINS))
    MASAacum = np.zeros((Nvoids, NBINS))
    Ninbin   = np.zeros((Nvoids, NBINS))
    nh = 0

    print('Calculando perfiles...')
    for j in np.arange(Nvoids):
        xv   = L[5][j]
        yv   = L[6][j]
        zv   = L[7][j]
        rv_j = L[1][j]

        MASAsum[j], MASAacum[j], Ninbin[j], nhalos = step_densidad(xv=xv, yv=yv, zv=xv, rv_j=rv_j, NBINS=NBINS, RMIN=RMIN, RMAX=RMAX, LOGM=LOGM)
        nh += nhalos 

    # realizamos el stacking de masa y calculo de densidad
    print(f'# halos: {nh}')

    masa_dif  = np.sum(MASAsum, axis=0)
    Nbin      = np.sum(Ninbin, axis=0)
    vol_dif   = np.array([(4*np.pi/3)*((bines[i+1])**3 - (bines[i])**3) for i in range(NBINS)])
    den_media = np.sum(masa_dif)/((4*np.pi/3)*(RMAX**3 - RMIN**3)) # masa total sobre volumen de la caja
    
    std_masa_dif = boot(MASAsum, nboot=nboot)

    # calculo de densidad acumulada
    masa_int = np.sum(MASAacum, axis=0)
    vol_acum = np.cumsum(vol_dif)
    
    std_masa_int = boot(MASAacum, nboot=nboot)

    output = np.array([masa_dif, masa_int, std_masa_dif, std_masa_int, vol_dif, vol_acum, Nbin, Nvoids, den_media], dtype=object)

    if interpolar:
        print('Interpolando...')

        poly_m_d = np.zeros((Nvoids,NBINS))
        poly_m_i = np.zeros((Nvoids,NBINS))
    
        for j in range(Nvoids):
            p_d = np.poly1d(np.polyfit(R,MASAsum[j],4))
            poly_m_d[j] = p_d(R)
        
            p_i = np.poly1d(np.polyfit(R,MASAacum[j],4))
            poly_m_i[j] = p_i(R)
        
        std_poly_m_d = boot(poly_m_d, nboot)
        std_poly_m_i = boot(poly_m_i, nboot)

        output_poly = np.array([poly_m_d, poly_m_i, std_poly_m_d, std_poly_m_i])

        return output, output_poly

    return output

def boot(poblacion, nboot=100):
    size,ndots = poblacion.shape
    
    index = np.arange(size)
    with NumpyRNGContext(1):
        bootresult = bootstrap(index, nboot)
    INDEX=bootresult.astype(int)

    std = np.std(poblacion[INDEX].sum(axis=1), axis=0)

    return std



if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', action='store', dest='sample',default='pru')
    parser.add_argument('-Rv_min', action='store', dest='Rv_min', default=12.)
    parser.add_argument('-Rv_max', action='store', dest='Rv_max', default=15.)
    parser.add_argument('-rho1_min', action='store', dest='rho1_min', default=-1.)
    parser.add_argument('-rho1_max', action='store', dest='rho1_max', default=1.)
    parser.add_argument('-rho2_min', action='store', dest='rho2_min', default=-1.)
    parser.add_argument('-rho2_max', action='store', dest='rho2_max', default=100.)
    parser.add_argument('-FLAG', action='store', dest='FLAG', default=2.)
    parser.add_argument('-z_min', action='store', dest='z_min', default=0.2)
    parser.add_argument('-z_max', action='store', dest='z_max', default=0.3)
    parser.add_argument('-LOGM', action='store', dest='LOGM', default=12.)
    parser.add_argument('-RMIN', action='store', dest='RMIN', default=0.05)
    parser.add_argument('-RMAX', action='store', dest='RMAX', default=4.)
    parser.add_argument('-NBINS', action='store', dest='NBINS', default=40)
    parser.add_argument('-NBOOT', action='store', dest='NBOOT', default=100)
    parser.add_argument('-INTP', action='store', dest='INTP', default=False)
    args = parser.parse_args()

    sample   = args.sample
    Rv_min   = float(args.Rv_min)
    Rv_max   = float(args.Rv_max) 
    rho1_min = float(args.rho1_min)
    rho1_max = float(args.rho1_max) 
    rho2_min = float(args.rho2_min)
    rho2_max = float(args.rho2_max) 
    FLAG     = float(args.FLAG) 
    z_min    = float(args.z_min) 
    z_max    = float(args.z_max) 
    LOGM     = float(args.LOGM) 
    RMIN     = float(args.RMIN)
    RMAX     = float(args.RMAX)
    NBINS    = int(args.NBINS)
    NBOOT    = int(args.NBOOT)
    INTP     = bool(args.INTP)


    M_halos = fits.open('/home/fcaporaso/cats/MICE/micecat2_halos.fits')[1].data

    if INTP:
        print('Calculando con interpolación...')
        print('Puede demorar unos segundos más...')
        resultado, res_poly = perfil_rho(NBINS=NBINS, RMIN=RMIN, RMAX=RMAX, LOGM=LOGM,
                                        Rv_min=Rv_min, Rv_max=Rv_max, z_min=z_min, z_max=z_max,
                                        rho1_min=rho1_min, rho1_max=rho1_max, rho2_min=rho2_min, rho2_max=rho2_max,
                                        FLAG=FLAG, nboot=NBOOT, interpolar=INTP)
    else:
        resultado = perfil_rho(NBINS=NBINS, RMIN=RMIN, RMAX=RMAX, LOGM=LOGM,
                               Rv_min=Rv_min, Rv_max=Rv_max, z_min=z_min, z_max=z_max,
                               rho1_min=rho1_min, rho1_max=rho1_max, rho2_min=rho2_min, rho2_max=rho2_max,
                               FLAG=FLAG, nboot=NBOOT, interpolar=INTP)

    bines = np.linspace(RMIN,RMAX,num=NBINS+1)
    r = (bines[:-1] + np.diff(bines)*0.5)

    h = fits.Header()
    h.append(('N_VOIDS',int(resultado[7])))
    h.append(('Rv_min',np.round(Rv_min,2)))
    h.append(('Rv_max',np.round(Rv_max,2)))
    h.append(('rho1_min',np.round(rho1_min,2)))
    h.append(('rho1_max',np.round(rho1_max,2)))
    h.append(('rho2_min',np.round(rho2_min,2)))
    h.append(('rho2_max',np.round(rho2_max,2)))
    h.append(('z_min',np.round(z_min,2)))
    h.append(('z_max',np.round(z_max,2)))
    h.append(('den_media', resultado[8]))

    table_p = [ fits.Column(name='r', format='E', array=r),
                fits.Column(name='masa_dif', format='E', array=resultado[0]),
                fits.Column(name='masa_int', format='E', array=resultado[1]),
                fits.Column(name='std_masa_dif', format='E', array=resultado[2]),
                fits.Column(name='std_masa_int', format='E', array=resultado[3]),
                fits.Column(name='vol_dif', format='E', array=resultado[4]),
                fits.Column(name='vol_int', format='E', array=resultado[5]),
                fits.Column(name='Nbin', format='E', array=resultado[6])]   

    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(table_p))

    if INTP:
        table_poly = [fits.Column(name='poly_mdif', format='E', array=res_poly[0]),
                      fits.Column(name='poly_mint', format='E', array=res_poly[1]),
                      fits.Column(name='std_poly_mdif', format='E', array=res_poly[2]),
                      fits.Column(name='std_poly_mint', format='E', array=res_poly[3])]
        
        tbhdu_poly = fits.BinTableHDU.from_columns(fits.ColDefs(table_poly))

    primary_hdu = fits.PrimaryHDU(header=h)
    
    if INTP:
        hdul = fits.HDUList([primary_hdu, tbhdu, tbhdu_poly])
    else:
        hdul = fits.HDUList([primary_hdu, tbhdu])


    try:
        os.mkdir(f'../profiles/voids/Rv_{int(Rv_min)}-{int(Rv_max)}/3D')
    except FileExistsError:
        pass

    output_folder = f'../profiles/voids/Rv_{int(Rv_min)}-{int(Rv_max)}/3D/'

    hdul.writeto(f'{output_folder+sample}.fits',overwrite=True)

    print(f'Archivo guardado en: {output_folder+sample}.fits !')
    print(f'Terminado!')



