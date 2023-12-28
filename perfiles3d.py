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

    MASAacum = np.zeros(NBINS)
    MASAacum[0] = MASAsum[0]
    
    for i in range(1,NBINS):
        MASAacum[i] = MASAacum[i-1] + MASAsum[i]

    return np.array([MASAsum, MASAacum, Ninbin, nhalos], dtype=object)


def perfil_rho(NBINS, RMIN, RMAX, LOGM = 12.,
              Rv_min = 12., Rv_max=15., z_min=0.2, z_max=0.3, rho1_min=-1., rho1_max=1., rho2_min=-1., rho2_max=100., FLAG=2,
              lcat = 'voids_MICE.dat', folder = '/mnt/simulations/MICE/', nboot=100):
    
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
    
    boot_masa_dif = boot(MASAsum, Nvoids, NBINS, nboot=nboot)
    std_masa_dif  = np.abs(np.std(boot_masa_dif, axis=0))

    # calculo de densidad acumulada
    masa_int = np.sum(MASAacum, axis=0)
    vol_acum = np.cumsum(vol_dif)
    
    boot_masa_int = boot(MASAacum, Nvoids, NBINS, nboot=nboot)
    std_masa_int  = np.abs(np.std(boot_masa_int, axis=0))

    output = np.array([masa_dif, masa_int, std_masa_dif, std_masa_int, vol_dif, vol_acum, Nbin, np.full_like(Nbin,Nvoids), np.full_like(Nbin, den_media)])

    return output

def boot(MASAsum,Nvoids,ndots,nboot=100):

    A = np.random.uniform(0,Nvoids,(nboot, Nvoids)).astype(np.int32)

    boot = np.sum(MASAsum[A], axis=1)/Nvoids

    std = np.std(boot, axis=0)

    return boot


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


    M_halos = fits.open('/home/fcaporaso/cats/MICE/micecat2_halos.fits')[1].data

    resultado = perfil_rho(NBINS=NBINS, RMIN=RMIN, RMAX=RMAX, LOGM=LOGM,
                Rv_min=Rv_min, Rv_max=Rv_max, z_min=z_min, z_max=z_max,
                rho1_min=rho1_min, rho1_max=rho1_max, rho2_min=rho2_min, rho2_max=rho2_max, FLAG=FLAG, nboot=NBOOT)


    import csv

    header = np.array(['masa_dif', 'masa_int', 'std_masa_dif', 'std_masa_int', 'vol_dif', 'vol_acum', 'Nbin', 'Nvoids', 'den_media'])
    data = resultado.T

    with open(f'/home/fcaporaso/tests/perfiles_3d/perfil3d_{sample}.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)
