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
import time
from multiprocessing import Pool, Process

h=1
cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)

def boot(poblacion, nboot=100):
    size,ndots = poblacion.shape
    
    index = np.arange(size)
    with NumpyRNGContext(1):
        bootresult = bootstrap(index, nboot)
    INDEX=bootresult.astype(int)

    std = np.std(poblacion[INDEX].mean(axis=1), axis=0)

    return std

def step_densidad(xv, yv, zv, rv_j,
              NBINS=10,RMIN=0.01,RMAX=3., LOGM=9.):
    '''calcula la masa en funcion de la distancia al centro para 1 void
    
    j (int): # void
    xv,yv,zv (float): posicion comovil del void en Mpc
    rv_j (float): radio del void en Mpc
    M_halos (array): catalogo de halos de MICE
    z (float): redshift del centro
    NBINS (int): cantidad de puntos a calcular del perfil
    RMIN,RMAX (float): radio minimo y maximo donde calcular el perfil'''
    
    NBINS = round(NBINS)
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
    # den_dif  = np.zeros(NBINS)  # en M_sun/ Mpc^3
    nhalos = len(halos_vj)

    for cascara in range(NBINS):
        
        mk = (r_halos_v >= rin)&(r_halos_v < rin+step)    
        
        MASAsum[cascara] = np.sum(mhalo[mk])
        # vol = (4*np.pi/3)*(step*(step*(3*rin+step)+3*rin**2))  # == ((rin+step)**3 - rin**3)
        # den_dif[cascara] = np.sum(mhalo[mk])/vol
        Ninbin[cascara] = np.sum(mk)
        rin += step

    MASAacum = np.cumsum(MASAsum)
    # den_media = np.mean(den_dif)
    # den_int = np.cumsum(den_dif)/den_media - 1
    # den_dif = den_dif/den_media - 1

    return np.array([MASAsum, MASAacum, Ninbin, nhalos], dtype=object)
    # return np.array([den_dif, den_int, Ninbin, nhalos], dtype=object)

def step_densidad_unpack(minput):
	return step_densidad(*minput)


def perfil_rho(NBINS, RMIN, RMAX, LOGM = 9.,
              Rv_min = 12., Rv_max=15., z_min=0.2, z_max=0.3, rho1_min=-1., rho1_max=1., rho2_min=-1., rho2_max=100., FLAG=2,
              lcat = 'voids_MICE.dat', folder = '/mnt/simulations/MICE/', nboot=100, ncores=10, interpolar=False):
    
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
    Nvoids = len(L.T)

    # corte del catalogo para paralelizado
    if Nvoids < ncores:
        ncores=Nvoids
    
    lbins = int(round(Nvoids/float(ncores), 0))
    slices = ((np.arange(lbins)+1)*ncores).astype(int)
    slices = slices[(slices < Nvoids)]
    Lsplit = np.split(L.T,slices)
    
    #calculamos los perfiles de cada void
    # Nvoids = len(L.T)
    print(f'# de voids: {Nvoids}')
    # MASAsum  = np.zeros((Nvoids, NBINS))
    # MASAacum = np.zeros((Nvoids, NBINS))
    # Ninbin   = np.zeros((Nvoids, NBINS))
    MASAsum  = np.array([])
    MASAacum = np.array([])
    # den_difsum = np.array([])
    # den_intsum = np.array([])
    Ninbin  = np.array([])
    nh = 0

    print('Calculando perfiles...')
    LARGO = len(Lsplit)

    tslice = np.array([])
        
    for l, Lsplit_l in enumerate(Lsplit):
                
        print(f'Vuelta {l+1} de {LARGO}')
                
        t1 = time.time()
        num = len(Lsplit_l)

        if num == 1:
            entrada = [Lsplit_l[5], Lsplit_l[6],
                       Lsplit_l[7], Lsplit_l[1],
                       NBINS, RMIN, RMAX,LOGM]
                    
            salida = [step_densidad(entrada)]
        else:                
            rmin   = np.full(num, RMIN)
            rmax   = np.full(num, RMAX)
            nbins  = np.full(num, NBINS, dtype=int)
            logm   = np.full(num, LOGM)
                    
            entrada = np.array([Lsplit_l.T[5],Lsplit_l.T[6],
                                Lsplit_l.T[7],Lsplit_l.T[1],
                                nbins,rmin,rmax,logm]).T
            with Pool(processes=num) as pool:
                salida = np.array(pool.map(step_densidad_unpack,entrada))
                pool.close()
                pool.join()

        for j, profilesums in enumerate(salida):
        
            MASAsum  = np.append(MASAsum,profilesums[0])
            MASAacum = np.append(MASAacum, profilesums[1])
            # den_difsum = np.append(den_difsum, profilesums[0])
            # den_intsum = np.append(den_intsum, profilesums[1])
            Ninbin  = np.append(Ninbin, profilesums[2])
            nh += profilesums[3]

        t2 = time.time()
        ts = (t2-t1)/60.
        tslice = np.append(tslice, ts)
        print('Tiempo del corte')
        print(f'{np.round(ts,4)} min')
        print('Timpo restante estimado')
        print(f'{np.round(np.mean(tslice)*(LARGO-(l+1)), 3)} min')

    # corrigiendo la forma de los arrays
    
    MASAsum  = MASAsum.reshape(Nvoids,NBINS)
    MASAacum = MASAacum.reshape(Nvoids,NBINS)
    Ninbin   = Ninbin.reshape(Nvoids,NBINS)
    
    # realizamos el stacking de masa y calculo de densidad
    print(f'# halos: {nh}')
    print(f'Calculo de perfiles terminado en {np.round(tslice.sum(), 3)} min')
    bines = np.linspace(RMIN,RMAX,num=NBINS+1)

    Nbin      = np.sum(Ninbin, axis=0)

    # densidad diferencial
    vol_dif    = np.array([(4*np.pi/3)*((bines[i+1])**3 - (bines[i])**3) for i in range(NBINS)])
    den_difsum = MASAsum/vol_dif #shape=(Nvoids,NBINS), cada fila es la densidad de c/void individual
    den_media = np.mean(den_difsum, axis=1) #shape=(Nvoids)

    den_dif = np.array([den_difsum[i]/den_media[i] for i in range(Nvoids)]) - 1 
    e_den_dif = boot(den_dif, nboot=nboot)
    den_dif = np.mean(den_dif, axis=0)
    
    # densidad acumulada/integrada
    den_intsum = MASAacum/np.cumsum(vol_dif)
    
    den_int = np.array([den_intsum[i]/den_media[i] for i in range(Nvoids)]) - 1
    e_den_int = boot(den_int, nboot=nboot)
    den_int = np.mean(den_int, axis=0)
 
    # output = np.array([masa_dif, masa_int, den_dif, den_int,
    #                    std_masa_dif, std_masa_int, e_den_dif, e_den_int,
    #                    vol_dif, vol_acum, Nbin, Nvoids, den_media, nh], dtype=object)

    # den_dif  = np.mean(den_difsum, axis=0)
    # Nbin      = np.sum(Ninbin, axis=0)
    # e_den_dif  = boot(den_difsum, nboot=nboot)   
    # den_int  = np.mean(den_intsum, axis=0)
    # e_den_int  = boot(den_intsum, nboot=nboot)

    output = np.array([den_dif, den_int,
                       e_den_dif, e_den_int,
                       Nbin, Nvoids, nh], dtype=object)

    return output


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
    parser.add_argument('-LOGM', action='store', dest='LOGM', default=9.)
    parser.add_argument('-RMIN', action='store', dest='RMIN', default=0.05)
    parser.add_argument('-RMAX', action='store', dest='RMAX', default=3.)
    parser.add_argument('-NBINS', action='store', dest='NBINS', default=40)
    parser.add_argument('-NBOOT', action='store', dest='NBOOT', default=100)
    parser.add_argument('-INTP', action='store', dest='INTP', default=False)
    parser.add_argument('-ncores', action='store', dest='ncores', default=10)
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
    ncores   = int(args.ncores)


    M_halos = fits.open('/home/fcaporaso/cats/MICE/micecat2_halos_full.fits')[1].data
    centrales = (M_halos.flag_central == 0)
    M_halos = M_halos[centrales]

    tin = time.time()

    resultado = perfil_rho(NBINS=NBINS, RMIN=RMIN, RMAX=RMAX, LOGM=LOGM,
                           Rv_min=Rv_min, Rv_max=Rv_max, z_min=z_min, z_max=z_max,
                           rho1_min=rho1_min, rho1_max=rho1_max, rho2_min=rho2_min, rho2_max=rho2_max,
                           FLAG=FLAG, nboot=NBOOT, ncores=ncores, interpolar=INTP)

    bines = np.linspace(RMIN,RMAX,num=NBINS+1)
    r = (bines[:-1] + np.diff(bines)*0.5)

    h = fits.Header()
    # h.append(('Nvoids',int(resultado[11])))
    h.append(('Nvoids',int(resultado[5])))
    h.append(('Rv_min',np.round(Rv_min,2)))
    h.append(('Rv_max',np.round(Rv_max,2)))
    h.append(('rho1_min',np.round(rho1_min,2)))
    h.append(('rho1_max',np.round(rho1_max,2)))
    h.append(('rho2_min',np.round(rho2_min,2)))
    h.append(('rho2_max',np.round(rho2_max,2)))
    h.append(('z_min',np.round(z_min,2)))
    h.append(('z_max',np.round(z_max,2)))
    h.append(('nhalos',resultado[-1]))
    # h.append(('den_media', np.float32(resultado[12])))

    primary_hdu = fits.PrimaryHDU(header=h)

    # table_p = [ fits.Column(name='r', format='E', array=r),
    #             fits.Column(name='masa_dif', format='E', array=resultado[0]),
    #             fits.Column(name='masa_int', format='E', array=resultado[1]),
    #             fits.Column(name='den_dif', format='E', array=resultado[2]),
    #             fits.Column(name='den_int', format='E', array=resultado[3]),
    #             fits.Column(name='std_masa_dif', format='E', array=resultado[4]),
    #             fits.Column(name='std_masa_int', format='E', array=resultado[5]),
    #             fits.Column(name='e_den_dif', format='E', array=resultado[6]),
    #             fits.Column(name='e_den_int', format='E', array=resultado[7]),
    #             fits.Column(name='vol_dif', format='E', array=resultado[8]),
    #             fits.Column(name='vol_int', format='E', array=resultado[9]),
    #             fits.Column(name='Nbin', format='E', array=resultado[10])]   
    table_p = [ fits.Column(name='r', format='E', array=r),
                fits.Column(name='den_dif', format='E', array=resultado[0]),
                fits.Column(name='den_int', format='E', array=resultado[1]),
                fits.Column(name='e_den_dif', format='E', array=resultado[2]),
                fits.Column(name='e_den_int', format='E', array=resultado[3]),
                fits.Column(name='Nbin', format='E', array=resultado[4])] 

    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(table_p))

    hdul = fits.HDUList([primary_hdu, tbhdu])

    try:
        os.makedirs(f'../profiles/voids/Rv_{round(Rv_min)}-{round(Rv_max)}/3D')
    except FileExistsError:
        pass

    output_folder = f'../profiles/voids/Rv_{round(Rv_min)}-{round(Rv_max)}/3D/'

    hdul.writeto(f'{output_folder+sample}.fits',overwrite=True)

    tfin = time.time()

    print(f'Archivo guardado en: {output_folder+sample}.fits !')
    print(f'Terminado en {(tfin-tin)/60} min!')



