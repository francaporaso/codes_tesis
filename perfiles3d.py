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

h=1
cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)

def step_densidad(j, xv, yv, zv, rv_j, M_halos,
                  NBINS=10,RMIN=0.01,RMAX=3.):
    '''calcula la masa en funcion de la distancia al centro para 1 void
    
    j (int): # void
    xv,yv,zv (float): posicion comovil del void en Mpc
    rv_j (float): radio del void en Mpc
    M_halos (array): catalogo de halos de MICE
    z (float): redshift del centro
    NBINS (int): cantidad de puntos a calcular del perfil
    RMIN,RMAX (float): radio minimo y maximo donde calcular el perfil'''
    
    
    delta = RMAX*rv_j  # tamaño de un lado de la caja centrada en el void

    #seleccionamos los halos dentro de la caja
    mask_j = (np.abs(M_halos.xhalo-xv)<=delta) & (np.abs(M_halos.yhalo-yv)<=delta) & (np.abs(M_halos.zhalo-zv)<=delta)
    halos_vj = M_halos[mask_j]
    
    xh = halos_vj.xhalo
    yh = halos_vj.yhalo
    zh = halos_vj.zhalo
    mhalo = 10**(halos_vj.lmhalo)
    
    r_halos_v = np.sqrt((xh-xv)**2+(yh-yv)**2+(zh-zv)**2) # distancia radial del centro del void a los halos
    
    #calculamos el perfil M(r)
    step = (RMAX-RMIN)*rv_j/NBINS # en Mpc
    rin = RMIN*rv_j               # en Mpc
    densidad_void = np.zeros(NBINS)  # en M_sun/ Mpc^3
    nhalos = 0

    for cascara in range(NBINS):
        
        mk = (r_halos_v > rin)&(r_halos_v <= rin+step)    
        # r[cascara] = rin + step/2
        v = 4*np.pi*(rin*step*(rin+step) + 3*step**3)
        densidad_void[cascara] = np.sum(mhalo[mk])/v
        rin += step
        nhalos += np.sum(mk)

    return densidad_void, nhalos

def perfil_rho(NBINS, RMIN, RMAX,
              Rv_min = 12., Rv_max=15., z_min=0.2, z_max=0.3, rho1_min=-1., rho1_max=1., rho2_min=-1., rho2_max=100., FLAG=2,
              lcat = 'voids_MICE.dat', folder = '/mnt/simulations/MICE/'):
    
    ## cargamos el catalogo de voids identificados
    L = np.loadtxt(folder+lcat).T
    
    Rv    = L[1]
    ra    = L[2]
    dec   = L[3]
    z     = L[4]
    rho_1 = L[8] #Sobredensidad integrada a un radio de void 
    rho_2 = L[9] #Sobredensidad integrada máxima entre 2 y 3 radios de void 
    flag  = L[11]

    MASKvoids = ((Rv >= Rv_min)&(Rv < Rv_max))&((z >= z_min)&(z < z_max))&((rho_1 >= rho1_min)&(rho_1 < rho1_max))&((rho_2 >= rho2_min)&(rho_2 < rho2_max))&(flag >= FLAG)
    L = L[:,MASKvoids]
    
    del ra, dec, z, rho_1, rho_2, flag
    
    # radio medio del ensemble
    rv_mean = np.mean(L[1])
    
    #rango de dist comovil en el corte de redshift
    xi_min = cosmo.comoving_distance(z_min).value
    xi_max = cosmo.comoving_distance(z_max).value
    
    
    ## cargamos el catalogo de halos MICE y seleccionamos los halos segun el rango del stacking
    M_halos = fits.open('../cats/MICE/micecat2_halos.fits')[1].data
    ## dist comovil a los halos de MICE
    x_halo = M_halos.xhalo
    y_halo = M_halos.yhalo
    z_halo = M_halos.zhalo
    r_halo = np.sqrt(x_halo**2+y_halo**2+z_halo**2)

    mask_halos = (r_halo >= xi_min-50)&(r_halo <= xi_max+50) #seleccionamos los halos en el rango de redshift +/- 50 Mpc

    M_halos = M_halos[mask_halos]
    
    #calculamos los perfiles de cada void
    Nvoids = len(L.T)
    print(f'# de voids: {Nvoids}')
    # masa = np.zeros((Nvoids, NBINS))
    # vol  = np.zeros((Nvoids, NBINS))
    densidad  = np.zeros((Nvoids, NBINS))
    nh = 0

    for j in np.arange(Nvoids):
        xv = L[5][j]
        yv = L[6][j]
        zv = L[7][j]
        rv_j = L[1][j]

        densidad[j] , nhalos = step_densidad(j=j, xv=xv, yv=yv, zv=xv, rv_j=rv_j, M_halos=M_halos, NBINS=NBINS,RMIN=RMIN,RMAX=RMAX)
        nh += nhalos 

    # realizamos el stacking de masa
    print(f'# halos: {nh}')
    
    densidad = np.sum(densidad, axis=0)/nh
    
    return densidad, rv_mean