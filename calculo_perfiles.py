from perfiles3d import *
import numpy as np
from astropy.io import fits

rv_min = np.array([6,9,12,15,18])
rv_max = np.array([9,12,15,18,50])

z_min = np.array([0.2,0.3])
z_max = np.array([0.3,0.4])

#tipo S: rho_2 = (0,100)
#tipo R: rho_2 = (-1,0)
#                     T    S   R
rho_2_min = np.array([-1 , 0 ,-1])
rho_2_max = np.array([100,100,0 ])

# sample   = args.sample
rho1_min = -1.
rho1_max = 1. 
FLAG     = 2 
LOGM     = 10. 
RMIN     = 0.05
RMAX     = 3.
NBINS    = 15
NBOOT    = 100
INTP     = False

M_halos = fits.open('/home/fcaporaso/cats/MICE/micecat2_halos.fits')[1].data

bines = np.linspace(RMIN,RMAX,num=NBINS+1)
r = (bines[:-1] + np.diff(bines)*0.5)

tipos = np.array(['Total', 'S', 'R'])

i=30 

for rvm,rvM in zip(rv_min,rv_max):
    for zm, zM in zip(z_min, z_max):
        for rho2m, rho2M, t in zip(rho_2_min, rho_2_max, tipos):
            print(f'Calculando el perfil {t}, con los parámetros')
            print(f'radio:{rvm}-{rvM}, z:{zm}-{zM}, rho_2:{rho2m}-{rho2M}')
        

            if INTP:
                print('Calculando con interpolación...')
                print('Puede demorar unos segundos más...')
                resultado, res_poly = perfil_rho(NBINS=NBINS, RMIN=RMIN, RMAX=RMAX, LOGM=LOGM,
                                                Rv_min=rvm, Rv_max=rvM, z_min=zm, z_max=zM,
                                                rho1_min=rho1_min, rho1_max=rho1_max, rho2_min=rho2m, rho2_max=rho2M,
                                                FLAG=FLAG, nboot=NBOOT, interpolar=INTP)
            else:
                resultado = perfil_rho(NBINS=NBINS, RMIN=RMIN, RMAX=RMAX, LOGM=LOGM,
                                       Rv_min=rvm, Rv_max=rvM, z_min=zm, z_max=zM,
                                       rho1_min=rho1_min, rho1_max=rho1_max, rho2_min=rho2m, rho2_max=rho2M,
                                       FLAG=FLAG, nboot=NBOOT, interpolar=INTP)


            h = fits.Header()
            h.append(('Nvoids',int(resultado[7])))
            h.append(('Rv_min',np.round(rvm,2)))
            h.append(('Rv_max',np.round(rvM,2)))
            h.append(('rho1_min',np.round(rho1_min,2)))
            h.append(('rho1_max',np.round(rho1_max,2)))
            h.append(('rho2_min',np.round(rho2m,2)))
            h.append(('rho2_max',np.round(rho2M,2)))
            h.append(('z_min',np.round(zm,2)))
            h.append(('z_max',np.round(zM,2)))
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
                os.mkdir(f'../profiles/voids/Rv_{int(rvm)}-{int(rvM)}/3D')
            except FileExistsError:
                pass
            
            output_folder = f'../profiles/voids/Rv_{int(rvm)}-{int(rvM)}/3D/'

            hdul.writeto(f'{output_folder+sample}.fits',overwrite=True)

            print(f'Archivo guardado en: {output_folder+sample}.fits !')
            print(f'Terminado, faltan {i-1} perfiles!')
            i-=1

print('Termiando!')
