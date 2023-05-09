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

def partial_profile(x,y,z,Rv,
                    RIN,ROUT,ndots,h):

        '''
        calcula el perfil de 1 solo void, tomando el centro del void y su redshift
        RA0,DEC0 (float): posicion del centro del void
        Z: redshift del void
        RIN,ROUT: bordes del perfil
        ndots: cantidad de puntos del perfil
        h: cosmologia
        addnoise(bool): agregar ruido (forma intrinseca) a las galaxias de fondo
        devuelve la densidad proyectada (Sigma), el contraste(DSigma), la cant de galaxias por bin (Ninbin) 
        y las totales (Ntot)'''
        
        ndots = int(ndots)

        Rv   = Rv/h *u.Mpc
        cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)
        
        DEGxMPC = cosmo.arcsec_per_kpc_proper(Z).to('deg/Mpc')
        delta = (DEGxMPC*(ROUT*Rv))

        pos_angles = 0*u.deg, 90*u.deg, 180*u.deg, 270*u.deg
        c1 = SkyCoord(RA0*u.deg, DEC0*u.deg)
        c2 = np.array([c1.directional_offset_by(pos_angle, delta) for pos_angle in pos_angles])

        mask = (S.dec_gal < c2[0].dec.deg)&(S.dec_gal > c2[2].dec.deg)&(S.ra_gal < c2[1].ra.deg)&(
                S.ra_gal > c2[3].ra.deg)&(S.z_cgal > (Z+0.1))
        
        # mask = (np.abs(S.ra_gal -RA0) < delta)& (np.abs(S.dec_gal-DEC0) < delta)&(S.z_cgal_v > (Z+0.1))
        catdata = S[mask]

        del mask, delta

        sigma_c = SigmaCrit(Z, catdata.z_cgal)
        
        rads, theta, *_ = eq2p2(np.deg2rad(catdata.ra_gal), np.deg2rad(catdata.dec_gal),
                                  np.deg2rad(RA0), np.deg2rad(DEC0))
                               
        
        e1     = catdata.gamma1
        e2     = -1.*catdata.gamma2

        # Add shape noise due to intrisic galaxy shapes        
        if addnoise:
            es1 = -1.*catdata.eps1
            es2 = catdata.eps2
            e1 += es1
            e2 += es2
        
        #get tangential ellipticities 
        et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
        #get cross ellipticities
        ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
               
        #get convergence
        k  = catdata.kappa*sigma_c

        r = (np.rad2deg(rads)/DEGxMPC.value)/(Rv.value)
        #r = (np.rad2deg(rads)*3600*KPCSCALE)/(Rv*1000.)
        Ntot = len(catdata)        

        del catdata
        del e1, e2, theta, sigma_c, rads

        bines = np.linspace(RIN,ROUT,num=ndots+1)
        dig = np.digitize(r,bines)
                
        SIGMAwsum    = np.empty(ndots)
        DSIGMAwsum_T = np.empty(ndots)
        DSIGMAwsum_X = np.empty(ndots)
        N_inbin      = np.empty(ndots)
                                             
        for nbin in range(ndots):
                mbin = dig == nbin+1              

                SIGMAwsum[nbin]    = k[mbin].sum()
                DSIGMAwsum_T[nbin] = et[mbin].sum()
                DSIGMAwsum_X[nbin] = ex[mbin].sum()
                N_inbin[nbin]      = np.count_nonzero(mbin)
        
        output = np.array([SIGMAwsum, DSIGMAwsum_T, DSIGMAwsum_X, N_inbin, Ntot], dtype=object)
        #output = (SIGMAwsum, DSIGMAwsum_T, DSIGMAwsum_X, N_inbin, Ntot)
        
        return output
