# from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import simpson, quad, cumulative_trapezoid
from scipy.special import erf

from fitting.constants import SQPI, Z_CMB, rho_mean, sigma_crit


def logistic(x, x0=1.0, k=10):
    return (1.0 + np.exp(-2.0 * k * (x - x0))) ** (-1)


# ====================
# Base models: sigma, delta_sigma integration
# ====================


class BaseModelFast:
    """
    Abstract base class for density models of cosmic structures. It calculates 
    the projected density and density contrast of a radial density model at
    blazing fast speeds, compared to a naive implementation with quadradute.
    
    This class should not be instantiated and used only as a parent to child 
    model classes. Subclasses must define the `density_contrast` method with 
    the proper radial density model.

    Attributes:
        redshift (float): Mean redshift of the structure. 
        rho_mean (float): Mean matter density of the Universe at `redshift`, in
            M_sun Mpc^-1 pc^-2, to match units of projected density in 
            literature. 
        sigma_crit_cmb (float): Critical surface density of the Observer-Lens-CMB
            system. Used only for the `kappa` profile method.

    Methods:
        density_contrast(): To be implemented in child class.
        sigma(): Projected density, related to convergence.
        delta_sigma: Projected density contrast, related to shear.
        kappa(): Convergence with the CMB as source.
    """

    def __init__(self, cosmo, redshift):
        """
        Class constructor.

        Args: 
            cosmo (astropy.cosmology.FLRW): Cosmological model instance (e.g., 
                FlatLambdaCDM). Used only to calculate `rho_mean`, `sigma_crit`
                and is not stored as an attribute.
        """


        self.redshift = redshift
        self.rho_mean = rho_mean(cosmo, redshift)
        self.sigma_crit_cmb = sigma_crit(cosmo, z_l=redshift, z_s=Z_CMB)

    ##  TODO: algun dia implementar clase abstracta usando abc

    # @abstractmethod
    def density_contrast(self, r, *params):
        """
        Density contrast $\\delta(r) = \\rho(r)/\\bar \\rho - 1$. This distribution is 
        spherically symmetric.

        Args:
            r (ndarray): Array of radial positions of the profile.
            *params (float): Parameters of the model.

        Raises:
            NotImplementedError: Subclasses must define this method.
        """
        raise NotImplementedError("Must be defined in child class")

    def sigma(self, R, *params):
        """
        Axially symmetric projected density profile Σ(R), calculated with an 
        Abel transfrom of the `density_contrast()`.

        Given that the model chosen decays to zero at great distances, the Abel
        transform limits can be switched from infinity to a large number. We 
        chose to integrate to 100 units (void radius for the void case, Mpc/h
        for the cluster case) in 500 steps.

        Args:
            R (ndarray of shape (N,)): Array of projected radial positions of 
                the profile.
            *params (float): Parameters of the density model.

        Returns:
            ndarray of shape (N,): The projected density at the `R` positions.
        """

        *p, sigma0 = params
        u_grid = np.linspace(0.0, 100.0, 500)
        radius_grid = np.hypot(u_grid[None, :], R[:, None])
        integrand_grid = self.density_contrast(radius_grid, *p)
        result = 2.0 * simpson(integrand_grid, u_grid, axis=1)

        return result * self.rho_mean + sigma0

    def kappa(self, R, *params):
        """
        Axially symmetric convergence profile κ(R), calculated as the projected
        density contrast over the critical surface density `self.sigma_crit_cmb`.
        It only calculates the convergence profile for CMB as source.

        Args:
            R (ndarray of shape (N,)): Array of projected radial positions of 
                the profile.
            *params (float): Parameters of the density model.

        Returns:
            ndarray of shape (N,): The convergence profile at the `R` positions.
        """
        return self.sigma(R, *params) / self.sigma_crit_cmb

    def delta_sigma(self, R, *params):
        """
        Axially symmetric projected density contrast profile ΔΣ(R), calculated
        as equation B.17 of Cromer et al. (2021), which changes the double
        integral of the Abel transform and the radial average with two, simpler
        integrals.

        The first integral over R is performed with 1000 fixed steps. The second, 
        using 200 steps. These were chosen to optimize speed.

        Args:
            R (ndarray of shape (N,)): Array of projected radial positions of 
                the profile.
            *params (float): Parameters of the density model.

        Returns:
            ndarray of shape (N,): The projected density contrast profile at 
                the `R` positions.
        """

        num_theta = 200
        num_x = 1000

        x_grid = np.linspace(1e-5, R.max(), num_x)
        # x_grid = np.geomspace(1e-5, R.max(), num_x)
        integrand_x = x_grid**2 * self.density_contrast(x_grid, *params)
        cumulative = cumulative_trapezoid(integrand_x, x_grid, initial=0.0)
        I1_interp = np.interp(R, x_grid, cumulative)

        theta = np.linspace(0.0, np.pi / 2.0 - 1e-6, num_theta)
        denom = 4.0 * np.sin(theta) + 3.0 - np.cos(2.0 * theta)

        r_mesh = R[:, None] / np.cos(theta[None, :])

        integrand_theta = self.density_contrast(r_mesh, *params) / denom[None, :]
        I2 = simpson(integrand_theta, theta, axis=1)

        return self.rho_mean * ((4.0 / R**2) * I1_interp - 4.0 * R * I2)


class BaseModelQuad:
    """
    Abstract base class for density models of cosmic structures. It calculates 
    the projected density and density contrast of a radial density model with
    quadradute.

    This class should not be instantiated and used only as a parent to child 
    model classes. Subclasses must define the `density_contrast` method with 
    the proper radial density model.

    Attributes:
        redshift (float): Mean redshift of the structure. 
        rho_mean (float): Mean matter density of the Universe at `redshift`, in
            M_sun Mpc^-1 pc^-2, to match units of projected density in 
            literature. 
        sigma_crit_cmb (float): Critical surface density of the Observer-Lens-CMB
            system. Used only for the `kappa` profile method.

    Methods:
        density_contrast(): To be implemented in child class.
        sigma(): NOT IMPLEMENTED. Projected density, related to convergence.
        delta_sigma: Projected density contrast, related to shear.
        kappa(): Convergence with the CMB as source.
    """

    def __init__(self, cosmo, redshift):
        self.redshift = redshift
        self.rho_mean = rho_mean(cosmo, redshift)
        self.sigma_crit_cmb = sigma_crit(cosmo, redshift, Z_CMB)

    # @abstractmethod
    def density_contrast(self, r, *params):
        """density contrast delta(r) = rho(x)/rho_mean - 1"""
        raise NotImplementedError("Must be defined in child class")

    def sigma(self, R, *params):
        raise NotImplementedError

    def kappa(self, R, *params):
        return self.sigma(R, *params) / self.sigma_crit_cmb

    def delta_sigma(self, R, *params):

        x_grid = np.linspace(0.0, R.max(), 1000)
        integrand = x_grid**2 * self.density_contrast(x_grid, *params)
        cumulative = cumulative_trapezoid(integrand, x_grid, initial=0.0)

        I1_interp = np.interp(R, x_grid, cumulative)

        result = np.zeros_like(R)

        for i, Ri in enumerate(R):

            def integrand2(theta):
                return self.density_contrast(Ri / np.cos(theta), *params) / (
                    4.0 * np.sin(theta) + 3 - np.cos(2.0 * theta)
                )

            I2, _ = quad(integrand2, 0.0, np.pi / 2.0 - 1e-6)
            result[i] = (4.0 / Ri**2) * I1_interp[i] - 4.0 * Ri * I2

        return result


# ====================
# Density Models
# ====================


class HSW(BaseModelFast):
    """
    Void density model, usually called the 'universal' density model, proposed 
    by Hamaus, Sutter & Wandelt (2014). It has four parameters controling 
    central density contrast, scale radius and inner and outer slopes.

    Inherits from `BaseModelFast`.

    Attributes: 
        params (dict): Dictionary of parameter names of each method.

    Methods:
        density_contrast(): Density definition of HSW. 
    """

    params = {
        "sigma": ["dc", "rs", "a", "b", "sigma0"],
        "kappa": ["dc", "rs", "a", "b", "sigma0"],
        "delta_sigma": ["dc", "rs", "a", "b"],
    }

    def density_contrast(self, r, dc, rs, a, b):
        """
        The shperical density contrast HSW.

        Args:
            r (ndarray of shape(N,)): Radial positions of the profile.
            dc (float): Value of the density contrast at r=0. Also called 
                central density contrast. Normal values between -1.0 to -0.2.
            rs (float): Radius at which the profile reaches the universe mean
                density, or the density contrast reaches zero. Normal values 
                between 0.8 to 4.0, depending on void type.
            a (float): Inner slope of the profile. Normal values between 1.0 to
                5.0, depending on void type.
            b (float): Outer slope of the profile. Normal values between 3.0 to
                10.0, depending on void type.
        
        Returns:
            ndarray of shape (N,): Density contrast profile values at radial 
            positions `r`.
        """
        return dc * (1 - (r / rs) ** a) / (1 + r**b)


class B15(BaseModelFast):
    """
    Void density model, proposed by Barreira et al. (2015). Similar to HSW, it 
    has five parameters controling central density contrast, void radius, scale 
    radius and inner and outer slopes.

    Inherits from `BaseModelFast`.

    Attributes: 
        params (dict): Dictionary of parameter names of each method.

    Methods:
        density_contrast(): Density definition of B15. 
    """

    params = {
        "sigma": ["dc", "rs", "rv", "a", "b", "sigma0"],
        "kappa": ["dc", "rs", "rv", "a", "b", "sigma0"],
        "delta_sigma": ["dc", "rs", "rv", "a", "b"],
    }

    def density_contrast(self, r, dc, rs, rv, a, b):
        """
        The shperical density contrast B15.

        Args:
            r (ndarray of shape(N,)): Radial positions of the profile.
            dc (float): Value of the density contrast at r=0. Also called 
                central density contrast. Normal values between -1.0 to -0.2.
            rs (float): Radius at which the profile reaches the universe mean
                density, or the density contrast reaches zero. Normal values 
                between 0.8 to 4.0, depending on void type.
            rv (float): Void radius. Used to normalize the denominator of the 
                model. Normal value 1.0.
            a (float): Inner slope of the profile. Normal values between 1.0 to
                5.0, depending on void type.
            b (float): Outer slope of the profile. Normal values between 3.0 to
                10.0, depending on void type.
        
        Returns:
            ndarray of shape (N,): Density contrast profile values at radial 
            positions `r`.
        """

        return dc * (1 - (r / rs) ** a) / (1 + (r / rv) ** b)


class ModifiedLW(BaseModelFast):
    """
    Void density model, proposed by Lavaux & Wandelt (20--) and modified by 
    Caporaso et al (2026) to represent voids with walls. It has three parameters
    controling central density contrast, wall density contrast, and scale of
    wall transition.

    Inherits from `BaseModelFast`.
    
    Instead of using discrete numerical approximations, this class leverages 
    the known analytical solution of the underlying system equations to compute
    exact projection values.

    Attributes:
        params (dict): Dictionary of parameter names of each method.

    Methods:
        density_contrast(): Density definition of mLW. 
        sigma_mean(): Radial projected density average.
        sigma(): Projected density.
        delta_sigma(): Projected density contrast.
    """

    params = {
        "sigma": ["dc", "dw", "rw", "sigma0"],
        "kappa": ["dc", "dw", "rw", "sigma0"],
        "delta_sigma": ["dc", "dw", "rw"],
    }

    def density_contrast(self, r, dc, dw, rw):
        """
        The shperical density contrast mLW.

        Args:
            r (ndarray of shape(N,)): Radial positions of the profile.
            dc (float): Value of the density contrast at r=0. Also called 
                central density contrast. Normal values between -1.0 to -0.2.
            dw (float): Value of the density contrast profile at the wall. It
                can be negative, representing a void inside a bigger structure.
                Normal values -0.5 to 0.5, depending on void type.
            rw (float): Radius at which the profile changes from inner void to 
                outer shells. Normal values between 1.5 to 4.0, depending on 
                void type.
        
        Returns:
            ndarray of shape (N,): Density contrast profile values at radial 
            positions `r`.
        """
        rv = 1.0
        return (
            np.where(r > 0.0, dc + (dw - dc) * (r / rv) ** 3, 0.0)
            + np.where(r > rv, (dc - dw) * (1.0 - (r / rv) ** 3), 0.0)
            - np.where(r > rw, dw, 0.0)
        )

    def _P(self, r, R):
        diff = np.maximum(r**2 - R**2, 0.0)
        return (1.0 / 3.0) * diff**1.5

    def _Q(self, r, R):
        diff = np.maximum(r**2 - R**2, 0.0)
        sqrt_diff = np.sqrt(diff)
        term1 = (r / 48.0) * (8 * r**4 - 2 * R**2 * r**2 - 3 * R**4) * sqrt_diff
        term2 = (R**6 / 16.0) * np.log(np.maximum(r + sqrt_diff, 1e-12))
        return term1 - term2

    def sigma_mean(self, R, dc, dw, rw):
        """
        Computes the mean enclosed projection S_bar(R). Used to calculate 
        Delta Sigma.

        Args:
            R (ndarray of shape (N,)): Projected radial positions of the
                profile.
            dc (float): Density contrast value at r=0.
            dw (float): Density contrast value between r=1 and r=rw.
            rw (float): Transition radius from inner to outer void.

        Returns:
            ndarray of shape (N,): Cumulative projected density at radial
                positions `R`.
        """

        def _W1_sbar(r, R):
            return dc * self._P(r, R) + (dw - dc) * self._Q(r, R)

        def _W2_sbar(r, R):
            return (dw - dc) * (self._P(r, R) - self._Q(r, R))

        R = np.atleast_1d(R)
        J = np.zeros_like(R, dtype=float)
        mask = R < rw
        Rm = R[mask]

        if np.any(mask):
            W1_rw = _W1_sbar(rw, Rm)
            W1_R = _W1_sbar(Rm, Rm)
            W2_rw = _W2_sbar(rw, Rm)
            W2_max = _W2_sbar(np.maximum(Rm, 1.0), Rm)
            J[mask] = (W1_rw - W1_R) + (W2_rw - W2_max)

        c_m = (dc - dw + 2 * dw * rw**3) / 6.0

        R_safe = np.maximum(R, 1e-12)
        sbar = (4.0 / R_safe**2) * (c_m - J)
        return self.rho_mean * sbar

    # ==========================================
    # Methods for S(R) (Standard Projection)
    # ==========================================
    def _I0(self, r, R):
        """Base integral I_0(r, R) for standard projection"""
        return np.sqrt(np.maximum(r**2 - R**2, 0.0))

    def _I3(self, r, R):
        """Base integral I_3(r, R) for standard projection"""
        diff = np.maximum(r**2 - R**2, 0.0)
        sqrt_diff = np.sqrt(diff)
        term1 = (r / 8.0) * (2 * r**2 + 3 * R**2) * sqrt_diff
        term2 = (3.0 * R**4 / 8.0) * np.log(np.maximum(r + sqrt_diff, 1e-12))
        return term1 + term2

    def sigma(self, R, dc, dw, rw, sigma0):
        """
        Computes the density projection S(R).

        Args:
            R (ndarray of shape (N,)): Projected radial positions of the
                profile.
            dc (float): Density contrast value at r=0.
            dw (float): Density contrast value between r=1 and r=rw.
            rw (float): Transition radius from inner to outer void.
            sigma0 (float): Projected density offset. Should be close to zero.

        Returns:
            ndarray of shape (N,): Projected density at radial positions `R`.
        """

        def _W1_s(r, R):
            return dc * self._I0(r, R) + (dw - dc) * self._I3(r, R)

        def _W2_s(r, R):
            return (dw - dc) * (self._I0(r, R) - self._I3(r, R))

        R = np.atleast_1d(R)
        S = np.zeros_like(R, dtype=float)
        mask = R < rw
        Rm = R[mask]

        if np.any(mask):
            W1_rw = _W1_s(rw, Rm)
            W1_R = _W1_s(Rm, Rm)
            W2_rw = _W2_s(rw, Rm)
            W2_max = _W2_s(np.maximum(Rm, 1.0), Rm)

            # Multiply by 2 because the integral was from R to infinity (half-line)
            S[mask] = 2.0 * ((W1_rw - W1_R) + (W2_rw - W2_max))

        return self.rho_mean * S + sigma0

    # ==========================================
    # The Final Target: D = S_bar - S
    # ==========================================
    def delta_sigma(self, R, dc, dw, rw):
        """
        Computes the excess surface density D(R) = S_bar(R) - S(R).

        Args:
            R (ndarray of shape (N,)): Projected radial positions of the
                profile.
            dc (float): Density contrast value at r=0.
            dw (float): Density contrast value between r=1 and r=rw.
            rw (float): Transition radius from inner to outer void.

        Returns:
            ndarray of shape (N,): Projected density contrast at radial 
                positions `R`.
        """
        return self.sigma_mean(R, dc, dw, rw) - self.sigma(R, dc, dw, rw, sigma0=0.0)


class TopHat(BaseModelFast):
    """
    Void density model, proposed by Higuchi et al. (20--) and modified by 
    Caporaso et al (2026) to represent voids with walls. It has three parameters
    controling central density contrast, wall density contrast, and scale of
    wall transition.

    Inherits from `BaseModelFast`.
    
    Instead of using discrete numerical approximations, this class leverages 
    the known analytical solution of the underlying system equations to compute
    exact projection values.

    Attributes:
        params (dict): Dictionary of parameter names of each method.

    Methods:
        density_contrast(): Density definition of mLW. 
        sigma_mean(): Radial projected density average.
        sigma(): Projected density.
        delta_sigma(): Projected density contrast.
    """
    params = {
        "sigma": ["dc", "dw", "rw", "sigma0"],
        "kappa": ["dc", "dw", "rw", "sigma0"],
        "delta_sigma": ["dc", "dw", "rw"],
    }

    def density_contrast(self, r, dc, dw, rw):
        """
        The shperical density contrast mLW.

        Args:
            r (ndarray of shape(N,)): Radial positions of the profile.
            dc (float): Value of the density contrast at r=0. Also called 
                central density contrast. Normal values between -1.0 to -0.2.
            dw (float): Value of the density contrast profile at the wall. It
                can be negative, representing a void inside a bigger structure.
                Normal values -0.5 to 0.5, depending on void type.
            rw (float): Radius at which the profile changes from inner void to 
                outer shells. Normal values between 1.5 to 4.0, depending on 
                void type.
        
        Returns:
            ndarray of shape (N,): Density contrast profile values at radial 
            positions `r`.
        """
        rv = 1.0
        return np.where(r < rv, dc - dw, 0.0) + np.where(r < rw, dw, 0.0)

    def _P(self, r, R):
        # np.maximum forces negative values to 0, naturally handling the Heaviside boundaries
        return (1.0 / 3.0) * np.maximum(r**2 - R**2, 0.0) ** 1.5

    def _I0(self, r, R):
        return np.sqrt(np.maximum(r**2 - R**2, 0.0))

    # ==========================================
    # Profile Calculations
    # ==========================================
    def sigma_mean(self, R, dc, dw, rw):
        """
        Computes the mean enclosed projection S_bar(R). Used to calculate 
        Delta Sigma.

        Args:
            R (ndarray of shape (N,)): Projected radial positions of the
                profile.
            dc (float): Density contrast value at r=0.
            dw (float): Density contrast value between r=1 and r=rw.
            rw (float): Transition radius from inner to outer void.

        Returns:
            ndarray of shape (N,): Cumulative projected density at radial
                positions `R`.
        """
        rv = 1.0

        R = np.atleast_1d(R)

        # Calculate J(R) using the two-term block logic
        J = dw * self._P(rw, R) + (dc - dw) * self._P(rv, R)

        # Protect against division by zero at R=0
        R_safe = np.maximum(R, 1e-12)
        C_M = (dc * rv**3 + dw * (rw**3 - rv**3)) / 3.0
        sbar = (4.0 / R_safe**2) * (C_M - J)

        return self.rho_mean * sbar

    def sigma(self, R, dc, dw, rw, sigma0):
        """
        Computes the density projection S(R).

        Args:
            R (ndarray of shape (N,)): Projected radial positions of the
                profile.
            dc (float): Density contrast value at r=0.
            dw (float): Density contrast value between r=1 and r=rw.
            rw (float): Transition radius from inner to outer void.
            sigma0 (float): Projected density offset. Should be close to zero.

        Returns:
            ndarray of shape (N,): Projected density at radial positions `R`.
        """

        rv = 1.0
        R = np.atleast_1d(R)

        # Calculate S(R) using the two-term block logic
        S = 2.0 * (dw * self._I0(rw, R) + (dc - dw) * self._I0(rv, R))

        return self.rho_mean * S + sigma0

    def delta_sigma(self, R, dc, dw, rw):
        """
        Computes the excess surface density D(R) = S_bar(R) - S(R).

        Args:
            R (ndarray of shape (N,)): Projected radial positions of the
                profile.
            dc (float): Density contrast value at r=0.
            dw (float): Density contrast value between r=1 and r=rw.
            rw (float): Transition radius from inner to outer void.

        Returns:
            ndarray of shape (N,): Projected density contrast at radial 
                positions `R`.
        """
        return self.sigma_mean(R, dc, dw, rw) - self.sigma(R, dc, dw, rw, sigma0=0.0)


class P13(BaseModelFast):
    """
    Void density model, proposed by Paz et al. (2013) and modified by 
    Caporaso et al (2026) to represent different central densities. It has five
    parameters controling central density contrast, wall density contrast, 
    steepnes and wall width.

    Inherits from `BaseModelFast`.
    
    Attributes:
        params (dict): Dictionary of parameter names of each method.

    Methods:
        density_contrast(): Density definition of P13 model, derived.
        integrated_density(): Integrated density contrast, original definition 
            of P13.
    """

    params = {
        "kappa": ["dc", "rs", "S", "P", "W", "sigma0"],
        "sigma": ["dc", "rs", "S", "P", "W", "sigma0"],
        "delta_sigma": ["dc", "rs", "S", "P", "W"],
    }

    def density_contrast(self, r, dc, rs, S, P, W):
        """
        The shperical density contrast P13.

        Args:
            r (ndarray of shape(N,)): Radial positions of the profile.
            dc (float): Value of the density contrast at r=0. Also called 
                central density contrast. Normal values between -1.0 to -0.2.
            rs (float): Scale radius. Normal values around 1.0.
            S (float): Steepness parameter. Normal values depend on void wall.
            P (float): Maximum density contrast of the profile, at the wall. 
                Normal values around 0.2
            W (float): Void wall width on the outer shells. 
                Normal values around 0.5.
        
        Returns:
            ndarray of shape (N,): Density contrast profile values at radial 
            positions `r`.
        """
        x = np.log(r / rs)
        A = np.where(r < rs, S, W)

        t1 = 0.5 * (1.0 - erf(S * x))
        t2 = S / (3 * SQPI) * np.exp(-((S * x) ** 2))
        rising = dc * (t1 - t2)

        shell = P * np.exp(-A * x**2) * (1.0 - 2 / 3 * A * x)
        return rising + shell

    def integrated_density(self, r, dc, rs, S, P, W):
        """
        The shperical integrated density contrast P13.

        Args:
            r (ndarray of shape(N,)): Radial positions of the profile.
            dc (float): Value of the density contrast at r=0. Also called 
                central density contrast. Normal values between -1.0 to -0.2.
            rs (float): Scale radius. Normal values around 1.0.
            S (float): Steepness parameter. Normal values depend on void wall.
            P (float): Maximum density contrast of the profile, at the wall. 
                Normal values around 0.2
            W (float): Void wall width on the outer shells. 
                Normal values around 0.5.
        
        Returns:
            ndarray of shape (N,): Density contrast profile values at radial 
            positions `r`.
        """
        x = np.log(r / rs)
        A = np.where(r < rs, S, W)
        rising = 0.5 * dc * (1.0 - erf(S * x))
        shell = P * np.exp(-A * x**2)
        return rising + shell

models_dict = {
    "HSW": HSW,
    "TH": TopHat,
    "mLW": ModifiedLW,
    "B15": B15,
    "P13": P13,
}
default_limits = {
    "HSW": {
        "dc": (-1.0, 0.0),
        "rs": (0.5, 5.0),
        "a": (1.0, 15.0),
        "b": (1.0, 15.0),
        "sigma0": (-0.5, 0.5),
    },
    "B15": {
        "dc": (-1.0, 0.0),
        "rs": (0.5, 5.0),
        "rv": (0.5, 5.0),
        "a": (1.0, 15.0),
        "b": (1.0, 15.0),
        "sigma0": (-0.5, 0.5),
    },
    "TH": {
        "dc": (-1.0, 0.0),
        "dw": (-0.5, 0.5),
        "rw": (1.0, 5.0),
        "sigma0": (-0.5, 0.5),
    },
    "mLW": {
        "dc": (-1.0, 0.0),
        "dw": (-0.5, 0.5),
        "rw": (1.0, 5.0),
        "sigma0": (-0.5, 0.5),
    },
    "P13": {
        "dc": (-1.0, 0.0),
        "S": (0.0, 10.0),
        "rs": (0.1, 5.0),
        "P": (-0.1, 1.0),
        "W": (0.0, 10.0),
        "sigma0": (-0.5, 0.5),
    },
}
default_guess = {
    "HSW": {"dc": -0.7, "rs": 0.9, "a": 3.0, "b": 6.0, "sigma0": 0.0},
    "B15": {"dc": -0.7, "rs": 0.9, "rv": 1.0, "a": 3.0, "b": 6.0, "sigma0": 0.0},
    "TH": {"dc": -0.7, "dw": 0.2, "rw": 2.5, "sigma0": 0.0},
    "mLW": {"dc": -0.7, "dw": 0.2, "rw": 2.5, "sigma0": 0.0},
    "P13": {"dc": -0.7, "S": 3.0, "rs": 1.0, "P": 0.1, "W": 2.0, "sigma0": 0.0},
}
