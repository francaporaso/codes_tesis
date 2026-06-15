# from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import simpson, quad, cumulative_trapezoid
from scipy.special import erf

from fitting.constants import SQPI, rho_mean


def logistic(x, x0=1.0, k=10):
    return (1.0 + np.exp(-2.0 * k * (x - x0))) ** (-1)


# ====================
# Base models: sigma, delta_sigma integration
# ====================


class BaseModelFast:
    def __init__(self, redshift):
        self.redshift = redshift
        self.rho_mean = rho_mean(redshift)

    ##  TODO: algun dia implementar clase abstracta usando abc

    # @abstractmethod
    def density_contrast(self, r, *params):
        """density contrast delta(r) = rho(x)/rho_mean - 1"""
        raise NotImplementedError("Must be defined in child class")

    def sigma(self, R, *params):

        *p, sigma0 = params
        u_grid = np.linspace(0.0, 100.0, 500)
        radius_grid = np.hypot(u_grid[None, :], R[:, None])
        integrand_grid = self.density_contrast(radius_grid, *p)
        result = 2.0 * simpson(integrand_grid, u_grid, axis=1)

        return result * self.rho_mean + sigma0

    def delta_sigma(self, R, *params):

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
    # @abstractmethod
    def density_contrast(self, r, *params):
        """density contrast delta(r) = rho(x)/rho_mean - 1"""
        raise NotImplementedError("Must be defined in child class")

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
    params = {
        "sigma": ["dc", "rs", "a", "b", "sigma0"],
        "delta_sigma": ["dc", "rs", "a", "b"],
    }

    def density_contrast(self, r, dc, rs, a, b):
        return dc * (1 - (r / rs) ** a) / (1 + r**b)


class B15(BaseModelFast):
    def density_contrast(self, r, dc, rs, rv, a, b):
        return dc * (1 - (r / rs) ** a) / (1 + (r / rv) ** b)


class ModifiedLW(BaseModelFast):
    params = {"sigma": ["dc", "dw", "rw", "sigma0"], "delta_sigma": ["dc", "dw", "rw"]}

    def density_contrast(self, r, dc, dw, rw):
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
        """Computes the mean enclosed projection S_bar(R)."""

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
        """Computes the standard projected profile S(R)."""

        def _W1_s(r, R):
            """Block 1 weight for standard projection"""
            return dc * self._I0(r, R) + (dw - dc) * self._I3(r, R)

        def _W2_s(r, R):
            """Block 2 weight for standard projection"""
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
        """
        return self.sigma_mean(R, dc, dw, rw) - self.sigma(R, dc, dw, rw, sigma0=0.0)


class TopHat(BaseModelFast):
    params = {"sigma": ["dc", "dw", "rw", "sigma0"], "delta_sigma": ["dc", "dw", "rw"]}

    def density_contrast(self, r, dc, dw, rw):
        rv = 1.0
        return np.where(r < rv, dc - dw, 0.0) + np.where(r < rw, dw, 0.0)

    # easier to compute since is integrable
    def sigma(self, R, dc, dw, rw, sigma0=0.0):
        rv = 1.0
        return (
            np.where(R < rv, (dc - dw) * np.sqrt(rv**2 - R**2), 0.0)
            + np.where(R < rw, dw * np.sqrt(rw**2 - R**2), 0.0)
            + sigma0
        )

    def delta_sigma(self, R, dc, dw, rw):
        rv = 1.0
        I1 = np.where(
            R < rv,
            1 / 3 * (dc - dw) * (rv**3 - (rv**2 - R**2) ** (3 / 2)),
            1 / 3 * (dc - dw) * rv**3,
        )
        I2 = np.where(
            R < rw, 1 / 3 * dw * (rw**3 - (rw**2 - R**2) ** (3 / 2)), 1 / 3 * dw * rw**3
        )

        return (2.0 / R**2) * (I1 + I2) - self.sigma(R, dc, dw, rw)


class Paz13(BaseModelFast):
    def density_contrast(self, r, S, Rs, P, W):
        x = np.log10(r / Rs)
        asym_gauss = np.where(r < Rs, np.exp(-S * x**2), np.exp(-W * x**2))

        Delta = 0.5 * (erf(S * x) - 1) + P * asym_gauss

        t1 = S * np.exp(-((S * x) ** 2)) / (SQPI * r)
        t2 = (-2.0 * P * x / r) * asym_gauss
        Delta_prime = t1 + t2

        return Delta + 1 / 3 * r * Delta_prime


class THLogistic(BaseModelFast):
    def density_contrast(self, r, dc, dw, rw):
        k = 15
        return (dc - dw) * (1.0 - logistic(r, x0=1, k=k)) + dw * (
            1.0 - logistic(r, x0=rw, k=k)
        )


class ModLWLogistic(BaseModelFast):
    # not tested! weird values at r=rv
    def density_contrast(self, r, dc, dw, rw):
        rv = 1.0
        k = 15
        return (dc - dw) * (1.0 - (r / rv) ** 3) * (
            1.0 - logistic(r, x0=rv, k=k)
        ) + dw * (1.0 - logistic(r, x0=rw, k=k))


models_dict = {
    "HSW": HSW,
    "TH": TopHat,
    "mLW": ModifiedLW,
    "B15": B15,
    "P13": Paz13,
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
        "S": (0.0, 10.0),
        "Rs": (0.1, 5.0),
        "P": (0.0, 1.0),
        "W": (0.1, 5.0),
        "sigma0": (-0.5, 0.5),
    },
}
default_guess = {
    "HSW": {"dc": -0.7, "rs": 0.9, "a": 3.0, "b": 6.0, "sigma0": 0.0},
    "B15": {"dc": -0.7, "rs": 0.9, "rv": 1.0, "a": 3.0, "b": 6.0, "sigma0": 0.0},
    "TH": {"dc": -0.7, "dw": 0.2, "rw": 2.5, "sigma0": 0.0},
    "mLW": {"dc": -0.7, "dw": 0.2, "rw": 2.5, "sigma0": 0.0},
    "P13": {"S": 1.0, "Rs": 4.5, "P": 0.6, "W": 3.0, "sigma0": 0.0},
}
