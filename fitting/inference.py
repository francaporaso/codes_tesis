import numpy as np


class Likelihood:
    OBSERVABLES = {
        "sigma": ("Sigma", "covS", "sigma"),
        "delta_sigma": ("DSigma_t", "covDSt", "delta_sigma"),
        "kappa": ("Kappa", "covK", "kappa"),
    }

    def __init__(
        self, data, model, param_limits, observable, cov_mode, fixed_params=None
    ):

        self.R = data.R
        # self.hartlap_factor = (data.Njk-len(self.R)-2)/(data.Njk-1)
        self.hartlap_factor = 1

        try:
            data_attr, cov_attr, func_name = self.OBSERVABLES[observable]
        except KeyError:
            raise ValueError(
                f"observable must be one of {list(self.OBSERVABLES)}, got {observable!r}"
            )

        self.ydata = getattr(data, data_attr)
        self.cov = getattr(data, cov_attr)
        self.func = getattr(model, func_name)

        if cov_mode == "full":
            self.yerr = np.linalg.inv(self.cov) * self.hartlap_factor
        elif cov_mode == "diag":
            # this allows to use log_likelihood with both diag or full covariance!
            self.yerr = np.zeros_like(self.cov)
            np.fill_diagonal(self.yerr, 1.0 / np.diag(self.cov))
        else:
            raise ValueError('cov_mode must be either "full" or "diag"')

        all_params = model.params[observable]
        fixed_params = fixed_params or {}
        self.param_name = [p for p in all_params if p not in fixed_params]
        self.limits = {k: param_limits[k] for k in self.param_name}
        self.nparams = len(self.param_name)

        # needed if there are fixed params...
        # creates a template for where to fill free params draw
        self._template = np.empty(len(all_params))
        self._freeidx = np.array([all_params.index(p) for p in self.param_name])
        for p, v in fixed_params.items():
            self._template[all_params.index(p)] = v

    def log_likelihood(self, theta):
        full = self._template.copy()
        full[self._freeidx] = theta
        model = self.func(self.R, *full)  # *self.rhomean
        dist = self.ydata - model
        return -0.5 * np.dot(dist, np.dot(self.yerr, dist))

    def log_prior(self, theta):
        ### tener cuidado con el orden de lims!
        if np.prod(
            [
                self.limits[self.param_name[j]][0]
                < theta[j]
                < self.limits[self.param_name[j]][1]
                for j in range(self.nparams)
            ],
            dtype=bool,
        ):
            return 0
        return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)


# - easy way to make a joint fit for different data but model with the same parameters.
class JointLikelihood:
    # should be a composition of two or more Likelihood instances
    pass
