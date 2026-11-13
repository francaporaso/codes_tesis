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

        self.all_params = list(model.params[observable])
        fixed_params = fixed_params or {}
        # keep only fixed params used by this observable (e.g. no sigma0 for delta_sigma)
        self.fixed = {k: v for k, v in fixed_params.items() if k in self.all_params}
        self.param_name = [p for p in self.all_params if p not in self.fixed]
        self.limits = {k: param_limits[k] for k in self.param_name}
        self.nparams = len(self.param_name)
        self.ndof = len(self.R) - self.nparams

    def params(self, theta):
        """free-parameter vector -> full {name: value} dict (fixed included),
        ordered as in the model signature"""
        values = {**dict(zip(self.param_name, theta)), **self.fixed}
        return {k: values[k] for k in self.all_params}

    def predict(self, theta):
        # positional: the base-class methods only accept *params
        return self.func(self.R, *self.params(theta).values())

    def chi2_red(self, theta):
        return -2.0 * self.log_likelihood(theta) / self.ndof

    def log_likelihood(self, theta):
        dist = self.ydata - self.predict(theta)
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
