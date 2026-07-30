import numpy as np
from dataclasses import dataclass
from astropy.io import fits
import pickle
# from astropy.table import Table


@dataclass
class DataProfile:
    redshift: np.float64
    Njk: np.float64
    R: np.ndarray
    Sigma: np.ndarray = None
    DSigma_t: np.ndarray = None
    DSigma_x: np.ndarray = None
    Kappa: np.ndarray = None
    covS: np.ndarray = None
    covDSt: np.ndarray = None
    covDSx: np.ndarray = None
    covK: np.ndarray = None

    def get(self, observable):
        if observable == "sigma":
            pass
        if observable == "delta_sigma":
            pass
        if observable == "kappa":
            return self.Kappa, np.sqrt(np.diag(self.covK))
        else:
            raise KeyError(f"observable not recognized. got {observable!r}")


# the **kwargs requires giving the arg name when calling this function
# ex: data = read_dataprofile_fits(name='myprofile.fits')
# this is not going to work: data = read_dataprofile_fits('myprofile.fits').
def read_dataprofile_fits(filename, binning="lin", **kwargs):

    binspace = np.linspace if binning == "lin" else np.geomspace
    with fits.open(name=filename, **kwargs) as f:
        hd = f[0].header
        dt = f["profiles"].data
        if "R" in f["profiles"].columns.names:
            R = dt["R"]
        else:
            R = binspace(hd["RIN"], hd["ROUT"], hd["NBINS"])

        data = DataProfile(
            R=R,
            redshift=hd["Z_MEAN"],
            Njk=hd["NJK"],
            Sigma=dt["Sigma"],
            DSigma_t=dt["DSigma_t"],
            DSigma_x=dt["DSigma_x"],
            covS=f["cov_sigma"].data,
            covDSt=f["cov_dsigma_t"].data,
            covDSx=f["cov_dsigma_x"].data,
        )
    return data


def save_chains_h5():
    pass


def read_dataprofile_pickle(filename):

    with open(filename, "rb") as f:
        data = pickle.load(f)

        dataobject = DataProfile(
            R=data["merged_data"][0]["r_frac"],
            Kappa=data["merged_data"][0]["profile"],
            covK=data["merged_data"][0]["cov_matrix"],
            redshift=data["merged_data"][0]["z_mean"],
            Njk=data["parameters"]["n_subsamples"],
        )

    return dataobject
