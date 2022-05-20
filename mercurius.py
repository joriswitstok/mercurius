#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for running MultiNest fits on dust FIR SEDs.

Joris Witstok, 17 November 2021
"""

import os, sys, shutil
from mock import patch
if __name__ == "__main__":
    print("Python", sys.version)

import numpy as np
rng = np.random.default_rng(seed=9)
import math

from scipy.stats import gamma, norm, gaussian_kde
from scipy.special import erf

from pymultinest.solve import Solver
from emcee import EnsembleSampler

import corner

import matplotlib
if __name__ == "__main__":
    print("Matplotlib", matplotlib.__version__, "(backend: " + matplotlib.get_backend() + ')')
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("ticks")

from aux.star_formation import SFR_L
from aux.infrared_luminosity import T_CMB_obs, inv_CMB_heating, CMB_correction, Planck_func, calc_FIR_SED
from aux.legend_handler import BTuple, BTupleHandler

# Import astropy cosmology, given H0 and Omega_matter
from astropy.cosmology import FLRW, FlatLambdaCDM



wl_CII = 157.73617000e-6 # m
nu_CII = 299792458.0 / wl_CII # Hz

# Dust mass absorption coefficient at frequency nu_star
# nu_star = 2998.0e9 # Hz
# k_nu_star = 52.2 # cm^2/g

# Values for dust ejected from SNe after reverse shock destruction from Hirashita et al. (2014): https://ui.adsabs.harvard.edu/abs/2014MNRAS.443.1704H/abstract
wl_star = wl_CII
nu_star = nu_CII # Hz
k_nu_star = 8.94 # cm^2/g (other values range from ~5 to ~30)

# Prior range on the dust emissivity, beta
beta_range = [1, 5]

# Solar luminosity in erg/s
L_sun_ergs = 3.828e26 * 1e7
# Solar mass in g
M_sun_g = 1.989e33

T_dusts_global = np.arange(20, 120, 10)
beta_IRs_global = np.arange(1.5, 2.05, 0.1)

dust_cmap = sns.color_palette("inferno", as_cmap=True)
dust_norm = matplotlib.colors.Normalize(vmin=0, vmax=T_dusts_global[-1])





def log_prob_Gauss(x, mu, cov):
    # Simple logarithmic probability function for Gaussian distribution
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

def mcmc_sampler(means, cov, n_dim=2, n_steps=10000, nwalkers=32):
    print("Running {:d}-dimensional MCMC sampler with {:d} walkers performing {:d} steps on function...".format(n_dim, nwalkers, n_steps))

    # Set up walkers and initial positions
    p0 = rng.normal(loc=means, scale=np.sqrt(np.diagonal(cov)), size=(nwalkers, n_dim))

    sampler = EnsembleSampler(nwalkers, n_dim, log_prob_Gauss, args=[means, cov])
    sampler.run_mcmc(p0, n_steps, progress=True)
    
    tau = sampler.get_autocorr_time()
    print("\nAutocorrelation times:", *tau)
    tau_max = math.ceil(0.5*np.max(tau))

    # Throw away a few times this number of steps as “burn-in”, discard the initial 3 tau_max, 
    # and thin by about half the autocorrelation time (15 steps), and flatten the chain
    flat_samples = sampler.get_chain(discard=3*tau_max, thin=int(0.5*tau_max), flat=True)

    return flat_samples

class lcoord_funcs:
    def __init__(self, rowi, coli, z):
        self.rowi = rowi
        self.coli = coli
        
        self.z = z
    
    def lamrf2lamobs(self, l_emit):
        # From micron to mm
        return l_emit / 1e3 * (1.0 + self.z)
    def lamobs2lamrf(self, l_obs):
        # From mm to micron
        return l_obs * 1e3 / (1.0 + self.z)
    def lamrf2nuobs(self, l_emit):
        # From micron to GHz
        return 299792.458 / l_emit / (1.0 + self.z)
    def nuobs2lamrf(self, nu_obs):
        # From GHz to micron
        return 299792.458 / nu_obs * (1.0 + self.z)
    def nuobs2lamobs(self, nu_obs):
        # From GHz to mm
        return 299.792458 / nu_obs
    def lamobs2nuobs(self, l_obs):
        # From mm to GHz
        return 299.792458 / l_obs

def FIR_SED_spectrum(theta, z, D_L, l0_dict, lambda_emit=None):
    logM_dust = theta[0]
    T_dust = theta[1]
    beta_IR = theta[2]
    
    if l0_dict.get("value", -1) is not None:
        # Dust area in kpc^2 converted to cm^2 (1 kpc = 3.085677e21 cm)
        l0_dict["cont_area_cm2"] = l0_dict["cont_area_kpc2"] * 3.085677e21**2
    
    if l0_dict["assumption"] == "fixed":
        optically_thick_lambda_0 = l0_dict["value"]
    elif l0_dict["assumption"] == "self-consistent":
        # Dust mass surface density in g/cm^2
        Sigma_dust = 10**logM_dust * M_sun_g / l0_dict["cont_area_cm2"]
        
        # Compute the wavelength where the optical depth, τ = κ_nu Σ, becomes 1 (κ in cm^2/g, Σ in g/cm^2)
        # κ_nu is given by κ_nu = κ_nu_star * (nu/nu_star)**beta_IR; κ Σ becomes 1 when (nu/nu_star)**beta_IR = 1/(κ_nu_star Σ)
        nu_0 = nu_star / (Sigma_dust * k_nu_star)**(1.0/beta_IR) # Hz

        optically_thick_lambda_0 = 299792458.0 * 1e6 / nu_0 # micron (nu_0 is in Hz)

    # Compute FIR SED fluxes only at the given (rest-frame) wavelengths
    lambda_emit, S_nu_emit = calc_FIR_SED(z=z, beta_IR=beta_IR, T_dust=T_dust,
                                            optically_thick_lambda_0=optically_thick_lambda_0,
                                            return_spectrum=True, lambda_emit=lambda_emit)
    nu_emit = 299792458.0 * 1e6 / lambda_emit # Hz (lambda is in micron)
    
    # Flux density needs to be corrected for observing against the the CMB (NB: can be negative if T_dust < T_CMB), and then normalised
    CMB_correction_factor = CMB_correction(z=z, nu0_emit=nu_emit, T_dust=T_dust)
    if np.all(CMB_correction_factor < 0):
        # No normalisation possible (T_dust < T_CMB) so 0 likelihood
        return -np.inf

    # Normalise fluxes using dust mass
    # S_nu in Jy, 1e-26 = W/m^2/Hz/Jy, D_L in cm, κ_nu in cm^2/g, Planck_func in W/m^2/Hz, M_sun = 1.989e33 g
    nu_ref = nu_CII # Hz
    k_nu = k_nu_star * (nu_ref/nu_star)**beta_IR
    
    # Compute (properly normalised) FIR SED flux at the reference (rest-frame) frequency, given the dust mass (NB: uncorrected for CMB effects)
    if l0_dict.get("value", -1) is None:
        # Compute flux density in optically thin limit, assuming 1 - exp(-τ) ~ τ = κ_nu Σ_dust
        S_nu_emit_norm_ref = 10**logM_dust * M_sun_g * k_nu * Planck_func(nu_ref, T_dust) / D_L**2 * 1e26
    else:
        assert l0_dict["assumption"] == "self-consistent" or l0_dict.get("value", None)
        # Compute flux density in general opacity case (see Jones et al. 2020), first computing the optical depth τ;
        # convert M_dust from units of M_sun to g, κ_nu is in cm^2/g, area is in cm^2
        if l0_dict["assumption"] == "self-consistent":
            tau = 10**logM_dust * M_sun_g * k_nu / l0_dict["cont_area_cm2"]
        else:
            # Compute the optical depth from the fixed value of lambda_0
            nu_0 = 299792458.0 * 1e6 / optically_thick_lambda_0 # Hz (lambda_0 is in micron)
            tau = (nu_ref/nu_0)**beta_IR
            # Calculate the dust surface area in cm^2 required to achieve the fixed lambda_0, given the dust mass
            l0_dict["cont_area_cm2"] = 10**logM_dust * M_sun_g * k_nu / tau
            Sigma_dust = 10**logM_dust * M_sun_g / l0_dict["cont_area_cm2"]
        
        S_nu_emit_norm_ref = (1.0 - np.exp(-tau)) * Planck_func(nu_ref, T_dust) * l0_dict["cont_area_cm2"] / D_L**2 * 1e26
    
    # Compute unnormalised FIR SED flux specifically for the reference (rest-frame) frequency (NB: also uncorrected for CMB effects)
    S_nu_emit_ref = calc_FIR_SED(z=z, beta_IR=beta_IR, T_dust=T_dust,
                                    optically_thick_lambda_0=optically_thick_lambda_0,
                                    return_spectrum=True, lambda_emit=299792458.0 * 1e6 / nu_ref)[1]
    # Compute the normalisation of the spectrum by the ratio of the two
    # NB: both are uncorrected for CMB effects, but this correction cancels out
    norm = S_nu_emit_norm_ref / S_nu_emit_ref
    
    # Normalise emitted flux density and observed, CMB-corrected flux density;
    # NB: S_nu_obs needs a factor (1+z) compared to emitted one
    S_nu_emit *= norm # Jy
    S_nu_obs = S_nu_emit * CMB_correction_factor * (1.0+z) # Jy

    return (lambda_emit, nu_emit, S_nu_emit, S_nu_obs)

class MN_FIR_SED_solver(Solver):
    def __init__(self, z, D_L, l0_dict, fluxes, flux_errs, uplims, wls,
                    fixed_beta=None, fit_uplims=True, uplim_nsig=None, M_star=np.nan, T_max=150.0, **solv_kwargs):
        print("Initialising MultiNest Solver object...")
        
        self.z = z
        self.D_L = D_L
        self.l0_dict = l0_dict

        self.T_CMB = T_CMB_obs * (1.0 + self.z) # K

        self.fluxes = fluxes
        self.flux_errs = flux_errs
        self.uplims = uplims
        self.wls = wls
        
        self.fixed_beta = fixed_beta
        self.fit_uplims = fit_uplims
        if uplim_nsig is None:
            assert not np.any(self.uplims)
        else:
            self.uplim_nsig = uplim_nsig

        self.set_prior(M_star=M_star, T_max=T_max)
        if solv_kwargs["verbose"]:
            super().__init__(**solv_kwargs)
        else:
            devnull = open(os.devnull, 'w')
            with patch("sys.stderr", devnull):
                super().__init__(**solv_kwargs)
    
    def set_prior(self, M_star, T_max):
        if np.isnan(M_star):
            logM_dust_range = [4, 12]
        else:
            logM_dust_range = [4, np.log10(M_star)]
        
        T_dust_range = [self.T_CMB, T_max]

        self.cube_range = [logM_dust_range, T_dust_range]
        if not self.fixed_beta:
            self.cube_range.append(beta_range)
        
        self.cube_range = np.array(self.cube_range)

    def Prior(self, cube):
        assert hasattr(self, "cube_range")
        # Scale the input unit cube to apply uniform priors across all parameters (except the temperature)
        for di in range(len(cube)):
            if di == 2:
                # Use a normal distribution for the dust emissivity (prior belief: beta likely around 1.8)
                cube[di] = norm.ppf(cube[di], loc=1.8, scale=0.25)
            elif di == 1:
                # Use a gamma distribution for the dust temperature (prior belief: unlikely to be near CMB or extremely high temperature)
                cube[di] = gamma.ppf(cube[di], a=1.5, loc=0, scale=self.cube_range[di, 0]/0.5) + self.cube_range[di, 0]
            else:
                cube[di] = cube[di] * (self.cube_range[di, 1] - self.cube_range[di, 0]) + self.cube_range[di, 0]
        
        return cube

    def LogLikelihood(self, cube):
        if self.fixed_beta:
            theta = (cube[0], cube[1], self.fixed_beta)
        else:
            theta = (cube[0], cube[1], cube[2])
        
        model_fluxes = FIR_SED_spectrum(theta=theta, z=self.z, D_L=self.D_L, l0_dict=self.l0_dict, lambda_emit=self.wls)[3]

        # Calculate the log likelihood given both detections and upper limits according to the formalism in Sawicki et al. (2012):
        # first part is a regular normalised least-squares sum, second part is integrating the Gaussian probability up to the 1σ detection limit
        # (see Sawicki et al. 2012 for details: https://ui.adsabs.harvard.edu/abs/2012PASP..124.1208S/abstract)
        ll = -0.5 * np.nansum(((self.fluxes[~self.uplims] - model_fluxes[~self.uplims])/self.flux_errs[~self.uplims])**2)

        if np.any(self.uplims) and self.fit_uplims:
            sigma_uplims = self.fluxes[self.uplims]/self.uplim_nsig
            ll += np.nansum( np.log( np.sqrt(np.pi) / 2.0 * (1.0 + erf((sigma_uplims - model_fluxes[self.uplims]) / (np.sqrt(2.0) * sigma_uplims))) ) )
        
        return ll

class FIR_SED_fit:
    def __init__(self, l0_list, analysis, mnrfol, fluxdens_unit="muJy", l_min=None, l_max=None,
                    fixed_T_dust=50.0, fixed_beta=None, cosmo=None,
                    T_lolim=False, T_uplim=False,
                    obj_color=None, l0_linestyles=None, pformat=None, dpi=None, mpl_style=None, verbose=True):
        """Class `FIR_SED_fit` for interacting with the fitting and plotting routines of `mercurius`.

        Parameters
        ----------
        l0_list : list
            A list of opacity model classifiers. Entries can be `None` for an optically thin model,
            `"self-consistent"` for a self-consistent opacity model, or a float setting a fixed
            value of `lambda_0`, the wavelength in micron setting the SED's transition point between
            optically thin and thick.
        analysis : bool
            Controls whether figures are made with slightly more detail (`True`) or  (`False`).
        mnrfol : str
            Name of the folder used for saving results of SED fits.
        fluxdens_unit : str, optional
            Unit of flux density in figures, by default `"muJy"`.
        l_min : {None, float}, optional
            Value in micron of the lower rest-frame wavelength bound in figures.
            Default is `None`, which results in `50.0`.
        l_max : {None, float}, optional
            Value in micron of the upper rest-frame wavelength bound in figures.
            Default is `None`, which results in `300.0`.
        fixed_T_dust : float, optional
            Fixed value of the dust temperature in Kelvin, set to `50.0` by default.
        fixed_beta : {`None`, float}, optional
            Fixed value of the dust emissivity beta, or `None` if variable (default).
        cosmo : instance of astropy.cosmology.FLRW, optional
            Custom cosmology (see `astropy` documentation for details).
        T_lolim : bool, optional
            Retrieve a lower limit (95% confidence) of the dust temperature? (Default: `False`.)
        T_uplim : bool, optional
            Retrieve an upper limit (95% confidence) of the dust temperature? (Default: `False`.)
        obj_color : {`None`, tuple, str}, optional
            A custom `matplotlib` colour highlighting the name of the object in figures (see
            `matplotlib` documentation on how to specify colours). Default is `None`: no
            colour is used.
        l0_linestyles : dict, optional
            Dictionary for custom `matplotlib` linestyles for each opacity model in figures (see
            `matplotlib` documentation on how to specify linestyles). Default is
            `{None: '--', "self-consistent": '-', 100.0: '-.', 200.0: ':'}`.
        pformat : str, optional
            Extension to be used for saving figures. Default is `None`, which sets the format to
            `".png"` (i.e. PNG image) if `analysis` is `True`, otherwise it is `".pdf"` (i.e. PDF).
        dpi : {int, float}, optional
            Quality to be used for saving figures. Default is `None`, which sets `dpi` to `150`.
        mpl_style : str, optional
            Path to custom `matplotlib` style file. Default is `None`, in which case no
            custom file is used.
        verbose : bool, optional
            Controls whether progress updates and results are printed. Default is `True`.

        """

        if verbose:
            print("\nInitialising FIR SED fitting object...")
        
        self.l0_list = l0_list
        self.analysis = analysis
        self.mnrfol = mnrfol

        if fluxdens_unit:
            assert fluxdens_unit in ["Jy", "mJy", "muJy", "nJy"]
            self.fluxdens_unit = fluxdens_unit
            # Setting this unit converter to 1 won't change any of the units, so flux density spectra are in Jy;
            # setting it to 1e3 will change the units of flux density spectra to mJy, 1e6 to μJy, etc.
            self.fd_conv = {"Jy": 1, "mJy": 1e3, "muJy": 1e6, "nJy": 1e9}[fluxdens_unit]
        else:
            self.fd_conv = 1

        if l_min is None:
            # Plot from rest-frame wavelength of 50 μm
            self.l_min = 50 # micron
        else:
            self.l_min = l_min
        if l_max is None:
            # Plot to rest-frame wavelength of 300 μm
            self.l_max = 300 # micron
        else:
            self.l_max = l_max
        
        self.set_fixed_values(fixed_T_dust, fixed_beta)
        if verbose:
            if self.fixed_beta:
                print("Chosen fiducial values T_dust = {:.0f} K, β = {:.2g}...".format(fixed_T_dust, fixed_beta))
            else:
                print("Chosen fiducial value T_dust = {:.0f} K (β is freely varying)...".format(fixed_T_dust))

        if cosmo is None:
            self.cosmo = FlatLambdaCDM(H0=70.0, Om0=0.300)
        else:
            assert isinstance(cosmo, FLRW)
            self.cosmo = cosmo

        self.T_lolim = T_lolim
        self.T_uplim = T_uplim
        
        self.fresh_calculation = {l0: False for l0 in self.l0_list}
        
        self.obj_color = obj_color

        if l0_linestyles is None:
            self.l0_linestyles = {None: '--', "self-consistent": '-', 100.0: '-.', 200.0: ':'}
        else:
            self.l0_linestyles = l0_linestyles
        
        if pformat is None:
            self.pformat = ".png" if self.analysis else ".pdf"
        else:
            self.pformat = pformat
        
        if dpi is None:
            self.dpi = 150
        else:
            self.dpi = dpi
        
        if mpl_style is not None:
            # Load style file
            plt.style.use(mpl_style)

        self.verbose = verbose
        if self.verbose:
            print("Initialisation done!")
    
    def set_fixed_values(self, fixed_T_dust, fixed_beta):
        """Function for changing the fixed values of the dust temperature and emissivity.

        Parameters
        ----------
        fixed_T_dust : float
            Fixed value of the dust temperature in Kelvin.
        fixed_beta : {`None`, float}
            Fixed value of the dust emissivity beta, or `None` if variable.

        """

        self.fixed_T_dust = fixed_T_dust
        self.fixed_beta = fixed_beta
        self.beta_str = "beta_{:.1f}".format(self.fixed_beta) if self.fixed_beta else "vary_beta"

    def set_data(self, obj, z,
                    lambda_emit_vals, S_nu_vals, S_nu_errs, cont_uplims, lambda_emit_ranges=None, cont_excludes=None, uplim_nsig=3.0,
                    obj_M=np.nan, obj_M_lowerr=np.nan, obj_M_uperr=np.nan, SFR_UV=np.nan, SFR_UV_err=np.nan,
                    cont_area=None, cont_area_uplim=False, reference=None):
        """Function for setting the photometric data of the object.

        Parameters
        ----------
        obj : str
            Object name.
        z : float
            Redshift of the object.
        lambda_emit_vals : array_like
            Values of the rest-frame wavelengths in microns.
        S_nu_vals : array_like
            Values of the flux density in Jy.
        S_nu_errs : array_like
            Values of the flux density uncertainty in Jy.
        cont_uplims : array_like
            Boolean array indicating which data points are upper limits.
        lambda_emit_ranges : {`None`, array_like}, optional
            Lower and upper wavelength bounds given as an offset from the main wavelength values
            (for visualisation only). Needs to be shape (N, 2) for N photometric data points.
            Default is `None`, in which case bounds will not be shown.
        cont_excludes : {`None`, array_like}, optional
            Boolean array indicating which data points are ignored in the fitting process.
            Default is `None` where none of the data are excluded.
        uplim_nsig : float, optional
            How many sigma is an upper limit given as? Default: `3.0`.
        obj_M : float, optional
            Stellar mass of object, in units of solar masses. Default: `np.nan` (i.e. unknown).
        obj_M_lowerr : float, optional
            Lower error on the stellar mass of the object, in units of solar masses.
            Default: `np.nan` (i.e. unknown).
        obj_M_uperr : float, optional
            Upper error on the stellar mass of the object, in units of solar masses.
            Default: `np.nan` (i.e. unknown).
        SFR_UV : float, optional
            The object's star formation rate (SFR) in the UV. Default: `np.nan` (i.e. unknown).
        SFR_UV_err : float, optional
            Uncertainty on the object's star formation rate (SFR) in the UV.
            Default: `np.nan` (i.e. unknown).
        cont_area : {`None`, float}, optional
            Area of the dust emission in square kiloparsec, to be used in the self-consistent
            opacity model. Default: `None` (i.e. unknown).
        cont_area_uplim : bool, optional
            Indicates whether the area is an upper limit. Default: `False`.

        """

        if self.verbose:
            print("\nSetting photometric data points and other properties of {}...".format(obj))
        
        self.obj = obj
        self.obj_fn = self.obj.replace(' ', '_')
        self.z = z
        self.D_L = self.cosmo.luminosity_distance(self.z).to("cm").value
        
        self.obj_M = obj_M
        self.obj_M_lowerr = obj_M_lowerr
        self.obj_M_uperr = obj_M_uperr
        
        self.SFR_UV = SFR_UV
        self.SFR_UV_err = SFR_UV_err
        
        valid_fluxes = np.isfinite(np.asarray(S_nu_vals))
        if cont_excludes is None:
            self.cont_excludes = np.tile(False, np.sum(valid_fluxes))
        else:
            self.cont_excludes = np.asarray(cont_excludes)[valid_fluxes]
        
        if self.verbose:
            if np.any(cont_uplims):
                print("Upper limits present ({:d}/{:d} data points), will be taken into account...".format(np.sum(cont_uplims), len(cont_uplims)))
            if np.any(self.cont_excludes):
                print("Excluded photometry present ({:d}/{:d} data points), will not be taken into account...".format(np.sum(self.cont_excludes), len(self.cont_excludes)))
            if np.any(~valid_fluxes):
                print("Warning: invalid fluxes present ({:d}/{:d} data points), will be ignored...".format(np.sum(~valid_fluxes), len(valid_fluxes)))
        
        self.lambda_emit_vals = np.asarray(lambda_emit_vals)[valid_fluxes]
        wl_order = np.argsort(self.lambda_emit_vals)
        self.lambda_emit_vals = self.lambda_emit_vals[wl_order]
        self.cont_excludes = self.cont_excludes[wl_order]
        
        if lambda_emit_ranges is None:
            self.lambda_emit_ranges = np.tile(np.nan, (np.sum(valid_fluxes), 2))
        else:
            self.lambda_emit_ranges = np.asarray(lambda_emit_ranges)[valid_fluxes][wl_order]
        self.S_nu_vals = np.asarray(S_nu_vals)[valid_fluxes][wl_order]
        self.S_nu_errs = np.asarray(S_nu_errs)[valid_fluxes][wl_order]
        self.cont_uplims = np.asarray(cont_uplims)[valid_fluxes][wl_order]
        self.all_uplims = np.all(self.cont_uplims[~self.cont_excludes])
        
        self.uplim_nsig = uplim_nsig
        
        if self.verbose:
            print('', "Wavelength (μm)\tFlux ({unit})\tError ({unit})\tUpper limit?\tExclude?".format(unit=self.fluxdens_unit.replace("mu", 'μ')),
                    *["{:.5g}\t\t{:.5g}\t\t{}\t\t{}\t\t{}".format(wl, f*self.fd_conv, 'N/A' if u else "{:.5g}".format(e*self.fd_conv), u, exc) \
                        for wl, f, e, u, exc in zip (self.lambda_emit_vals, self.S_nu_vals, self.S_nu_errs, self.cont_uplims, self.cont_excludes)], sep='\n')
        
        self.valid_cont_area = cont_area is not None and np.isfinite(cont_area) and cont_area > 0.0
        self.cont_area = cont_area
        self.cont_area_uplim = cont_area_uplim
        
        self.reference = reference

        # Combined detections for object
        self.n_meas = self.lambda_emit_vals.size
        self.cont_det = ~self.cont_uplims * ~self.cont_excludes
    
    def fit_data(self, pltfol=None, fit_uplims=True, return_samples=False, save_results=True, lambda_emit=None,
                    n_live_points=400, evidence_tolerance=0.5, sampling_efficiency=0.8, max_iter=0,
                    force_run=False, skip_redundant_calc=False, ann_size="small", mnverbose=False):
        """Function for fitting the photometric data of the object with greybody spectra for
        all opacity models set in `l0_list`.

        Parameters
        ----------
        pltfol : str, optional
            Path to folder in which a corner plot of the results is saved. Default is
            `None`, so that no corner plot is saved.
        fit_uplims : bool, optional
            Include upper limits in the fitting routine?
        return_samples : bool, optional
            Return samples directly? Default: `False`.
        save_results : bool, optional
            Save results produced by MultiNest after a run and save resulting samples
            in compressed NumPy format (used for plotting afterwards)? Default: `True`.
        lambda_emit : array_like, optional
            Values in micron to be used as the rest-frame wavelengths in results,
            including the (F)IR luminosities. Default is `None`, which will default
            to a linearly spaced array of 10,000 points ranging between 4 and 1100
            micron in the function `calc_FIR_SED` (located in
            `mercurius/aux/infrared_luminosity.py`).
        n_live_points : int, optional
            Number of live points used in the MultiNest run. See the `pymultinest`
            documentation for details.
        evidence_tolerance : float, optional
            Evidence tolerance value used in the MultiNest run. See the `pymultinest`
            documentation for details.
        sampling_efficiency : float, optional
            Sampling efficiency value used in the MultiNest run. See the `pymultinest`
            documentation for details.
        max_iter : int, optional
            Maximum number of iterations of the MultiNest run. See the `pymultinest`
            documentation for details.
        force_run : bool, optional
            Force a new run of the fitting routine, even if previous results are found.
        skip_redundant_calc : bool, optional
            Entirely skip the data fitting (including the calculation of ) if results
            are already present?
        ann_size : float or {'xx-small', 'x-small', 'small', 'medium',
                    'large', 'x-large', 'xx-large'}, optional
            Argument used by `matplotlib.text.Text` for the font size of the annotation.
            Default: `"small"`.
        mnverbose : bool, optional
            Controls whether the main output of the `pymultinest` solver is shown.
        
        Returns
        ----------
        flat_samples : tuple
            If `return_samples` is set to `True`, a tuple containing two lists is returned:
            one of arrays containing N samples of various parameters and one of the
            parameter names (the exact parameters depending on whether the dust emissivity
            is varied, with `fixed_beta` equal to `None`, or kept constant).

        """
        
        if self.all_uplims:
            print("Warning: only upper limits for {} specified! No MultiNest fit performed...".format(self.obj))
            return np.ones((2, 0)) if return_samples else 1
        
        if not return_samples and not save_results:
            print("Warning: results neither returned nor saved...")

        # Set percentiles to standard ±1σ confidence intervals around the median value
        percentiles = [0.5*(100-68.2689), 50, 0.5*(100+68.2689)]
        
        for l0 in self.l0_list:
            # Without knowing the source's area, can only fit an optically thin SED or one with fixed lambda_0
            if l0 == "self-consistent" and not self.valid_cont_area:
                print("\nWithout a valid area, cannot run MultiNest fit with a self-consistent lambda_0 for {}...".format(self.obj))
                continue
            
            l0_dict = {"cont_area_kpc2": self.cont_area}
            if l0 == "self-consistent":
                l0_dict["assumption"] = l0
            else:
                l0_dict["assumption"] = "fixed"
                l0_dict["value"] = l0
            
            l0_str, l0_txt = self.get_l0string(l0)

            # Run fit on data
            
            n_dim = 3 - bool(self.fixed_beta)

            samples_fname = self.mnrfol + "{}_MN_FIR_SED_flat_samples_{}{}.npz".format(self.obj_fn, self.beta_str, l0_str)
            if not force_run and skip_redundant_calc and os.path.isfile(samples_fname):
                print("\nResults already present: skipping {:d}-dimensional MultiNest fit".format(n_dim),
                        "with {}{}".format("β = {:.1f}".format(self.fixed_beta) if self.fixed_beta else "varying β", l0_txt),
                        "for {}...".format(self.obj))
                continue
            obtain_MN_samples = force_run or not os.path.isfile(samples_fname)
            
            omnrfol = self.mnrfol + "MultiNest_{}/".format(self.obj)
            if not os.path.exists(omnrfol):
                os.makedirs(omnrfol)

            if obtain_MN_samples:
                if self.verbose:
                    print("\nRunning {:d}-dimensional MultiNest fit".format(n_dim),
                            "with {}{}".format("β = {:.1f}".format(self.fixed_beta) if self.fixed_beta else "varying β", l0_txt),
                            "for {}...".format(self.obj))
                
                currentdir = os.getcwd()
                
                try:
                    os.chdir(omnrfol)
                    MN_solv = MN_FIR_SED_solver(z=self.z, D_L=self.D_L, l0_dict=l0_dict,
                                                fluxes=self.S_nu_vals[~self.cont_excludes], flux_errs=self.S_nu_errs[~self.cont_excludes],
                                                uplims=self.cont_uplims[~self.cont_excludes], wls=self.lambda_emit_vals[~self.cont_excludes],
                                                fixed_beta=self.fixed_beta, fit_uplims=fit_uplims, uplim_nsig=self.uplim_nsig,
                                                n_dims=n_dim, outputfiles_basename="MN", n_live_points=n_live_points,
                                                evidence_tolerance=evidence_tolerance, sampling_efficiency=sampling_efficiency, max_iter=max_iter,
                                                resume=False, verbose=mnverbose and self.verbose)
                except Exception as e:
                    os.chdir(currentdir)
                    raise RuntimeError("error occurred while running MultiNest fit...\n{}".format(e))
                
                os.chdir(currentdir)
                if not save_results:
                    shutil.rmtree(omnrfol)
                
                # Note results are also saved as MNpost_equal_weights.dat; load with np.loadtxt(omnrfol + "MNpost_equal_weights.dat")[:, :n_dim]
                flat_samples = MN_solv.samples
                del MN_solv
                
                if save_results:
                    # Save results
                    np.savez_compressed(samples_fname, flat_samples=flat_samples)
                if self.verbose:
                    print("\nFreshly calculated MultiNest samples with {}{}".format("β = {:.1f}".format(self.fixed_beta) if self.fixed_beta else "varying β", l0_txt),
                            "for {}!\nNumber of samples: {:d}, array size: {:.2g} MB".format(self.obj, flat_samples.shape[0], flat_samples.nbytes/1e6))
            else:
                # Read in samples from the MN run
                flat_samples = np.load(samples_fname)["flat_samples"]

                if self.verbose:
                    print("\nFreshly loaded MultiNest samples with {}{}".format("β = {:.1f}".format(self.fixed_beta) if self.fixed_beta else "varying β", l0_txt),
                            "for {}!\nNumber of samples: {:d}, array size: {:.2g} MB".format(self.obj, flat_samples.shape[0], flat_samples.nbytes/1e6))
            
            n_samples = flat_samples.shape[0]
            logM_dust_samples = flat_samples[:, 0]
            T_dust_samples = flat_samples[:, 1]

            rdict = {}

            # There is at least one detection, so set upper limits to False
            rdict["M_dust_uplim"] = False
            rdict["L_IR_uplim"] = False
            
            if self.fixed_beta:
                beta_IR, rdict["beta_IR_lowerr"], rdict["beta_IR_uperr"] = self.fixed_beta, np.nan, np.nan
                flat_samples = np.concatenate((flat_samples, np.tile(beta_IR, n_samples).reshape(n_samples, 1)), axis=1)
            else:
                # Calculate percentiles of dust emissivity, masking any non-finite values
                beta_samples = flat_samples[:, 2]
                beta_perc = np.percentile(beta_samples[np.isfinite(beta_samples)], percentiles, axis=0)
                beta_IR, rdict["beta_IR_lowerr"], rdict["beta_IR_uperr"] = beta_perc[1], *np.diff(beta_perc)
            rdict["beta_IR"] = beta_IR
            
            M_dust_perc = np.percentile(10**logM_dust_samples, percentiles, axis=0)
            rdict["M_dust"], rdict["M_dust_lowerr"], rdict["M_dust_uperr"] = M_dust_perc[1], *np.diff(M_dust_perc)
            
            if self.valid_cont_area:
                # Dust mass surface density in M_sun/pc^2 (area in kpc^2 converted to pc^2 by multiplying by (10^3)^2)
                Sigma_dust_samples = 10**logM_dust_samples / (self.cont_area * 1e6)
                Sigma_dust_perc = np.percentile(Sigma_dust_samples, percentiles, axis=0)
                rdict["Sigma_dust"], rdict["Sigma_dust_lowerr"], rdict["Sigma_dust_uperr"] = Sigma_dust_perc[1], *np.diff(Sigma_dust_perc)
                
                # Convert from M_sun/pc^2 to g/cm^2 (1 pc = 3.085677e18 cm)
                Sigma_dust_samples *= M_sun_g / (3.085677e18)**2

                # Compute the wavelength where the optical depth, τ ~ κ_nu Σ, becomes 1 (κ in cm^2/g, Σ in g/cm^2)
                # κ_nu is given by κ_nu = κ_nu_star * (nu/nu_star)**beta_IR; κ Σ becomes 1 when (nu/nu_star)**beta_IR = 1/(κ_nu_star Σ)
                nu_0_samples = nu_star / (Sigma_dust_samples * k_nu_star)**(1.0/beta_IR) # Hz
                del Sigma_dust_samples

                lambda_0_samples = 299792458.0 * 1e6 / nu_0_samples # micron (nu_0 is in Hz)
                lambda_0_perc = np.percentile(lambda_0_samples, percentiles, axis=0)
                rdict["lambda_0"], rdict["lambda_0_lowerr"], rdict["lambda_0_uperr"] = lambda_0_perc[1], *np.diff(lambda_0_perc)
                del nu_0_samples
            else:
                rdict["Sigma_dust"], rdict["Sigma_dust_lowerr"], rdict["Sigma_dust_uperr"] = np.nan, np.nan, np.nan
                rdict["lambda_0"], rdict["lambda_0_lowerr"], rdict["lambda_0_uperr"] = np.nan, np.nan, np.nan
            
            if np.isnan(self.obj_M):
                dust_frac_perc = np.tile(np.nan, len(percentiles))
            else:
                if np.isnan(self.obj_M_lowerr) or np.isnan(self.obj_M_uperr):
                    dust_frac_samples = 10**logM_dust_samples/self.obj_M
                else:
                    M_star_samples = mcmc_sampler([self.obj_M], [[(0.5*(self.obj_M_lowerr+self.obj_M_uperr))**2]],
                                                    n_dim=1, n_steps=2500, nwalkers=32)[:, 0]
                    dust_frac_samples = np.clip(10**logM_dust_samples/rng.choice(M_star_samples, size=logM_dust_samples.size, replace=True), 0, 1)
                    del M_star_samples
                dust_frac_perc = np.percentile(dust_frac_samples, percentiles, axis=0)
                del dust_frac_samples
            
            rdict["dust_frac"], rdict["dust_frac_lowerr"], rdict["dust_frac_uperr"] = dust_frac_perc[1], *np.diff(dust_frac_perc)
            rdict["dust_yield_AGB"], rdict["dust_yield_AGB_lowerr"], rdict["dust_yield_AGB_uperr"] = 29 * np.array([rdict["dust_frac"],
                                                                                rdict["dust_frac_lowerr"], rdict["dust_frac_uperr"]])
            rdict["dust_yield_SN"], rdict["dust_yield_SN_lowerr"], rdict["dust_yield_SN_uperr"] = 84 * np.array([rdict["dust_frac"],
                                                                                rdict["dust_frac_lowerr"], rdict["dust_frac_uperr"]])
            
            T_perc = np.percentile(T_dust_samples, percentiles, axis=0)
            T_dust, rdict["T_dust_lowerr"], rdict["T_dust_uperr"] = T_perc[1], *np.diff(T_perc)
            dcolor = dust_cmap(dust_norm(T_dust))
            rdict["T_dust"] = T_dust
            if self.T_lolim:
                rdict["T_lolim"] = np.percentile(T_dust_samples, 5)
                T_lim = rdict["T_lolim"]
            elif self.T_uplim:
                rdict["T_uplim"] = np.percentile(T_dust_samples, 95)
                T_lim = rdict["T_uplim"]
            else:
                T_lim = np.nan
            
            rdict["T_dust_z0"] = inv_CMB_heating(self.z, T_dust, beta_IR)
            rdict["T_dust_z0_lowerr"], rdict["T_dust_z0_uperr"] = np.abs(np.sort(inv_CMB_heating(self.z, np.array([T_dust-rdict["T_dust_lowerr"], T_dust+rdict["T_dust_uperr"]]), beta_IR) - rdict["T_dust_z0"]))
            if self.T_lolim:
                rdict["T_lolim_z0"] = inv_CMB_heating(self.z, rdict["T_lolim"], beta_IR)
            elif self.T_uplim:
                rdict["T_uplim_z0"] = inv_CMB_heating(self.z, rdict["T_uplim"], beta_IR)
            
            # Get the normalised spectrum for the best-fit parameters
            data = [logM_dust_samples, T_dust_samples] if self.fixed_beta else [logM_dust_samples, T_dust_samples, beta_samples]
            H, edges = np.histogramdd(data, bins=50, range=[np.percentile(d, [5, 95]) for d in data])
            argmax = np.unravel_index(H.argmax(), H.shape)
            rdict["theta_ML"] = np.array([np.mean([edges[ai][a:a+2]]) for ai, a in enumerate(argmax)])
            if self.fixed_beta:
                rdict["theta_ML"] = np.append(rdict["theta_ML"], self.fixed_beta)
            # rdict["theta_ML"] = [np.log10(rdict["M_dust"]), T_dust, beta_IR]

            lambda_emit, nu_emit, rdict["S_nu_emit"], rdict["S_nu_obs"] = FIR_SED_spectrum(theta=rdict["theta_ML"],
                                                                                            z=self.z, D_L=self.D_L, l0_dict=l0_dict, lambda_emit=lambda_emit)
            rdict["lambda_emit"] = lambda_emit
            rdict["nu_emit"] = nu_emit

            S_nu_emit_samples = np.array([FIR_SED_spectrum(theta=sample, z=self.z, D_L=self.D_L,
                                                            l0_dict=l0_dict, lambda_emit=lambda_emit)[2] for sample in flat_samples])

            # Loop over samples of S_nu_emit to find uncertainty of L_IR
            L_IR_Lsun_samples = []
            L_FIR_Lsun_samples = []
            lambda_IR = (lambda_emit > 8.0) * (lambda_emit < 1000.0)
            lambda_FIR = (lambda_emit > 42.5) * (lambda_emit < 122.5)

            for S_nu_emit in S_nu_emit_samples:
                # Calculate the integrated IR luminosity between 8 and 1000 μm (integrate Jy over Hz, so convert result to erg/s/cm^2;
                # 1 Jy = 10^-23 erg/s/cm^2/Hz)
                F_IR = np.trapz(S_nu_emit[lambda_IR][np.argsort(nu_emit[lambda_IR])], x=np.sort(nu_emit[lambda_IR])) * 1e-23
                F_FIR = np.trapz(S_nu_emit[lambda_FIR][np.argsort(nu_emit[lambda_FIR])], x=np.sort(nu_emit[lambda_FIR])) * 1e-23
                L_IR_Lsun_samples.append(F_IR * 4.0 * np.pi * self.D_L**2 / L_sun_ergs) # L_sun
                L_FIR_Lsun_samples.append(F_FIR * 4.0 * np.pi * self.D_L**2 / L_sun_ergs) # L_sun

            # Wien's displacement law to find the observed (after correction for CMB attenuation) peak temperature
            T_peak_samples = np.array([2.897771955e3/lambda_emit[np.argmax(S_nu_emit)] for S_nu_emit in S_nu_emit_samples])
            T_perc = np.percentile(T_peak_samples, percentiles)
            rdict["T_peak_val"], rdict["T_peak_lowerr"], rdict["T_peak_uperr"] = T_perc[1], *np.diff(T_perc)
            if self.T_lolim:
                rdict["T_peak_lolim"] = np.percentile(T_peak_samples, 5)
                rdict["T_peak"] = rdict["T_peak_lolim"]
                rdict["T_peak_err"] = np.nan
                rdict["T_peak_constraint"] = "lolim"
            elif self.T_uplim:
                rdict["T_peak_uplim"] = np.percentile(T_peak_samples, 95)
                rdict["T_peak"] = rdict["T_peak_uplim"]
                rdict["T_peak_err"] = np.nan
                rdict["T_peak_constraint"] = "uplim"
            else:
                rdict["T_peak"] = rdict["T_peak_val"]
                rdict["T_peak_err"] = [[rdict["T_peak_lowerr"]], [rdict["T_peak_uperr"]]]
                rdict["T_peak_constraint"] = "range"
            
            del S_nu_emit_samples
            
            S_nu_obs_samples = np.array([FIR_SED_spectrum(theta=sample, z=self.z, D_L=self.D_L,
                                                            l0_dict=l0_dict, lambda_emit=lambda_emit)[3] for sample in flat_samples])
            del flat_samples
            
            S_nu_obs_median = np.median(S_nu_obs_samples, axis=0)
            rdict["S_nu_obs_lowerr"] = S_nu_obs_median - np.percentile(S_nu_obs_samples, 0.5*(100-68.2689), axis=0)
            rdict["S_nu_obs_uperr"] = np.percentile(S_nu_obs_samples, 0.5*(100+68.2689), axis=0) - S_nu_obs_median
            
            del S_nu_obs_samples

            L_IR_perc = np.percentile(L_IR_Lsun_samples, percentiles)
            rdict["L_IR_Lsun"], rdict["L_IR_Lsun_lowerr"], rdict["L_IR_Lsun_uperr"] = L_IR_perc[1], *np.diff(L_IR_perc)
            L_FIR_perc = np.percentile(L_FIR_Lsun_samples, percentiles)
            rdict["L_FIR_Lsun"], rdict["L_FIR_Lsun_lowerr"], rdict["L_FIR_Lsun_uperr"] = L_FIR_perc[1], *np.diff(L_FIR_perc)

            rdict["SFR_IR"] = SFR_L(rdict["L_IR_Lsun"] * L_sun_ergs, band="TIR")
            rdict["SFR_IR_err"] = SFR_L(np.array([rdict["L_IR_Lsun_lowerr"], rdict["L_IR_Lsun_uperr"]]) * L_sun_ergs, band="TIR")

            rdict["SFR"] = self.SFR_UV + rdict["SFR_IR"]
            rdict["SFR_err"] = np.sqrt(np.tile(self.SFR_UV_err, 2)**2 + (rdict["SFR_IR"]/3.0 if rdict["L_IR_uplim"] else rdict["SFR_IR_err"])**2)

            extra_dim = 3
            names = ["logM_dust", "logL_IR", "logL_FIR", "T_dust", "T_peak"]
            main_names = ["logM_dust", "T_dust"]
            data = [logM_dust_samples, np.log10(L_IR_Lsun_samples), np.log10(L_FIR_Lsun_samples), T_dust_samples, T_peak_samples]
            del logM_dust_samples, L_IR_Lsun_samples, L_FIR_Lsun_samples, T_dust_samples, T_peak_samples
            
            n_bins = max(50, n_samples//500)
            bins = [n_bins, n_bins, n_bins, n_bins, n_bins]
            
            labels = [r"$\log_{10} \left( M_\mathrm{dust} \, (\mathrm{M_\odot}) \right)$", r"$\log_{10} \left( L_\mathrm{IR} \, (\mathrm{L_\odot}) \right)$",
                        r"$\log_{10} \left( L_\mathrm{FIR} \, (\mathrm{L_\odot}) \right)$",
                        r"$T_\mathrm{dust} \, (\mathrm{K})$", r"$T_\mathrm{peak} \, (\mathrm{K})$"]
            
            if l0 == "self-consistent":
                extra_dim += 1
                names.insert(1, "lambda_0")
                data.insert(1, lambda_0_samples)
                del lambda_0_samples
                bins.insert(1, n_bins)
                labels.insert(1, r"$\lambda_0$")
            if not self.fixed_beta:
                names.append("beta")
                main_names.append("beta")
                data.append(beta_samples)
                del beta_samples
                bins.append(n_bins)
                labels.append(r"$\beta_\mathrm{IR}$")
            
            # Deselect non-finite data for histograms
            select_data = np.product([np.isfinite(d) for d in data], axis=0).astype(bool)
            if not np.any(select_data):
                print("Warning: MultiNest fit of {} resulted in non-finite parameters...".format(self.obj))
                return np.ones((2, 0)) if return_samples else 1
            data = [d[select_data] for d in data]

            ranges = []
            for n, d in zip(names, data):
                if n in ["logM_dust", "logL_IR", "logL_FIR"]:
                    ranges.append((math.floor(np.min(d)), math.ceil(np.max(d))))
                elif n in ["T_dust", "T_peak"]:
                    ranges.append((math.floor(0.8*T_CMB_obs*(1.0+self.z)) if np.percentile(d, 99) < 40 else 0, math.ceil(np.percentile(d, 99.5))))
            
            if l0 == "self-consistent":
                ranges.insert(1, 0.9)
            if not self.fixed_beta:
                ranges.append((0.75, 5.25))
            
            if pltfol:
                cfig = corner.corner(np.transpose(data), labels=labels, bins=bins, range=ranges,
                                        quantiles=[0.5*(1-0.682689), 0.5, 0.5*(1+0.682689)],
                                        color=dcolor, show_titles=True, title_kwargs=dict(size="small"))

                text = self.obj
                if self.reference:
                    text += " ({})".format(self.reference.replace('(', '').replace(')', ''))
                    size = "medium"
                else:
                    size = "large"
                text += '\n' + r"$z = {:.6g}$, $T_\mathrm{{ CMB }} = {:.2f} \, \mathrm{{ K }}$".format(self.z, T_CMB_obs*(1.0+self.z))
                if not np.isnan(self.obj_M):
                    text += '\n' + r"$M_* = {:.1f}_{{-{:.1f}}}^{{+{:.1f}}} \cdot 10^{{{:d}}} \, \mathrm{{M_\odot}}$".format(self.obj_M/10**math.floor(np.log10(self.obj_M)),
                                self.obj_M_lowerr/10**math.floor(np.log10(self.obj_M)), self.obj_M_uperr/10**math.floor(np.log10(self.obj_M)), math.floor(np.log10(self.obj_M)))
                if not np.isnan(self.SFR_UV):
                    text += r", $\mathrm{{ SFR_{{UV}} }} = {:.0f}{}".format(self.SFR_UV, r'' if np.isnan(self.SFR_UV_err) else r" \pm {:.0f}".format(self.SFR_UV_err)) + \
                            r" \, \mathrm{{M_\odot yr^{{-1}}}}$"
                
                cfig.suptitle(text, size=size)

                # Extract the axes
                axes_c = np.array(cfig.axes).reshape((n_dim+extra_dim, n_dim+extra_dim))

                # Loop over the histograms
                for ri in range(n_dim+extra_dim):
                    for ci in range(ri):
                        axes_c[ri, ci].vlines(np.percentile(data[ci], percentiles), ymin=0, ymax=1,
                                                transform=axes_c[ri, ci].get_xaxis_transform(), linestyles=['--', '-', '--'], color="grey")
                        axes_c[ri, ci].hlines(np.percentile(data[ri], percentiles), xmin=0, xmax=1,
                                                transform=axes_c[ri, ci].get_yaxis_transform(), linestyles=['--', '-', '--'], color="grey")
                        axes_c[ri, ci].plot(np.percentile(data[ci], 50), np.percentile(data[ri], 50), color="grey", marker='s', mfc="None", mec="grey")

                        if names[ri] in main_names and names[ci] in main_names:
                            axes_c[ri, ci].plot(rdict["theta_ML"][main_names.index(names[ci])], rdict["theta_ML"][main_names.index(names[ri])],
                                                color='k', marker='o', mfc="None", mec='k', mew=1.5)
                
                ax_c = axes_c[names.index("T_peak"), names.index("T_dust")]
                ax_c.plot(np.linspace(-10, 200, 10), np.linspace(-10, 200, 10), linestyle='--', color="lightgrey", alpha=0.6)
                
                ax_c = axes_c[names.index("T_dust"), names.index("T_dust")]
                ax_c.axvline(T_CMB_obs*(1.0+self.z), linestyle='--', color='k', alpha=0.6)
                ax_c.annotate(text=r"$T_\mathrm{{ CMB }} (z = {:.6g})$".format(self.z), xy=(T_CMB_obs*(1.0+self.z), 0.5), xytext=(-2, 0),
                                xycoords=ax_c.get_xaxis_transform(), textcoords="offset points", rotation="vertical",
                                va="center", ha="right", size="xx-small", alpha=0.8).set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
                if self.T_lolim or self.T_uplim:
                    ax_c.axvline(T_lim, color="grey", alpha=0.6)
                    ax_c.annotate(text=("Lower" if self.T_lolim else "Upper") + " limit (95% conf.)", xy=(T_lim, 1), xytext=(2, -4),
                                    xycoords=ax_c.get_xaxis_transform(), textcoords="offset points", rotation="vertical",
                                    va="top", ha="left", size="xx-small", alpha=0.8).set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
                
                ax_c = axes_c[names.index("T_peak"), names.index("T_peak")]
                ax_c.axvline(T_CMB_obs*(1.0+self.z), linestyle='--', color='k', alpha=0.6)
                ax_c.annotate(text=r"$T_\mathrm{{ CMB }} (z = {:.6g})$".format(self.z), xy=(T_CMB_obs*(1.0+self.z), 0.5), xytext=(-2, 0),
                                xycoords=ax_c.get_xaxis_transform(), textcoords="offset points", rotation="vertical",
                                va="center", ha="right", size="xx-small", alpha=0.8).set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
                if self.T_lolim or self.T_uplim:
                    ax_c.axvline(rdict["T_peak"], color="grey", alpha=0.6)
                    ax_c.annotate(text=("Lower" if self.T_lolim else "Upper") + " limit (95% conf.)", xy=(rdict["T_peak"], 1), xytext=(2, -4),
                                    xycoords=ax_c.get_xaxis_transform(), textcoords="offset points", rotation="vertical",
                                    va="top", ha="left", size="xx-small", alpha=0.8).set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
                
                self.annotate_results(rdict, [axes_c[0, -1], axes_c[1, -1]], ax_type="corner", ann_size=ann_size)
                cfig.savefig(pltfol + "Corner_MN_" + self.obj_fn + self.get_mstring(l0_list=[l0], inc_astr=False) + self.pformat,
                                dpi=self.dpi, bbox_inches="tight")
        
                # plt.show()
                plt.close(cfig)
            
            if save_results:
                # Save results
                np.savez_compressed(self.mnrfol + "{}_MN_FIR_SED_fit_{}{}.npz".format(self.obj_fn, self.beta_str, l0_str), **rdict)
                np.savez_compressed(self.mnrfol + "{}_MN_FIR_SED_data_samples_{}{}.npz".format(self.obj_fn, self.beta_str, l0_str),
                                    data=data, names=names)
                self.fresh_calculation[l0] = True
            else:
                # Keep results for plotting purposes
                self.rdict = rdict
            
            if self.verbose:
                self.print_results(rdict, l0_txt, rtype="calculated")
            
            if return_samples:
                return (data, names)
    
    def print_results(self, rdict, l0_txt, rtype):
        """Function for printing the results; designed for internal use.

        Parameters
        ----------
        rdict : dict
            Dictionary containing the main MultiNest results.
        l0_txt : str
            A string containing a description of the opacity model classifier.
        rtype : str
            Type of estimates.

        """

        M_dust_log10 = math.floor(np.log10(rdict["M_dust"]))
        L_IR_log10 = math.floor(np.log10(rdict["L_IR_Lsun"]))
        L_FIR_log10 = math.floor(np.log10(rdict["L_FIR_Lsun"]))

        print("\nFreshly {} MultiNest estimates of {}".format(rtype, self.obj))
        print("(with {}{}):".format("β = {:.1f}".format(self.fixed_beta) if self.fixed_beta else "varying β", l0_txt))
        print("M_dust = ({:.1f} -{:.1f} +{:.1f}) x 10^{:d} M_sun".format(rdict["M_dust"]/10**M_dust_log10,
                    rdict["M_dust_lowerr"]/10**M_dust_log10, rdict["M_dust_uperr"]/10**M_dust_log10, M_dust_log10))
        print("L_IR = ({:.1f} -{:.1f} +{:.1f}) x 10^{:d} L_sun".format(rdict["L_IR_Lsun"]/10**L_IR_log10,
                    rdict["L_IR_Lsun_lowerr"]/10**L_IR_log10, rdict["L_IR_Lsun_uperr"]/10**L_IR_log10, L_IR_log10))
        print("L_FIR = ({:.1f} -{:.1f} +{:.1f}) x 10^{:d} L_sun".format(rdict["L_FIR_Lsun"]/10**L_FIR_log10,
                    rdict["L_FIR_Lsun_lowerr"]/10**L_FIR_log10, rdict["L_FIR_Lsun_uperr"]/10**L_FIR_log10, L_FIR_log10))
        print("T_dust = {:.1f} -{:.1f} +{:.1f} K ({:.1f} -{:.1f} +{:.1f} K at z = 0)".format(rdict["T_dust"],
                    rdict["T_dust_lowerr"], rdict["T_dust_uperr"], rdict["T_dust_z0"], rdict["T_dust_z0_lowerr"], rdict["T_dust_z0_uperr"]))
        print("T_peak = {:.1f} -{:.1f} +{:.1f} K".format(rdict["T_peak_val"], rdict["T_peak_lowerr"], rdict["T_peak_uperr"]))
        if self.T_lolim:
            print("T_dust > {:.1f} K (95% confidence),".format(rdict["T_lolim"]))
            print("\tor T_dust > {:.1f} K at z = 0".format(rdict["T_lolim_z0"]))
            print("T_peak > {:.1f} K (95% confidence)".format(rdict["T_peak_lolim"]))
        if self.T_uplim:
            print("T_dust < {:.1f} K (95% confidence),".format(rdict["T_uplim"]))
            print("\tor T_dust < {:.1f} K at z = 0".format(rdict["T_uplim_z0"]))
            print("T_peak < {:.1f} K (95% confidence)".format(rdict["T_peak_uplim"]))
        if self.fixed_beta:
            print("beta_IR = {:.2g} (fixed)".format(rdict["beta_IR"]))
        else:
            print("beta_IR = {:.2g} -{:.2g} +{:.2g}".format(rdict["beta_IR"], rdict["beta_IR_lowerr"], rdict["beta_IR_uperr"]))
        print('')
        
        if not np.isnan(self.obj_M):
            fmt = ".2f" if rdict["dust_frac"] > 0.01 else ".2g"
            print("Dust-to-stellar mass fraction: {:{fmt}} -{:{fmt}} +{:{fmt}}".format(rdict["dust_frac"],
                                                rdict["dust_frac_lowerr"], rdict["dust_frac_uperr"], fmt=fmt))
            print("Dust yield (AGB): {:{fmt}} -{:{fmt}} +{:{fmt}} M_sun".format(rdict["dust_yield_AGB"],
                                                rdict["dust_yield_AGB_lowerr"], rdict["dust_yield_AGB_uperr"], fmt=fmt))
            print("Dust yield (SN): {:{fmt}} -{:{fmt}} +{:{fmt}} M_sun".format(rdict["dust_yield_SN"],
                                                rdict["dust_yield_SN_lowerr"], rdict["dust_yield_SN_uperr"], fmt=fmt))
            print('')
    
    def plot_MN_fit(self, l0_list=None, fig=None, ax=None, ax_res=None, pltfol=None, obj_str=None, single_plot=None,
                    annotate_title=True, plot_data=True, bot_axis="wl_emit", add_top_axis="wl_obs",
                    set_xrange=True, set_xlabel="both", set_ylabel=True, low_yspace_mult=0.05, up_yspace_mult=4,
                    ann_size="small", show_T_peak=False, leg_framealpha=0, rowi=0, coli=0):
        """Function for plotting the results of a MultiNest fit.

        Parameters
        ----------
        l0_list : list, optional
            A list of opacity model classifiers. Entries can be `None` for an optically thin model,
            `"self-consistent"` for a self-consistent opacity model, or a float setting a fixed
            value of `lambda_0`, the wavelength in micron setting the SED's transition point between
            optically thin and thick. Default of `l0_list` is `None`, in which case the main `l0_list`,
            given when creating the `FIR_SED_fit` instance, is used.
        fig : {`None`, instance of `matplotlib.figure.Figure`}, optional
            Figure instance to use for plotting. If `fig` and `ax` are not given, `plt.subplots()`
            will be used to create them. Otherwise, if `fig` is given but `ax` is not,
            `fig.add_subplot()` will be used to create `ax`. Default for `fig` is `None` (not set).
        ax : {`None`, instance of `matplotlib.axes.Axes`}, optional
            Axes instance to use for plotting. If `fig` and `ax` are not given, `plt.subplots()`
            will be used to create them. Otherwise, if `ax` is given but `fig` is not,
            `ax.get_figure()` will be used to retrieve `fig`. Default for `ax` is `None` (not set).
        ax_res : {`None`, bool, instance of `matplotlib.axes.Axes`}, optional
            Axes instance to use for plotting residual fluxes. When `ax_res` is `None` (default) or
            `True`, a new axis will be produced with `plt.subplots()`.
        pltfol : {`None`, str}, optional
            Path to folder in which figures are saved. Default is `None` (plots are not saved
            directly).
        single_plot : {`None`, bool}, optional
            Controls whether a single plot is made for different opacity models. Default is `None`,
            in which case a single plot will be made when `analysis` is `False`.
        annotate_title : bool, optional
            Annotate general information (e.g. name of the object and redshift)? Default is `True`.
        plot_data : bool, optional
            Show data points? Default is `True`.
        bot_axis : {`"wl_emit"`, `"nu_obs"`}, optional
            Choice for the main x-axis to show the rest-frame wavelength (`"wl_emit"`) or observed
            frequency (`"nu_obs"`). Default is to show rest-frame wavelength.
        add_top_axis : {`"wl_obs"`, `"nu_obs"`, `False`}, optional
            Add second axis showing the observed wavelength (`"wl_obs"`) or frequency (`"nu_obs"`)
            on the top x-axis? Default is to show observed wavelength.
        set_xrange : bool, optional
            Manually set bounds on the rest-frame wavelength axis in the plots? Default is `True`.
        set_xlabel : {`"top"`, `"bottom"`, `"both"`, `False`}, optional
            Set labels on the rest-frame (bottom) and/or observed (top) wavelength axes in this plot?
            Default is `"both"`.
        set_ylabel : bool, optional
            Set labels on the flux density axis in this plot? Default is `True`.
        low_yspace_mult : float, optional
            Multiplier of the lower flux density bound to create more vertical space in this plot.
            Default is `0.05`.
        up_yspace_mult : float, optional
            Multiplier of the upper flux density bound to create more vertical space in this plot.
            Default is `4`.
        ann_size : float or {'xx-small', 'x-small', 'small', 'medium',
                    'large', 'x-large', 'xx-large'}, optional
            Argument used by `matplotlib.text.Text` for the font size of the annotation.
            Default: `"small"`.
        show_T_peak : {`False`, `"all"`, `None`, `"self-consistent"`, float}, optional
            Indicate the peak temperature on the plot for all (`"all"`) or only one specific opacity
            model classifier (see `l0_list` for details)? Default is `False`, where it is not indicated.
        leg_framealpha : float, optional
            Alpha value used by `matplotlib` for the frame of the legend. Default is `None`, in which
            case the value is taken from `plt.rcParams`.
        rowi : int, optional
            Row number of this plot in a multi-panel figure. Default is `0`.
        coli : int, optional
            Column number of this plot in a multi-panel figure. Default is `0`.

        """

        if self.all_uplims:
            print("Warning: only upper limits for {} specified! No MultiNest results plotted...".format(self.obj))
            return 1
        
        if l0_list is None:
            l0_list = self.l0_list
        create_fig = fig is None and ax is None
        
        # Minimum/maximum observed flux (for plot scaling)
        self.F_nu_obs_min, self.F_nu_obs_max = np.inf, -np.inf
        
        if single_plot is None:
            single_plot = not self.analysis
        if single_plot:
            handles = []
        if leg_framealpha is None:
            leg_framealpha = plt.rcParams["legend.framealpha"]
        
        plot_ax_res = ax_res is None or ax_res is True

        if plot_ax_res:
            self.residuals_min = []
            self.residuals_max = []

        for l0 in l0_list:
            height_ratios = [4, 1]
            if create_fig:
                # Prepare figure for plotting FIR SED
                if plot_ax_res:
                    fig, (ax, ax_res) = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True, gridspec_kw={"hspace": 0, "height_ratios": height_ratios})
                else:
                    fig, ax = plt.subplots()
                
                if single_plot:
                    create_fig = False
            elif fig is None:
                fig = ax.get_figure()
                if plot_ax_res and not isinstance(ax_res, matplotlib.axes.Axes):
                    # Restructure gridspec layout to fit in residual axes
                    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=height_ratios)
                    ax.set_position(gs[0].get_position(fig))
                    ax_res = fig.add_subplot()
            elif ax is None:
                if plot_ax_res:
                    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=height_ratios)
                    ax = fig.add_subplot(gs[0])
                    ax_res = fig.add_subplot(gs[1])
                else:
                    ax = fig.add_subplot()
            else:
                if plot_ax_res and not isinstance(ax_res, matplotlib.axes.Axes):
                    # Restructure gridspec layout to fit in residual axes
                    gs_orig = ax.get_gridspec()
                    gs = fig.add_gridspec(nrows=2*gs_orig._nrows, ncols=gs_orig._ncols, height_ratios=gs_orig._nrows*height_ratios)
                    sb_params = {sb_param: gs_orig.get_subplot_params().__dict__[sb_param] for sb_param in gs_orig.locally_modified_subplot_params()}
                    gs.update(**sb_params)
                    
                    ax.set_position(gs[2*rowi, coli].get_position(fig))
                    ax_res = fig.add_subplot(gs[2*rowi+1, coli])

            self.fig, self.ax = fig, ax
            if plot_ax_res:
                self.ax_res = ax_res

            if annotate_title:
                self.annotate_title()
            
            l0_str, l0_txt = self.get_l0string(l0)
            
            rdict = self.load_rdict(l0)
            
            if rdict is None:
                continue
            
            M_dust_log10 = math.floor(np.log10(rdict["M_dust"]))
            T_dust = rdict["T_dust"]
            dcolor = dust_cmap(dust_norm(T_dust))
            
            lambda_emit = rdict["lambda_emit"]
            if bot_axis == "wl_emit":
                self.x = lambda_emit
            elif bot_axis == "nu_obs":
                nu_obs = 299792.458 / lambda_emit / (1.0 + self.z) # GHz (lambda_emit is in micron)
                self.x = nu_obs
            self.S_nu_obs = rdict["S_nu_obs"]
            S_nu_obs_lowerr = rdict["S_nu_obs_lowerr"]
            S_nu_obs_uperr = rdict["S_nu_obs_uperr"]
            
            # Plot the observed spectrum (intrinsic spectrum is nearly the same apart from at the very red wavelengths above ~100 micron)
            ax.plot(self.x, self.S_nu_obs*self.fd_conv, linewidth=1.5,
                    linestyle=self.l0_linestyles.get(l0, '-'), color=dcolor, alpha=0.8)
            
            if plot_ax_res:
                # Plot the residuals of the observed spectrum
                ax_res.axhline(y=0, linewidth=1.5, linestyle=self.l0_linestyles.get(l0, '-'), color=dcolor, alpha=0.8)
            
            if show_T_peak and (show_T_peak == "all" or show_T_peak == l0):
                peak_idx = np.argmax(rdict["S_nu_emit"])
                prec = max(0, 2-math.floor(np.log10(min(rdict["T_dust_lowerr"], rdict["T_dust_uperr"]))))
                
                ax.annotate(text=r"$T_\mathrm{{ peak }} = {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \, \mathrm{{ K }}$".format(rdict["T_peak_val"],
                            rdict["T_peak_lowerr"], rdict["T_peak_uperr"], prec=prec) + \
                                (r" (${} {:.{prec}f} \, \mathrm{{ K }}$)".format(r'>' if self.T_lolim else r'<', rdict["T_peak"], prec=prec) if self.T_lolim or self.T_uplim else r''),
                            xy=(self.x[peak_idx], self.S_nu_obs[peak_idx]*self.fd_conv), xytext=(0, -40), xycoords="data", textcoords="offset points",
                            va="top", ha="center", size=ann_size, alpha=0.8,
                            arrowprops={"arrowstyle": '-', "shrinkA": 0, "shrinkB": 0, "color": dcolor, "alpha": 0.8},
                            zorder=6).set_bbox(dict(boxstyle="Round, pad=0.1", linestyle=self.l0_linestyles.get(l0, '-'), facecolor='w', edgecolor=dcolor, alpha=0.8))
            
            ax.fill_between(self.x, y1=(self.S_nu_obs-S_nu_obs_lowerr)*self.fd_conv, y2=(self.S_nu_obs+S_nu_obs_uperr)*self.fd_conv,
                            facecolor=dcolor, edgecolor="None", alpha=0.1)
            
            if plot_ax_res:
                # Plot the residuals of the observed spectrum
                ax_res.fill_between(self.x, y1=-S_nu_obs_lowerr*self.fd_conv, y2=S_nu_obs_uperr*self.fd_conv, facecolor=dcolor, edgecolor="None", alpha=0.1)
                self.residuals_min.append(np.min(-1.25*S_nu_obs_lowerr[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]*self.fd_conv))
                self.residuals_max.append(np.max(1.25*S_nu_obs_uperr[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]*self.fd_conv))
            
            # if np.min((self.S_nu_obs-S_nu_obs_lowerr)[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]) * self.fd_conv < self.F_nu_obs_min:
            #     self.F_nu_obs_min = np.nanmin((self.S_nu_obs-S_nu_obs_lowerr)[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]) * self.fd_conv
            if np.max((self.S_nu_obs+S_nu_obs_uperr)[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]) * self.fd_conv > self.F_nu_obs_max:
                self.F_nu_obs_max = np.nanmax((self.S_nu_obs+S_nu_obs_uperr)[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]) * self.fd_conv
            
            M_dust_log10 = math.floor(np.log10(rdict["M_dust"]))
            L_IR_log10 = math.floor(np.log10(rdict["L_IR_Lsun"]))
            
            if not self.fresh_calculation[l0]:
                if self.verbose:
                    self.print_results(rdict, l0_txt, rtype="loaded")

            if self.analysis and not single_plot:
                self.annotate_results(rdict, [ax, ax], ax_type="regular", ann_size=ann_size)
            else:
                if l0 == "self-consistent":
                    l0_lab = l0.capitalize() + '\n' + r"$\lambda_0 = {:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }} \, \mathrm{{ \mu m }}$".format(rdict["lambda_0"],
                                                                                                    rdict["lambda_0_lowerr"], rdict["lambda_0_uperr"])
                else:
                    l0_lab = "Fixed: " + r"$\lambda_0 = {:.0f} \, \mathrm{{ \mu m }}$".format(l0) if l0 else "Optically thin"
                    if not np.isnan(rdict["lambda_0"]):
                        l0_lab += '\n' + r"$\lambda_0^\mathrm{{AP}} = {:.1f} \, \mathrm{{ \mu m }}$".format(rdict["lambda_0"])
                
                label = '\n'.join([l0_lab,
                                    r"$M_\mathrm{{ dust }} = {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \cdot 10^{{ {:d} }} \, \mathrm{{ M_\odot }}$".format(rdict["M_dust"]/10**M_dust_log10,
                                            rdict["M_dust_lowerr"]/10**M_dust_log10, rdict["M_dust_uperr"]/10**M_dust_log10, M_dust_log10,
                                            prec=1 if (rdict["M_dust"]-rdict["M_dust_lowerr"])/10**M_dust_log10 < 1 else 0),
                                    r"$T_\mathrm{{ dust }} = {:.0f}_{{ -{:.0f} }}^{{ +{:.0f} }} \, \mathrm{{ K }}$".format(T_dust,
                                            rdict["T_dust_lowerr"], rdict["T_dust_uperr"]),
                                    r"$L_\mathrm{{ IR }} = {:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }} \cdot 10^{{ {:d} }} \, \mathrm{{ L_\odot }}$".format(rdict["L_IR_Lsun"]/10**L_IR_log10,
                                            rdict["L_IR_Lsun_lowerr"]/10**L_IR_log10, rdict["L_IR_Lsun_uperr"]/10**L_IR_log10, L_IR_log10),
                                    r"$\mathrm{{ SFR_{{IR}} }} = {:.0f}_{{ -{:.0f} }}^{{ +{:.0f} }} \, \mathrm{{ M_\odot \, yr^{{-1}} }}$".format(rdict["SFR_IR"],
                                            *rdict["SFR_IR_err"])])
                
                if not self.fixed_beta:
                    label += '\n' + r"$\beta_\mathrm{{ IR }} = {:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }}$".format(rdict["beta_IR"],
                                        rdict["beta_IR_lowerr"], rdict["beta_IR_uperr"])
                if single_plot:
                    handles.append(BTuple(([dcolor], [0.1], 1.5, self.l0_linestyles.get(l0, '-'), dcolor, 0.8), label + '\n'))

            if not single_plot:
                self.plot_data(bot_axis=bot_axis, plot_ax_res=plot_ax_res)
                self.set_axes(bot_axis=bot_axis, add_top_axis=add_top_axis, set_xrange=set_xrange, set_xlabel=set_xlabel, set_ylabel=set_ylabel,
                                low_yspace_mult=low_yspace_mult, up_yspace_mult=up_yspace_mult, rowi=rowi, coli=coli, plot_ax_res=plot_ax_res)
                if pltfol:
                    self.save_fig(pltfol=pltfol, ptype="MN_fit", l0_list=[l0], single_plot=single_plot)
        
        if single_plot:
            leg = ax.legend(handles=handles, handler_map={BTuple: BTupleHandler()}, ncol=len(self.l0_list), loc="lower center",
                            handlelength=0.7*plt.rcParams["legend.handlelength"], handleheight=5*plt.rcParams["legend.handleheight"],
                            columnspacing=0.3*plt.rcParams["legend.columnspacing"], framealpha=leg_framealpha)
            
            # Show which beta_IRs have been used
            leg.set_title("MN fits of dust emission" + r", fixed $\beta_\mathrm{{ IR }} = {:.2g}$".format(self.fixed_beta) if self.fixed_beta else '',
                            prop={"size": "small"})
            
            if plot_data:
                self.plot_data(bot_axis=bot_axis, plot_ax_res=plot_ax_res)
            self.set_axes(bot_axis=bot_axis, add_top_axis=add_top_axis, set_xrange=set_xrange, set_xlabel=set_xlabel, set_ylabel=set_ylabel,
                            low_yspace_mult=low_yspace_mult, up_yspace_mult=up_yspace_mult, rowi=rowi, coli=coli, plot_ax_res=plot_ax_res)
            if pltfol:
                self.save_fig(pltfol=pltfol, ptype="MN_fit", l0_list=l0_list, obj_str=obj_str, single_plot=single_plot)
    
    def plot_ranges(self, l0, T_dusts=T_dusts_global, beta_IRs=beta_IRs_global, fixed_T_dust=None, fixed_beta=None, lambda_emit=None,
                    save_results=True, fig=None, ax=None, pltfol=None, obj_str=None,
                    annotate_title=True, annotate_results=True, show_legend=True,
                    plot_data=True, bot_axis="wl_emit", add_top_axis="wl_obs", set_xrange=True, set_xlabel="both", set_ylabel=True,
                    low_yspace_mult=0.05, up_yspace_mult=4, rowi=0, coli=0):
        """Function for plotting a range of greybody spectra tuned to the photometric data.

        Notes
        ----------
        Normalisation of each greybody curve is performed according to the most constraining
        upper limit, if available, or the highest SNR detection. Curves are considered
        compatible if they fall below upper limits and within ±1σ of other detections.

        Parameters
        ----------
        l0 : {`None`, "self-consistent", float}
            Opacity model classifiers: `None` for an optically thin model, `"self-consistent"`
            for a self-consistent opacity model, or a float setting a fixed value of `lambda_0`,
            the wavelength in micron setting the SED's transition point between optically thin
            and thick.
        T_dusts : array_like, optional
            Range of dust temperatures to be shown. Default is `T_dusts_global`, a range from
            10 K to 110 K in steps of 10 K.
        beta_IRs : array_like, optional
            Range of dust emissivities to be shown. Default is `beta_IRs_global`, a range from
            1.5 to 2 in steps of 0.1.
        fixed_T_dust : {`None`, float}, optional
            Fixed value of the dust temperature in Kelvin used if saving results. Default is
            `None`, which will revert to the main value of fixed_T_dust, given when creating
            the `FIR_SED_fit` instance.
        fixed_beta : {`None`, float}, optional
            Fixed value of the dust emissivity beta used if saving results. Default is `None`,
            which will revert to the main value of fixed_beta, given when creating the
            `FIR_SED_fit` instance.
        lambda_emit : array_like, optional
            Values in micron to be used as the rest-frame wavelengths in results,
            including the (F)IR luminosities. Default is `None`, which will default
            to a linearly spaced array of 10,000 points ranging between 4 and 1100
            micron in the function `calc_FIR_SED` (located in
            `mercurius/aux/infrared_luminosity.py`).
        save_results : bool, optional
            Save results for a greybody spectrum with dust temperature `fixed_T_dust` and dust
            emissivity `fixed_beta`? Default: `True`.
        fig : {`None`, instance of `matplotlib.figure.Figure`}, optional
            Figure instance to use for plotting. If `fig` and `ax` are not given, `plt.subplots()`
            will be used to create them. Otherwise, if `fig` is given but `ax` is not,
            `fig.add_subplot()` will be used to create `ax`. Default for `fig` is `None` (not set).
        ax : {`None`, instance of `matplotlib.axes.Axes`}, optional
            Axes instance to use for plotting. If `fig` and `ax` are not given, `plt.subplots()`
            will be used to create them. Otherwise, if `ax` is given but `fig` is not,
            `ax.get_figure()` will be used to retrieve `fig`. Default for `ax` is `None` (not set).
        pltfol : {`None`, str}, optional
            Path to folder in which figures are saved. Default is `None` (plots are not saved
            directly).
        obj_str : {`None`, str}, optional
            String to be added to the filename of the figure. Default is `None`,
            in which case the object name, preceded by an underscore, will be used.
        annotate_title : bool, optional
            Annotate general information (e.g. name of the object and redshift)? Default is `True`.
        annotate_results : bool, optional
            Annotate the (range of) infrared luminosities found to be compatible for a given
            dust temperature? Default is `True`.
        show_legend : bool, optional
            Show legend with all dust temperatures plotted? Default is `True`.
        plot_data : bool, optional
            Show data points? Default is `True`.
        bot_axis : {`"wl_emit"`, `"nu_obs"`}, optional
            Choice for the main x-axis to show the rest-frame wavelength (`"wl_emit"`) or observed
            frequency (`"nu_obs"`). Default is to show rest-frame wavelength.
        add_top_axis : {`"wl_obs"`, `"nu_obs"`, `False`}, optional
            Add second axis showing the observed wavelength (`"wl_obs"`) or frequency (`"nu_obs"`)
            on the top x-axis? Default is to show observed wavelength.
        set_xrange : bool, optional
            Manually set bounds on the rest-frame wavelength axis in the plots? Default is `True`.
        set_xlabel : {`"top"`, `"bottom"`, `"both"`, `False`}, optional
            Set labels on the rest-frame (bottom) and/or observed (top) wavelength axes in this plot?
            Default is `"both"`.
        set_ylabel : bool, optional
            Set labels on the flux density axis in this plot? Default is `True`.
        low_yspace_mult : float, optional
            Multiplier of the lower flux density bound to create more vertical space in this plot.
            Default is `0.05`.
        up_yspace_mult : float, optional
            Multiplier of the upper flux density bound to create more vertical space in this plot.
            Default is `4`.
        rowi : int, optional
            Row number of this plot in a multi-panel figure. Default is `0`.
        coli : int, optional
            Column number of this plot in a multi-panel figure. Default is `0`.

        """

        if l0 == "self-consistent":
            print("Warning: ranges cannot be shown for a self-consistent opacity model! Continuing...")
            return 1
        if save_results:
            if fixed_T_dust is None:
                assert self.fixed_T_dust
                fixed_T_dust = self.fixed_T_dust
            if fixed_beta is None:
                assert self.fixed_beta
                fixed_beta = self.fixed_beta

        # Minimum/maximum observed flux (for plot scaling)
        self.F_nu_obs_min, self.F_nu_obs_max = np.inf, -np.inf
        
        if fig is None and ax is None:
            # Prepare figure for plotting FIR SED
            fig, ax = plt.subplots()
        elif fig is None:
            fig = ax.get_figure()
        elif ax is None:
            ax = fig.add_subplot()

        self.fig, self.ax = fig, ax

        SNR_ratios = [np.nan if exc else S/N for S, N, exc in zip(self.S_nu_vals, self.S_nu_errs, self.cont_excludes)]

        if annotate_title:
            prop_ann = self.annotate_title()

        rdict = {}
        rdict["Sigma_dust"], rdict["Sigma_dust_lowerr"], rdict["Sigma_dust_uperr"] = np.nan, np.nan, np.nan
        rdict["T_dust"], rdict["T_dust_lowerr"], rdict["T_dust_uperr"] = np.nan, np.nan, np.nan
        rdict["T_peak_val"], rdict["T_peak_lowerr"], rdict["T_peak_uperr"] = np.nan, np.nan, np.nan

        # Minimum and maximum peak dust temperature
        T_peak_minmax = [np.inf, -np.inf]

        # Minimum/maximum observed flux (for plot scaling)
        self.F_nu_obs_min, self.F_nu_obs_max = np.inf, -np.inf
        
        # Counter and list for L_IR annotations
        cb_ann_text_offset_colours = []

        fixed_vals_cb = False
        
        if self.verbose:
            print("\nPlotting ranges on FIR SED of {}...".format(self.obj))
            if save_results:
                print("Selected fixed T_dust = {:.1f}, β = {:.1f}{}".format(fixed_T_dust, fixed_beta, self.get_l0string(l0)[1]))
        
        T_dust_handles = []

        for di, T_dust in enumerate(T_dusts):
            if T_dust < T_CMB_obs*(1.0+self.z):
                continue

            x_betas = []
            S_nu_obs_betas = []
            L_IR_Lsun_betas = []
            L_FIR_Lsun_betas = []
            T_peak_betas = []

            compatible_betas = []
            
            dcolor = dust_cmap(dust_norm(T_dust))

            for bi, beta_IR in enumerate(beta_IRs):
                lambda_emit, S_nu_emit = calc_FIR_SED(z=self.z, beta_IR=beta_IR, T_dust=T_dust,
                                                        optically_thick_lambda_0=l0, return_spectrum=True, lambda_emit=lambda_emit)
                nu_emit = 299792458.0 * 1e6 / lambda_emit # Hz (lambda is in micron)
                if bot_axis == "wl_emit":
                    x = lambda_emit
                elif bot_axis == "nu_obs":
                    nu_obs = nu_emit / 1e9 / (1.0 + self.z) # GHz (nu_emit is in Hz)
                    x = nu_obs
                
                # Flux density needs to be corrected for observing against the the CMB (NB: can be negative if T_dust < T_CMB), and then normalised
                CMB_correction_factor = CMB_correction(z=self.z, nu0_emit=nu_emit, T_dust=T_dust)
                if np.all(CMB_correction_factor < 0):
                    # No normalisation possible (T_dust < T_CMB) and so any upper limit is compatible, continue
                    continue
                else:
                    if self.analysis and np.any(CMB_correction_factor < 0.9):
                        # Show where the correction is 90%
                        ax.axvline(x=x[np.argmin(np.abs(CMB_correction_factor - 0.9))], linestyle='--', color=dcolor, alpha=0.8)
                        if di == 2 and bi == 0:
                            ax.annotate(text="10% CMB background", xy=(x[np.argmin(np.abs(CMB_correction_factor - 0.9))], 1), xytext=(-4, -4),
                                        xycoords=ax.get_xaxis_transform(), textcoords="offset points", rotation="vertical",
                                        va="top", ha="right", size="x-small", color=dcolor, alpha=0.8)

                S_nu_emit_CMB_att = S_nu_emit * CMB_correction_factor # Jy
                
                # Wien's displacement law to find the observed (after correction for CMB attenuation) peak temperature
                T_peak_betas.append(2.897771955e3/lambda_emit[np.argmax(S_nu_emit)])

                def normalise(wl, flux, norm=1):
                    return (flux*self.fd_conv) / np.interp(wl, lambda_emit, S_nu_emit_CMB_att * norm)
                
                uplim = np.all(self.cont_uplims[~self.cont_excludes])
                
                if uplim:
                    normalisations = []

                    for cont_wl, cont_flux, exclude in zip(self.lambda_emit_vals, self.S_nu_vals, self.cont_excludes):
                        if exclude:
                            continue

                        # Normalise by observed (upper limit of) [CII]/[OIII]/[NII] continuum flux density
                        norm = normalise(cont_wl, cont_flux)
                        
                        # See if this normalisation implies the other data points/upper limits fall below the curve
                        normalisations.append(np.mean([normalise(wl, f, norm=norm) for wl, f in zip(self.lambda_emit_vals, self.S_nu_vals)]))

                    cont_idx = np.nanargmax(normalisations)
                    normalisation = normalise(self.lambda_emit_vals[cont_idx], self.S_nu_vals[cont_idx])
                    normalisation_lowerr = np.nan
                    normalisation_uperr = np.nan

                    # With only upper limits, all betas are always compatible
                    compatible_beta = True
                else:
                    if np.sum(self.cont_det) == 1:
                        # Only one detection, use as normalisation
                        cont_idx = list(self.cont_det).index(True)
                    else:
                        # Choose highest SNR detection for normalisation
                        cont_idx = np.nanargmax(SNR_ratios)

                    # Normalise by detected continuum flux density (or upper limit thereof)
                    normalisation = normalise(self.lambda_emit_vals[cont_idx], self.S_nu_vals[cont_idx])
                    normalisation_lowerr = normalise(self.lambda_emit_vals[cont_idx], self.S_nu_vals[cont_idx]-self.S_nu_errs[cont_idx])
                    normalisation_uperr = normalise(self.lambda_emit_vals[cont_idx], self.S_nu_vals[cont_idx]+self.S_nu_errs[cont_idx])

                    # See if this normalisation violates the other measurements (whether they are detections or upper limits)
                    compatible_beta = True
                    for ci in range(self.n_meas):
                        if self.cont_excludes[ci]:
                            continue
                        elif self.cont_uplims[ci]:
                            # See if normalisation falls below the upper limit
                            compatible_beta = compatible_beta and normalise(self.lambda_emit_vals[ci], self.S_nu_vals[ci], norm=normalisation) > 1
                        else:
                            # See if normalisation falls between ±1σ of the detection
                            low_ratio = normalise(self.lambda_emit_vals[ci], self.S_nu_vals[ci] - self.S_nu_errs[ci], norm=normalisation)
                            up_ratio = normalise(self.lambda_emit_vals[ci], self.S_nu_vals[ci] + self.S_nu_errs[ci], norm=normalisation)
                            compatible_beta = compatible_beta and (low_ratio <= 1 and up_ratio >= 1)

                compatible_betas.append(compatible_beta)
                
                # Note the normalisation is based on the observed flux density, which is related to the emitted flux density via
                # S_(ν, obs) = S_(ν, emit) * (1+z) as ν itself scales as ν_obs = ν_emit / (1+z) while the flux (S_ν dν) is invariant
                S_nu_emit_lowerr = S_nu_emit * normalisation_lowerr / (1.0+self.z) / self.fd_conv # Jy
                S_nu_emit_uperr = S_nu_emit * normalisation_uperr / (1.0+self.z) / self.fd_conv # Jy
                S_nu_emit *= normalisation / (1.0+self.z) / self.fd_conv # Jy
                S_nu_obs = S_nu_emit_CMB_att * normalisation # (m/μ)Jy (depending on self.fd_conv)

                if np.min(S_nu_obs[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]) < self.F_nu_obs_min:
                    self.F_nu_obs_min = np.nanmin(S_nu_obs[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))])
                if np.max(S_nu_obs[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]) > self.F_nu_obs_max:
                    self.F_nu_obs_max = np.nanmax(S_nu_obs[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))])
                
                # Calculate the integrated IR luminosity between 8 and 1000 μm (integrate Jy over Hz, so convert result to erg/s/cm^2;
                # 1 Jy = 10^-23 erg/s/cm^2/Hz)
                lambda_IR = (lambda_emit > 8.0) * (lambda_emit < 1000.0)
                lambda_FIR = (lambda_emit > 42.5) * (lambda_emit < 122.5)
                F_IR = np.trapz(S_nu_emit[lambda_IR][np.argsort(nu_emit[lambda_IR])], x=np.sort(nu_emit[lambda_IR])) * 1e-23
                F_IR_lowerr = np.trapz(S_nu_emit_lowerr[lambda_IR][np.argsort(nu_emit[lambda_IR])], x=np.sort(nu_emit[lambda_IR])) * 1e-23
                F_IR_uperr = np.trapz(S_nu_emit_uperr[lambda_IR][np.argsort(nu_emit[lambda_IR])], x=np.sort(nu_emit[lambda_IR])) * 1e-23
                F_FIR = np.trapz(S_nu_emit[lambda_FIR][np.argsort(nu_emit[lambda_FIR])], x=np.sort(nu_emit[lambda_FIR])) * 1e-23
                F_FIR_lowerr = np.trapz(S_nu_emit_lowerr[lambda_FIR][np.argsort(nu_emit[lambda_FIR])], x=np.sort(nu_emit[lambda_FIR])) * 1e-23
                F_FIR_uperr = np.trapz(S_nu_emit_uperr[lambda_FIR][np.argsort(nu_emit[lambda_FIR])], x=np.sort(nu_emit[lambda_FIR])) * 1e-23

                # Convert flux to (solar) luminosity
                L_IR_Lsun = F_IR * 4.0 * np.pi * self.D_L**2 / L_sun_ergs # L_sun
                L_IR_Lsun_lowerr = F_IR_lowerr * 4.0 * np.pi * self.D_L**2 / L_sun_ergs # L_sun
                L_IR_Lsun_uperr = F_IR_uperr * 4.0 * np.pi * self.D_L**2 / L_sun_ergs # L_sun
                L_FIR_Lsun = F_FIR * 4.0 * np.pi * self.D_L**2 / L_sun_ergs # L_sun
                L_FIR_Lsun_lowerr = F_FIR_lowerr * 4.0 * np.pi * self.D_L**2 / L_sun_ergs # L_sun
                L_FIR_Lsun_uperr = F_FIR_uperr * 4.0 * np.pi * self.D_L**2 / L_sun_ergs # L_sun

                # Add a 0.4 dex systematic uncertainty
                if uplim:
                    L_IR_Lsun *= np.sqrt(1 + (10**0.4)**2)
                    L_FIR_Lsun *= np.sqrt(1 + (10**0.4)**2)
                else:
                    L_IR_Lsun_lowerr *= np.sqrt(1 + (10**0.4)**2)
                    L_IR_Lsun_uperr *= np.sqrt(1 + (10**0.4)**2)
                    L_FIR_Lsun_lowerr *= np.sqrt(1 + (10**0.4)**2)
                    L_FIR_Lsun_uperr *= np.sqrt(1 + (10**0.4)**2)

                if save_results and np.abs(T_dust - fixed_T_dust) < 1e-8 and np.abs(beta_IR - fixed_beta) < 1e-8:
                    fixed_vals_cb = compatible_beta
                    if not compatible_beta:
                        if self.verbose:
                            print("Warning: T_dust = {:.0f} K, β = {:.2g} are incompatible with measurements...".format(fixed_T_dust,
                                                                                                                        fixed_beta, self.uplim_nsig))

                    # Estimate of the dust mass (S_nu in Jy, 1e-26 = W/m^2/Hz/Jy, self.D_L in cm, κ_nu in cm^2/g, Planck_func in W/m^2/Hz, M_sun = 1.989e33 g)
                    S_nu_emit_ref = np.interp(wl_star*1e6, lambda_emit, S_nu_emit)
                    rdict["M_dust_uplim"] = uplim
                    rdict["M_dust"] = S_nu_emit_ref * self.D_L**2 / (M_sun_g * k_nu_star * Planck_func(nu_star, T_dust) * 1e26)
                    if uplim:
                        rdict["M_dust_lowerr"], rdict["M_dust_uperr"] = np.nan, np.nan
                    else:
                        S_nu_emit_ref_errs = np.array([S_nu_emit_ref, S_nu_emit_ref]) / np.nanmean(SNR_ratios)
                        rdict["M_dust_lowerr"], rdict["M_dust_uperr"] = S_nu_emit_ref_errs * self.D_L**2 / (M_sun_g * k_nu_star * Planck_func(nu_star, T_dust) * 1e26)
            
                    rdict["dust_frac"] = rdict["M_dust"] / self.obj_M
                    rdict["dust_frac_lowerr"] = rdict["dust_frac"] * np.sqrt((rdict["M_dust_lowerr"]/rdict["M_dust"])**2 + (self.obj_M_uperr/self.obj_M)**2)
                    rdict["dust_frac_uperr"] = rdict["dust_frac"] * np.sqrt((rdict["M_dust_uperr"]/rdict["M_dust"])**2 + (self.obj_M_lowerr/self.obj_M)**2)

                    rdict["dust_yield_AGB"], rdict["dust_yield_AGB_lowerr"], rdict["dust_yield_AGB_uperr"] = 29 * np.array([rdict["dust_frac"],
                                                                                        rdict["dust_frac_lowerr"], rdict["dust_frac_uperr"]])
                    rdict["dust_yield_SN"], rdict["dust_yield_SN_lowerr"], rdict["dust_yield_SN_uperr"] = 84 * np.array([rdict["dust_frac"],
                                                                                        rdict["dust_frac_lowerr"], rdict["dust_frac_uperr"]])

                    rdict["L_IR_uplim"] = uplim
                    rdict["L_IR_Lsun"], rdict["L_IR_Lsun_lowerr"], rdict["L_IR_Lsun_uperr"] = L_IR_Lsun, L_IR_Lsun_lowerr, L_IR_Lsun_uperr
                    L_IR_log10 = math.floor(np.log10(rdict["L_IR_Lsun"]))
                    rdict["L_FIR_Lsun"], rdict["L_FIR_Lsun_lowerr"], rdict["L_FIR_Lsun_uperr"] = L_FIR_Lsun, L_FIR_Lsun_lowerr, L_FIR_Lsun_uperr

                    rdict["SFR_IR"] = SFR_L(rdict["L_IR_Lsun"] * L_sun_ergs, band="TIR")
                    rdict["SFR_IR_err"] = SFR_L(np.array([rdict["L_IR_Lsun_lowerr"], rdict["L_IR_Lsun_uperr"]]) * L_sun_ergs, band="TIR")

                    rdict["SFR"] = self.SFR_UV + rdict["SFR_IR"]
                    rdict["SFR_err"] = np.sqrt(np.tile(self.SFR_UV_err, 2)**2 + (rdict["SFR_IR"]/3.0 if rdict["L_IR_uplim"] else rdict["SFR_IR_err"])**2)

                    if uplim:
                        IR_ann_list = [r"$L_\mathrm{{ IR }} \lesssim {:.1f} \cdot 10^{{ {:d} }} \, \mathrm{{ L_\odot }}$".format(rdict["L_IR_Lsun"]/10**L_IR_log10,
                                                    L_IR_log10),
                                        r"$\mathrm{{ SFR_{{IR}} }} \lesssim {:.0f} \, \mathrm{{ M_\odot \, yr^{{-1}} }}$".format(rdict["SFR_IR"])]
                    else:
                        IR_ann_list = [r"$L_\mathrm{{ IR }} \simeq {:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }} \cdot 10^{{ {:d} }} \, \mathrm{{ L_\odot }}$".format(rdict["L_IR_Lsun"]/10**L_IR_log10,
                                                rdict["L_IR_Lsun_lowerr"]/10**L_IR_log10, rdict["L_IR_Lsun_uperr"]/10**L_IR_log10, L_IR_log10),
                                        r"$\mathrm{{ SFR_{{IR}} }} \simeq {:.0f}_{{ -{:.0f} }}^{{ +{:.0f} }} \, \mathrm{{ M_\odot \, yr^{{-1}} }}$".format(rdict["SFR_IR"],
                                                *rdict["SFR_IR_err"])]
                    
                    if compatible_beta:
                        if annotate_title:
                            prop_ann.set_text(prop_ann.get_text() + "\n\nFor " + r"$T_\mathrm{{ dust }} = {:.0f} \, \mathrm{{ K }}$, ".format(fixed_T_dust) + \
                                                r"$\beta_\mathrm{{ IR }} = {:.2g}$:".format(fixed_beta) + '\n' + \
                                                '\n'.join(IR_ann_list))
                    else:
                        for key in rdict.keys():
                            if not "uplim" in key:
                                rdict[key] = np.nan

                T_dust_handles.append(T_dust)
                x_betas.append(x)
                S_nu_obs_betas.append(S_nu_obs)
                L_IR_Lsun_betas.append(L_IR_Lsun)
                L_FIR_Lsun_betas.append(L_FIR_Lsun)

            low_beta_idx = min([ci for ci, compatible in enumerate(compatible_betas) if compatible], default=0)
            high_beta_idx = max([ci for ci, compatible in enumerate(compatible_betas) if compatible], default=len(beta_IRs)-1)
            
            n_compatible_betas = np.sum(compatible_betas)
            any_compatible_betas = n_compatible_betas > 0
            all_compatible_betas = n_compatible_betas == len(compatible_betas)
            
            if any_compatible_betas:
                # Adjust minimum and maximum peak dust temperature
                if np.min(T_peak_betas[low_beta_idx:high_beta_idx+1]) < T_peak_minmax[0]:
                    T_peak_minmax[0] = np.min(T_peak_betas[low_beta_idx:high_beta_idx+1])
                if np.max(T_peak_betas[low_beta_idx:high_beta_idx+1]) > T_peak_minmax[1]:
                    T_peak_minmax[1] = np.max(T_peak_betas[low_beta_idx:high_beta_idx+1])

                # Plot the observed spectrum (intrinsic spectrum is nearly the same apart from at the very red wavelengths above ~100 micron)
                # ax.plot(lambda_emit, S_nu_emit, linestyle='--', color=dcolor, alpha=0.8)
                ax.plot(x_betas[low_beta_idx], S_nu_obs_betas[low_beta_idx], linewidth=1.5, color=dcolor, alpha=0.8)
                ax.plot(x_betas[high_beta_idx], S_nu_obs_betas[high_beta_idx], linewidth=1.5, color=dcolor, alpha=0.8)

                # Also fill in area in between curves of minimum/maximum beta_IR that is compatible
                assert np.all([x == x_betas[0] for l in x_betas])
                ax.fill_between(x_betas[0], y1=S_nu_obs_betas[low_beta_idx], y2=S_nu_obs_betas[high_beta_idx],
                                facecolor=dcolor, edgecolor="None", alpha=0.2)
            
            if not any_compatible_betas:
                if x_betas and S_nu_obs_betas:
                    ax.plot(x_betas[0], S_nu_obs_betas[0], linestyle='--', linewidth=1.0, color=dcolor, alpha=0.5)
                    ax.plot(x_betas[-1], S_nu_obs_betas[-1], linestyle='--', linewidth=1.0, color=dcolor, alpha=0.5)
                    ax.fill_between(x_betas[0], y1=S_nu_obs_betas[0], y2=S_nu_obs_betas[-1],
                                    facecolor=dcolor, edgecolor="None", alpha=0.05)
            elif not all_compatible_betas:
                if not compatible_betas[0]:
                    ax.plot(x_betas[0], S_nu_obs_betas[0], linestyle='--', linewidth=1.0, color=dcolor, alpha=0.5)
                    ax.fill_between(x_betas[0], y1=S_nu_obs_betas[0], y2=S_nu_obs_betas[low_beta_idx],
                                    facecolor=dcolor, edgecolor="None", alpha=0.05)
                if not compatible_betas[-1]:
                    ax.plot(x_betas[-1], S_nu_obs_betas[-1], linestyle='--', linewidth=1.0, color=dcolor, alpha=0.5)
                    ax.fill_between(x_betas[0], y1=S_nu_obs_betas[high_beta_idx], y2=S_nu_obs_betas[-1],
                                    facecolor=dcolor, edgecolor="None", alpha=0.05)

            if any_compatible_betas and annotate_results:
                # Annotate results of L_IR
                L_IR_log10 = math.floor(np.log10(np.min([L_IR_Lsun_betas[0], L_IR_Lsun_betas[-1]])))
                L_FIR_log10 = math.floor(np.log10(np.min([L_FIR_Lsun_betas[0], L_FIR_Lsun_betas[-1]])))
                text = r"$L_\mathrm{{ IR }} {} ( {:.1f}$-${:.1f} )".format('=' if np.sum(self.cont_det) > 0 else r"\leq",
                        *np.sort([L_IR_Lsun_betas[0], L_IR_Lsun_betas[-1]])/10**L_IR_log10) + \
                            r"\cdot 10^{{ {:d} }} \, \mathrm{{ L_\odot }}$".format(L_IR_log10)

                if not all_compatible_betas:
                    if low_beta_idx == high_beta_idx:
                        text += "\n\t" + r"(for $\beta_\mathrm{{ IR }} = {:.2g}$)".format(beta_IRs[low_beta_idx])
                    else:
                        text += "\n\t" + r"(for $\beta_\mathrm{{ IR }} \in [{:.2g}$, ${:.2g}]$)".format(beta_IRs[low_beta_idx], beta_IRs[high_beta_idx])

                # Save text for later annotation and update position of all previous annotations
                for cb_ann_text_offset_colour in cb_ann_text_offset_colours:
                    cb_ann_text_offset_colour[1] += 1 + int(not all_compatible_betas)
                cb_ann_text_offset_colours.append([text, 0, dcolor])

        for cb_ann_text_offset_colour in cb_ann_text_offset_colours:
            ax.annotate(text=cb_ann_text_offset_colour[0], xy=(0.025, 0.025), xytext=(0, 16*cb_ann_text_offset_colour[1]),
                        xycoords="axes fraction", textcoords="offset points", va="bottom", ha="left", color='k',
                        size="x-small", alpha=0.8).set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor=cb_ann_text_offset_colour[2], alpha=0.8))
        
        rdict["T_peak"] = np.nan
        rdict["T_peak_err"] = np.nan
        rdict["T_peak_constraint"] = "none"
        
        if save_results:
            # Save results
            np.savez_compressed(self.mnrfol + "{}_FIR_SED_parameters_{}{}{}.npz".format(self.obj_fn, "T_{:.0f}".format(fixed_T_dust),
                                "_beta_{:.1f}".format(fixed_beta), "_l0_{:.0f}".format(l0) if l0 else ''), **rdict)

        if show_legend:
            handles = [matplotlib.patches.Rectangle(xy=(np.nan, np.nan), width=1, height=1, edgecolor="None", facecolor=dust_cmap(dust_norm(T_dust)), alpha=0.8,
                        label=r"$T_\mathrm{{ dust }} = {:.0f} \, \mathrm{{ K }}$".format(T_dust)) for T_dust in T_dusts if T_dust in T_dust_handles]
            
            leg = ax.legend(handles=handles, ncol=2, loc="lower right", frameon=True, framealpha=0.8, fontsize="small")
            
            # Show which beta_IRs have been used
            if l0 == "self-consistent":
                l0_lab = l0.capitalize() + " opacity, "
            else:
                l0_lab = "Dust emission with fixed " + r"$\lambda_0 = {:.0f} \, \mathrm{{ \mu m }}$, ".format(l0) if l0 else "Optically thin dust emission, "
            leg.set_title(l0_lab + r"$\beta_\mathrm{{ IR }} \in [{:.2g}$, ${:.2g}]$".format(beta_IRs[0], beta_IRs[-1]), prop={"size": "small"})
        
        if self.verbose:
            print("T_dust: {:.1f}-{:.1f} K".format(T_dusts[0], T_dusts[-1]))
            print("beta_IR: {:.1f}-{:.1f}".format(beta_IRs[0], beta_IRs[-1]))
            if save_results:
                print("\nFiducial results (T_dust = {:.0f} K, beta_IR = {:.2g}) for {}:".format(fixed_T_dust, fixed_beta, self.obj), end='')
                if fixed_vals_cb:
                    M_dust_log10 = math.floor(np.log10(rdict["M_dust"]))
                    L_IR_log10 = math.floor(np.log10(rdict["L_IR_Lsun"]))
                    L_FIR_log10 = math.floor(np.log10(rdict["L_FIR_Lsun"]))

                    valstr = lambda fmt, *vals: "< {}{:{fmt}}{}".format(vals[0], vals[1], vals[-1], fmt=fmt) if uplim \
                                            else "= {}{:{fmt}} -{:{fmt}} +{:{fmt}}{}".format(*vals, fmt=fmt)
                    print("\nM_dust {} x 10^{:d} M_sun".format(valstr(".1f", '(', rdict["M_dust"]/10**M_dust_log10,
                                rdict["M_dust_lowerr"]/10**M_dust_log10, rdict["M_dust_uperr"]/10**M_dust_log10, ')'), M_dust_log10))
                    print("L_IR {} x 10^{:d} L_sun".format(valstr(".1f", '(', rdict["L_IR_Lsun"]/10**L_IR_log10,
                                rdict["L_IR_Lsun_lowerr"]/10**L_IR_log10, rdict["L_IR_Lsun_uperr"]/10**L_IR_log10, ')'), L_IR_log10))
                    print("L_FIR {} x 10^{:d} L_sun".format(valstr(".1f", '(', rdict["L_FIR_Lsun"]/10**L_FIR_log10,
                                rdict["L_FIR_Lsun_lowerr"]/10**L_FIR_log10, rdict["L_FIR_Lsun_uperr"]/10**L_FIR_log10, ')'), L_FIR_log10))
                    print('')
                    print("Dust-to-stellar mass fraction: {} per cent".format(valstr(".2g", '', rdict["dust_frac"],
                                                        rdict["dust_frac_lowerr"], rdict["dust_frac_uperr"], '')))
                    print("Dust yield (AGB): {} M_sun".format(valstr(".2g", '', rdict["dust_yield_AGB"],
                                                        rdict["dust_yield_AGB_lowerr"], rdict["dust_yield_AGB_uperr"], '')))
                    print("Dust yield (SN): {} M_sun".format(valstr(".2g", '', rdict["dust_yield_SN"],
                                                        rdict["dust_yield_SN_lowerr"], rdict["dust_yield_SN_uperr"], '')))
                else:
                    print("\nincompatible with measurements!")
    
        if plot_data:
            self.plot_data(bot_axis=bot_axis)
        self.set_axes(bot_axis=bot_axis, add_top_axis=add_top_axis, set_xrange=set_xrange, set_xlabel=set_xlabel, set_ylabel=set_ylabel,
                        low_yspace_mult=low_yspace_mult, up_yspace_mult=up_yspace_mult, rowi=rowi, coli=coli)
        if pltfol:
            self.save_fig(pltfol=pltfol, ptype="ranges", obj_str=obj_str, l0_list=[l0], single_plot=False)
    
    def annotate_title(self):
        """Function for annotating a figure with the main object properties; designed for internal use.

        """

        ax = self.ax

        text = self.obj
        if self.reference:
            text += " ({})".format(self.reference.replace('(', '').replace(')', ''))
            size = "medium"
            bbox_col = 'w'
        else:
            size = "large"
            bbox_col = self.obj_color if self.obj_color else 'w'
        
        ax.annotate(text=text, xy=(0, 1), xytext=(8, -8), xycoords="axes fraction", textcoords="offset points",
                    va="top", ha="left", color='k', size=size, alpha=0.8, zorder=6).set_bbox(dict(boxstyle="Round, pad=0.05", facecolor=bbox_col, edgecolor="None", alpha=0.8))

        text = r"$z = {:.6g}$, $T_\mathrm{{ CMB }} = {:.2f} \, \mathrm{{ K }}$".format(self.z, T_CMB_obs*(1.0+self.z))
        if not np.isnan(self.obj_M):
            text += '\n' + r"$M_* = {:.1f}_{{-{:.1f}}}^{{+{:.1f}}} \cdot 10^{{{:d}}} \, \mathrm{{M_\odot}}$".format(self.obj_M/10**math.floor(np.log10(self.obj_M)),
                        self.obj_M_lowerr/10**math.floor(np.log10(self.obj_M)), self.obj_M_uperr/10**math.floor(np.log10(self.obj_M)), math.floor(np.log10(self.obj_M)))
        if not np.isnan(self.SFR_UV):
            text += '\n' + r"$\mathrm{{ SFR_{{UV}} }} = {:.0f}{}".format(self.SFR_UV, r'' if np.isnan(self.SFR_UV_err) else r" \pm {:.0f}".format(self.SFR_UV_err)) + \
                    r" \, \mathrm{{M_\odot yr^{{-1}}}}$"
        prop_ann = ax.annotate(text=text, xy=(1, 1), xytext=(-8, -8), xycoords="axes fraction", textcoords="offset points",
                                va="top", ha="right", color='k', size="small", alpha=0.8, zorder=6)
        prop_ann.set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))

        return prop_ann
    
    def annotate_results(self, rdict, axes_ann, ax_type, ann_size):
        """Function for annotating a figure with the main results of a greybody fit; designed for internal use.

        Parameters
        ----------
        rdict : dict
            Dictionary containing the main MultiNest results.
        axes_ann : list
            Sequence of two `matplotlib.axes.Axes` to be annotated.
        ax_type : {`"regular"`, `"corner"`}
            String indicating the type of figure being annotated.
        ann_size : float or {'xx-small', 'x-small', 'small', 'medium',
                    'large', 'x-large', 'xx-large'}
            Argument used by `matplotlib.text.Text` for the font size of the annotation.

        """

        M_dust_log10 = math.floor(np.log10(rdict["M_dust"]))
        L_IR_log10 = math.floor(np.log10(rdict["L_IR_Lsun"]))
        L_FIR_log10 = math.floor(np.log10(rdict["L_FIR_Lsun"]))
        
        if self.valid_cont_area:
            Sigma_sign = '>' if self.cont_area_uplim else '='
            lambda_0_sign = '<' if self.cont_area_uplim else '='
        
        if self.T_lolim:
            T_lim_str = r" > {:.0f} \, \mathrm{{ K }}$".format(rdict["T_lolim"])
            T_z0_lim_str = r" > {:.0f} \, \mathrm{{ K }}$".format(rdict["T_lolim_z0"])
        elif self.T_uplim:
            T_lim_str = r" < {:.0f} \, \mathrm{{ K }}$".format(rdict["T_uplim"])
            T_z0_lim_str = r" < {:.0f} \, \mathrm{{ K }}$".format(rdict["T_uplim_z0"])
        else:
            T_lim_str = r'$'
            T_z0_lim_str = r'$'
        
        # Annotate results
        prec_IR = max(0, 2-math.floor(np.log10(min(rdict["L_IR_Lsun_lowerr"]/10**L_IR_log10, rdict["L_IR_Lsun_uperr"]/10**L_IR_log10))))
        prec_FIR = max(0, 2-math.floor(np.log10(min(rdict["L_FIR_Lsun_lowerr"]/10**L_FIR_log10, rdict["L_FIR_Lsun_uperr"]/10**L_FIR_log10))))
        prec_SFR = max(0, 2-math.floor(np.log10(np.min(rdict["SFR_IR_err"])))) if np.min(rdict["SFR_IR_err"]) < 1 else 0
        prec_M = max(1, 2-math.floor(np.log10(min(rdict["M_dust_lowerr"]/10**M_dust_log10, rdict["M_dust_uperr"]/10**M_dust_log10))))
        
        text = r"$L_\mathrm{{ IR }} = {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \cdot 10^{{ {:d} }} \, \mathrm{{ L_\odot }}$".format(rdict["L_IR_Lsun"]/10**L_IR_log10,
                    rdict["L_IR_Lsun_lowerr"]/10**L_IR_log10, rdict["L_IR_Lsun_uperr"]/10**L_IR_log10, L_IR_log10, prec=prec_IR) + \
                '\n' + r"$L_\mathrm{{ FIR }} = {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \cdot 10^{{ {:d} }} \, \mathrm{{ L_\odot }}$".format(rdict["L_FIR_Lsun"]/10**L_FIR_log10,
                    rdict["L_FIR_Lsun_lowerr"]/10**L_FIR_log10, rdict["L_FIR_Lsun_uperr"]/10**L_FIR_log10, L_FIR_log10, prec=prec_FIR) + \
                '\n' + r"$\mathrm{{ SFR_{{IR}} }} = {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \, \mathrm{{ M_\odot \, yr^{{-1}} }}$".format(rdict["SFR_IR"],
                    *rdict["SFR_IR_err"], prec=prec_SFR) + \
                "\n\n" + r"$M_\mathrm{{ dust }} = {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \cdot 10^{{ {:d} }} \, \mathrm{{ M_\odot }}$".format(rdict["M_dust"]/10**M_dust_log10,
                    rdict["M_dust_lowerr"]/10**M_dust_log10, rdict["M_dust_uperr"]/10**M_dust_log10, M_dust_log10, prec=prec_M)
        if self.valid_cont_area:
            prec_Sig = max(0, 2-math.floor(np.log10(min(rdict["Sigma_dust_lowerr"], rdict["Sigma_dust_uperr"]))))
            prec_l0 = max(0, 2-math.floor(np.log10(min(rdict["lambda_0_lowerr"], rdict["lambda_0_uperr"]))))

            text += '\n' + r"$\Sigma_\mathrm{{ dust }} {} {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \, \mathrm{{ M_\odot \, pc^{{-2}} }}$".format(Sigma_sign,
                    rdict["Sigma_dust"], rdict["Sigma_dust_lowerr"], rdict["Sigma_dust_uperr"], prec=prec_Sig) + \
                '\n' + r"$\lambda_0 {} {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \, \mathrm{{ \mu m }}$".format(lambda_0_sign, rdict["lambda_0"],
                    rdict["lambda_0_lowerr"], rdict["lambda_0_uperr"], prec=prec_l0)
        
        if not np.isnan(rdict["dust_frac"]):
            prec_fr = max(0, 2-math.floor(np.log10(min(rdict["dust_frac_lowerr"], rdict["dust_frac_uperr"]))))
            prec_yAGB = max(0, 2-math.floor(np.log10(min(rdict["dust_yield_AGB_lowerr"], rdict["dust_yield_AGB_uperr"]))))
            prec_ySN = max(0, 2-math.floor(np.log10(min(rdict["dust_yield_SN_lowerr"], rdict["dust_yield_SN_uperr"]))))

            text += '\n' + r"$M_\mathrm{{ dust }} / M_* = {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }}$".format(rdict["dust_frac"],
                                rdict["dust_frac_lowerr"], rdict["dust_frac_uperr"], prec=prec_fr) + \
                    '\n' + r"Dust yield (AGB, SN): ${:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \, \mathrm{{ M_\odot }}$, ".format(rdict["dust_yield_AGB"],
                                rdict["dust_yield_AGB_lowerr"], rdict["dust_yield_AGB_uperr"], prec=prec_yAGB) + \
                            r"${:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \, \mathrm{{ M_\odot }}$".format(rdict["dust_yield_SN"],
                                rdict["dust_yield_SN_lowerr"], rdict["dust_yield_SN_uperr"], prec=prec_ySN)
        
        if ax_type == "regular":
            xy = (0, 0)
            xytext = (8, 8)
            va = "bottom"
            ha = "left"
        elif ax_type == "corner":
            xy = (1, 1)
            xytext = (-8, -8)
            va = "top"
            ha = "right"
        ann = axes_ann[0].annotate(text=text, xy=xy, xytext=xytext, xycoords="axes fraction", textcoords="offset points",
                                    va=va, ha=ha, color='k', size=ann_size, alpha=0.8, zorder=6)
        ann.set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
        
        if self.fixed_beta:
            beta_str = r"$\beta_\mathrm{{ IR }} = {:.2g}$ (fixed)".format(rdict["beta_IR"])
        else:
            prec = max(0, 2-math.floor(np.log10(min(rdict["beta_IR_lowerr"], rdict["beta_IR_uperr"]))))
            beta_str = r"$\beta_\mathrm{{ IR }} = {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }}$".format(rdict["beta_IR"],
                        rdict["beta_IR_lowerr"], rdict["beta_IR_uperr"], prec=prec)
        
        prec = max(0, 2-math.floor(np.log10(min(rdict["T_dust_lowerr"], rdict["T_dust_uperr"]))))

        text = '\n' + r"$T_\mathrm{{ dust }} = {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \, \mathrm{{ K }}".format(rdict["T_dust"],
                    rdict["T_dust_lowerr"], rdict["T_dust_uperr"], prec=prec) + T_lim_str + \
                '\n' + r"$T_\mathrm{{ dust }}^{{ z=0 }} = {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \, \mathrm{{ K }}".format(rdict["T_dust_z0"],
                            rdict["T_dust_z0_lowerr"], rdict["T_dust_z0_uperr"], prec=prec) + T_z0_lim_str + \
                '\n' + r"$T_\mathrm{{ peak }} = {:.{prec}f}_{{ -{:.{prec}f} }}^{{ +{:.{prec}f} }} \, \mathrm{{ K }}".format(rdict["T_peak_val"],
                            rdict["T_peak_lowerr"], rdict["T_peak_uperr"], prec=prec) + \
                                (r" {} {:.{prec}f} \, \mathrm{{ K }}$".format(r'>' if self.T_lolim else r'<', rdict["T_peak"], prec=prec) if self.T_lolim or self.T_uplim else r'$') + \
                '\n' + beta_str
        
        if ax_type == "regular":
            ann = axes_ann[1].annotate(text=text, xy=(1, 0), xytext=(-8, 8), xycoords="axes fraction", textcoords="offset points",
                                        va="bottom", ha="right", color='k', size=ann_size, alpha=0.8, zorder=6)
            ann.set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
        elif ax_type == "corner":
            ann.set_text(ann.get_text() + '\n' + text)

    def plot_data(self, bot_axis, plot_ax_res=False):
        """Function for plotting the photometric data; designed for internal use.

        Parameters
        ----------
        bot_axis : {`"wl_emit"`, `"nu_obs"`}
            Choice for the main x-axis to show the rest-frame wavelength (`"wl_emit"`) or observed
            frequency (`"nu_obs"`).
        plot_ax_res : bool, optional
            Plot residual fluxes? Default: `True`.

        """
        
        for l, lrange, s_nu, s_nuerr, uplim, exclude in zip(self.lambda_emit_vals, self.lambda_emit_ranges, self.S_nu_vals, self.S_nu_errs,
                                                            self.cont_uplims, self.cont_excludes):
            if bot_axis == "wl_emit":
                x = l
                xrange = lrange
            elif bot_axis == "nu_obs":
                nu_obs = 299792.458 / l / (1.0 + self.z) # GHz (l is in micron)
                x = nu_obs
                xrange = np.abs(np.sort((299792.458 / (l + np.array([-1, 1]) * lrange) / (1.0 + self.z)) - nu_obs))
            
            self.ax.errorbar(x, s_nu*self.fd_conv, xerr=xrange.reshape(2, 1),
                                yerr=0.5*s_nu*self.fd_conv if uplim else s_nuerr*self.fd_conv, uplims=uplim,
                                marker='o', linestyle="None", color='k', alpha=0.4 if exclude else 0.8, zorder=5)
            if uplim:
                self.ax.errorbar(x, s_nu/self.uplim_nsig*self.fd_conv,
                                    marker='_', linestyle="None", color='k', alpha=0.4 if exclude else 0.8, zorder=5)
            
            if s_nu / (self.uplim_nsig if uplim else 1) * self.fd_conv < self.F_nu_obs_min:
                self.F_nu_obs_min = s_nu/(self.uplim_nsig if uplim else 1) * self.fd_conv
            if s_nu / (self.uplim_nsig if uplim else 1) * self.fd_conv > self.F_nu_obs_max:
                self.F_nu_obs_max = s_nu/(self.uplim_nsig if uplim else 1) * self.fd_conv
            
            if plot_ax_res:
                # Plot the residuals of the observed spectrum
                s_nu_model = np.interp(x, self.x, self.S_nu_obs, left=np.nan, right=np.nan)
                
                self.ax_res.errorbar(x, (s_nu-s_nu_model)*self.fd_conv, xerr=xrange.reshape(2, 1),
                                        yerr=0.5*s_nu*self.fd_conv if uplim else s_nuerr*self.fd_conv, uplims=uplim,
                                        marker='o', linestyle="None", color='k', alpha=0.4 if exclude else 0.8, zorder=5)
                
                if not exclude:
                    self.residuals_min.append((s_nu-s_nu_model-(2 if uplim else 1.25)*s_nuerr)*self.fd_conv)
                    self.residuals_max.append((s_nu-s_nu_model+(2 if uplim else 1.25)*s_nuerr)*self.fd_conv)
                
                if uplim:
                    self.ax_res.errorbar(x, (s_nu/self.uplim_nsig-s_nu_model)*self.fd_conv,
                                            marker='_', linestyle="None", color='k', alpha=0.4 if exclude else 0.8, zorder=5)
    
    def set_axes(self, bot_axis, add_top_axis, set_xrange, set_xlabel, set_ylabel, low_yspace_mult, up_yspace_mult, rowi, coli, plot_ax_res=False):
        """Function for setting up the axes in an SED plot; designed for internal use.

        Parameters
        ----------
        bot_axis : {`"wl_emit"`, `"nu_obs"`}
            Choice for the main x-axis to show the rest-frame wavelength (`"wl_emit"`) or observed
            frequency (`"nu_obs"`).
        add_top_axis : {`"wl_obs"`, `"nu_obs"`, `False`}
            Add second axis showing the observed wavelength (`"wl_obs"`) or frequency (`"nu_obs"`)
            on the top x-axis?
        set_xrange : bool
            Manually set bounds on the rest-frame wavelength axis in the plots?
        set_xlabel : {`"top"`, `"bottom"`, `"both"`, `False`}
            Set labels on the rest-frame (bottom) and/or observed (top) wavelength axes in this plot?
        set_ylabel : bool
            Set labels on the flux density axis in this plot?
        low_yspace_mult : float
            Multiplier of the lower flux density bound to create more vertical space in this plot.
        up_yspace_mult : float
            Multiplier of the upper flux density bound to create more vertical space in this plot.
        rowi : int
            Row number of this plot in a multi-panel figure.
        coli : int
            Column number of this plot in a multi-panel figure.
        plot_ax_res : bool, optional
            Plot residual fluxes? Default: `True`.

        """

        lfunc = lcoord_funcs(rowi=rowi, coli=coli, z=self.z)

        if add_top_axis == "wl_obs":
            if bot_axis == "wl_emit":
                functions = (lfunc.lamrf2lamobs, lfunc.lamobs2lamrf)
            elif bot_axis == "nu_obs":
                functions = (lfunc.nuobs2lamobs, lfunc.lamobs2nuobs)
                
            self.top_ax = self.ax.secondary_xaxis("top", functions=functions)
            self.top_ax.tick_params(axis='x', which="both", bottom=False, top=True, labelbottom=False, labeltop=True)
            self.ax.tick_params(axis='x', which="both", top=False, labelbottom=not plot_ax_res)
        elif add_top_axis == "nu_obs":
            if bot_axis == "wl_emit":
                self.top_ax = self.ax.secondary_xaxis("top", functions=(lambda l_emit: 299.792458/lfunc.lamrf2lamobs(l_emit), lambda nu_obs: lfunc.lamobs2lamrf(299.792458/nu_obs)))
                self.top_ax.tick_params(axis='x', which="both", bottom=False, top=True, labelbottom=False, labeltop=True)
                self.ax.tick_params(axis='x', which="both", top=False, labelbottom=not plot_ax_res)
            elif bot_axis == "nu_obs":
                add_top_axis = False
                self.top_ax = None
                self.ax.tick_params(axis='x', which="both", top=True, labelbottom=not plot_ax_res)

        if plot_ax_res:
            # Link x-axis of the two plots
            self.ax.get_shared_x_axes().join(self.ax, self.ax_res)

        self.ax.set_xscale("log")
        self.ax.set_yscale("log")

        if set_xrange:
            if bot_axis == "wl_emit":
                self.ax.set_xlim(self.l_min, self.l_max)
            elif bot_axis == "nu_obs":
                self.ax.set_xlim(lfunc.lamrf2nuobs(self.l_min), lfunc.lamrf2nuobs(self.l_max))
        
        if np.isfinite(self.F_nu_obs_min) and np.isfinite(self.F_nu_obs_max):
            self.ax.set_ylim(low_yspace_mult*self.F_nu_obs_min, up_yspace_mult*self.F_nu_obs_max)
        if hasattr(self, "residuals_min") and hasattr(self, "residuals_max"):
            if np.isfinite(np.min(self.residuals_min)) and np.isfinite(np.max(self.residuals_max)):
                self.ax_res.set_ylim(np.min(self.residuals_min), np.max(self.residuals_max))
        
        if set_xlabel == "top" or set_xlabel == "both":
            assert add_top_axis in ["wl_obs", "nu_obs"]
            if add_top_axis == "wl_obs":
                self.top_ax.set_xlabel(r"$\lambda_\mathrm{obs} \, (\mathrm{mm})$")
            elif add_top_axis == "nu_obs":
                self.top_ax.set_xlabel(r"$\nu_\mathrm{obs} \, (\mathrm{GHz})$")
        if set_xlabel == "bottom" or set_xlabel == "both":
            if plot_ax_res:
                if bot_axis == "wl_emit":
                    self.ax_res.set_xlabel(r"$\lambda_\mathrm{emit} \, (\mathrm{\mu m})$")
                elif bot_axis == "nu_obs":
                    self.ax_res.set_xlabel(r"$\nu_\mathrm{obs} \, (\mathrm{GHz})$")
            else:
                if bot_axis == "wl_emit":
                    self.ax.set_xlabel(r"$\lambda_\mathrm{emit} \, (\mathrm{\mu m})$")
                elif bot_axis == "nu_obs":
                    self.ax.set_xlabel(r"$\nu_\mathrm{obs} \, (\mathrm{GHz})$")
        if set_ylabel:
            self.ax.set_ylabel(r"$F_\mathrm{{ \nu, \, obs }} \, (\mathrm{{ {} }})$".format(self.fluxdens_unit.replace("mu", r"\mu ")))
            if plot_ax_res:
                self.ax_res.set_ylabel(r"$\Delta F_\mathrm{{ \nu, \, obs }} \, (\mathrm{{ {} }})$".format(self.fluxdens_unit.replace("mu", r"\mu ")))
    
    def save_fig(self, pltfol, fig=None, ptype="constraints", obj_str=None, l0_list=None, single_plot=None):
        """Function for saving a figure; designed for internal use, but can also be used
        externally.

        Parameters
        ----------
        pltfol : str
            Path to folder in which figures are saved.
        fig : {`None`, instance of `matplotlib.figure.Figure`}, optional
            Figure instance to be saved. Default is `None`, which will save the figure last
            supplied or created in either `plot_MN_fit` or `plot_ranges`.
        ptype : str, optional
            String to be added to the filename of the figure, after `"FIR_SED_"`. Default
            is `"constraints"`.
        obj_str : {`None`, str}, optional
            String to be added to the filename of the figure. Default is `None`,
            in which case the object name, preceded by an underscore, will be used.
        l0_list : {`None`, list}, optional
            A list of opacity model classifiers to be added to the filename of the figure.
            Default is `None`, which will revert to the main `l0_list`, given when creating
            the `FIR_SED_fit` instance.
        single_plot : {`None`, bool}, optional
            Indicates whether a single plot has been made for different opacity models. Default is `None`,
            in which case a single plot has been made when `analysis` is `False`.

        """

        if fig is None:
            fig = self.fig
        if obj_str is None:
            obj_str = '_' + self.obj_fn
        if single_plot is None:
            single_plot = not self.analysis
        
        fig.savefig(pltfol + "FIR_SED_{}{}".format(ptype, obj_str) + self.get_mstring(l0_list=l0_list, single_plot=single_plot) + self.pformat,
                    dpi=self.dpi, bbox_inches="tight")
        
        # plt.show()
        plt.close(fig)
    
    def get_l0string(self, l0):
        """Function for creating a string of a list of opacity model classifiers; designed for
        internal use.

        Parameters
        ----------
        l0 : {`None`, "self-consistent", float}
            Opacity model classifiers: `None` for an optically thin model, `"self-consistent"`
            for a self-consistent opacity model, or a float setting a fixed value of `lambda_0`,
            the wavelength in micron setting the SED's transition point between optically thin
            and thick.
        
        Returns
        ----------
        l0_tuple : tuple
            A tuple containing `(l0_str, l0_txt)`.

        """

        if l0 == "self-consistent":
            l0_str = "_{}-l0".format(l0)
            l0_txt = " and {} λ_0".format(l0)
        else:
            l0_str = "_l0_{:.0f}".format(l0) if l0 else ''
            l0_txt = " and λ_0 = {:.0f} μm".format(l0) if l0 else " under a fully optically thin SED"
        
        return (l0_str, l0_txt)
    
    def get_mstring(self, l0_list=None, analysis=None, single_plot=None, inc_astr=True):
        """Function for creating a string to be used in a filename; designed for internal
        use.

        Parameters
        ----------
        l0_list : {`None`, list}, optional
            A list of opacity model classifiers to be added to the filename of the figure.
            Default is `None`, which will revert to the main `l0_list`, given when creating
            the `FIR_SED_fit` instance.
        analysis : {`None`, bool}, optional
            Manually set the value of `analysis` to be used in the string. Default is `None`,
            such that the main `analysis`, given when creating the `FIR_SED_fit` instance,
            is used.
        single_plot : {`None`, bool}, optional
            Indicates whether a single plot has been made for different opacity models. Default is `None`,
            in which case a single plot has been made when `analysis` is `False`.
        inc_astr : bool, optional
            Controls whether the string will contain an identifier if `analysis` is `True`.
            Default is `True`.
        
        Returns
        ----------
        mstring : str
            A string to be used in a filename.

        """

        if l0_list is None:
            l0_list = self.l0_list
        if analysis is None:
            analysis = self.analysis
        if single_plot is None:
            single_plot = not self.analysis
        
        if single_plot:
            lstr = ''
        else:
            lstr = '_' + '_'.join(sorted(set([("scl0" if l0 == "self-consistent" else "l0_{:.0f}".format(l0)) for l0 in l0_list if l0]))) if any(l0_list) else ''
        
        return lstr + ("_beta_{:.1f}".format(self.fixed_beta) if self.fixed_beta else '') + ("_analysis" if analysis and inc_astr else '')
    
    def load_rdict(self, l0):
        """Function to obtain a saved dictionary of results.

        Parameters
        ----------
        l0 : {`None`, "self-consistent", float}
            Opacity model classifiers: `None` for an optically thin model, `"self-consistent"`
            for a self-consistent opacity model, or a float setting a fixed value of `lambda_0`,
            the wavelength in micron setting the SED's transition point between optically thin
            and thick.
        
        Returns
        ----------
        rdict : dictionary
            A dictionary containing results of the fitting routine.

        """

        if hasattr(self, "rdict"):
            rdict = self.rdict
        else:
            l0_str, l0_txt = self.get_l0string(l0)
            rdict_fname = self.mnrfol + "{}_MN_FIR_SED_fit_{}{}.npz".format(self.obj_fn, self.beta_str, l0_str)
            if os.path.isfile(rdict_fname):
                rdict = np.load(rdict_fname)
            else:
                if self.verbose:
                    print("Warning: MultiNest results with {}{}".format("β = {:.1f}".format(self.fixed_beta) if self.fixed_beta else "varying β", l0_txt),
                            "for {} not found! Filename:\n{}\nContinuing...".format(self.obj, rdict_fname.split('/')[-1]))
                rdict = None
        
        return rdict

