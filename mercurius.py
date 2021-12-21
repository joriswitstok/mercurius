#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for running MultiNest fits on dust FIR SEDs.

Joris Witstok, 17 November 2021
"""

import os, sys
from mock import patch
if __name__ == "__main__":
    print("Python", sys.version)

import numpy as np
rng = np.random.default_rng(seed=9)
import math

from scipy.stats import gamma
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

# Load style file
plt.style.use("aux/PaperDoubleFig.mplstyle")
fontsize = plt.rcParams["font.size"]
def_linewidth = plt.rcParams["lines.linewidth"]
def_markersize = plt.rcParams["lines.markersize"]

from aux.star_formation import SFR_L
from aux.infrared_luminosity import T_CMB_obs, inv_CMB_heating, CMB_correction, Planck_func, calc_FIR_SED

from General.mpl.Legend.legend_handler import BTuple, BTupleHandler

# Import astropy cosmology, given H0 and Omega_matter
from astropy.cosmology import FLRW, FlatLambdaCDM



wl_CII = 157.73617000e-6 # m
nu_CII = 299792458.0 / wl_CII # Hz

# Dust mass absorption coefficient at frequency nu_star
# nu_star = 2998.0e9 # Hz
# k_nu_star = 52.2 # cm^2/g
wl_star = wl_CII
nu_star = nu_CII # Hz
k_nu_star = 8.94 # cm^2/g (ranges from ~5 to ~30; see Hirashita et al. 2014)
# k_nu = 8.94 # cm^2/g

# Solar luminosity in erg/s
L_sun_ergs = 3.828e26 * 1e7
# Solar mass in g
M_sun_g = 1.989e33

T_dusts_global = np.arange(20, 120, 10)
beta_IRs_global = np.arange(1.5, 2.05, 0.1)

dust_colors = sns.color_palette("inferno", len(T_dusts_global))
dust_cmap = sns.color_palette("inferno", as_cmap=True)
dust_norm = matplotlib.colors.Normalize(vmin=0, vmax=T_dusts_global[-1])

l0_linestyles = {None: '--', "self-consistent": '-', 100.0: '-.', 200.0: ':'}

obj_colors = {obj: sns.color_palette("Set1", 9)[obji] for obji, obj in enumerate(["COS-2987030247", "COS-3018555981", "UVISTA-Z-007", "UVISTA-Z-019"])}





def log_prob_Gauss(x, mu, cov):
    # Simple logarithmic probability function for Gaussian distribution
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

def mcmc_sampler(means, cov, n_dim=2, n_steps=10000, nwalkers=32):
    print("Running {:d}D MCMC sampler with {:d} walkers performing {:d} steps on function...".format(n_dim, nwalkers, n_steps))

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
    
    def rf2obs(self, l_emit):
        return l_emit / 1e3 * (1.0 + self.z)
    def obs2rf(self, l_obs):
        return l_obs * 1e3 / (1.0 + self.z)

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

    # Compute IR SED fluxes only for the given (rest-frame) wavelengths
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
    
    # Compute IR SED flux at the reference (rest-frame) frequency, given the dust mass (NB: uncorrected for CMB effects)
    if l0_dict.get("value", -1) is None:
        # Compute flux density in optically thin limit, assuming 1 - exp(-τ) ~ τ = κ_nu Σ_dust
        S_nu_emit_ref = 10**logM_dust * M_sun_g * k_nu * Planck_func(nu_ref, T_dust) / D_L**2 * 1e26
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
        
        S_nu_emit_ref = (1.0 - np.exp(-tau)) * Planck_func(nu_ref, T_dust) * l0_dict["cont_area_cm2"] / D_L**2 * 1e26
    
    # Compute IR SED flux specifically for the reference (rest-frame) frequency (NB: also uncorrected for CMB effects)
    S_nu_emit_CII = calc_FIR_SED(z=z, beta_IR=beta_IR, T_dust=T_dust,
                                        optically_thick_lambda_0=optically_thick_lambda_0,
                                        return_spectrum=True, lambda_emit=299792458.0 * 1e6 / nu_ref)[1]
    # Compute the normalisation of the spectrum by the ratio of the two
    # NB: both are uncorrected for CMB effects, but this correction cancels out
    norm = S_nu_emit_ref / S_nu_emit_CII
    
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
            beta_range = [1, 3]
        
            self.cube_range.append(beta_range)
        
        self.cube_range = np.array(self.cube_range)

    def Prior(self, cube):
        assert hasattr(self, "cube_range")
        # Scale the input unit cube to apply uniform priors across all parameters (except the temperature)
        for di in range(len(cube)):
            if di == 1:
                # Use a gamma distribution for the dust temperature (prior belief: unlikely to be near CMB or extremely high temperature)
                cube[di] = gamma.ppf(cube[di], 1.5, 0, self.cube_range[di, 0]/0.5) + self.cube_range[di, 0]
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
        ll = -0.5 * np.nansum(((self.fluxes[~self.uplims] - model_fluxes[~self.uplims])/self.flux_errs[~self.uplims])**2)

        if np.any(self.uplims) and self.fit_uplims:
            sigma_uplims = self.fluxes[self.uplims]/self.uplim_nsig
            ll += np.nansum( np.log( np.sqrt(np.pi) / 2.0 * (1.0 + erf((sigma_uplims - model_fluxes[self.uplims]) / (np.sqrt(2.0) * sigma_uplims))) ) )
        
        return ll

class FIR_SED_fit:
    def __init__(self, l0_list, analysis, mcrfol, fd_conv=None, l_min=None, l_max=None,
                    fixed_T_dust=50.0, fixed_beta=None, cosmo=None,
                    T_lolim=False, T_uplim=False, pformat=None, dpi=None, verbose=True):
        if verbose:
            print("\nInitialising FIR SED fitting object...")
        
        self.l0_list = l0_list
        self.analysis = analysis
        self.mcrfol = mcrfol

        if fd_conv is None:
            # Setting this unit converter to 1 won't change any of the units, so flux density spectra are in Jy;
            # setting it to 1e3 will change the units of flux density spectra to mJy, 1e6 to μJy, etc.
            self.fd_conv = 1e6

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
        
        if pformat is None:
            self.pformat = ".png" if self.analysis else ".pdf"
        else:
            self.pformat = pformat
        
        if dpi is None:
            self.dpi = 150
        else:
            self.dpi = dpi

        self.verbose = verbose
        if self.verbose:
            print("Initialisation done!")
    
    def set_fixed_values(self, fixed_T_dust, fixed_beta):
        self.fixed_T_dust = fixed_T_dust
        self.fixed_beta = fixed_beta
        self.beta_str = "beta_{:.1f}".format(self.fixed_beta) if self.fixed_beta else "vary_beta"

    def set_data(self, obj, z, obj_M, obj_M_lowerr, obj_M_uperr, SFR_UV, SFR_UV_err,
                    lambda_emit_vals, lambda_emit_ranges, S_nu_vals, S_nu_errs, cont_uplims, cont_excludes=None, uplim_nsig=3.0,
                    cont_area=None, cont_area_uplim=None, reference=None):
        if self.verbose:
            print("\nSetting photometric data points...")
        self.obj = obj
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
        self.lambda_emit_ranges = np.asarray(lambda_emit_ranges)[valid_fluxes]
        self.S_nu_vals = np.asarray(S_nu_vals)[valid_fluxes]
        self.S_nu_errs = np.asarray(S_nu_errs)[valid_fluxes]
        self.cont_uplims = np.asarray(cont_uplims)[valid_fluxes]
        self.all_uplims = np.all(self.cont_uplims)
        
        self.uplim_nsig = uplim_nsig
        
        if self.verbose:
            print('', "Wavelength (μm)\tFlux (μJy)\tError (μJy)\tUpper limit?\tExclude?",
                    *["{:.5g}\t\t{:.5g}\t\t{}\t\t{}\t\t{}".format(wl, f*1e6, 'N/A' if u else "{:.5g}".format(e*1e6), u, exc) \
                        for wl, f, e, u, exc in zip (self.lambda_emit_vals, self.S_nu_vals, self.S_nu_errs, self.cont_uplims, self.cont_excludes)], sep='\n')
        
        self.valid_cont_area = cont_area is not None and np.isfinite(cont_area)
        self.cont_area = cont_area
        self.cont_area_uplim = cont_area_uplim
        
        self.reference = reference

        # Combined detections for object
        self.n_meas = self.lambda_emit_vals.size
        self.cont_det = ~self.cont_uplims * ~self.cont_excludes
    
    def fit_data(self, pltfol, force_run=False, fit_uplims=True, return_samples=False,
                    n_live_points=400, evidence_tolerance=0.5, sampling_efficiency=0.8, max_iter=0, mnverbose=False):
        
        if self.all_uplims:
            print("Warning: only upper limits for {} specified! No MultiNest fit performed...".format(self.obj))
            return 1

        if not return_samples:
            # Set percentiles to standard ±1σ confidence intervals around the median value
            percentiles = [0.5*(100-68.2689), 50, 0.5*(100+68.2689)]
        
        for l0 in self.l0_list:
            # Without knowing the source's area, can only fit an optically thin SED or one with fixed lambda_0
            if l0 == "self-consistent" and not self.valid_cont_area:
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

            samples_fname = self.mcrfol + "{}_MN_FIR_SED_samples_{}{}.npz".format(self.obj, self.beta_str, l0_str)
            obtain_MN_samples = force_run or not os.path.isfile(samples_fname)
            
            omcrfol = self.mcrfol + "MultiNest_{}/".format(self.obj)
            if not os.path.exists(omcrfol):
                os.makedirs(omcrfol)

            if obtain_MN_samples:
                if self.verbose:
                    print("\nRunning {:d}-dimensional MultiNest fit".format(n_dim),
                            "with {}{}".format("β = {:.1f}".format(self.fixed_beta) if self.fixed_beta else "varying β", l0_txt),
                            "for {}...".format(self.obj))
                
                try:
                    os.chdir(omcrfol)
                    MN_solv = MN_FIR_SED_solver(z=self.z, D_L=self.D_L, l0_dict=l0_dict,
                                                fluxes=self.S_nu_vals[~self.cont_excludes], flux_errs=self.S_nu_errs[~self.cont_excludes],
                                                uplims=self.cont_uplims[~self.cont_excludes], wls=self.lambda_emit_vals[~self.cont_excludes],
                                                fixed_beta=self.fixed_beta, fit_uplims=fit_uplims, uplim_nsig=self.uplim_nsig,
                                                n_dims=n_dim, outputfiles_basename="MN", n_live_points=n_live_points,
                                                evidence_tolerance=evidence_tolerance, sampling_efficiency=sampling_efficiency, max_iter=max_iter,
                                                resume=False, verbose=mnverbose and self.verbose)
                except Exception as e:
                    raise RuntimeError("error occurred while running MultiNest fit...\n{}".format(e))
                
                # Note results are also saved as MNpost_equal_weights.dat; load with np.loadtxt(omcrfol + "MNpost_equal_weights.dat")[:, :n_dim]
                flat_samples = MN_solv.samples
                
                # Save results
                np.savez_compressed(samples_fname, flat_samples=flat_samples)
                if self.verbose:
                    print("\nFreshly calculated MultiNest samples with {}{}".format("β = {:.1f}".format(self.fixed_beta) if self.fixed_beta else "varying β", l0_txt),
                            "for {}! Array size: {:.2g} MB".format(self.obj, flat_samples.nbytes/1e6))
            else:
                # Read in samples from the MN run
                flat_samples = np.load(samples_fname)["flat_samples"]

                if self.verbose:
                    print("\nFreshly loaded MultiNest samples with {}{}".format("β = {:.1f}".format(self.fixed_beta) if self.fixed_beta else "varying β", l0_txt),
                            "for {}! Array size: {:.2g} MB".format(self.obj, flat_samples.nbytes/1e6))
            
            if return_samples:
                return flat_samples
            
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
                beta_samples = flat_samples[:, 2]
                beta_perc = np.percentile(beta_samples, percentiles, axis=0)
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
                M_star_samples = mcmc_sampler([self.obj_M], [[(0.5*(self.obj_M_lowerr+self.obj_M_uperr))**2]],
                                                n_dim=1, n_steps=2500, nwalkers=32)[:, 0]
                dust_frac_samples = 10**logM_dust_samples/rng.choice(M_star_samples, size=logM_dust_samples.size, replace=True)
                dust_frac_perc = np.percentile(dust_frac_samples, percentiles, axis=0)
                del M_star_samples, dust_frac_samples
            
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
            lambda_emit, nu_emit, rdict["S_nu_emit"], rdict["S_nu_obs"] = FIR_SED_spectrum(theta=(np.log10(rdict["M_dust"]), T_dust, beta_IR),
                                                                                            z=self.z, D_L=self.D_L, l0_dict=l0_dict)
            rdict["lambda_emit"] = lambda_emit
            rdict["nu_emit"] = nu_emit

            S_nu_emit_samples = np.array([FIR_SED_spectrum(theta=sample, z=self.z, D_L=self.D_L, l0_dict=l0_dict)[2] for sample in flat_samples])

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
            T_peak_samples = [2.897771955e3/lambda_emit[np.argmax(S_nu_emit)] for S_nu_emit in S_nu_emit_samples]
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
            
            S_nu_obs_samples = np.array([FIR_SED_spectrum(theta=sample, z=self.z, D_L=self.D_L, l0_dict=l0_dict)[3] for sample in flat_samples])
            del flat_samples
            
            rdict["y1"], rdict["y2"] = np.percentile(S_nu_obs_samples, [0.5*(100-68.2689), 0.5*(100+68.2689)], axis=0)
            
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
            names = ["logM_dust", "L_IR", "L_FIR", "T_dust", "T_peak"]
            data = [logM_dust_samples, np.log10(L_IR_Lsun_samples), np.log10(L_FIR_Lsun_samples), T_dust_samples, T_peak_samples]
            del logM_dust_samples, L_IR_Lsun_samples, L_FIR_Lsun_samples
            bins = [100, 100, 100, 100, 100]
            ranges = [0.95, 0.95, 0.95,
                        (0.8*T_CMB_obs*(1.0+self.z) if np.percentile(T_dust_samples, 99) < 40 else 0, np.percentile(T_dust_samples, 99)),
                        (0.8*T_CMB_obs*(1.0+self.z) if np.percentile(T_peak_samples, 99) < 40 else 0, np.percentile(T_peak_samples, 99))]
            del T_dust_samples, T_peak_samples
            labels = [r"$\log_{10} \left( M_\mathrm{dust} \, (\mathrm{M_\odot}) \right)$", r"$\log_{10} \left( L_\mathrm{IR} \, (\mathrm{L_\odot}) \right)$",
                        r"$\log_{10} \left( L_\mathrm{FIR} \, (\mathrm{L_\odot}) \right)$",
                        r"$T_\mathrm{dust} \, (\mathrm{K})$", r"$T_\mathrm{peak} \, (\mathrm{K})$"]
            
            if l0 == "self-consistent":
                extra_dim += 1
                names.insert(1, "lambda_0")
                data.insert(1, lambda_0_samples)
                del lambda_0_samples
                bins.insert(1, 100)
                ranges.insert(1, 0.9)
                labels.insert(1, r"$\lambda_0$")
            if not self.fixed_beta:
                names.append("beta")
                data.append(beta_samples)
                del beta_samples
                bins.append(100)
                ranges.append(0.9)
                labels.append(r"$\beta_\mathrm{IR}$")
            
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
                text += '\n' + r"$M_* = {:.1f}_{{-{:.1f}}}^{{+{:.1f}}} \cdot 10^{{{:d}}} \, \mathrm{{M_\odot}}$".format(self.obj_M/10**int(np.log10(self.obj_M)),
                            self.obj_M_lowerr/10**int(np.log10(self.obj_M)), self.obj_M_uperr/10**int(np.log10(self.obj_M)), int(np.log10(self.obj_M)))
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
            del data
            
            ax_c = axes_c[names.index("T_peak"), names.index("T_dust")]
            ax_c.plot(np.linspace(-10, 200, 10), np.linspace(-10, 200, 10), linestyle='--', color="lightgrey", alpha=0.6)
            
            ax_c = axes_c[names.index("T_dust"), names.index("T_dust")]
            ax_c.axvline(T_CMB_obs*(1.0+self.z), linestyle='--', color='k', alpha=0.6)
            ax_c.annotate(text=r"$T_\mathrm{{ CMB }} (z = {:.6g})$".format(self.z), xy=(T_CMB_obs*(1.0+self.z), 0.5), xytext=(-2, 0),
                            xycoords=ax_c.get_xaxis_transform(), textcoords="offset points", rotation="vertical",
                            size="xx-small", va="center", ha="right").set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
            if self.T_lolim or self.T_uplim:
                ax_c.axvline(T_lim, color="grey", alpha=0.6)
                ax_c.annotate(text=("Lower" if self.T_lolim else "Upper") + " limit (95% conf.)", xy=(T_lim, 1), xytext=(2, -4),
                                xycoords=ax_c.get_xaxis_transform(), textcoords="offset points", rotation="vertical",
                                size="xx-small", va="top", ha="left").set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
            
            ax_c = axes_c[names.index("T_peak"), names.index("T_peak")]
            ax_c.axvline(T_CMB_obs*(1.0+self.z), linestyle='--', color='k', alpha=0.6)
            ax_c.annotate(text=r"$T_\mathrm{{ CMB }} (z = {:.6g})$".format(self.z), xy=(T_CMB_obs*(1.0+self.z), 0.5), xytext=(-2, 0),
                            xycoords=ax_c.get_xaxis_transform(), textcoords="offset points", rotation="vertical",
                            size="xx-small", va="center", ha="right").set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
            if self.T_lolim or self.T_uplim:
                ax_c.axvline(rdict["T_peak"], color="grey", alpha=0.6)
                ax_c.annotate(text=("Lower" if self.T_lolim else "Upper") + " limit (95% conf.)", xy=(rdict["T_peak"], 1), xytext=(2, -4),
                                xycoords=ax_c.get_xaxis_transform(), textcoords="offset points", rotation="vertical",
                                size="xx-small", va="top", ha="left").set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
            
            self.annotate_results(rdict, [axes_c[0, -1], axes_c[1, -1]], ax_type="corner")
            cfig.savefig(pltfol + "Corner_MN_" + self.obj + self.get_mstring(l0_list=[l0], inc_astr=False) + self.pformat,
                            dpi=self.dpi, bbox_inches="tight")
    
            # plt.show()
            plt.close(cfig)
            
            # Save results
            np.savez_compressed(self.mcrfol + "{}_MN_FIR_SED_fit_{}{}.npz".format(self.obj, self.beta_str, l0_str), **rdict)
            self.fresh_calculation[l0] = True
            if self.verbose:
                self.print_results(rdict, rtype="calculated")
    
    def print_results(self, rdict, rtype):
        M_dust_log10 = int(np.log10(rdict["M_dust"]))
        L_IR_log10 = int(np.log10(rdict["L_IR_Lsun"]))
        L_FIR_log10 = int(np.log10(rdict["L_FIR_Lsun"]))

        print("\nFreshly {} MultiNest estimates of {}:".format(rtype, self.obj))
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
            print("beta_IR = {:.1f} (fixed)".format(rdict["beta_IR"]))
        else:
            print("beta_IR = {:.1f} -{:.1f} +{:.1f}".format(rdict["beta_IR"], rdict["beta_IR_lowerr"], rdict["beta_IR_uperr"]))
        print('')
        
        if not np.isnan(self.obj_M):
            print("Dust-to-stellar mass fraction: {:.1f} -{:.1f} +{:.1f} per cent".format(100.0*rdict["dust_frac"],
                                                100.0*rdict["dust_frac_lowerr"], 100.0*rdict["dust_frac_uperr"]))
            print("Dust yield (AGB): {:.1f} -{:.1f} +{:.1f} M_sun".format(rdict["dust_yield_AGB"],
                                                rdict["dust_yield_AGB_lowerr"], rdict["dust_yield_AGB_uperr"]))
            print("Dust yield (SN): {:.1f} -{:.1f} +{:.1f} M_sun".format(rdict["dust_yield_SN"],
                                                rdict["dust_yield_SN_lowerr"], rdict["dust_yield_SN_uperr"]))
            print('')
    
    def plot_MN_fit(self, l0_list=None, fig=None, ax=None, pltfol=None, obj_str=None, single_plot=None,
                    set_xrange=True, set_xlabel="both", set_ylabel=True, extra_yspace=True, rowi=0, coli=0):
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

        for l0 in l0_list:
            if create_fig:
                # Prepare figure for plotting FIR SED
                fig, ax = plt.subplots()
            elif fig is None:
                fig = ax.get_figure()
            elif ax is None:
                ax = fig.add_subplot()

            self.fig, self.ax = fig, ax

            self.annotate_title()
            
            l0_str, l0_txt = self.get_l0string(l0)

            rdict_fname = self.mcrfol + "{}_MN_FIR_SED_fit_{}{}.npz".format(self.obj, self.beta_str, l0_str)
            if os.path.isfile(rdict_fname):
                rdict = np.load(rdict_fname)
            else:
                if self.verbose:
                    print("Warning: MultiNest results with {}{}".format("β = {:.1f}".format(self.fixed_beta) if self.fixed_beta else "varying β", l0_txt),
                            "for {} not found! Filename:\n{}\nContinuing...".format(self.obj, rdict_fname.split('/')[-1]))
                continue
            
            M_dust_log10 = int(np.log10(rdict["M_dust"]))
            T_dust = rdict["T_dust"]
            dcolor = dust_cmap(dust_norm(T_dust))
            
            lambda_emit = rdict["lambda_emit"]
            S_nu_obs = rdict["S_nu_obs"]

            y1 = rdict["y1"]
            y2 = rdict["y2"]
            
            # Plot the observed spectrum (intrinsic spectrum is nearly the same apart from at the very red wavelengths above ~100 micron)
            ax.plot(lambda_emit, S_nu_obs*self.fd_conv, linewidth=1.5, linestyle=l0_linestyles.get(l0, '-'), color=dcolor, alpha=0.8)
            
            ax.fill_between(lambda_emit, y1=y1*self.fd_conv, y2=y2*self.fd_conv, facecolor=dcolor, edgecolor="None", alpha=0.1)
            
            if np.min(y1[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]) * self.fd_conv < self.F_nu_obs_min:
                self.F_nu_obs_min = np.nanmin(y1[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]) * self.fd_conv
            if np.max(y2[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]) * self.fd_conv > self.F_nu_obs_max:
                self.F_nu_obs_max = np.nanmax(y2[(lambda_emit >= max(20, self.l_min)) * (lambda_emit <= min(1e3, self.l_max))]) * self.fd_conv
            
            M_dust_log10 = int(np.log10(rdict["M_dust"]))
            L_IR_log10 = int(np.log10(rdict["L_IR_Lsun"]))
            
            if not self.fresh_calculation[l0]:
                if self.verbose:
                    self.print_results(rdict, rtype="loaded")

            if self.analysis and not single_plot:
                self.annotate_results(rdict, [ax, ax], ax_type="regular")
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
                    handles.append(BTuple(([dcolor], [0.1], 1.5, l0_linestyles[l0], dcolor, 0.8), label + '\n'))

            if not single_plot:
                self.plot_data()
                self.set_axes(set_xrange=set_xrange, set_xlabel=set_xlabel, set_ylabel=set_ylabel, extra_yspace=extra_yspace, rowi=rowi, coli=coli)
                if pltfol:
                    self.save_fig(pltfol=pltfol, ptype="MN_fit", l0_list=[l0])
        
        if single_plot:
            leg = ax.legend(handles=handles, handler_map={BTuple: BTupleHandler()}, ncol=len(self.l0_list), loc="lower center",
                            handlelength=0.7*plt.rcParams["legend.handlelength"], handleheight=5*plt.rcParams["legend.handleheight"],
                            columnspacing=0.3*plt.rcParams["legend.columnspacing"])
            
            # Show which beta_IRs have been used
            leg.set_title("MN fits of dust emission" + r", fixed $\beta_\mathrm{{ IR }} = {:.2g}$".format(self.fixed_beta) if self.fixed_beta else '',
                            prop={"size": "small"})
            
            self.plot_data()
            self.set_axes(set_xrange=set_xrange, set_xlabel=set_xlabel, set_ylabel=set_ylabel, extra_yspace=extra_yspace, rowi=rowi, coli=coli)
            if pltfol:
                self.save_fig(pltfol=pltfol, ptype="MN_fit", obj_str=obj_str)
    
    def plot_ranges(self, l0, T_dusts=T_dusts_global, beta_IRs=beta_IRs_global, fixed_T_dust=None, fixed_beta=None, save_results=True,
                    fig=None, ax=None, pltfol=None, obj_str=None,
                    annotate_results=True, set_xrange=True, set_xlabel="both", set_ylabel=True, extra_yspace=True, rowi=0, coli=0):
        if l0 == "self-consistent":
            print("Warning: ranges cannot be shown for a self-consistent opacity model! Continuing...")
            return 1
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
            print("\nPlotting ranges on FIR SED of {}:".format(self.obj))
        
        T_dust_handles = []

        for di, T_dust in enumerate(T_dusts):
            if T_dust < T_CMB_obs*(1.0+self.z) or (T_dust > 100 and self.obj in ["COS-2987030247", "UVISTA-Z-007"]):
                continue

            lambda_emit_betas = []
            S_nu_obs_betas = []
            L_IR_Lsun_betas = []
            L_FIR_Lsun_betas = []
            T_peak_betas = []

            compatible_betas = []

            for bi, beta_IR in enumerate(beta_IRs):
                lambda_emit, S_nu_emit = calc_FIR_SED(z=self.z, beta_IR=beta_IR, T_dust=T_dust,
                                                                optically_thick_lambda_0=l0, return_spectrum=True)
                nu_emit = 299792458.0 * 1e6 / lambda_emit # Hz (lambda is in micron)
                
                # Flux density needs to be corrected for observing against the the CMB (NB: can be negative if T_dust < T_CMB), and then normalised
                CMB_correction_factor = CMB_correction(z=self.z, nu0_emit=nu_emit, T_dust=T_dust)
                if np.all(CMB_correction_factor < 0):
                    # No normalisation possible (T_dust < T_CMB) and so any upper limit is compatible, continue
                    continue
                else:
                    if self.analysis:
                        # Show where the correction is 90%
                        ax.axvline(x=lambda_emit[np.argmin(np.abs(CMB_correction_factor - 0.9))], linestyle='--', color=dust_colors[di], alpha=0.8)
                        if di == 2 and bi == 0:
                            ax.annotate(text="10% CMB background", xy=(lambda_emit[np.argmin(np.abs(CMB_correction_factor - 0.9))], 1), xytext=(-4, -4),
                                        xycoords=ax.get_xaxis_transform(), textcoords="offset points", rotation="vertical",
                                        va="top", ha="right", size="x-small", color=dust_colors[di], alpha=0.8)

                S_nu_emit_CMB_att = S_nu_emit * CMB_correction_factor # Jy
                
                # Wien's displacement law to find the observed (after correction for CMB attenuation) peak temperature
                T_peak_betas.append(2.897771955e3/lambda_emit[np.argmax(S_nu_emit)])

                def normalise(wl, flux, norm=1):
                    return (flux*self.fd_conv) / np.interp(wl, lambda_emit, S_nu_emit_CMB_att * norm)
                
                uplim = np.all(self.cont_uplims)
                
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
                    normalisation_lowerr = normalise(self.lambda_emit_vals[cont_idx], self.S_nu_vals[cont_idx]-self.S_nu_errs[cont_idx])
                    normalisation_uperr = normalise(self.lambda_emit_vals[cont_idx], self.S_nu_vals[cont_idx]+self.S_nu_errs[cont_idx])

                    # With only upper limits, all betas are always compatible
                    compatible_beta = True
                else:
                    if np.sum(self.cont_det) == 1:
                        # Only one detection, use as normalisation
                        cont_idx = list(self.cont_uplims).index(False)
                    else:
                        # Choose highest S/N detection for normalisation
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

                if np.abs(T_dust - fixed_T_dust) < 1e-8 and np.abs(beta_IR - fixed_beta) < 1e-8:
                    fixed_vals_cb = compatible_beta
                    if not compatible_beta:
                        if self.verbose:
                            print("Warning: T_dust = {:.0f} K, β = {:.2g} are incompatible with measurements...".format(fixed_T_dust,
                                                                                                                        fixed_beta, self.uplim_nsig))

                    # Estimate of the dust mass (S_nu in Jy, 1e-26 = W/m^2/Hz/Jy, self.D_L in cm, κ_nu in cm^2/g, Planck_func in W/m^2/Hz, M_sun = 1.989e33 g)
                    S_nu_emit_ref = np.interp(wl_star/1e4, lambda_emit, S_nu_emit)
                    rdict["M_dust_uplim"] = uplim
                    rdict["M_dust"] = S_nu_emit_ref * self.D_L**2 / (M_sun_g * 8.94 * Planck_func(nu_star, T_dust) * 1e26)
                    if uplim:
                        rdict["M_dust_lowerr"], rdict["M_dust_uperr"] = np.nan, np.nan
                    else:
                        S_nu_emit_ref_errs = np.array([S_nu_emit_ref, S_nu_emit_ref]) / np.nanmean(SNR_ratios)
                        rdict["M_dust_lowerr"], rdict["M_dust_uperr"] = S_nu_emit_ref_errs * self.D_L**2 / (M_sun_g * 8.94 * Planck_func(nu_star, T_dust) * 1e26)
            
                    # Add an (asymmetric) systematic uncertainty, since κ_ν actually ranges from 28.4 (28.4/8.94 ~ 0.5 dex)
                    # to 5.57 (8.94/5.57 ~ 0.2 dex) in cm^2/g (note a higher κ_ν means a lower dust mass)
                    # rdict["M_dust_lowerr"] = rdict["M_dust"] - (rdict["M_dust"] - rdict["M_dust_lowerr"]) * 8.94/28.4
                    # rdict["M_dust_uperr"] = (rdict["M_dust"] + rdict["M_dust_uperr"]) * 8.94/5.57 - rdict["M_dust"]

                    rdict["dust_frac"] = rdict["M_dust"] / self.obj_M
                    rdict["dust_frac_lowerr"] = rdict["dust_frac"] * np.sqrt((rdict["M_dust_lowerr"]/rdict["M_dust"])**2 + (self.obj_M_uperr/self.obj_M)**2)
                    rdict["dust_frac_uperr"] = rdict["dust_frac"] * np.sqrt((rdict["M_dust_uperr"]/rdict["M_dust"])**2 + (self.obj_M_lowerr/self.obj_M)**2)

                    rdict["dust_yield_AGB"], rdict["dust_yield_AGB_lowerr"], rdict["dust_yield_AGB_uperr"] = 29 * np.array([rdict["dust_frac"],
                                                                                        rdict["dust_frac_lowerr"], rdict["dust_frac_uperr"]])
                    rdict["dust_yield_SN"], rdict["dust_yield_SN_lowerr"], rdict["dust_yield_SN_uperr"] = 84 * np.array([rdict["dust_frac"],
                                                                                        rdict["dust_frac_lowerr"], rdict["dust_frac_uperr"]])

                    rdict["L_IR_uplim"] = uplim
                    rdict["L_IR_Lsun"], rdict["L_IR_Lsun_lowerr"], rdict["L_IR_Lsun_uperr"] = L_IR_Lsun, L_IR_Lsun_lowerr, L_IR_Lsun_uperr
                    L_IR_log10 = int(np.log10(rdict["L_IR_Lsun"]))
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
                        prop_ann.set_text(prop_ann.get_text() + "\n\nFor " + r"$T_\mathrm{{ dust }} = {:.0f} \, \mathrm{{ K }}$, ".format(fixed_T_dust) + \
                                            r"$\beta_\mathrm{{ IR }} = {:.2g}$:".format(fixed_beta) + '\n' + \
                                            '\n'.join(IR_ann_list))
                    else:
                        for key in rdict.keys():
                            if not "uplim" in key:
                                rdict[key] = np.nan

                T_dust_handles.append(T_dust)
                lambda_emit_betas.append(lambda_emit)
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
                # ax.plot(lambda_emit, S_nu_emit, linestyle='--', color=dust_colors[di], alpha=0.8)
                ax.plot(lambda_emit_betas[low_beta_idx], S_nu_obs_betas[low_beta_idx], linewidth=1.5, color=dust_colors[di], alpha=0.8)
                ax.plot(lambda_emit_betas[high_beta_idx], S_nu_obs_betas[high_beta_idx], linewidth=1.5, color=dust_colors[di], alpha=0.8)

                # Also fill in area in between curves of minimum/maximum beta_IR that is compatible
                assert np.all([l == lambda_emit_betas[0] for l in lambda_emit_betas])
                ax.fill_between(lambda_emit_betas[0], y1=S_nu_obs_betas[low_beta_idx], y2=S_nu_obs_betas[high_beta_idx],
                                facecolor=dust_colors[di], edgecolor="None", alpha=0.2)
            
            if not any_compatible_betas:
                if lambda_emit_betas and S_nu_obs_betas:
                    ax.plot(lambda_emit_betas[0], S_nu_obs_betas[0], linestyle='--', linewidth=1.0, color=dust_colors[di], alpha=0.5)
                    ax.plot(lambda_emit_betas[-1], S_nu_obs_betas[-1], linestyle='--', linewidth=1.0, color=dust_colors[di], alpha=0.5)
                    ax.fill_between(lambda_emit_betas[0], y1=S_nu_obs_betas[0], y2=S_nu_obs_betas[-1],
                                    facecolor=dust_colors[di], edgecolor="None", alpha=0.05)
            elif not all_compatible_betas:
                if not compatible_betas[0]:
                    ax.plot(lambda_emit_betas[0], S_nu_obs_betas[0], linestyle='--', linewidth=1.0, color=dust_colors[di], alpha=0.5)
                    ax.fill_between(lambda_emit_betas[0], y1=S_nu_obs_betas[0], y2=S_nu_obs_betas[low_beta_idx],
                                    facecolor=dust_colors[di], edgecolor="None", alpha=0.05)
                if not compatible_betas[-1]:
                    ax.plot(lambda_emit_betas[-1], S_nu_obs_betas[-1], linestyle='--', linewidth=1.0, color=dust_colors[di], alpha=0.5)
                    ax.fill_between(lambda_emit_betas[0], y1=S_nu_obs_betas[high_beta_idx], y2=S_nu_obs_betas[-1],
                                    facecolor=dust_colors[di], edgecolor="None", alpha=0.05)

            if any_compatible_betas and annotate_results:
                # Annotate results of L_IR
                L_IR_log10 = int(np.log10(np.min([L_IR_Lsun_betas[0], L_IR_Lsun_betas[-1]])))
                L_FIR_log10 = int(np.log10(np.min([L_FIR_Lsun_betas[0], L_FIR_Lsun_betas[-1]])))
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
                cb_ann_text_offset_colours.append([text, 0, dust_colors[di]])

        for cb_ann_text_offset_colour in cb_ann_text_offset_colours:
            ax.annotate(text=cb_ann_text_offset_colour[0], xy=(0.025, 0.025), xytext=(0, 16*cb_ann_text_offset_colour[1]),
                        xycoords="axes fraction", textcoords="offset points", va="bottom", ha="left", color='k',
                        size="x-small").set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor=cb_ann_text_offset_colour[2], alpha=0.8))
        
        rdict["T_peak"] = np.nan
        rdict["T_peak_err"] = np.nan
        rdict["T_peak_constraint"] = "none"
        
        if save_results:
            # Save results
            np.savez_compressed(self.mcrfol + "{}_FIR_SED_parameters_{}{}{}.npz".format(self.obj, "T_{:.0f}".format(fixed_T_dust),
                                "_beta_{:.1f}".format(fixed_beta), "_l0_{:.0f}".format(l0) if l0 else ''), **rdict)

        handles = [matplotlib.patches.Rectangle(xy=(np.nan, np.nan), width=1, height=1, edgecolor="None", facecolor=dust_colors[di], alpha=0.8,
                    label=r"$T_\mathrm{{ dust }} = {:.0f} \, \mathrm{{ K }}$".format(T_dust)) for di, T_dust in enumerate(T_dusts) if T_dust in T_dust_handles]
        
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
                    M_dust_log10 = int(np.log10(rdict["M_dust"]))
                    L_IR_log10 = int(np.log10(rdict["L_IR_Lsun"]))
                    L_FIR_log10 = int(np.log10(rdict["L_FIR_Lsun"]))

                    valstr = lambda *vals: "< {}{:.1f}{}".format(vals[0], vals[1], vals[-1]) if uplim else "= {}{:.1f} -{:.1f} +{:.1f}{}".format(*vals)
                    print("\nM_dust {} x 10^{:d} M_sun".format(valstr('(', rdict["M_dust"]/10**M_dust_log10,
                                rdict["M_dust_lowerr"]/10**M_dust_log10, rdict["M_dust_uperr"]/10**M_dust_log10, ')'), M_dust_log10))
                    print("L_IR {} x 10^{:d} L_sun".format(valstr('(', rdict["L_IR_Lsun"]/10**L_IR_log10,
                                rdict["L_IR_Lsun_lowerr"]/10**L_IR_log10, rdict["L_IR_Lsun_uperr"]/10**L_IR_log10, ')'), L_IR_log10))
                    print("L_FIR {} x 10^{:d} L_sun".format(valstr('(', rdict["L_FIR_Lsun"]/10**L_FIR_log10,
                                rdict["L_FIR_Lsun_lowerr"]/10**L_FIR_log10, rdict["L_FIR_Lsun_uperr"]/10**L_FIR_log10, ')'), L_FIR_log10))
                    print('')
                    print("Dust-to-stellar mass fraction: {} per cent".format(valstr('', 100.0*rdict["dust_frac"],
                                                        100.0*rdict["dust_frac_lowerr"], 100.0*rdict["dust_frac_uperr"], '')))
                    print("Dust yield (AGB): {} M_sun".format(valstr('', rdict["dust_yield_AGB"],
                                                        rdict["dust_yield_AGB_lowerr"], rdict["dust_yield_AGB_uperr"], '')))
                    print("Dust yield (SN): {} M_sun".format(valstr('', rdict["dust_yield_SN"],
                                                        rdict["dust_yield_SN_lowerr"], rdict["dust_yield_SN_uperr"], '')))
                else:
                    print(" incompatible with measurements!")
    
        self.plot_data()
        self.set_axes(set_xrange=set_xrange, set_xlabel=set_xlabel, set_ylabel=set_ylabel, extra_yspace=extra_yspace, rowi=rowi, coli=coli)
        if pltfol:
            self.save_fig(pltfol=pltfol, ptype="ranges", obj_str=obj_str, l0_list=[l0])
    
    def annotate_title(self):
        ax = self.ax

        text = self.obj
        if self.reference:
            text += " ({})".format(self.reference.replace('(', '').replace(')', ''))
            size = "medium"
            bbox_col = 'w'
        else:
            size = "large"
            bbox_col = obj_colors[self.obj]
        
        ax.annotate(text=text, xy=(0, 1), xytext=(8, -8), xycoords="axes fraction", textcoords="offset points",
                    color='k', size=size, va="top", ha="left").set_bbox(dict(boxstyle="Round, pad=0.05", facecolor=bbox_col, edgecolor="None", alpha=0.8))

        text = r"$z = {:.6g}$, $T_\mathrm{{ CMB }} = {:.2f} \, \mathrm{{ K }}$".format(self.z, T_CMB_obs*(1.0+self.z))
        if not np.isnan(self.obj_M):
            text += '\n' + r"$M_* = {:.1f}_{{-{:.1f}}}^{{+{:.1f}}} \cdot 10^{{{:d}}} \, \mathrm{{M_\odot}}$".format(self.obj_M/10**int(np.log10(self.obj_M)),
                        self.obj_M_lowerr/10**int(np.log10(self.obj_M)), self.obj_M_uperr/10**int(np.log10(self.obj_M)), int(np.log10(self.obj_M)))
        if not np.isnan(self.SFR_UV):
            text += '\n' + r"$\mathrm{{ SFR_{{UV}} }} = {:.0f}{}".format(self.SFR_UV, r'' if np.isnan(self.SFR_UV_err) else r" \pm {:.0f}".format(self.SFR_UV_err)) + \
                    r" \, \mathrm{{M_\odot yr^{{-1}}}}$"
        prop_ann = ax.annotate(text=text, xy=(1, 1), xytext=(-8, -8), xycoords="axes fraction", textcoords="offset points",
                                va="top", ha="right", color='k', size="small")
        prop_ann.set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))

        return prop_ann
    
    def annotate_results(self, rdict, axes_ann, ax_type):
        M_dust_log10 = int(np.log10(rdict["M_dust"]))
        L_IR_log10 = int(np.log10(rdict["L_IR_Lsun"]))
        L_FIR_log10 = int(np.log10(rdict["L_FIR_Lsun"]))
        
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
        text = r"$L_\mathrm{{ IR }} = {:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }} \cdot 10^{{ {:d} }} \, \mathrm{{ L_\odot }}$".format(rdict["L_IR_Lsun"]/10**L_IR_log10,
                    rdict["L_IR_Lsun_lowerr"]/10**L_IR_log10, rdict["L_IR_Lsun_uperr"]/10**L_IR_log10, L_IR_log10) + \
                '\n' + r"$L_\mathrm{{ FIR }} = {:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }} \cdot 10^{{ {:d} }} \, \mathrm{{ L_\odot }}$".format(rdict["L_FIR_Lsun"]/10**L_FIR_log10,
                    rdict["L_FIR_Lsun_lowerr"]/10**L_FIR_log10, rdict["L_FIR_Lsun_uperr"]/10**L_FIR_log10, L_FIR_log10) + \
                '\n' + r"$\mathrm{{ SFR_{{IR}} }} = {:.0f}_{{ -{:.0f} }}^{{ +{:.0f} }} \, \mathrm{{ M_\odot \, yr^{{-1}} }}$".format(rdict["SFR_IR"],
                    *rdict["SFR_IR_err"]) + \
                "\n\n" + r"$M_\mathrm{{ dust }} = {:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }} \cdot 10^{{ {:d} }} \, \mathrm{{ M_\odot }}$".format(rdict["M_dust"]/10**M_dust_log10,
                    rdict["M_dust_lowerr"]/10**M_dust_log10, rdict["M_dust_uperr"]/10**M_dust_log10, M_dust_log10)
        if self.valid_cont_area:
            text += '\n' + r"$\Sigma_\mathrm{{ dust }} {} {:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }} \, \mathrm{{ M_\odot \, pc^{{-2}} }}$".format(Sigma_sign,
                    rdict["Sigma_dust"], rdict["Sigma_dust_lowerr"], rdict["Sigma_dust_uperr"]) + \
                '\n' + r"$\lambda_0 {} {:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }} \, \mathrm{{ \mu m }}$".format(lambda_0_sign, rdict["lambda_0"],
                    rdict["lambda_0_lowerr"], rdict["lambda_0_uperr"])
        
        if not np.isnan(rdict["dust_frac"]):
            text += '\n' + r"$M_\mathrm{{ dust }} / M_* = {:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }} \%$".format(100.0*rdict["dust_frac"],
                                100.0*rdict["dust_frac_lowerr"], 100.0*rdict["dust_frac_uperr"]) + \
                    '\n' + r"Dust yield (AGB, SN): ${:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }} \, \mathrm{{ M_\odot }}$, ".format(rdict["dust_yield_AGB"],
                                rdict["dust_yield_AGB_lowerr"], rdict["dust_yield_AGB_uperr"]) + \
                            r"${:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }} \, \mathrm{{ M_\odot }}$".format(rdict["dust_yield_SN"],
                                rdict["dust_yield_SN_lowerr"], rdict["dust_yield_SN_uperr"])
        
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
                                    va=va, ha=ha, color='k', size="small", alpha=0.8, zorder=6)
        ann.set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
        
        if self.fixed_beta:
            beta_str = r"$\beta_\mathrm{{ IR }} = {:.1f}$ (fixed)".format(rdict["beta_IR"])
        else:
            beta_str = r"$\beta_\mathrm{{ IR }} = {:.1f}_{{ -{:.1f} }}^{{ +{:.1f} }}$".format(rdict["beta_IR"],
                        rdict["beta_IR_lowerr"], rdict["beta_IR_uperr"])
        
        text = '\n' + r"$T_\mathrm{{ dust }} = {:.0f}_{{ -{:.0f} }}^{{ +{:.0f} }} \, \mathrm{{ K }}".format(rdict["T_dust"],
                    rdict["T_dust_lowerr"], rdict["T_dust_uperr"]) + T_lim_str + \
                '\n' + r"$T_\mathrm{{ dust }}^{{ z=0 }} = {:.0f}_{{ -{:.0f} }}^{{ +{:.0f} }} \, \mathrm{{ K }}".format(rdict["T_dust_z0"],
                            rdict["T_dust_z0_lowerr"], rdict["T_dust_z0_uperr"]) + T_z0_lim_str + \
                '\n' + r"$T_\mathrm{{ peak }} = {:.0f}_{{ -{:.0f} }}^{{ +{:.0f} }} \, \mathrm{{ K }}".format(rdict["T_peak_val"],
                            rdict["T_peak_lowerr"], rdict["T_peak_uperr"]) + \
                                (r" {} {:.0f} \, \mathrm{{ K }}$".format(r'>' if self.T_lolim else r'<', rdict["T_peak"]) if self.T_lolim or self.T_uplim else r'$') + \
                '\n' + beta_str
        
        if ax_type == "regular":
            ann = axes_ann[1].annotate(text=text, xy=(1, 0), xytext=(-8, 8), xycoords="axes fraction", textcoords="offset points", color='k', size="small",
                                        va="bottom", ha="right", alpha=0.8, zorder=6)
            ann.set_bbox(dict(boxstyle="Round, pad=0.05", facecolor='w', edgecolor="None", alpha=0.8))
        elif ax_type == "corner":
            ann.set_text(ann.get_text() + '\n' + text)

    def plot_data(self):
        for l, lrange, s_nu, s_nuerr, uplim, exclude in zip(self.lambda_emit_vals, self.lambda_emit_ranges, self.S_nu_vals, self.S_nu_errs,
                                                            self.cont_uplims, self.cont_excludes):
            self.ax.errorbar(l, s_nu*self.fd_conv, xerr=lrange.reshape(2, 1),
                                yerr=(s_nu-s_nu/self.uplim_nsig)*self.fd_conv if uplim else s_nuerr*self.fd_conv, uplims=uplim,
                                marker='o', markersize=def_markersize, linestyle="None", color='k', alpha=0.4 if exclude else 0.8, zorder=5)
            if uplim:
                self.ax.errorbar(l, s_nu/self.uplim_nsig*self.fd_conv,
                                    marker='_', markersize=def_markersize, linestyle="None", color='k', alpha=0.4 if exclude else 0.8, zorder=5)
    
    def set_axes(self, set_xrange, set_xlabel, set_ylabel, extra_yspace, rowi, coli):
        lfunc = lcoord_funcs(rowi=rowi, coli=coli, z=self.z)
        l_obs_ax = self.ax.secondary_xaxis("top", functions=(lfunc.rf2obs, lfunc.obs2rf))
        l_obs_ax.tick_params(axis='x', which="both", bottom=False, top=True, labelbottom=False, labeltop=True)
        self.ax.tick_params(axis="both", which="both", top=False, labelleft=True, labelbottom=True)

        self.ax.set_xscale("log")
        self.ax.set_yscale("log")

        if set_xrange:
            self.ax.set_xlim(self.l_min, self.l_max)
        
        if np.isfinite(self.F_nu_obs_min) and np.isfinite(self.F_nu_obs_max):
            self.ax.set_ylim((0.05 if extra_yspace else 0.1)*self.F_nu_obs_min, 4*self.F_nu_obs_max)
        
        if set_xlabel == "top" or set_xlabel == "both":
            l_obs_ax.set_xlabel(r"$\lambda_\mathrm{{ obs }} \, (\mathrm{mm})$")
        if set_xlabel == "bottom" or set_xlabel == "both":
            self.ax.set_xlabel(r"$\lambda_\mathrm{{ emit }} \, (\mathrm{\mu m})$")
        if set_ylabel:
            self.ax.set_ylabel(r"$F_\mathrm{\nu, \, obs} \, (\mathrm{\mu Jy})$")
    
    def save_fig(self, pltfol, fig=None, ptype="constraints", obj_str=None, l0_list=None):
        if fig is None:
            fig = self.fig
        if obj_str is None:
            obj_str = '_' + self.obj
        
        fig.savefig(pltfol + "FIR_SED_{}{}".format(ptype, obj_str) + self.get_mstring(l0_list=l0_list) + self.pformat,
                    dpi=self.dpi, bbox_inches="tight")
        
        # plt.show()
        plt.close(fig)
    
    def get_l0string(self, l0):
        if l0 == "self-consistent":
            l0_str = "_{}-l0".format(l0)
            l0_txt = " and {} λ_0".format(l0)
        else:
            l0_str = "_l0_{:.0f}".format(l0) if l0 else ''
            l0_txt = " and λ_0 = {:.0f} μm".format(l0) if l0 else " under a fully optically thin SED"
        
        return (l0_str, l0_txt)
    
    def get_mstring(self, l0_list=None, analysis=None, inc_astr=True):
        if l0_list is None:
            l0_list = self.l0_list
        if analysis is None:
            analysis = self.analysis
        
        if self.analysis:
            lstr = '_' + '_'.join(sorted(set([("scl0" if l0 == "self-consistent" else "l0_{:.0f}".format(l0)) for l0 in l0_list if l0]))) if any(l0_list) else ''
        else:
            lstr = ''
        
        return lstr + ("_beta_{:.1f}".format(self.fixed_beta) if self.fixed_beta else '') + ("_analysis" if analysis and inc_astr else '')

