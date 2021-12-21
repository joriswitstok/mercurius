#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for IR luminosities from continuum detections (with e.g. ALMA).

Joris Witstok, 11 February 2021
"""

import os, sys
import numpy as np

# Import astropy cosmology module
from astropy.cosmology import FLRW, FlatLambdaCDM

# Physical constants
c = 299792458.0 * 1e6 # micron/s
h_Planck = 6.62607015e-34 # J/Hz
k_B = 1.380649e-23 # J/K
h_kB = h_Planck / k_B # # K/Hz

# Solar luminosity in erg/s
L_sun_ergs = 3.828e26 * 1e7

# Observed CMB temperature
T_CMB_obs = 2.725 # K

def CMB_heating(z, T_dust_z0, beta_IR, T_CMB_obs=T_CMB_obs):
    # Convert a dust temperature at z = 0 to one affected by CMB heating at higher redshift
    bp4 = beta_IR + 4.0

    T_dust = ( (T_dust_z0)**bp4 + T_CMB_obs**bp4 * ((1.0 + z)**bp4 - 1.0) )**(1.0/bp4)

    return T_dust

def inv_CMB_heating(z, T_dust, beta_IR, T_CMB_obs=T_CMB_obs):
    # Convert a dust temperature at redshift z to one at redshift z = 0
    bp4 = beta_IR + 4.0

    if np.any(T_CMB_obs**bp4 * ((1.0 + z)**bp4 - 1.0) > T_dust**bp4):
        raise ValueError("dust temperature " + ("({:.3g} K) ".format(T_dust) if isinstance(T_dust, float) else '') + "too low in comparison to " + \
                            "that of the CMB at z = {:.3g} ({:.3g} K)".format(z, T_CMB_obs * (1.0 + z)))

    T_dust_z0 = ( T_dust**bp4 - T_CMB_obs**bp4 * ((1.0 + z)**bp4 - 1.0) )**(1.0/bp4)

    return T_dust_z0

def CMB_correction(z, nu0_emit, T_dust, T_CMB_obs=T_CMB_obs, verbose=False):
    # Convert an observed flux density to the intrinsic value, correcting for observing it against the CMB
    T_CMB = T_CMB_obs * (1.0 + z) # K

    # There is a nu0_emit^3 in the numerator, but this cancels out when taking the ratio
    F_nu_BB = 1.0 / (np.exp(h_kB * nu0_emit / T_dust) - 1.0)
    F_nu_BB_CMB = 1.0 / (np.exp(h_kB * nu0_emit / T_CMB) - 1.0)
    if verbose:
        print("CMB correction (T_CMB = {:.1f} K at z = {:.2f}): {:.2f}%".format(T_CMB, z, 100.0 / (F_nu_BB / F_nu_BB_CMB - 1.0)))

    # Correction factor: intrinsic flux is higher by this factor, i.e. F_obs = F_int * corr
    # but note corr < 1 and if T_CMB > T_dust, corr < 0 and so F_obs < 0
    CMB_correction_factor = (1.0 - F_nu_BB_CMB / F_nu_BB)

    return CMB_correction_factor

def Planck_func(nu, T):
    # Return B_nu(nu, T) in W/m^2/Hz
    return 2.0 * 6.626e-34 * nu**3 / 299792458.0**2 / (np.exp(6.626e-34 * nu / (1.380649e-23 * T)) - 1.0)

def calc_FIR_SED(z, cont_flux=None, lambda0_emit=None,
                    T_CMB_obs=T_CMB_obs, beta_IR=1.5, T_dust=50.0, optically_thick_lambda_0=None,
                    lambda_min=8.0, lambda_max=1000.0, cosmo=None,
                    return_spectrum=False, lambda_emit=None, verbose=True):
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70.0, Om0=0.300)
    else:
        assert isinstance(cosmo, FLRW)
    
    if not return_spectrum:
        assert lambda0_emit is not None and cont_flux is not None

        nu0_emit = c / lambda0_emit # Hz

        if verbose:
            print("\nCalculating IR luminosity of a source at redshift {:.2f} with observed continuum flux density {:.2f} μJy".format(z, cont_flux*1e6))
            print("Assuming a dust emissivity beta of {:.3g} and dust temperature of {:.3g} K...\n".format(beta_IR, T_dust))

        CMB_correction_factor = CMB_correction(z, nu0_emit, T_dust, T_CMB_obs, verbose)
        cont_flux_corr = cont_flux / CMB_correction_factor

    # Set up wavelength/frequency range for the theoretically expected greybody emission
    if lambda_emit is None or not return_spectrum:
        lambda_emit = np.linspace(0.5*lambda_min, 1.1*lambda_max, 10000) # micron
    
    lambda_obs = lambda_emit * (1.0 + z) # micron
    nu_emit = c / lambda_emit # Hz

    # Calculate the observed dust continuum flux density from the theoretically expected greybody emission (e.g. Casey, 2012, MNRAS, 425, 3094);
    # the observed flux density should be scaled by (1+z) but can simply be normalised by the measurement at lambda0_emit
    if optically_thick_lambda_0 is None:
        # Assume the optical depth is small τ << 1, so that (1 - e^-τ) becomes τ ~ nu^beta
        S_nu_emit = nu_emit**(beta_IR+3.0) / (np.exp(h_kB * nu_emit / T_dust) - 1.0)
    elif np.isinf(optically_thick_lambda_0):
        # Fully optically thick spectrum: perfect blackbody
        S_nu_emit = nu_emit**3.0 / (np.exp(h_kB * nu_emit / T_dust) - 1.0)
    else:
        # Do not assume anything about the optical depth by taking the full term (1 - e^-τ) with τ = (lambda/lambda_0)^beta (lambda_0 is where τ=1)
        S_nu_emit = (1.0 - np.exp(-(optically_thick_lambda_0/lambda_emit)**beta_IR)) * nu_emit**3.0 / (np.exp(h_kB * nu_emit / T_dust) - 1.0)

    if return_spectrum:
        # Return emission spectrum (without normalisation)
        return (lambda_emit, S_nu_emit)
    
    # Scale both the observed flux density and the (rest-frame) emitted flux density
    S_nu_obs = S_nu_emit * cont_flux_corr / np.interp(lambda0_emit, lambda_emit, S_nu_emit) # Jy
    S_nu_emit *= (cont_flux_corr / (1.0+z)) / np.interp(lambda0_emit, lambda_emit, S_nu_emit) # Jy
    S_lambda_obs = S_nu_obs * c / lambda_obs**2 * 1e-23 # erg/s/cm^2/micron

    # Integrate observed flux density within infrared luminosity wavelength boundaries;
    # note that since flux is conserved, integrating over emitted or observed
    # flux density is equivalent (F_lambda and lambda differ by factor of (1+z) and 1/(1+z))
    lambda_IR = (lambda_emit > lambda_min) * (lambda_emit < lambda_max)
    F_IR = np.trapz(S_lambda_obs[lambda_IR], x=lambda_obs[lambda_IR]) # erg/s/cm^2

    # Convert flux to (solar) luminosity
    D_L = cosmo.luminosity_distance(z).to("cm").value

    L_IR = F_IR * 4.0 * np.pi * D_L**2 # erg/s
    L_IR_Lsun = L_IR / L_sun_ergs # L_sun

    return (L_IR, L_IR_Lsun)