# -*- coding: utf-8 -*-
"""
Script for storing functions that allow conversions between luminosities and star formation rates (SFRs).

Joris Witstok, 5 February 2021
"""

import numpy as np

# SFR conversions from Kennicutt & Evans (2012)
bands = ["FUV", "NUV", "H_alpha", "TIR", "24micron", "70micron", "1.4GHz", "2-10keV"]
ages = {"FUV": (0, 10, 100), "NUV": (0, 10, 200), "H_alpha": (0, 3, 10), "TIR": (0, 5, 100),
        "24micron": (0, 5, 100), "70micron": (0, 5, 100), "1.4GHz": (0, 100, np.nan), "2-10keV": (0, 100, np.nan)}
units = {"FUV": "erg/s (nuLnu)", "NUV": "erg/s (nuLnu)", "H_alpha": "erg/s", "TIR": "erg/s (3–1100 micron)",
        "24micron": "erg/s (nuLnu)", "70micron": "erg/s (nuLnu)", "1.4GHz": "erg/s/Hz", "2-10keV": "erg/s"}
logC_xs = {"FUV": 43.35, "NUV": 43.17, "H_alpha": 41.27, "TIR": 43.41,
        "24micron": 42.69, "70micron": 43.23, "1.4GHz": 28.20, "2-10keV": 39.77}
references = {"FUV": "Hao et al. (2011), Murphy et al. (2011)", "NUV": "Hao et al. (2011), Murphy et al. (2011)",
                "H_alpha": "Hao et al. (2011), Murphy et al. (2011)", "TIR": "Hao et al. (2011), Murphy et al. (2011)",
                "24micron": "Rieke et al. (2009)", "70micron": "Calzetti et al. (2010b)",
                "1.4GHz": "Murphy et al. (2011)", "2-10keV": "Ranalli et al. (2003)"}

def SFR_L(L, band="FUV"):
    """

    SFR conversions from Kennicutt & Evans (2012):
    
    log(SFR) = log(L_x) + log(C_x)
    
    with SFR in M_sun/year

    Bands:
    FUV: far-ultraviolet
    NUV: near-ultraviolet
    H_alpha
    TIR: total infrared
    24 micron
    70 micron
    1.4 GHz
    2-10 keV

    Returns SFR in M_sun/year

    """

    assert band in bands

    logL = np.log10(L)
    logC_x = logC_xs[band]
    
    logSFR = logL - logC_x
    
    SFR = 10**logSFR

    return SFR