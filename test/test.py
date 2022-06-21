#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for testing MERCURIUS.

Joris Witstok, 23 December 2021
"""

import os, sys
import numpy as np

sys.path.append("..")
from mercurius import FIR_SED_fit

mnrfol = "MN_results/"
if not os.path.exists(mnrfol):
    os.makedirs(mnrfol)
pltfol = "MN_plots/"
if not os.path.exists(pltfol):
    os.makedirs(pltfol)

f = FIR_SED_fit(l0_list=[None], analysis=True, mnrfol=mnrfol)
f.set_data(obj="A1689-zD1", z=7.13,
            lambda_emit_vals=[53.04702183893297, 89.45738131653835, 107.22593320533173, 163.04962431996177, 402.5],
            S_nu_vals=[155.10381273644117e-6, 178.9438898238026e-6, 146.97017233573683e-6, 59.69243176098955e-6, 3*4.046324352847791e-6],
            S_nu_errs=[71.30750887192343e-6, 19.627630566036796e-6, 19.129462212757556e-6, 6.327415246314415e-6, np.nan],
            cont_uplims=[False, False, False, False, True],
            reference="Watson et al. (2015); Knudsen et al. (2017);\nInoue et al. (2020); Bakx et al. (2021); Akins et al. (2022)")

for l0 in f.l0_list:
    f.plot_ranges(l0=l0, save_results=False, pltfol=pltfol)

f.fit_data(pltfol=pltfol, fit_uplims=True, n_live_points=2000, evidence_tolerance=0.001,
            force_run=False, mnverbose=False)
f.plot_MN_fit(pltfol=pltfol)