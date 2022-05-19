#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for testing MERCURIUS.

Joris Witstok, 23 December 2021
"""

import os, sys
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
            lambda_emit_vals=[1.33e3/(1.0+7.13), 0.873e3/(1.0+7.13), 0.728e3/(1.0+7.13), 0.427e3/(1.0+7.13)],
            S_nu_vals=[60e-6, 143e-6, 180e-6, 154e-6],
            S_nu_errs=[11e-6, 15e-6, 39e-6, 37e-6],
            cont_uplims=[False, False, False, False],
            reference="Knudsen et al. (2017);\nInoue et al. (2020); Bakx et al. (2021)")

for l0 in f.l0_list:
    f.plot_ranges(l0=l0, save_results=False, pltfol=pltfol)

f.fit_data(pltfol=pltfol, fit_uplims=True, n_live_points=2000, evidence_tolerance=0.001,
            force_run=False, mnverbose=False)
f.plot_MN_fit(pltfol=pltfol)