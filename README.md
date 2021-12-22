# mercurius

## Description

Multimodal Estimation Routine for the Cosmological Unravelment of Rest-frame Infrared Uniformised Spectra (MERCURIUS) uses the `pymultinest` package ([Feroz et al. 2009](https://ui.adsabs.harvard.edu/abs/2009MNRAS.398.1601F/abstract); [Buchner et al. 2014](https://ui.adsabs.harvard.edu/abs/2014A%26A...564A.125B/abstract)) to fit and plot greybody spectra given far-infrared (FIR) photometry of an object. Below, its usage is illustrated with an example.

## Example usage

```python
from mercurius import FIR_SED_fit

fixed_beta = 2.0

f = FIR_SED_fit(l0_list=[None, "self-consistent"], analysis=True, mcrfol="MN_results/", fixed_beta=fixed_beta)
f.set_data(obj="A1689-zD1", z=7.13,
            lambda_emit_vals=[1.33e3/(1.0+7.13), 0.873e3/(1.0+7.13), 0.728e3/(1.0+7.13), 0.427e3/(1.0+7.13)],
            S_nu_vals=[60e-6, 143e-6, 180e-6, 154e-6], S_nu_errs=[11e-6, 15e-6, 39e-6, 37e-6], cont_uplims=[False, False, False, False], reference="Knudsen et al. (2017);\nInoue et al. (2020); Bakx et al. (2021)")

for l0 in l0_list:
    if l0 == "self-consistent":
        continue

    f.plot_ranges(l0=l0, fixed_beta=fixed_beta, save_results=False, pltfol="MN_plots/")

f.fit_data(pltfol="MN_plots/", force_run=force_run,
            fit_uplims=True, n_live_points=2000, evidence_tolerance=0.01, mnverbose=False)

# Plot MultiNest fit results in various plots (individually and in overview figures)
f.plot_MN_fit(pltfol=spltfol)
```
