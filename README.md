# MERCURIUS

## Contents
1. [Description](#Description)
2. [Installation and setup](#Installation)
    - [Cloning](#Cloning)
    - [Package requirements](#Package_requirements)
3. [Example usage](#Example_usage)

## <a name="Description"></a>Description

Multimodal Estimation Routine for the Cosmological Unravelment of Rest-frame Infrared Uniformised Spectra (MERCURIUS) uses the `pymultinest` package ([Feroz et al. 2009](https://ui.adsabs.harvard.edu/abs/2009MNRAS.398.1601F/abstract); [Buchner et al. 2014](https://ui.adsabs.harvard.edu/abs/2014A%26A...564A.125B/abstract)) to fit and plot greybody spectra given far-infrared (FIR) photometry of an object. Below, its usage is illustrated with an example.

## <a name="Installation"></a>Installation and setup

### <a name="Cloning"></a>Cloning

First, clone the repository using e.g.

```
git clone https://github.com/joriswitstok/mercurius.git
```

### <a name="Package_requirements"></a>Package requirements

Running `mercurius` requires the following Python packages:
- `numpy`
- `scipy`
- `astropy`
- `emcee`
- `pymultinest`
- `corner`
- `matplotlib`
- `seaborn`
- `mock`
  
The file `mercurius3.yml` can be used to create an `conda` environment in Python 3 (see the `conda` [documentation on environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details) containing all the required modules:

```
conda env create -f mercurius3.yml
```

Before running the code, activate the environment using

```
conda activate mercurius3
```

## <a name="Example_usage"></a>Example usage

This section runs through an example usage case of `mercurius`. The main functionality is accessed through an `FIR_SED_fit` object, which can be imported after the main folder has been added to the `PYTHONPATH`. We also create two subfolders for saving the results and figures created in the next steps.

```python
import sys
sys.path.append("..")
from mercurius import FIR_SED_fit

mcrfol = "MN_results/"
if not os.path.exists(mcrfol):
    os.makedirs(mcrfol)
pltfol = "MN_plots/"
if not os.path.exists(pltfol):
    os.makedirs(pltfol)

f = FIR_SED_fit(l0_list=[None], analysis=True, mcrfol=mcrfol)
```

The variable `l0_list` is a list of opacity model classifiers, where entries can be `None` for an optically thin model, `"self-consistent"` for a self-consistent opacity model (which requires `cont_area`, the deconvolved area of the dust emission in square kiloparsec, to be given in the next step), or a float setting a fixed value of `lambda_0`, the wavelength in micron setting the SED's transition point between optically thin and thick.

Now, a source and its corresponding FIR photometry can be specified through the function `set_data`. In this example, we consider the star-forming galaxy A1689-zD1 at a redshift of 7.13 (e.g. [Bakx et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508L..58B/abstract)). This source has been detected in four ALMA bands, so all upper limits (`cont_uplims`) are set to `False`:

```python
f.set_data(obj="A1689-zD1", z=7.13,
            lambda_emit_vals=[1.33e3/(1.0+7.13), 0.873e3/(1.0+7.13), 0.728e3/(1.0+7.13), 0.427e3/(1.0+7.13)],
            S_nu_vals=[60e-6, 143e-6, 180e-6, 154e-6], S_nu_errs=[11e-6, 15e-6, 39e-6, 37e-6], cont_uplims=[False, False, False, False], reference="Knudsen et al. (2017);\nInoue et al. (2020); Bakx et al. (2021)")
```

Here, we have not set `cont_area` (precluding the use of a self-consistent opacity model, identified via `"self-consistent"` entry in `l0_list`). Instead, we only look at an entirely optically thin SED (hence `l0_list = [None]` in the first step). It is possible, however, to consider a fixed value of `lambda_0`, without specifying `cont_area`.

For a first look, the `plot_ranges` function creates a figure with a range of dust temperatures and emissivities. Note that if `save_results=True`, this will also print and save the results for a given temperature and emissivity, specified with `fixed_T_dust` and `fixed_beta` in calling `plot_ranges` or earlier in the creation of the `FIR_SED_fit` instance.

```python
for l0 in f.l0_list:
    f.plot_ranges(l0=l0, save_results=False, pltfol=pltfol)
```

A MultiNest fit for each the opacity models in `f.l0_list` can be initiated with `fit_data` (although `fit_uplims=True` here, there are no upper limits to be taken into account):

```python
f.fit_data(pltfol=pltfol, force_run=False, fit_uplims=True, remove_mnfiles=True,
            n_live_points=2000, evidence_tolerance=0.001, mnverbose=False)
```

Finally, the results of the fitting routine are visualised with `plot_MN_fit`:

```python
f.plot_MN_fit(pltfol=pltfol)
```
