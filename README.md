# MERCURIUS
<br>
<img src="/aux/MERCURIUS.png" width="100%">
<br>

## Contents
1. [Description](#Description)
2. [Installation and setup](#Installation)
    - [Cloning](#Cloning)
    - [Package requirements](#Package_requirements)
3. [Example usage](#Example_usage)

## <a name="Description"></a>Description

Multimodal Estimation Routine for the Cosmological Unravelling of Rest-frame Infrared Uniformised Spectra (MERCURIUS) uses the `pymultinest` package ([Feroz et al. 2009](https://ui.adsabs.harvard.edu/abs/2009MNRAS.398.1601F/abstract); [Buchner et al. 2014](https://ui.adsabs.harvard.edu/abs/2014A%26A...564A.125B/abstract)) to fit and plot greybody spectra given far-infrared (FIR) photometry of an object. Below, its usage is illustrated with an example.

## <a name="Installation"></a>Installation and setup

### <a name="Cloning"></a>Cloning

First, obtain the latest version of the `mercurius` code. For example, you can clone the repository by navigating to your desired installation folder and using

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
  
One way to ensure all these modules are installed is via the file `mercurius3.yml` provided in the main folder, which can be used to create an `conda` environment in Python 3 (see the `conda` [documentation on environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details) containing all the required packages.

If you have `conda` installed and set up as a `python` distribution, this can be achieved with:

```
conda env create -f mercurius3.yml
```

Before running the code, the environment needs to be activated using

```
conda activate mercurius3
```

By default, the terminal will indicate the environment is active by showing a prompt similar to:

```
(mercurius3) $ 
```

## <a name="Example_usage"></a>Example usage

### <a name="Running_the_test_script"></a>Running the test script

This section goes through an example usage case of `mercurius` by running the file `test.py` (located in the `test` folder). The first step is to activate the environment as explained in [the previous section](#Package_requirements). So, starting from the main folder, the script would be run as follows:

```
$ conda activate mercurius3
(mercurius3) $ cd test/
(mercurius3) $ python test.py
```

While the code is running, this should produce output in the terminal informing the user about the input choices and photometry as well as the progress of the fitting procedure. If it has finished successfully, several figures will have been saved in the `MN_plots` folder. It will also have created data files containing results of the fitting routine (in `MN_results`) that can be loaded in subsequent runs to speed up the code. Below, the functionality is explained step by step.

### <a name="Understanding_the_code_usage"></a>Understanding the code's usage

#### <a name="Initialisation"></a>Initialisation

The main functionality is accessed through an `FIR_SED_fit` object, which can be imported after the main folder has been added to the `PYTHONPATH`. We also create two subfolders for saving the results and figures created in the next steps.

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

#### <a name="Specifying_input_photometry"></a>Specifying input photometry

Now, a source and its corresponding FIR photometry can be specified through the function `set_data`. In this example, we consider the star-forming galaxy A1689-zD1 at a redshift of 7.13 (e.g. [Watson et al. 2015](https://ui.adsabs.harvard.edu/abs/2015Natur.519..327W/abstract)). This source has been detected in four ALMA bands, whose upper limits (`cont_uplims`) are set to `False`, while the band-4 upper limit is set to `True`:

```python
f.set_data(obj="A1689-zD1", z=7.13,
            lambda_emit_vals=[53.04702183893297, 89.45738131653835, 107.22593320533173, 163.04962431996177, 402.5],
            S_nu_vals=[155.10381273644117e-6, 178.9438898238026e-6, 146.97017233573683e-6, 59.69243176098955e-6, 3*4.046324352847791e-6],
            S_nu_errs=[71.30750887192343e-6, 19.627630566036796e-6, 19.129462212757556e-6, 6.327415246314415e-6, np.nan],
            cont_uplims=[False, False, False, False, True],
            reference="Watson et al. (2015); Knudsen et al. (2017);\nInoue et al. (2020); Bakx et al. (2021); Akins et al. (2022)")
```

Here, we have not set `cont_area` (precluding the use of a self-consistent opacity model, identified via `"self-consistent"` entry in `l0_list`). Instead, we only look at an entirely optically thin SED (hence `l0_list = [None]` in the first step). It is possible, however, to consider a fixed value of `lambda_0`, without specifying `cont_area`.

#### <a name="First_look_of_the_data"></a>First look of the data

For a first look, the `plot_ranges` function creates a figure with a range of dust temperatures and emissivities. Note that if `save_results=True`, this will also print and save the results for a given temperature and emissivity, specified with `fixed_T_dust` and `fixed_beta` in calling `plot_ranges` or earlier in the creation of the `FIR_SED_fit` instance.

```python
for l0 in f.l0_list:
    f.plot_ranges(l0=l0, save_results=False, pltfol=pltfol)
```

This creates the figure below, in which the various coloured greybody curves indicate that out of the default range of dust temperatures (starting at 20 K, increasing in steps of 10 K up to 110 K; note this can be adjusted in the call to `plot_ranges` with the optional keyword `T_dusts`), only 40 K and 50 K are compatible with the photometric data points for certain dust emissivities (similarly specified by the optional keyword `beta_IRs`, by default a range between 1.5 and 2 with steps of 0.1):
<br>
<img src="/test/MN_plots/FIR_SED_ranges_A1689-zD1_analysis.png" width="800">
<br>

#### <a name="Executing_the_fitting_routine"></a>Executing the fitting routine

A MultiNest fit for each the opacity models in `f.l0_list` can be initiated with `fit_data` (although `fit_uplims=True` here, there are no upper limits to be taken into account):

```python
f.fit_data(pltfol=pltfol, fit_uplims=True, n_live_points=2000, force_run=False, mnverbose=False)
```

Since `pltfol` is specified in the call to `fit_data`, the posterior distributions will be shown in a figure produced with the `corner` module:
<br>
<img src="/test/MN_plots/Corner_MN_A1689-zD1.png" width="800">
<br>

#### <a name="Plotting_results_of_the_fitting_routine"></a>Plotting results of the fitting routine

Finally, the results of the fitting routine are visualised with `plot_MN_fit`:

```python
f.plot_MN_fit(pltfol=pltfol)
```

This produces the figure shown below:
<br>
<img src="/test/MN_plots/FIR_SED_MN_fit_A1689-zD1_analysis.png" width="800">
<br>