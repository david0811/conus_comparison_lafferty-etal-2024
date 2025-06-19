# conus_comparison_lafferty-etal-2025

**Varying sources of uncertainty in risk-relevant hazard projections across the United States**

*David C. Lafferty<sup>1\*</sup>, Samantha H. Hartke<sup>2,3</sup>, Ryan L. Sriver<sup>4</sup>, Andrew J. Newman<sup>2</sup>, Ethan D. Gutmann<sup>2</sup>, Flavio Lehner<sup>2,5</sup>, Paul A. Ullrich <sup>6,7</sup>, Vivek Srikrishnan<sup>1</sup>*

<sup>1 </sup>Department of Biological \& Environmental Engineering, Cornell University\
<sup>2 </sup>NSF National Center for Atmospheric Research\
<sup>3 </sup>U.S. Army Corps of Engineers\
<sup>4 </sup>Department of Climate, Meteorology \& Atmospheric Sciences, University of Illinois
<sup>5 </sup>Department of Earth \& Atmospheric Sciences, Cornell University
<sup>6 </sup>Division of Physical and Life Sciences, Lawrence Livermore National Laboratory
<sup>7 </sup>Department of Land, Air, and Water Resources, University of California, Davis

\* corresponding author:  `dcl257@cornell.edu`\

## Abstract
Physical climate risk assessment requires understanding how different sources of uncertainty affect hazard projections, yet these uncertainties can manifest differently across use-cases. Here, we combine three state-of-the-art downscaled climate ensembles to characterize how different uncertainties affect projections of several temperature- and precipitation-based risk metrics across the contiguous United States. We focus on long-term trends of aggregate indices as well as the intensity of rare events with 10- to 100-year return periods. By leveraging new downscaled initial condition ensembles, we characterize the role of internal variability at local scales and estimate its importance relative to other sources of uncertainty. Our results demonstrate systematic differences in patterns of uncertainty between average and extreme indices, across recurrence intervals, and between temperature- and precipitation-derived variables. We show that temperature metrics are more sensitive to the choice of emissions scenario and Earth system model, while internal variability is often dominant for precipitation-based metrics. Additionally, we find that the statistical uncertainty from extreme value distribution fitting can often exceed climate-related uncertainties, particularly at recurrence intervals of 50 years or longer. Our results can provide guidance for researchers and practitioners conducting climate risk assessment.

## Journal reference
TBD

## Code reference
TBD

## Data reference

### Input data
| Dataset | Data download link | Reference | Notes |
|---------|------|-----|-------|
| LOCA2 | https://loca.ucsd.edu/ | https://doi.org/10.1175/JHM-D-22-0194.1 |  |
| GARD-LENS | https://oidc.rda.ucar.edu/datasets/d619000/ | https://doi.org/10.1038/s41597-024-04205-z | - | 
| STAR-ESDM | TBD | https://doi.org/10.1029/2023EF004107 | - |
| Livneh-unsplit | https://cirrus.ucsd.edu/~pierce/nonsplit_precip/ | https://doi.org/10.1175/JHM-D-20-0212.1 | Training data for LOCA2, used in SI figures only. |
| GMET | https://www.earthsystemgrid.org/dataset/gridded_precip_and_temp.html | https://doi.org/10.1175/JHM-D-15-0026.1 | Training data for GARD-LENS, used in SI figures only. |
| NClimGrid-Daily | https://www.ncei.noaa.gov/products/land-based-station/nclimgrid-daily | https://doi.org/10.1175/JTECH-D-22-0024.1 | Training data for STAR-ESDM, used in SI figures only. |

### Output data
TBD

## Reproduce my experiment
Project dependencies are specified in `pyproject.toml`. You can clone this directory and install via pip by running `pip install -e .` from the root directory. You'll also need to download all of the input data sets and update the appropriate paths in `src/utils.py`.

The following scripts can then be used to reproduce the experiment:

| Script | Description |
|--------|-------------|
| 01a_metrics_star-esdm.ipynb | Calculates metrics for STAR-ESDM. |
| 01b_metrics_loca2.ipynb | Calculates metrics for LOCA2. |
| 01c_metrics_gard-lens.ipynb | Calculates metrics for GARD-LENS. |
| 01d_metrics_tgw.ipynb | Calculates metrics for TGW. |
| 01e_metrics_obs.ipynb | Calculates metrics for observations. |
| 02a_eva_nonstat.sh | Fits the non-stationary GEV to all datasets. |
| 02b_eva_stat.ipynb | Fits the stationary GEV to all datasets. |
| 02c_trends.ipynb | Calculates trends for all datasets. |
| 02d_averages.ipynb | Calculates averages for all datasets. |
| 02e_cities.ipynb | Performs EVA and trend fitting for city locations. |
| 03a_sa-eva.ipynb | Performs sensitivity analysis for EVA. |
| 03b_sa-trends.ipynb | Performs sensitivity analysis for trends. |
| 03c_sa-averages.ipynb | Performs sensitivity analysis for averages. |
| 03d_sa_cities.ipynb | Performs sensitivity analysis for city-level analyses. |
| 99_testing.ipynb | Testing and validation scripts for GEV fitting and coverage analysis. |
| 99_other_checks.ipynb | Additional validation and quality control checks. |
